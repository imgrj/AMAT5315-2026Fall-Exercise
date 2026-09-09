//! End-to-end checks of the md CLI operations.

use std::path::PathBuf;

use md::cli::{CheckCfg, RunCfg, VideoCfg};
use md::ops::run_sim;
use md::trajectory::read;

fn run_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(name);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Run a default simulation into `dir` unless one is already there, so tests
/// that inspect the output never race with the test that created it.
fn ensure_run(dir: &PathBuf) {
    if !dir.join("trajectory.txt").exists() {
        run_sim(&run_cfg(dir.to_str().unwrap())).unwrap();
    }
}

fn run_cfg(out: &str) -> RunCfg {
    RunCfg {
        force: md::ForceStrategy::Cells,
        n: 100,
        temp: 1.0,
        dt: 0.005,
        steps: 500,
        equil: 100,
        sample_every: 1,
        ramp_to: None,
        out: out.to_string(),
        seed: 1,
    }
}

fn check_cfg(path: &str) -> CheckCfg {
    CheckCfg {
        file: path.into(),
        force: md::ForceStrategy::Cells,
        temp_tol: 0.05,
        drift_tol: 1e-3,
        ks_tol: 0.05,
    }
}

fn mean_t(frame: &[[f64; 4]]) -> f64 {
    frame.iter().map(|a| a[2] * a[2] + a[3] * a[3]).sum::<f64>() / 200.0
}

#[test]
fn run_writes_a_checkable_trajectory() {
    let dir = run_dir("md_cli_run_test");
    let out = dir.to_str().unwrap().to_string();
    run_sim(&run_cfg(&out)).unwrap();
    let traj = dir.join("trajectory.txt");
    let t = read(traj.to_str().unwrap()).unwrap();
    assert_eq!(t.meta.n, 100);
    assert_eq!(t.frames.len(), 500);
    let mean: f64 = t.frames.iter().map(|f| mean_t(f)).sum::<f64>() / t.frames.len() as f64;
    assert!((mean - 1.0).abs() < 0.05, "mean T {mean}");
}

#[test]
fn check_passes_on_our_own_run() {
    let dir = run_dir("md_cli_check_test");
    ensure_run(&dir);
    let report = md::ops::check(&check_cfg(dir.to_str().unwrap())).unwrap();
    assert!(report.pass, "report {report:?}");
}

#[test]
fn check_rejects_doctored_temperature() {
    // Re-scale all velocities by 1.5: mean temperature rises ~2.25x, out of tolerance.
    let dir = run_dir("md_cli_doctored");
    ensure_run(&dir);
    let mut t = read(dir.join("trajectory.txt").to_str().unwrap()).unwrap();
    for frame in &mut t.frames {
        for a in frame {
            a[2] *= 1.5;
            a[3] *= 1.5;
        }
    }
    let path = dir.join("doctored.txt");
    md::trajectory::write(path.to_str().unwrap(), &t).unwrap();
    let report = md::ops::check(&check_cfg(path.to_str().unwrap())).unwrap();
    assert!(!report.pass);
}

#[test]
fn energy_of_frames_is_consistent() {
    let dir = run_dir("md_cli_energy_test");
    ensure_run(&dir);
    let report = md::ops::check(&check_cfg(dir.to_str().unwrap())).unwrap();
    assert!(report.drift < 1e-3, "drift {}", report.drift);
}

#[test]
fn frames_render_as_ppm() {
    let dir = run_dir("md_cli_frames_test");
    ensure_run(&dir);
    let t = read(dir.join("trajectory.txt").to_str().unwrap()).unwrap();
    let frame = &t.frames[0];
    let positions: Vec<[f64; 2]> = frame.iter().map(|a| [a[0], a[1]]).collect();
    let ppm = md::video::render_frame(t.meta.box_l, &positions, &[], 450);
    const HEADER: &[u8] = b"P6\n900 450 255\n";
    assert_eq!(&ppm[..HEADER.len()], HEADER);
    assert_eq!(ppm.len(), HEADER.len() + 900 * 450 * 3);
    let a = frame[0];
    let scale = 405.0 / t.meta.box_l;
    let px = 22.5 + a[0] * scale;
    let py = 22.5 + a[1] * scale;
    let idx = HEADER.len() + (py as usize) * 900 * 3 + (px as usize) * 3;
    assert!(ppm[idx] < 250 || ppm[idx + 1] < 250 || ppm[idx + 2] < 250);
}

#[test]
fn render_wraps_positions_into_the_box() {
    let dir = run_dir("md_cli_wrap_test");
    ensure_run(&dir);
    let t = read(dir.join("trajectory.txt").to_str().unwrap()).unwrap();
    let frame = &t.frames[0];
    let box_l = t.meta.box_l;
    let positions: Vec<[f64; 2]> = frame.iter().map(|a| [a[0], a[1]]).collect();
    let shifted: Vec<[f64; 2]> = frame
        .iter()
        .map(|a| [a[0] + box_l, a[1] + 2.0 * box_l])
        .collect();
    let a = md::video::render_frame(box_l, &positions, &[], 300);
    let b = md::video::render_frame(box_l, &shifted, &[], 300);
    assert_eq!(a, b);
}

#[test]
fn rdf_gets_its_own_panel() {
    let box_l = 10.0;
    let positions = vec![[1.0, 5.0], [4.0, 5.0]];
    let curve = vec![(1.0, 2.0), (2.0, 1.5), (3.0, 1.0)];
    let ppm = md::video::render_frame(box_l, &positions, &curve, 300);
    let is_blue = |i: usize| ppm[i] < 100 && ppm[i + 2] > 180;
    let (mut left_blue, mut right_blue) = (0, 0);
    for y in 0..300 {
        for x in 0..300 {
            let o = (y * 600 + x) * 3;
            left_blue += is_blue(o) as usize;
        }
        for x in 300..600 {
            let o = (y * 600 + x) * 3;
            right_blue += is_blue(o) as usize;
        }
    }
    assert_eq!(left_blue, 0, "left panel must not contain the RDF");
    assert!(right_blue > 0, "right panel should show the RDF curve");
}

#[test]
fn make_video_produces_an_mp4() {
    let dir = run_dir("md_cli_video_test");
    ensure_run(&dir);
    let out = std::env::temp_dir().join("md_cli_video_test.mp4");
    let cfg = VideoCfg {
        file: dir.to_str().unwrap().into(),
        out: out.to_str().unwrap().into(),
        fps: 10.0,
    };
    md::ops::make_video(&cfg).unwrap();
    let meta = std::fs::metadata(&out).unwrap();
    assert!(meta.len() > 1000, "mp4 too small: {}", meta.len());
}

#[test]
fn sample_every_thins_frames() {
    let dir = run_dir("md_cli_sample_test");
    let mut cfg = run_cfg(dir.to_str().unwrap());
    cfg.steps = 500;
    cfg.sample_every = 100;
    run_sim(&cfg).unwrap();
    let t = read(dir.join("trajectory.txt").to_str().unwrap()).unwrap();
    assert_eq!(t.frames.len(), 5);
    assert_eq!(t.meta.sample_every, 100);
}

#[test]
fn ramp_to_heats_along_schedule_and_is_recorded() {
    let dir = run_dir("md_ramp_test");
    let out = dir.join("ramp_out");
    let mut cfg = run_cfg(out.to_str().unwrap());
    cfg.steps = 200;
    cfg.equil = 10;
    cfg.ramp_to = Some(2.0);
    run_sim(&cfg).unwrap();
    let t = read(out.join("trajectory.txt").to_str().unwrap()).unwrap();
    assert_eq!(t.frames.len(), 200);
    // Rescaling every recorded step makes the instantaneous temperature equal
    // the ramp target: frame k sits after step k+1, fraction (k+1)/steps.
    let first = mean_t(&t.frames[0]);
    let target_first = 1.0 + (2.0 - 1.0) * (1.0 / 200.0);
    assert!((first - target_first).abs() < 1e-6, "first T {first} vs {target_first}");
    let last = mean_t(&t.frames[199]);
    assert!((last - 2.0).abs() < 1e-6, "last T {last} vs 2.0");
    // ramp_to recorded in run.json inside the output directory.
    let text = std::fs::read_to_string(out.join("run.json")).unwrap();
    assert!(text.contains("ramp_to"), "run.json: {text}");
    assert!(text.contains("2"), "run.json: {text}");
}
