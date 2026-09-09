//! End-to-end tests of the public run/check artifact contract.

use std::path::PathBuf;

use md::cli::{CheckCfg, RunCfg};
use md::trajectory::{self, TRAJECTORY_FILE};

fn temp_dir(tag: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("md-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    std::fs::create_dir_all(&path).unwrap();
    path
}

fn short_cfg(tag: &str) -> RunCfg {
    RunCfg {
        n: 36,
        eq_steps: 100,
        steps: 100,
        sample_every: 10,
        out: temp_dir(tag).to_string_lossy().into_owned(),
        ..RunCfg::default()
    }
}

#[test]
fn binary_writes_readable_run_json_and_jsonl_frames() {
    let cfg = short_cfg("binary-output");
    let executable = env!("CARGO_BIN_EXE_md");
    let status = std::process::Command::new(executable)
        .args([
            "run",
            "--n",
            "36",
            "--eq-steps",
            "100",
            "--steps",
            "100",
            "--sample-every",
            "10",
            "--out",
            &cfg.out,
        ])
        .status()
        .unwrap();
    assert!(status.success());
    let trajectory = trajectory::read(PathBuf::from(&cfg.out).as_path()).unwrap();
    assert_eq!(trajectory.frames.len(), 10);
    assert_eq!(trajectory.frames[0].step, 10);
    assert_eq!(trajectory.frames[9].step, 100);
    let raw = std::fs::read_to_string(PathBuf::from(&cfg.out).join(TRAJECTORY_FILE)).unwrap();
    assert_eq!(raw.lines().count(), 10);
    assert!(raw
        .lines()
        .all(|line| line.starts_with('{') && line.ends_with('}')));
}

#[test]
fn contract_run_passes_all_three_recomputed_physics_checks() {
    let mut cfg = RunCfg::default();
    cfg.out = temp_dir("contract").to_string_lossy().into_owned();
    md::ops::run_sim(&cfg).unwrap();
    let check_cfg = CheckCfg {
        input: cfg.out,
        drift_limit: 2e-3,
        temperature_limit: 0.05,
        shape_limit: 2.0,
    };
    let report = md::ops::check(&check_cfg).unwrap();
    assert!(report.pass, "{report:?}");
}

#[test]
fn stored_energy_tampering_is_rejected() {
    let cfg = short_cfg("tamper");
    md::ops::run_sim(&cfg).unwrap();
    let path = PathBuf::from(&cfg.out).join(TRAJECTORY_FILE);
    let mut lines: Vec<String> = std::fs::read_to_string(&path)
        .unwrap()
        .lines()
        .map(str::to_owned)
        .collect();
    let marker = "\"E_kin\":";
    let start = lines[0].find(marker).unwrap();
    lines[0].truncate(start);
    lines[0].push_str("\"E_kin\":12345}");
    std::fs::write(&path, lines.join("\n") + "\n").unwrap();
    let result = md::ops::check(&CheckCfg {
        input: cfg.out,
        drift_limit: 2e-3,
        temperature_limit: 0.05,
        shape_limit: 2.0,
    });
    assert!(result.is_err());
}

#[test]
fn heating_schedule_reaches_ramp_target_and_is_recorded() {
    let mut cfg = short_cfg("ramp");
    cfg.temperature = 0.2;
    cfg.ramp_to = Some(1.2);
    md::ops::run_sim(&cfg).unwrap();
    let trajectory = trajectory::read(PathBuf::from(&cfg.out).as_path()).unwrap();
    assert_eq!(trajectory.meta.ramp_to, Some(1.2));
    let thermostat_temperature = |frame: &md::trajectory::Frame| {
        let kinetic: f64 = frame.vel.iter().map(|v| 0.5 * (v[0] * v[0] + v[1] * v[1])).sum();
        kinetic / (frame.vel.len() as f64 - 1.0)
    };
    assert!((thermostat_temperature(trajectory.frames.last().unwrap()) - 1.2).abs() < 1e-12);
    assert!(thermostat_temperature(&trajectory.frames[0]) < thermostat_temperature(trajectory.frames.last().unwrap()));
}
