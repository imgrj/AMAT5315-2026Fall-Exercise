use std::fs::{self, File};
use std::io::BufWriter;
use std::path::PathBuf;

use seismic::experiment::Experiment;
use seismic::npy::{read_npy_f64, write_npy_f32, write_npy_f64};
use seismic::solver::{
    run_adjoint_shot_full, run_adjoint_shot_treeverse, run_born_shot, run_forward_shot,
};
use seismic::treeverse::build_treeverse_schedule;
use serde_json::json;

struct CliArgs {
    experiment: PathBuf,
    mode: String,
    every: Option<usize>,
    out: PathBuf,
    data: Option<PathBuf>,
    storage: String,
    checkpoints: Option<usize>,
}

fn parse_args() -> CliArgs {
    let args: Vec<String> = std::env::args().collect();
    let mut experiment = PathBuf::new();
    let mut mode = String::from("forward");
    let mut every = None;
    let mut out = PathBuf::from("artifacts");
    let mut data = None;
    let mut storage = String::from("full");
    let mut checkpoints = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--experiment" => {
                i += 1;
                experiment = PathBuf::from(&args[i]);
            }
            "--mode" => {
                i += 1;
                mode = args[i].clone();
            }
            "--every" => {
                i += 1;
                every = Some(args[i].parse().expect("valid integer for --every"));
            }
            "--out" => {
                i += 1;
                out = PathBuf::from(&args[i]);
            }
            "--data" => {
                i += 1;
                data = Some(PathBuf::from(&args[i]));
            }
            "--storage" => {
                i += 1;
                storage = args[i].clone();
            }
            "--checkpoints" => {
                i += 1;
                checkpoints = Some(args[i].parse().expect("valid integer for --checkpoints"));
            }
            _ => {}
        }
        i += 1;
    }

    CliArgs {
        experiment,
        mode,
        every,
        out,
        data,
        storage,
        checkpoints,
    }
}

fn l2_norm(slice: &[f64]) -> f64 {
    let sum_sq: f64 = slice.iter().map(|&x| x * x).sum();
    sum_sq.sqrt()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();
    let exp = Experiment::load(&args.experiment)?;
    fs::create_dir_all(&args.out)?;

    println!("shot\tmode\tdata L2 norm");

    let n_shots = exp.shots.len();
    let n_recvs = exp.receivers.len();
    let steps = exp.steps;

    match args.mode.as_str() {
        "forward" => {
            let mut all_traces = vec![0.0f64; n_shots * steps * n_recvs];
            let mut recorded_wavefield = None;
            let mut recorded_echo = None;
            let mut frame_steps = None;
            let mut frame_times = None;

            for shot_idx in 0..n_shots {
                let is_first = shot_idx == 0;
                let rec_opt = if is_first { args.every } else { None };

                let (traces, wf, f_steps, f_times) = run_forward_shot(&exp, &exp.background, shot_idx, rec_opt);

                let start_idx = shot_idx * steps * n_recvs;
                all_traces[start_idx..start_idx + steps * n_recvs].copy_from_slice(&traces);

                let shot_l2 = l2_norm(&traces);
                println!("{}\tforward\t{:.6}", shot_idx, shot_l2);

                if is_first && args.every.is_some() {
                    recorded_wavefield = wf;
                    frame_steps = f_steps;
                    frame_times = f_times;

                    // If perturbation exists, compute echo
                    if let Some(ref pert) = exp.perturbation {
                        let mut total_c = exp.background.clone();
                        for i in 0..total_c.len() {
                            total_c[i] += pert[i];
                        }
                        let (_, wf_pert, _, _) = run_forward_shot(&exp, &total_c, 0, args.every);
                        if let (Some(ref bg_f), Some(ref pt_f)) = (&recorded_wavefield, &wf_pert) {
                            let mut echo = vec![0.0f32; bg_f.len()];
                            for k in 0..bg_f.len() {
                                echo[k] = pt_f[k] - bg_f[k];
                            }
                            recorded_echo = Some(echo);
                        }
                    }
                }
            }

            // Save traces.npy
            write_npy_f64(args.out.join("traces.npy"), &[n_shots, steps, n_recvs], &all_traces)?;

            // Save wavefield.npy and echo.npy if recording
            if let Some(ref wf) = recorded_wavefield {
                let n_frames = frame_steps.as_ref().unwrap().len();
                write_npy_f32(args.out.join("wavefield.npy"), &[n_frames, exp.nz, exp.nx], wf)?;
            }
            if let Some(ref echo) = recorded_echo {
                let n_frames = frame_steps.as_ref().unwrap().len();
                write_npy_f32(args.out.join("echo.npy"), &[n_frames, exp.nz, exp.nx], echo)?;
            }

            // Write run.json
            let mut run_json = json!({
                "experiment_file": args.experiment.to_string_lossy(),
                "experiment": exp.raw_meta
            });
            if let Some(ev) = args.every {
                run_json["recording"] = json!({
                    "every": ev,
                    "steps": frame_steps.unwrap(),
                    "times": frame_times.unwrap()
                });
            }
            let run_file = File::create(args.out.join("run.json"))?;
            serde_json::to_writer_pretty(BufWriter::new(run_file), &run_json)?;

            // Write result.json
            let result_json = json!({
                "mode": "forward",
                "nx": exp.nx,
                "nz": exp.nz,
                "dx": exp.dx,
                "dt": exp.dt,
                "steps": exp.steps,
                "shots": exp.shots,
                "receivers": exp.receivers
            });
            let res_file = File::create(args.out.join("result.json"))?;
            serde_json::to_writer_pretty(BufWriter::new(res_file), &result_json)?;
        }
        "born" => {
            let mut all_born = vec![0.0f64; n_shots * steps * n_recvs];
            for shot_idx in 0..n_shots {
                let born_traces = run_born_shot(&exp, shot_idx);
                let start_idx = shot_idx * steps * n_recvs;
                all_born[start_idx..start_idx + steps * n_recvs].copy_from_slice(&born_traces);

                let shot_l2 = l2_norm(&born_traces);
                println!("{}\tborn\t{:.6}", shot_idx, shot_l2);
            }

            write_npy_f64(args.out.join("born_data.npy"), &[n_shots, steps, n_recvs], &all_born)?;

            let run_json = json!({
                "experiment_file": args.experiment.to_string_lossy(),
                "experiment": exp.raw_meta
            });
            let run_file = File::create(args.out.join("run.json"))?;
            serde_json::to_writer_pretty(BufWriter::new(run_file), &run_json)?;

            let result_json = json!({
                "mode": "born",
                "nx": exp.nx,
                "nz": exp.nz,
                "dx": exp.dx,
                "dt": exp.dt,
                "steps": exp.steps,
                "shots": exp.shots,
                "receivers": exp.receivers
            });
            let res_file = File::create(args.out.join("result.json"))?;
            serde_json::to_writer_pretty(BufWriter::new(res_file), &result_json)?;
        }
        "adjoint" => {
            let data_path = args.data.expect("--data is required for adjoint mode");
            let (shape, data) = read_npy_f64(&data_path)?;
            if shape.len() != 3 || shape[0] != n_shots || shape[1] != steps || shape[2] != n_recvs {
                panic!("Data shape mismatch: expected [{}, {}, {}], got {:?}", n_shots, steps, n_recvs, shape);
            }

            let is_treeverse = args.storage == "treeverse";
            let delta = args.checkpoints.unwrap_or(5);

            let mut total_image = vec![0.0f64; exp.nz * exp.nx];
            let mut per_shot_stats = Vec::new();
            let mut total_reverse_calls = 0;
            let mut total_forward_calls = 0;
            let mut max_peak_saved_states = 0;

            let mut recorded_wavefield = None;
            let mut frame_steps = None;
            let mut frame_times = None;

            for shot_idx in 0..n_shots {
                let is_first = shot_idx == 0;
                let rec_opt = if is_first { args.every } else { None };
                let start_idx = shot_idx * steps * n_recvs;
                let weights = &data[start_idx..start_idx + steps * n_recvs];

                let result = if is_treeverse {
                    let schedule = build_treeverse_schedule(steps, delta);
                    let act_filename = format!("actions-{}.json", shot_idx);
                    let act_file = File::create(args.out.join(&act_filename))?;
                    serde_json::to_writer_pretty(BufWriter::new(act_file), &schedule.actions)?;

                    let res = run_adjoint_shot_treeverse(&exp, shot_idx, weights, delta, &schedule.actions, rec_opt);
                    per_shot_stats.push(json!({
                        "reverse_calls": res.reverse_calls,
                        "scheduler_forward_calls": res.forward_calls,
                        "peak_saved_states": res.peak_saved_states,
                        "actions_file": act_filename
                    }));
                    res
                } else {
                    let res = run_adjoint_shot_full(&exp, shot_idx, weights, rec_opt);
                    per_shot_stats.push(json!({
                        "reverse_calls": res.reverse_calls,
                        "scheduler_forward_calls": res.forward_calls,
                        "peak_saved_states": res.peak_saved_states
                    }));
                    res
                };

                for i in 0..total_image.len() {
                    total_image[i] += result.image_contrib[i];
                }

                total_reverse_calls += result.reverse_calls;
                total_forward_calls += result.forward_calls;
                max_peak_saved_states = max_peak_saved_states.max(result.peak_saved_states);

                let shot_l2 = l2_norm(&result.image_contrib);
                println!("{}\tadjoint\t{:.6}", shot_idx, shot_l2);

                if is_first && args.every.is_some() {
                    recorded_wavefield = result.recorded_frames;
                    frame_steps = result.frame_steps;
                    frame_times = result.frame_times;
                }
            }

            write_npy_f64(args.out.join("image.npy"), &[exp.nz, exp.nx], &total_image)?;

            if let Some(ref wf) = recorded_wavefield {
                let n_frames = frame_steps.as_ref().unwrap().len();
                write_npy_f32(args.out.join("wavefield.npy"), &[n_frames, exp.nz, exp.nx], wf)?;
            }

            let mut run_json = json!({
                "experiment_file": args.experiment.to_string_lossy(),
                "experiment": exp.raw_meta
            });
            if let Some(ev) = args.every {
                run_json["recording"] = json!({
                    "every": ev,
                    "steps": frame_steps.unwrap(),
                    "times": frame_times.unwrap()
                });
            }
            let run_file = File::create(args.out.join("run.json"))?;
            serde_json::to_writer_pretty(BufWriter::new(run_file), &run_json)?;

            let state_bytes = 2 * exp.nx * exp.nz * 8;
            let stats_obj = json!({
                "storage": args.storage,
                "checkpoints": if is_treeverse { json!(delta) } else { json!(null) },
                "reverse_calls": total_reverse_calls,
                "scheduler_forward_calls": total_forward_calls,
                "peak_saved_states": max_peak_saved_states,
                "peak_saved_bytes": max_peak_saved_states * state_bytes,
                "per_shot": per_shot_stats
            });

            let result_json = json!({
                "mode": "adjoint",
                "nx": exp.nx,
                "nz": exp.nz,
                "dx": exp.dx,
                "dt": exp.dt,
                "steps": exp.steps,
                "shots": exp.shots,
                "receivers": exp.receivers,
                "statistics": stats_obj
            });
            let res_file = File::create(args.out.join("result.json"))?;
            serde_json::to_writer_pretty(BufWriter::new(res_file), &result_json)?;
        }
        _ => panic!("Unknown mode: {}", args.mode),
    }

    Ok(())
}
