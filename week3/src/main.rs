mod lattice;
mod metropolis;
mod wolff;
mod rng;

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use lattice::Lattice;
use metropolis::{metropolis_sweep, MetropolisTable};
use wolff::WolffState;
use rng::Rng;

struct Config {
    update: String,
    l: usize,
    t_from: f64,
    t_to: f64,
    t_step: f64,
    discard: usize,
    measure: usize,
    seed: u64,
    every: usize,
    out: PathBuf,
}

fn parse_args() -> Result<Option<Config>, String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() == 1 || args.contains(&"--help".to_string()) || args.contains(&"-h".to_string()) {
        println!("Usage: ising --update <metropolis|wolff> --l <L> --t-from <T> --t-to <T> --t-step <dT> --discard <N> --measure <N> [--every <N>] --seed <S> --out <dir>");
        return Ok(None);
    }

    let mut update = None;
    let mut l = None;
    let mut t_from = None;
    let mut t_to = None;
    let mut t_step = None;
    let mut discard = None;
    let mut measure = None;
    let mut seed = None;
    let mut every = 0usize;
    let mut out = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--update" => {
                i += 1;
                update = Some(args[i].clone());
            }
            "--l" => {
                i += 1;
                l = Some(args[i].parse::<usize>().map_err(|e| e.to_string())?);
            }
            "--t-from" => {
                i += 1;
                t_from = Some(args[i].parse::<f64>().map_err(|e| e.to_string())?);
            }
            "--t-to" => {
                i += 1;
                t_to = Some(args[i].parse::<f64>().map_err(|e| e.to_string())?);
            }
            "--t-step" => {
                i += 1;
                t_step = Some(args[i].parse::<f64>().map_err(|e| e.to_string())?);
            }
            "--discard" => {
                i += 1;
                discard = Some(args[i].parse::<usize>().map_err(|e| e.to_string())?);
            }
            "--measure" => {
                i += 1;
                measure = Some(args[i].parse::<usize>().map_err(|e| e.to_string())?);
            }
            "--seed" => {
                i += 1;
                seed = Some(args[i].parse::<u64>().map_err(|e| e.to_string())?);
            }
            "--every" => {
                i += 1;
                every = args[i].parse::<usize>().map_err(|e| e.to_string())?;
            }
            "--out" => {
                i += 1;
                out = Some(PathBuf::from(&args[i]));
            }
            other => return Err(format!("Unknown argument: {}", other)),
        }
        i += 1;
    }

    Ok(Some(Config {
        update: update.ok_or("Missing --update")?,
        l: l.ok_or("Missing --l")?,
        t_from: t_from.ok_or("Missing --t-from")?,
        t_to: t_to.ok_or("Missing --t-to")?,
        t_step: t_step.ok_or("Missing --t-step")?,
        discard: discard.ok_or("Missing --discard")?,
        measure: measure.ok_or("Missing --measure")?,
        seed: seed.ok_or("Missing --seed")?,
        every,
        out: out.ok_or("Missing --out")?,
    }))
}

fn round6(v: f64) -> f64 {
    (v * 1e6).round() / 1e6
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = match parse_args().map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))? {
        Some(cfg) => cfg,
        None => return Ok(()),
    };

    fs::create_dir_all(&config.out)?;

    // Build temperature grid
    let mut t_grid = Vec::new();
    let mut cur_t = config.t_from;
    while cur_t <= config.t_to + 1e-8 * config.t_step.abs().max(1.0) {
        t_grid.push((cur_t * 1e6).round() / 1e6);
        cur_t += config.t_step;
    }

    let time_unit = if config.update == "metropolis" {
        "sweep"
    } else {
        "cluster_flip"
    };

    // 1. Write run.json
    let run_json_path = config.out.join("run.json");
    let mut run_file = BufWriter::new(File::create(run_json_path)?);
    let t_grid_strs: Vec<String> = t_grid.iter().map(|t| t.to_string()).collect();
    writeln!(run_file, "{{")?;
    writeln!(run_file, "  \"L\": {},", config.l)?;
    writeln!(run_file, "  \"update\": \"{}\",", config.update)?;
    writeln!(run_file, "  \"t_grid\": [{}],", t_grid_strs.join(", "))?;
    writeln!(run_file, "  \"discard\": {},", config.discard)?;
    writeln!(run_file, "  \"measure\": {},", config.measure)?;
    writeln!(run_file, "  \"seed\": {},", config.seed)?;
    writeln!(run_file, "  \"sample_every\": 1,")?;
    writeln!(run_file, "  \"time_unit\": \"{}\"", time_unit)?;
    writeln!(run_file, "}}")?;
    run_file.flush()?;

    // Prepare series.jsonl
    let series_path = config.out.join("series.jsonl");
    let mut series_writer = BufWriter::new(File::create(series_path)?);

    // Prepare spins.jsonl if every > 0
    let mut spins_writer = if config.every > 0 {
        let spins_path = config.out.join("spins.jsonl");
        Some(BufWriter::new(File::create(spins_path)?))
    } else {
        None
    };

    let mut rng = Rng::seed_from_u64(config.seed);
    let mut lattice = Lattice::new_all_up(config.l);
    let mut wolff_state = WolffState::new(lattice.n);

    // Print stdout header
    if config.update == "metropolis" {
        println!("T\tmean |M|\tacceptance");
    } else {
        println!("T\tmean |M|\tmean cluster size");
    }

    let mut cumulative_sweep = 0usize;

    for &t in &t_grid {
        if config.update == "metropolis" {
            let table = MetropolisTable::new(t);
            let mut accepted_total = 0usize;

            // Discard sweeps
            for _ in 0..config.discard {
                let acc = metropolis_sweep(&mut lattice, &table, &mut rng);
                accepted_total += acc;
                cumulative_sweep += 1;
            }

            // Measure sweeps
            let mut sum_abs_m = 0.0;
            for step in 1..=config.measure {
                let acc = metropolis_sweep(&mut lattice, &table, &mut rng);
                accepted_total += acc;
                cumulative_sweep += 1;

                let m = lattice.magnetization();
                let e = lattice.energy_per_site();
                sum_abs_m += m.abs();

                let m_r = round6(m);
                let e_r = round6(e);

                writeln!(
                    series_writer,
                    "{{\"L\":{},\"T\":{},\"sweep\":{},\"M\":{:.6},\"E\":{:.6}}}",
                    config.l, t, step, m_r, e_r
                )?;

                if config.every > 0 && step % config.every == 0 {
                    if let Some(ref mut sw) = spins_writer {
                        write!(
                            sw,
                            "{{\"L\":{},\"T\":{},\"sweep\":{},\"m\":{:.6},\"spins\":[",
                            config.l, t, cumulative_sweep, m_r
                        )?;
                        for (idx, &s) in lattice.spins.iter().enumerate() {
                            if idx > 0 {
                                write!(sw, ",")?;
                            }
                            write!(sw, "{}", s)?;
                        }
                        writeln!(sw, "]}}")?;
                    }
                }
            }

            let mean_abs_m = sum_abs_m / (config.measure as f64);
            let total_proposals = ((config.discard + config.measure) * lattice.n) as f64;
            let acc_rate = (accepted_total as f64) / total_proposals;

            println!("{}\t{:.4}\t{:.4}", t, mean_abs_m, acc_rate);
        } else if config.update == "wolff" {
            let mut total_cluster_size = 0usize;

            // Discard cluster moves
            for _ in 0..config.discard {
                wolff_state.step(&mut lattice, t, &mut rng);
                cumulative_sweep += 1;
            }

            // Measure cluster moves
            let mut sum_abs_m = 0.0;
            for step in 1..=config.measure {
                let c = wolff_state.step(&mut lattice, t, &mut rng);
                total_cluster_size += c;
                cumulative_sweep += 1;

                let m = lattice.magnetization();
                let e = lattice.energy_per_site();
                sum_abs_m += m.abs();

                let m_r = round6(m);
                let e_r = round6(e);

                writeln!(
                    series_writer,
                    "{{\"L\":{},\"T\":{},\"sweep\":{},\"M\":{:.6},\"E\":{:.6},\"cluster_size\":{}}}",
                    config.l, t, step, m_r, e_r, c
                )?;

                if config.every > 0 && step % config.every == 0 {
                    if let Some(ref mut sw) = spins_writer {
                        write!(
                            sw,
                            "{{\"L\":{},\"T\":{},\"sweep\":{},\"m\":{:.6},\"spins\":[",
                            config.l, t, cumulative_sweep, m_r
                        )?;
                        for (idx, &s) in lattice.spins.iter().enumerate() {
                            if idx > 0 {
                                write!(sw, ",")?;
                            }
                            write!(sw, "{}", s)?;
                        }
                        writeln!(sw, "]}}")?;
                    }
                }
            }

            let mean_abs_m = sum_abs_m / (config.measure as f64);
            let mean_c = (total_cluster_size as f64) / (config.measure as f64);

            println!("{}\t{:.4}\t{:.4}", t, mean_abs_m, mean_c);
        } else {
            return Err(format!("Unsupported update method: {}", config.update).into());
        }
    }

    series_writer.flush()?;
    if let Some(ref mut sw) = spins_writer {
        sw.flush()?;
    }

    Ok(())
}
