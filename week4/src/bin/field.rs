use std::env;
use serde::Serialize;
use week4::fluid_solver::{taylor_green, random_field};

#[derive(Serialize)]
struct FieldOutput {
    case: String,
    n: usize,
    seed: Option<u64>,
    k_band: Option<[f64; 2]>,
    u: Vec<f64>,
    v: Vec<f64>,
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: field <taylor-green|random> [args]");
        std::process::exit(1);
    }

    let subcommand = &args[1];
    match subcommand.as_str() {
        "taylor-green" => {
            let mut n: Option<usize> = None;
            let mut nu: Option<f64> = None;
            let mut t: f64 = 0.0;

            let mut i = 2;
            while i < args.len() {
                match args[i].as_str() {
                    "--n" => { n = Some(args[i+1].parse().expect("Invalid --n")); i += 2; }
                    "--nu" => { nu = Some(args[i+1].parse().expect("Invalid --nu")); i += 2; }
                    "--t" => { t = args[i+1].parse().expect("Invalid --t"); i += 2; }
                    _ => {
                        eprintln!("Unknown argument: {}", args[i]);
                        std::process::exit(1);
                    }
                }
            }

            let n_val = n.expect("--n is required");
            let nu_val = if t > 0.0 {
                nu.expect("--nu is required when t > 0")
            } else {
                nu.unwrap_or(0.0)
            };

            let (u, v, _) = taylor_green(n_val, nu_val, t);
            let out = FieldOutput {
                case: "taylor-green".to_string(),
                n: n_val,
                seed: None,
                k_band: None,
                u,
                v,
            };
            println!("{}", serde_json::to_string(&out).unwrap());
        }
        "random" => {
            let mut n: Option<usize> = None;
            let mut seed: Option<u64> = None;
            let mut k_min: Option<f64> = None;
            let mut k_max: Option<f64> = None;

            let mut i = 2;
            while i < args.len() {
                match args[i].as_str() {
                    "--n" => { n = Some(args[i+1].parse().expect("Invalid --n")); i += 2; }
                    "--seed" => { seed = Some(args[i+1].parse().expect("Invalid --seed")); i += 2; }
                    "--k-min" => { k_min = Some(args[i+1].parse().expect("Invalid --k-min")); i += 2; }
                    "--k-max" => { k_max = Some(args[i+1].parse().expect("Invalid --k-max")); i += 2; }
                    _ => {
                        eprintln!("Unknown argument: {}", args[i]);
                        std::process::exit(1);
                    }
                }
            }

            let n_val = n.expect("--n is required");
            let seed_val = seed.expect("--seed is required");
            let k_min_val = k_min.expect("--k-min is required");
            let k_max_val = k_max.expect("--k-max is required");

            let (u, v, _) = random_field(n_val, seed_val, k_min_val, k_max_val);
            let out = FieldOutput {
                case: "random".to_string(),
                n: n_val,
                seed: Some(seed_val),
                k_band: Some([k_min_val, k_max_val]),
                u,
                v,
            };
            println!("{}", serde_json::to_string(&out).unwrap());
        }
        _ => {
            eprintln!("Unknown subcommand: {}", subcommand);
            std::process::exit(1);
        }
    }
}
