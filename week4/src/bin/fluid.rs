use std::env;
use std::fs::{self, File};
use std::io::{self, Read, Write, BufWriter};
use serde::{Deserialize, Serialize};
use week4::{Integrator, ForwardEuler, ExplicitMidpoint, Rk4};
use week4::fluid_solver::Spectral2D;

#[derive(Deserialize, Serialize)]
struct FieldInput {
    case: String,
    n: usize,
    seed: Option<u64>,
    k_band: Option<[f64; 2]>,
    u: Vec<f64>,
    v: Vec<f64>,
}

#[derive(Serialize)]
struct RunMetadata {
    case: String,
    n: usize,
    seed: Option<u64>,
    k_band: Option<[f64; 2]>,
    method: String,
    nu: f64,
    dt: f64,
    t_end: f64,
    snapshot_every: f64,
}

#[derive(Serialize)]
struct SnapshotFrame {
    t: f64,
    step: usize,
    u: Vec<f64>,
    v: Vec<f64>,
    omega: Vec<f64>,
}

fn round6(val: f64) -> f64 {
    (val * 1e6).round() / 1e6
}

fn format_t(t: f64) -> String {
    let t_10 = t * 10.0;
    if (t_10 - t_10.round()).abs() < 1e-4 {
        format!("{:.1}", t)
    } else {
        format!("{:.4}", t)
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let mut method: Option<String> = None;
    let mut nu: Option<f64> = None;
    let mut dt: Option<f64> = None;
    let mut t_end: Option<f64> = None;
    let mut every: Option<f64> = None;
    let mut out: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--method" => { method = Some(args[i+1].clone()); i += 2; }
            "--nu" => { nu = Some(args[i+1].parse().expect("Invalid --nu")); i += 2; }
            "--dt" => { dt = Some(args[i+1].parse().expect("Invalid --dt")); i += 2; }
            "--t-end" => { t_end = Some(args[i+1].parse().expect("Invalid --t-end")); i += 2; }
            "--every" => { every = Some(args[i+1].parse().expect("Invalid --every")); i += 2; }
            "--out" => { out = Some(args[i+1].clone()); i += 2; }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                std::process::exit(1);
            }
        }
    }

    let method_str = method.expect("--method is required");
    let nu_val = nu.expect("--nu is required");
    let dt_val = dt.expect("--dt is required");
    let t_end_val = t_end.expect("--t-end is required");
    let every_val = every.expect("--every is required");
    let out_dir = out.expect("--out is required");

    // Read field from stdin
    let mut input_str = String::new();
    io::stdin().read_to_string(&mut input_str).expect("Failed to read stdin");
    let field_data: FieldInput = serde_json::from_str(&input_str).expect("Failed to parse field JSON");

    fs::create_dir_all(&out_dir).expect("Failed to create out dir");

    // Write run.json
    let meta = RunMetadata {
        case: field_data.case.clone(),
        n: field_data.n,
        seed: field_data.seed,
        k_band: field_data.k_band,
        method: method_str.clone(),
        nu: nu_val,
        dt: dt_val,
        t_end: t_end_val,
        snapshot_every: every_val,
    };
    let run_json_path = format!("{}/run.json", out_dir);
    let mut f_run = File::create(&run_json_path).expect("Failed to create run.json");
    f_run.write_all(serde_json::to_string_pretty(&meta).unwrap().as_bytes()).unwrap();

    let fields_jsonl_path = format!("{}/fields.jsonl", out_dir);
    let f_fields = File::create(&fields_jsonl_path).expect("Failed to create fields.jsonl");
    let mut fields_writer = BufWriter::new(f_fields);

    let n = field_data.n;
    let spec = Spectral2D::new(n);

    // Initial vorticity from velocity
    let mut omega = spec.omega_from_velocity(&field_data.u, &field_data.v);
    let (mut u_curr, mut v_curr) = spec.velocity_from_omega(&omega);

    let initial_e = spec.energy(&u_curr, &v_curr);
    let initial_z = spec.enstrophy(&omega);

    // Write snapshot 0
    let frame0 = SnapshotFrame {
        t: 0.0,
        step: 0,
        u: u_curr.iter().map(|&x| round6(x)).collect(),
        v: v_curr.iter().map(|&x| round6(x)).collect(),
        omega: omega.iter().map(|&x| round6(x)).collect(),
    };
    writeln!(fields_writer, "{}", serde_json::to_string(&frame0).unwrap()).unwrap();

    // Select integrator
    let integrator: Box<dyn Integrator> = match method_str.as_str() {
        "euler" => Box::new(ForwardEuler),
        "rk2" => Box::new(ExplicitMidpoint),
        "rk4" => Box::new(Rk4),
        _ => {
            eprintln!("Unknown method: {}", method_str);
            std::process::exit(1);
        }
    };

    println!("t\tEnergy E\tEnstrophy Z");
    println!("0.0\t{:.6}\t{:.6}", initial_e, initial_z);

    let snapshot_interval = ((every_val / dt_val).round()) as usize;
    let total_steps = ((t_end_val / dt_val).round()) as usize;

    for step in 1..=total_steps {
        integrator.step(&mut omega, dt_val, &|w, out_r| spec.rate(w, out_r, nu_val));
        let (u_next, v_next) = spec.velocity_from_omega(&omega);
        u_curr = u_next;
        v_curr = v_next;

        let e = spec.energy(&u_curr, &v_curr);
        let z = spec.enstrophy(&omega);
        let t = (step as f64) * dt_val;

        if !e.is_finite() || !z.is_finite() || e.abs() > 1e10 {
            // Unstable blowup: print line and exit 1
            println!("{}\t{:.6}\t{:.6}", format_t(t), e, z);
            fields_writer.flush().unwrap();
            std::process::exit(1);
        }

        if step % snapshot_interval == 0 {
            println!("{}\t{:.6}\t{:.6}", format_t(t), e, z);
            let frame = SnapshotFrame {
                t: round6(t),
                step,
                u: u_curr.iter().map(|&x| round6(x)).collect(),
                v: v_curr.iter().map(|&x| round6(x)).collect(),
                omega: omega.iter().map(|&x| round6(x)).collect(),
            };
            writeln!(fields_writer, "{}", serde_json::to_string(&frame).unwrap()).unwrap();
        }
    }

    fields_writer.flush().unwrap();
}
