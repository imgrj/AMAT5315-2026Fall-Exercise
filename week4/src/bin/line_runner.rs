use std::env;
use std::fs::File;
use std::io::Write;
use std::f64::consts::PI;
use week4::{Integrator, ForwardEuler, ExplicitMidpoint, Rk4, Rk4EqualWeights};
use week4::line::{fourier_rate, fd_rate, gaussian_pulse, exact_gaussian_pulse};
use serde::Serialize;

#[derive(Serialize)]
struct StabilityRun {
    dt: f64,
    n: usize,
    times: Vec<f64>,
    u_xt: Vec<Vec<f64>>,
}

#[derive(Serialize)]
struct PulseProfileData {
    x: Vec<f64>,
    exact: Vec<f64>,
    rk4_fourier: Vec<f64>,
    rk4_fd: Vec<f64>,
    euler_fourier: Vec<f64>,
    start: Vec<f64>,
    err_rk4_fourier: f64,
    err_rk4_fd: f64,
    err_euler_fourier: f64,
}

#[derive(Serialize)]
struct LineConvergenceData {
    dts: Vec<f64>,
    euler_errs: Vec<f64>,
    midpoint_errs: Vec<f64>,
    rk4_errs: Vec<f64>,
    rk4_equal_errs: Vec<f64>,
}

fn integrate_to(stepper: &dyn Integrator, state: &mut [f64], dt: f64, t_end: f64, rate: &dyn Fn(&[f64], &mut [f64])) {
    let mut t = 0.0;
    while t < t_end - 1e-12 {
        let step = if t + dt > t_end { t_end - t } else { dt };
        stepper.step(state, step, rate);
        t += step;
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: line_runner <stability|pulse-profiles|convergence-line> [args]");
        std::process::exit(1);
    }

    match args[1].as_str() {
        "stability" => {
            let mut dt: f64 = 0.045;
            let mut t_end: f64 = 6.0;
            let mut out_file = "stability_out.json".to_string();
            let mut i = 2;
            while i < args.len() {
                match args[i].as_str() {
                    "--dt" => { dt = args[i+1].parse::<f64>().unwrap(); i += 2; }
                    "--t-end" => { t_end = args[i+1].parse::<f64>().unwrap(); i += 2; }
                    "--out" => { out_file = args[i+1].clone(); i += 2; }
                    _ => i += 1,
                }
            }
            let n = 64usize;
            let c: f64 = 1.0;
            let nu: f64 = 0.05;
            let sigma: f64 = 0.35;
            let mut state = gaussian_pulse(n, sigma);

            let mut times = vec![0.0];
            let mut u_xt = vec![state.clone()];

            let stepper = Rk4;
            let mut cur_t = 0.0;
            let total_steps = ((t_end / dt).round()) as usize;

            for _step in 1..=total_steps {
                stepper.step(&mut state, dt, &|u, out| fourier_rate(u, out, c, nu));
                cur_t += dt;
                times.push(cur_t);
                u_xt.push(state.clone());
            }

            let result = StabilityRun { dt, n, times, u_xt };
            let json = serde_json::to_string(&result).unwrap();
            let mut f = File::create(&out_file).unwrap();
            f.write_all(json.as_bytes()).unwrap();
            println!("Wrote stability run to {}", out_file);
        }
        "pulse-profiles" => {
            let mut out_file = "pulse_profiles.json".to_string();
            if args.len() >= 4 && args[2] == "--out" {
                out_file = args[3].clone();
            }
            let n = 64usize;
            let c: f64 = 1.0;
            let nu: f64 = 0.002;
            let sigma: f64 = 0.25;
            let t_end = 2.0 * PI;
            let dx = 2.0 * PI / (n as f64);
            let x: Vec<f64> = (0..n).map(|j| (j as f64) * dx).collect();
            let start = gaussian_pulse(n, sigma);
            let exact = exact_gaussian_pulse(n, sigma, c, nu, t_end);

            let stepper_rk4 = Rk4;
            let mut state_rk4_fourier = start.clone();
            integrate_to(&stepper_rk4, &mut state_rk4_fourier, 0.02, t_end, &|u, out| fourier_rate(u, out, c, nu));
            let err_rk4_fourier = state_rk4_fourier.iter().zip(&exact).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);

            let mut state_rk4_fd = start.clone();
            integrate_to(&stepper_rk4, &mut state_rk4_fd, 0.02, t_end, &|u, out| fd_rate(u, out, c, nu));
            let err_rk4_fd = state_rk4_fd.iter().zip(&exact).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);

            let stepper_euler = ForwardEuler;
            let mut state_euler = start.clone();
            integrate_to(&stepper_euler, &mut state_euler, 0.005, t_end, &|u, out| fourier_rate(u, out, c, nu));
            let err_euler_fourier = state_euler.iter().zip(&exact).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);

            println!("Maximum errors at t = 2*pi:");
            println!("  RK4 Fourier (dt=0.02): {:e}", err_rk4_fourier);
            println!("  RK4 Centred FD (dt=0.02): {:e}", err_rk4_fd);
            println!("  Euler Fourier (dt=0.005): {:e}", err_euler_fourier);

            let data = PulseProfileData {
                x,
                exact,
                rk4_fourier: state_rk4_fourier,
                rk4_fd: state_rk4_fd,
                euler_fourier: state_euler,
                start,
                err_rk4_fourier,
                err_rk4_fd,
                err_euler_fourier,
            };
            let json = serde_json::to_string(&data).unwrap();
            let mut f = File::create(&out_file).unwrap();
            f.write_all(json.as_bytes()).unwrap();
            println!("Wrote pulse profiles to {}", out_file);
        }
        "convergence-line" => {
            let mut out_file = "line_convergence.json".to_string();
            if args.len() >= 4 && args[2] == "--out" {
                out_file = args[3].clone();
            }
            let n = 64usize;
            let c: f64 = 1.0;
            let nu: f64 = 0.05;
            let sigma: f64 = 0.35;
            let t_end = 1.0;
            let start = gaussian_pulse(n, sigma);
            let exact = exact_gaussian_pulse(n, sigma, c, nu, t_end);

            let dts = vec![0.02f64, 0.01f64, 0.005f64, 0.0025f64];
            let mut euler_errs = Vec::new();
            let mut midpoint_errs = Vec::new();
            let mut rk4_errs = Vec::new();
            let mut rk4_equal_errs = Vec::new();

            let s_euler = ForwardEuler;
            let s_mid = ExplicitMidpoint;
            let s_rk4 = Rk4;
            let s_rk4_eq = Rk4EqualWeights;

            for &dt in &dts {
                let steps = ((t_end / dt).round()) as usize;

                let mut u_e = start.clone();
                for _ in 0..steps { s_euler.step(&mut u_e, dt, &|u, out| fourier_rate(u, out, c, nu)); }
                let err_e = u_e.iter().zip(&exact).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
                euler_errs.push(err_e);

                let mut u_m = start.clone();
                for _ in 0..steps { s_mid.step(&mut u_m, dt, &|u, out| fourier_rate(u, out, c, nu)); }
                let err_m = u_m.iter().zip(&exact).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
                midpoint_errs.push(err_m);

                let mut u_4 = start.clone();
                for _ in 0..steps { s_rk4.step(&mut u_4, dt, &|u, out| fourier_rate(u, out, c, nu)); }
                let err_4 = u_4.iter().zip(&exact).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
                rk4_errs.push(err_4);

                let mut u_4eq = start.clone();
                for _ in 0..steps { s_rk4_eq.step(&mut u_4eq, dt, &|u, out| fourier_rate(u, out, c, nu)); }
                let err_4eq = u_4eq.iter().zip(&exact).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
                rk4_equal_errs.push(err_4eq);
            }

            let data = LineConvergenceData {
                dts,
                euler_errs,
                midpoint_errs,
                rk4_errs,
                rk4_equal_errs,
            };
            let json = serde_json::to_string(&data).unwrap();
            let mut f = File::create(&out_file).unwrap();
            f.write_all(json.as_bytes()).unwrap();
            println!("Wrote line convergence to {}", out_file);
        }
        _ => {
            eprintln!("Unknown subcommand: {}", args[1]);
            std::process::exit(1);
        }
    }
}
