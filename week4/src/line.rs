use std::f64::consts::PI;
use num_complex::Complex64;
use rustfft::FftPlanner;

pub fn gaussian_pulse(n: usize, sigma: f64) -> Vec<f64> {
    let dx = 2.0 * PI / (n as f64);
    let mut u = vec![0.0; n];
    let x0 = PI / 2.0;
    for j in 0..n {
        let x = (j as f64) * dx;
        let mut sum = 0.0;
        for p in -5..=5 {
            let d = x - x0 - (p as f64) * 2.0 * PI;
            sum += (-d * d / (2.0 * sigma * sigma)).exp();
        }
        u[j] = sum;
    }
    u
}

pub fn exact_gaussian_pulse(n: usize, sigma: f64, c: f64, nu: f64, t: f64) -> Vec<f64> {
    let dx = 2.0 * PI / (n as f64);
    let mut u = vec![0.0; n];
    let x0 = PI / 2.0;
    let s2 = sigma * sigma + 2.0 * nu * t;
    let amp = sigma / s2.sqrt();
    for j in 0..n {
        let x = (j as f64) * dx;
        let mut sum = 0.0;
        for p in -5..=5 {
            let d = x - x0 - c * t - (p as f64) * 2.0 * PI;
            sum += amp * (-d * d / (2.0 * s2)).exp();
        }
        u[j] = sum;
    }
    u
}

pub fn fourier_rate(u: &[f64], out: &mut [f64], c: f64, nu: f64) {
    let n = u.len();
    let mut planner = FftPlanner::new();
    let fft_forward = planner.plan_fft_forward(n);
    let fft_inverse = planner.plan_fft_inverse(n);

    let mut buf: Vec<Complex64> = u.iter().map(|&val| Complex64::new(val, 0.0)).collect();
    fft_forward.process(&mut buf);

    let half = n / 2;
    for m in 0..n {
        if m == half {
            // Nyquist mode k = -n/2 does not travel: odd derivative is 0
            let k = -(half as f64);
            let mult = -nu * k * k;
            buf[m] *= mult;
        } else {
            let k = if m < half {
                m as f64
            } else {
                (m as f64) - (n as f64)
            };
            // rate multiplier = -nu * k^2 - i * c * k
            let mult = Complex64::new(-nu * k * k, -c * k);
            buf[m] *= mult;
        }
    }

    fft_inverse.process(&mut buf);
    let inv_n = 1.0 / (n as f64);
    for j in 0..n {
        out[j] = buf[j].re * inv_n;
    }
}

pub fn fd_rate(u: &[f64], out: &mut [f64], c: f64, nu: f64) {
    let n = u.len();
    let dx = 2.0 * PI / (n as f64);
    let inv_2dx = 1.0 / (2.0 * dx);
    let inv_dx2 = 1.0 / (dx * dx);

    for j in 0..n {
        let jp1 = if j + 1 == n { 0 } else { j + 1 };
        let jm1 = if j == 0 { n - 1 } else { j - 1 };
        let ux = (u[jp1] - u[jm1]) * inv_2dx;
        let uxx = (u[jp1] - 2.0 * u[j] + u[jm1]) * inv_dx2;
        out[j] = -c * ux + nu * uxx;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrator::{Integrator, ForwardEuler, ExplicitMidpoint, Rk4};

    #[test]
    fn test_single_wave_convergence() {
        let n = 32usize;
        let c: f64 = 1.0;
        let nu: f64 = 0.05;
        let k: f64 = 2.0;
        let dx: f64 = 2.0 * PI / (n as f64);
        let t_end: f64 = 0.4;

        let exact: Vec<f64> = (0..n)
            .map(|j| {
                let x = (j as f64) * dx;
                let decay: f64 = (-1.0 * nu * k * k * t_end).exp();
                decay * (k * (x - c * t_end)).cos()
            })
            .collect();

        let initial: Vec<f64> = (0..n).map(|j| (k * (j as f64) * dx).cos()).collect();

        // Test Euler (order 1)
        let mut errs_euler = Vec::new();
        for &dt in &[0.01f64, 0.005f64] {
            let mut state = initial.clone();
            let steps = (t_end / dt).round() as usize;
            let stepper = ForwardEuler;
            for _ in 0..steps {
                stepper.step(&mut state, dt, &|u, out| fourier_rate(u, out, c, nu));
            }
            let err = state.iter().zip(&exact).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
            errs_euler.push(err);
        }
        let slope_euler = (errs_euler[0] / errs_euler[1]).log2();
        assert!((slope_euler - 1.0).abs() < 0.1, "Euler slope: {}", slope_euler);

        // Test Midpoint (order 2)
        let mut errs_rk2 = Vec::new();
        for &dt in &[0.02f64, 0.01f64] {
            let mut state = initial.clone();
            let steps = (t_end / dt).round() as usize;
            let stepper = ExplicitMidpoint;
            for _ in 0..steps {
                stepper.step(&mut state, dt, &|u, out| fourier_rate(u, out, c, nu));
            }
            let err = state.iter().zip(&exact).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
            errs_rk2.push(err);
        }
        let slope_rk2 = (errs_rk2[0] / errs_rk2[1]).log2();
        assert!((slope_rk2 - 2.0).abs() < 0.1, "RK2 slope: {}", slope_rk2);

        // Test RK4 (order 4)
        let mut errs_rk4 = Vec::new();
        for &dt in &[0.04f64, 0.02f64] {
            let mut state = initial.clone();
            let steps = (t_end / dt).round() as usize;
            let stepper = Rk4;
            for _ in 0..steps {
                stepper.step(&mut state, dt, &|u, out| fourier_rate(u, out, c, nu));
            }
            let err = state.iter().zip(&exact).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
            errs_rk4.push(err);
        }
        let slope_rk4 = (errs_rk4[0] / errs_rk4[1]).log2();
        assert!((slope_rk4 - 4.0).abs() < 0.1, "RK4 slope: {}", slope_rk4);
    }
}
