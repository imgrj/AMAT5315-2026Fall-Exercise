//! Simulation, independent physics checks, and video operations.

use std::path::Path;

use crate::cli::{CheckCfg, IntegratorChoice, RunCfg, VideoCfg};
use crate::rng::Rng;
use crate::trajectory::{Frame, RunMetadata, Trajectory};
use crate::{Euler, ForceStrategy, Integrator, System, VelocityVerlet, triangular_lattice};

pub const RC: f64 = 2.5;

fn remove_mean_and_rescale(velocities: &mut [[f64; 2]], target: f64) {
    let n = velocities.len() as f64;
    let mean = velocities.iter().fold([0.0; 2], |sum, v| [sum[0] + v[0], sum[1] + v[1]]);
    for v in velocities.iter_mut() {
        v[0] -= mean[0] / n;
        v[1] -= mean[1] / n;
    }
    rescale_thermostat(velocities, target);
}

fn rescale_thermostat(velocities: &mut [[f64; 2]], target: f64) {
    let kinetic: f64 = velocities
        .iter()
        .map(|v| 0.5 * (v[0] * v[0] + v[1] * v[1]))
        .sum();
    let current = kinetic / (velocities.len() as f64 - 1.0);
    let factor = (target / current).sqrt();
    for v in velocities {
        v[0] *= factor;
        v[1] *= factor;
    }
}

fn initial_velocities(n: usize, temperature: f64, rng: &mut Rng) -> Vec<[f64; 2]> {
    let mut velocities: Vec<_> = (0..n)
        .map(|_| [rng.gaussian() * temperature.sqrt(), rng.gaussian() * temperature.sqrt()])
        .collect();
    remove_mean_and_rescale(&mut velocities, temperature);
    velocities
}

pub fn run_sim(cfg: &RunCfg) -> Result<String, String> {
    let (positions, box_size) = triangular_lattice(cfg.n, cfg.rho)?;
    if box_size.iter().any(|side| *side <= 2.0 * RC) {
        return Err(format!("box {box_size:?} must be wider than twice rc={RC}"));
    }
    let mut rng = Rng::new(cfg.seed);
    let velocities = initial_velocities(cfg.n, cfg.temperature, &mut rng);
    let mut system = System::periodic(positions, velocities, box_size, RC);
    system.set_force_strategy(cfg.force);
    let integrator: &dyn Integrator = match cfg.integrator {
        IntegratorChoice::VelocityVerlet => &VelocityVerlet,
        IntegratorChoice::Euler => &Euler,
    };

    for step in 1..=cfg.eq_steps {
        integrator.step(&mut system, cfg.dt);
        if step % 50 == 0 {
            rescale_thermostat(&mut system.velocities, cfg.temperature);
        }
    }

    let mut frames = Vec::with_capacity(cfg.steps / cfg.sample_every);
    for step in 1..=cfg.steps {
        integrator.step(&mut system, cfg.dt);
        if let Some(final_temperature) = cfg.ramp_to {
            let fraction = step as f64 / cfg.steps as f64;
            let target = cfg.temperature + fraction * (final_temperature - cfg.temperature);
            rescale_thermostat(&mut system.velocities, target);
        }
        if step % cfg.sample_every == 0 {
            frames.push(Frame {
                step,
                t: step as f64 * cfg.dt,
                pos: system.positions.clone(),
                vel: system.velocities.clone(),
                e_pot: system.pair_energy(),
                e_kin: system.kinetic_energy(),
            });
        }
    }
    let trajectory = Trajectory {
        meta: RunMetadata {
            n: cfg.n,
            rho: cfg.rho,
            box_size,
            dt: cfg.dt,
            temperature: cfg.temperature,
            eq_steps: cfg.eq_steps,
            steps: cfg.steps,
            sample_every: cfg.sample_every,
            seed: cfg.seed,
            integrator: cfg.integrator.name().into(),
            force: cfg.force,
            ramp_to: cfg.ramp_to,
        },
        frames,
    };
    crate::trajectory::validate(&trajectory)?;
    crate::trajectory::write(Path::new(&cfg.out), &trajectory)?;
    Ok(format!(
        "wrote {} frames for N={} to {}",
        trajectory.frames.len(),
        cfg.n,
        cfg.out
    ))
}

#[derive(Clone, Debug)]
pub struct Report {
    pub drift: f64,
    pub temperature: f64,
    pub target_temperature: f64,
    pub chi2_per_dof: f64,
    pub maximum_energy_mismatch: f64,
    pub pass: bool,
}

fn recomputed_energy(meta: &RunMetadata, frame: &Frame) -> (f64, f64) {
    let mut system = System::periodic(
        frame.pos.clone(),
        frame.vel.clone(),
        meta.box_size,
        RC,
    );
    system.set_force_strategy(ForceStrategy::Cells);
    (system.pair_energy(), system.kinetic_energy())
}

fn speed_shape(speeds_squared: &[f64], temperature: f64) -> f64 {
    let mut observed = [0usize; 24];
    for &v2 in speeds_squared {
        let cdf = 1.0 - (-v2 / (2.0 * temperature)).exp();
        let bin = ((24.0 * cdf).floor() as usize).min(23);
        observed[bin] += 1;
    }
    let expected = speeds_squared.len() as f64 / 24.0;
    observed
        .iter()
        .map(|&count| {
            let delta = count as f64 - expected;
            delta * delta / expected
        })
        .sum::<f64>()
        / 22.0
}

pub fn check(cfg: &CheckCfg) -> Result<Report, String> {
    let trajectory = crate::trajectory::read(Path::new(&cfg.input))?;
    if trajectory.frames.is_empty() {
        return Err("trajectory has no saved frames".into());
    }

    let mut energies = Vec::with_capacity(trajectory.frames.len());
    let mut speeds_squared = Vec::with_capacity(trajectory.frames.len() * trajectory.meta.n);
    let mut maximum_energy_mismatch = 0.0_f64;
    for (index, frame) in trajectory.frames.iter().enumerate() {
        let (e_pot, e_kin) = recomputed_energy(&trajectory.meta, frame);
        let mismatch = (e_pot - frame.e_pot)
            .abs()
            .max((e_kin - frame.e_kin).abs());
        let scale = e_pot.abs().max(e_kin.abs()).max(1.0);
        maximum_energy_mismatch = maximum_energy_mismatch.max(mismatch / scale);
        if mismatch > 1e-9 * scale {
            return Err(format!(
                "frame {} stored energies disagree with raw positions/velocities: relative mismatch {:.3e}",
                index + 1,
                mismatch / scale
            ));
        }
        energies.push(e_pot + e_kin);
        speeds_squared.extend(frame.vel.iter().map(|v| v[0] * v[0] + v[1] * v[1]));
    }

    let k = (trajectory.frames.len() / 10).max(1);
    let first_mean = energies[..k].iter().sum::<f64>() / k as f64;
    let last_mean = energies[energies.len() - k..].iter().sum::<f64>() / k as f64;
    let drift = (last_mean - first_mean).abs() / energies[0].abs();
    let temperature = speeds_squared.iter().sum::<f64>() / (2.0 * speeds_squared.len() as f64);
    let chi2_per_dof = speed_shape(&speeds_squared, temperature);
    let pass = drift < cfg.drift_limit
        && (temperature - trajectory.meta.temperature).abs() < cfg.temperature_limit
        && chi2_per_dof < cfg.shape_limit;
    Ok(Report {
        drift,
        temperature,
        target_temperature: trajectory.meta.temperature,
        chi2_per_dof,
        maximum_energy_mismatch,
        pass,
    })
}

pub fn format_report(report: &Report, cfg: &CheckCfg) -> String {
    let drift_ok = report.drift < cfg.drift_limit;
    let temperature_ok =
        (report.temperature - report.target_temperature).abs() < cfg.temperature_limit;
    let shape_ok = report.chi2_per_dof < cfg.shape_limit;
    format!(
        "secular drift: {:.6e} < {:.1e} ... {}\n\
         T_speed: {:.6} (|T_speed - {:.3}| < {:.3}) ... {}\n\
         chi2/dof: {:.6} < {:.3} ... {}\n\
         stored-energy cross-check: max relative mismatch {:.3e} ... PASS\n\
         {}\n",
        report.drift,
        cfg.drift_limit,
        if drift_ok { "PASS" } else { "FAIL" },
        report.temperature,
        report.target_temperature,
        cfg.temperature_limit,
        if temperature_ok { "PASS" } else { "FAIL" },
        report.chi2_per_dof,
        cfg.shape_limit,
        if shape_ok { "PASS" } else { "FAIL" },
        report.maximum_energy_mismatch,
        if report.pass { "PASS" } else { "FAIL" },
    )
}

pub fn make_video(cfg: &VideoCfg) -> Result<String, String> {
    let trajectory = crate::trajectory::read(Path::new(&cfg.input))?;
    crate::video::write_video(&trajectory, Path::new(&cfg.out), cfg.fps)?;
    let size = std::fs::metadata(&cfg.out).map_err(|e| format!("inspect {}: {e}", cfg.out))?.len();
    Ok(format!(
        "wrote {} ({} frames, {:.2} MiB)",
        cfg.out,
        trajectory.frames.len(),
        size as f64 / (1024.0 * 1024.0)
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thermostat_removes_center_of_mass_and_uses_two_n_minus_two_dof() {
        let mut velocities = vec![[1.0, 3.0], [2.0, -1.0], [-1.0, 2.0]];
        remove_mean_and_rescale(&mut velocities, 0.5);
        let mean = velocities.iter().fold([0.0; 2], |s, v| [s[0] + v[0], s[1] + v[1]]);
        let kinetic: f64 = velocities.iter().map(|v| 0.5 * (v[0] * v[0] + v[1] * v[1])).sum();
        assert!(mean[0].abs() < 1e-14 && mean[1].abs() < 1e-14);
        assert!((kinetic / 2.0 - 0.5).abs() < 1e-14);
    }

    #[test]
    fn equal_probability_speed_bins_accept_exact_quantiles() {
        let temperature = 0.5;
        let samples: Vec<f64> = (0..2400)
            .map(|i| {
                let probability = (i as f64 + 0.5) / 2400.0;
                -2.0 * temperature * (1.0 - probability).ln()
            })
            .collect();
        assert!(speed_shape(&samples, temperature) < 1e-12);
    }
}
