//! Two-dimensional Lennard-Jones molecular dynamics in reduced units.

pub mod cli;
mod integrator;
pub mod ops;
pub mod rdf;
pub mod rng;
mod system;
pub mod trajectory;
pub mod video;

pub use integrator::{Euler, Integrator, VelocityVerlet};
pub use system::{ForceStrategy, System};

/// Greeting shared by the library test and the no-argument executable.
pub fn greeting() -> &'static str {
    "Hello, world!"
}

/// Lennard-Jones pair energy U(r) = 4(r^-12 - r^-6).
pub fn energy(r: f64) -> f64 {
    let inv_r2 = 1.0 / (r * r);
    let inv_r6 = inv_r2 * inv_r2 * inv_r2;
    4.0 * inv_r6 * (inv_r6 - 1.0)
}

/// Radial pair-force magnitude F(r) = -dU/dr.
pub fn force(r: f64) -> f64 {
    let inv_r2 = 1.0 / (r * r);
    let inv_r6 = inv_r2 * inv_r2 * inv_r2;
    24.0 / r * (2.0 * inv_r6 * inv_r6 - inv_r6)
}

/// Shifted-potential cutoff from the learning sheet.
pub fn energy_cutoff(r: f64, rc: f64) -> f64 {
    if r < rc { energy(r) - energy(rc) } else { 0.0 }
}

/// The cutoff leaves the LJ force unchanged inside and zero outside.
pub fn force_cutoff(r: f64, rc: f64) -> f64 {
    if r < rc { force(r) } else { 0.0 }
}

/// Run one experiment through any integrator implementing the shared trait.
pub fn run<I: Integrator + ?Sized>(
    integrator: &I,
    system: &mut System,
    dt: f64,
    steps: usize,
) -> Vec<f64> {
    let e0 = system.total_energy();
    (0..steps)
        .map(|_| {
            integrator.step(system, dt);
            (system.total_energy() - e0) / e0.abs()
        })
        .collect()
}

/// Two atoms released from rest at separation 1.2.
pub fn dimer() -> System {
    System::new(
        vec![[0.0, 0.0], [1.2, 0.0]],
        vec![[0.0, 0.0], [0.0, 0.0]],
    )
}

/// A square-count triangular lattice in its natural rectangular periodic box.
pub fn triangular_lattice(n: usize, rho: f64) -> Result<(Vec<[f64; 2]>, [f64; 2]), String> {
    let side = (n as f64).sqrt() as usize;
    if side * side != n || side < 2 || side % 2 != 0 {
        return Err(format!("--n {n} must be an even square (4, 16, 36, ...)"));
    }
    if !rho.is_finite() || rho <= 0.0 {
        return Err("--rho must be finite and positive".into());
    }
    let a = (2.0 / (3.0_f64.sqrt() * rho)).sqrt();
    let h = 0.5 * 3.0_f64.sqrt() * a;
    let box_size = [side as f64 * a, side as f64 * h];
    let mut positions = Vec::with_capacity(n);
    for row in 0..side {
        for col in 0..side {
            positions.push([
                (col as f64 + 0.5 * (row % 2) as f64) * a,
                row as f64 * h,
            ]);
        }
    }
    Ok((positions, box_size))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greeting_is_hello_world() {
        assert_eq!(greeting(), "Hello, world!");
    }

    #[test]
    fn energy_well_depth_is_minus_one() {
        let r0 = 2.0_f64.powf(1.0 / 6.0);
        assert!((energy(r0) + 1.0).abs() < 1e-12);
    }

    #[test]
    fn force_matches_the_energy_slope() {
        let h = 1e-5;
        for r in [1.0, 1.1, 1.2, 1.5, 2.2] {
            let numerical = -(energy(r + h) - energy(r - h)) / (2.0 * h);
            let tolerance = 1e-6 * force(r).abs().max(1.0);
            assert!((force(r) - numerical).abs() < tolerance, "r={r}");
        }
    }

    #[test]
    fn shifted_potential_is_continuous_at_cutoff() {
        let rc = 2.5;
        assert_eq!(energy_cutoff(rc, rc), 0.0);
        assert!(energy_cutoff(rc - 1e-9, rc).abs() < 1e-9);
        assert_eq!(force_cutoff(rc, rc), 0.0);
        assert!((force_cutoff(rc - 1e-9, rc) - force(rc - 1e-9)).abs() < 1e-15);
    }

    #[test]
    fn lattice_matches_sheet_geometry_and_density() {
        let (positions, b) = triangular_lattice(100, 0.8).unwrap();
        let a = (2.0 / (3.0_f64.sqrt() * 0.8)).sqrt();
        let h = 0.5 * 3.0_f64.sqrt() * a;
        assert_eq!(positions.len(), 100);
        assert!((b[0] - 10.0 * a).abs() < 1e-12);
        assert!((b[1] - 10.0 * h).abs() < 1e-12);
        assert!((100.0 / (b[0] * b[1]) - 0.8).abs() < 1e-12);
        assert_eq!(positions[10], [0.5 * a, h]);
    }

    #[test]
    fn dimer_conservation_uses_one_trait_driver() {
        fn experiment(method: &dyn Integrator, steps: usize) -> Vec<f64> {
            run(method, &mut dimer(), 0.01, steps)
        }
        let verlet = experiment(&VelocityVerlet, 500);
        let euler = experiment(&Euler, 500);
        let long_verlet = experiment(&VelocityVerlet, 5000);
        let maximum = |xs: &[f64]| xs.iter().map(|x| x.abs()).fold(0.0, f64::max);
        assert!(maximum(&verlet) < 1e-3);
        assert!(*euler.last().unwrap() > 0.5);
        assert!(maximum(&long_verlet) < 1e-3);
    }

    #[test]
    fn verlet_is_time_reversible() {
        let mut s = dimer();
        for _ in 0..200 {
            VelocityVerlet.step(&mut s, 0.01);
        }
        for v in &mut s.velocities {
            v[0] = -v[0];
            v[1] = -v[1];
        }
        for _ in 0..200 {
            VelocityVerlet.step(&mut s, 0.01);
        }
        assert!(s.positions[0][0].abs() < 1e-10);
        assert!((s.positions[1][0] - 1.2).abs() < 1e-10);
    }
}
