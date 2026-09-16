#[cfg(test)]
mod tests {
    use ising::lattice::Lattice;
    use ising::metropolis::{metropolis_sweep, MetropolisTable};
    use ising::wolff::WolffState;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn test_lattice_energy_and_magnetization() {
        let l = 8;
        let lattice = Lattice::new_all_up(l);
        assert_eq!(lattice.spin_sum, 64);
        assert_eq!(lattice.energy, -128);
        assert_eq!(lattice.compute_total_energy(), -128);
        assert!((lattice.magnetization() - 1.0).abs() < 1e-12);
        assert!((lattice.energy_per_site() - (-2.0)).abs() < 1e-12);
    }

    #[test]
    fn test_metropolis_energy_consistency() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let mut lattice = Lattice::new_all_up(8);
        let table = MetropolisTable::new(2.3);
        for _ in 0..10 {
            metropolis_sweep(&mut lattice, &table, &mut rng);
            let exact_e = lattice.compute_total_energy();
            assert_eq!(lattice.energy, exact_e, "Metropolis energy mismatch!");
            let exact_m: i64 = lattice.spins.iter().map(|&s| s as i64).sum();
            assert_eq!(lattice.spin_sum, exact_m, "Metropolis spin sum mismatch!");
        }
    }

    #[test]
    fn test_wolff_energy_consistency() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1042);
        let mut lattice = Lattice::new_all_up(8);
        let mut wolff = WolffState::new(lattice.n);
        for _ in 0..50 {
            wolff.step(&mut lattice, 2.3, &mut rng);
            let exact_e = lattice.compute_total_energy();
            assert_eq!(lattice.energy, exact_e, "Wolff energy mismatch!");
            let exact_m: i64 = lattice.spins.iter().map(|&s| s as i64).sum();
            assert_eq!(lattice.spin_sum, exact_m, "Wolff spin sum mismatch!");
        }
    }
}
