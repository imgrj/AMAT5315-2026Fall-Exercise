use crate::lattice::Lattice;
use crate::rng::Rng;

pub struct MetropolisTable {
    pub exp_table: [f64; 5], // for de in [-8, -4, 0, 4, 8]
}

impl MetropolisTable {
    pub fn new(t: f64) -> Self {
        Self {
            exp_table: [
                1.0,
                1.0,
                1.0,
                (-4.0 / t).exp(),
                (-8.0 / t).exp(),
            ],
        }
    }
}

pub fn metropolis_sweep(
    lattice: &mut Lattice,
    table: &MetropolisTable,
    rng: &mut Rng,
) -> usize {
    let mut accepted = 0usize;
    let n = lattice.n;
    for _ in 0..n {
        let site = rng.gen_range(n);
        let s = lattice.spins[site];
        let nbrs = &lattice.nbrs[site];
        let sum_nbrs = lattice.spins[nbrs[0]] as i32
            + lattice.spins[nbrs[1]] as i32
            + lattice.spins[nbrs[2]] as i32
            + lattice.spins[nbrs[3]] as i32;
        let de = 2 * (s as i32) * sum_nbrs;
        let pass = if de <= 0 {
            true
        } else {
            let p = if de == 4 {
                table.exp_table[3]
            } else {
                table.exp_table[4]
            };
            rng.gen_f64() < p
        };
        if pass {
            lattice.spins[site] = -s;
            lattice.energy += de as i64;
            lattice.spin_sum -= 2 * (s as i64);
            accepted += 1;
        }
    }
    accepted
}
