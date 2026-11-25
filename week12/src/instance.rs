//! Random spin glasses, drawn exactly as `branchopt.design.toml` prescribes:
//! an integer-only SplitMix64 stream, a minimum degree of 2, a fixed bond count,
//! couplings in `{+1, -1}` and fields in `{-1, 0, +1}`.

use crate::rng::SplitMix64;
use std::collections::BTreeSet;

pub const ID_SALT: u64 = 0xD1B5_4A32_D192_ED03;

#[derive(Clone, Debug)]
pub struct Instance {
    pub id: usize,
    pub n: usize,
    pub m: usize,
    /// Undirected bonds `(u, v)` with `u < v`, sorted lexicographically.
    pub edges: Vec<(usize, usize)>,
    /// Coupling of every bond, parallel to `edges`.
    pub j: Vec<f64>,
    /// Local field of every spin.
    pub h: Vec<f64>,
}

impl Instance {
    /// `H(s) = sum_{(i,j) in E} J_ij s_i s_j + sum_i h_i s_i` for spins in `{-1,+1}`.
    pub fn energy(&self, spins: &[f64]) -> f64 {
        let mut e = 0.0;
        for (e_idx, &(u, v)) in self.edges.iter().enumerate() {
            e += self.j[e_idx] * spins[u] * spins[v];
        }
        for i in 0..self.n {
            e += self.h[i] * spins[i];
        }
        e
    }

    /// Energy of a `0/1` bit vector `x`, using `s = 1 - 2x`.
    pub fn energy_from_bits(&self, x: &[f64]) -> f64 {
        let spins: Vec<f64> = x.iter().map(|&xi| 1.0 - 2.0 * xi).collect();
        self.energy(&spins)
    }

    /// The squared-free nearest rounding: `x_i >= 1/2` becomes `1`, else `0`.
    pub fn nearest_rounding(&self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .map(|&xi| if xi >= 0.5 { 1.0 } else { 0.0 })
            .collect()
    }

    /// Incident bond count of a spin.
    #[allow(dead_code)]
    pub fn degree(&self, v: usize) -> usize {
        self.edges.iter().filter(|&&(a, b)| a == v || b == v).count()
    }
}

/// Draw instance `id` of a family `(n, m)` with the given `seed`.
pub fn generate(seed: u64, id: usize, n: usize, m: usize) -> Instance {
    // The instance stream starts at the first output of the salted generator.
    let mut salted = SplitMix64::new(seed ^ (id as u64).wrapping_mul(ID_SALT));
    let start = salted.next_u64();
    let mut rng = SplitMix64::new(start);

    let mut set: BTreeSet<(usize, usize)> = BTreeSet::new();
    let min_deg = 2.min(n.saturating_sub(1));
    let mut deg = vec![0usize; n];
    for i in 0..n {
        while deg[i] < min_deg {
            let j = rng.below(n as u64) as usize;
            if j == i {
                continue;
            }
            let (u, v) = (i.min(j), i.max(j));
            if set.insert((u, v)) {
                deg[u] += 1;
                deg[v] += 1;
            }
        }
    }
    while set.len() < m {
        let i = rng.below(n as u64) as usize;
        let j = rng.below(n as u64) as usize;
        if i == j {
            continue;
        }
        let (u, v) = (i.min(j), i.max(j));
        set.insert((u, v));
    }

    let edges: Vec<(usize, usize)> = set.into_iter().collect();
    let mut j = Vec::with_capacity(edges.len());
    for _ in 0..edges.len() {
        j.push(if rng.next_u64() & 1 == 0 { 1.0 } else { -1.0 });
    }
    let mut h = Vec::with_capacity(n);
    for _ in 0..n {
        h.push(rng.below(3) as f64 - 1.0);
    }

    Instance {
        id,
        n,
        m: edges.len(),
        edges,
        j,
        h,
    }
}

/// The opening four-spin square of the learning sheet: couplings `+1, +1, +1, -1`
/// on the cycle `0-1-2-3-0` and zero fields.
pub fn square() -> Instance {
    let edges = vec![(0usize, 1usize), (0, 3), (1, 2), (2, 3)];
    // 0-1: +1, 1-2: +1, 2-3: +1, 3-0: -1  (mapped onto the sorted edge list)
    let j = vec![1.0, -1.0, 1.0, 1.0];
    Instance {
        id: 0,
        n: 4,
        m: edges.len(),
        edges,
        j,
        h: vec![0.0; 4],
    }
}

/// The four-spin frustrated tetrahedron `K4` with all six couplings `+1` and a
/// field `h_0 = +1`: the five-node pruned search of Part 2.
pub fn tetrahedron() -> Instance {
    let mut edges = Vec::new();
    for u in 0..4usize {
        for v in (u + 1)..4 {
            edges.push((u, v));
        }
    }
    let m = edges.len();
    Instance {
        id: 0,
        n: 4,
        m,
        edges,
        j: vec![1.0; m],
        h: vec![1.0, 0.0, 0.0, 0.0],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_graphs_respect_the_family() {
        for id in 0..4 {
            let inst = generate(2026, id, 22, 52);
            assert_eq!(inst.n, 22);
            assert_eq!(inst.edges.len(), 52);
            assert_eq!(inst.j.len(), 52);
            assert_eq!(inst.h.len(), 22);
            assert!(inst.edges.windows(2).all(|w| w[0] < w[1]));
            assert!(inst.edges.iter().all(|&(u, v)| u < v && v < 22));
            for v in 0..22 {
                assert!(inst.degree(v) >= 2);
            }
            assert!(inst.j.iter().all(|&x| x == 1.0 || x == -1.0));
            assert!(inst.h.iter().all(|&x| x == -1.0 || x == 0.0 || x == 1.0));
        }
    }

    #[test]
    fn instances_differ_between_ids() {
        let a = generate(2026, 0, 22, 52);
        let b = generate(2026, 1, 22, 52);
        assert_ne!(a.edges, b.edges);
    }

    #[test]
    fn square_ground_state_is_minus_two() {
        let sq = square();
        // [0,0,1,0] obeys the 0/1 convention: x = 1 means spin -1.
        assert_eq!(sq.energy_from_bits(&[0.0, 0.0, 1.0, 0.0]), -2.0);
        // all spins -1 costs +2
        assert_eq!(sq.energy_from_bits(&[1.0, 1.0, 1.0, 1.0]), 2.0);
    }
}
