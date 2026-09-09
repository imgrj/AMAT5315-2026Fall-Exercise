//! Owned particle state, periodic geometry, and pair-force engines.

use crate::{energy, energy_cutoff, force, force_cutoff};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ForceStrategy {
    Naive,
    #[default]
    Cells,
}

impl ForceStrategy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Naive => "naive",
            Self::Cells => "cells",
        }
    }

    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "naive" => Ok(Self::Naive),
            "cells" => Ok(Self::Cells),
            _ => Err(format!("unknown force strategy {value:?}")),
        }
    }
}

pub struct System {
    pub positions: Vec<[f64; 2]>,
    pub velocities: Vec<[f64; 2]>,
    pub(crate) accelerations: Vec<[f64; 2]>,
    pub(crate) box_size: Option<[f64; 2]>,
    pub(crate) rc: Option<f64>,
    force_strategy: ForceStrategy,
}

impl System {
    pub fn new(positions: Vec<[f64; 2]>, velocities: Vec<[f64; 2]>) -> Self {
        Self::build(positions, velocities, None, None)
    }

    pub fn periodic(
        positions: Vec<[f64; 2]>,
        velocities: Vec<[f64; 2]>,
        box_size: [f64; 2],
        rc: f64,
    ) -> Self {
        assert!(box_size.iter().all(|x| x.is_finite() && *x > 2.0 * rc));
        Self::build(positions, velocities, Some(box_size), Some(rc))
    }

    fn build(
        positions: Vec<[f64; 2]>,
        velocities: Vec<[f64; 2]>,
        box_size: Option<[f64; 2]>,
        rc: Option<f64>,
    ) -> Self {
        assert_eq!(positions.len(), velocities.len());
        let mut result = Self {
            positions,
            velocities,
            accelerations: Vec::new(),
            box_size,
            rc,
            force_strategy: ForceStrategy::Naive,
        };
        result.wrap_positions();
        result.refresh_accelerations();
        result
    }

    pub fn n_atoms(&self) -> usize {
        self.positions.len()
    }

    pub fn box_size(&self) -> Option<[f64; 2]> {
        self.box_size
    }

    pub fn set_force_strategy(&mut self, strategy: ForceStrategy) {
        self.force_strategy = strategy;
        self.refresh_accelerations();
    }

    pub fn force_strategy(&self) -> ForceStrategy {
        self.force_strategy
    }

    pub(crate) fn wrap_positions(&mut self) {
        if let Some(b) = self.box_size {
            for p in &mut self.positions {
                p[0] = p[0].rem_euclid(b[0]);
                p[1] = p[1].rem_euclid(b[1]);
            }
        }
    }

    fn displacement(&self, i: usize, j: usize) -> [f64; 2] {
        let mut d = [
            self.positions[i][0] - self.positions[j][0],
            self.positions[i][1] - self.positions[j][1],
        ];
        if let Some(b) = self.box_size {
            d[0] -= b[0] * (d[0] / b[0]).round();
            d[1] -= b[1] * (d[1] / b[1]).round();
        }
        d
    }

    fn scan_pairs(
        &self,
        strategy: ForceStrategy,
        mut visit: impl FnMut(usize, usize, [f64; 2], f64),
    ) {
        if strategy == ForceStrategy::Cells && self.box_size.is_some() && self.rc.is_some() {
            self.scan_cells(&mut visit);
        } else {
            self.scan_naive(&mut visit);
        }
    }

    fn scan_naive(&self, visit: &mut impl FnMut(usize, usize, [f64; 2], f64)) {
        for i in 0..self.n_atoms() {
            for j in (i + 1)..self.n_atoms() {
                self.visit_if_interacting(i, j, visit);
            }
        }
    }

    fn visit_if_interacting(
        &self,
        i: usize,
        j: usize,
        visit: &mut impl FnMut(usize, usize, [f64; 2], f64),
    ) {
        let d = self.displacement(i, j);
        let r2 = d[0] * d[0] + d[1] * d[1];
        if r2 == 0.0 {
            return;
        }
        let r = r2.sqrt();
        if self.rc.is_none_or(|rc| r < rc) {
            visit(i, j, d, r);
        }
    }

    fn scan_cells(&self, visit: &mut impl FnMut(usize, usize, [f64; 2], f64)) {
        let b = self.box_size.unwrap();
        let rc = self.rc.unwrap();
        let nx = (b[0] / rc).floor() as usize;
        let ny = (b[1] / rc).floor() as usize;
        assert!(nx >= 2 && ny >= 2);
        let cell = [b[0] / nx as f64, b[1] / ny as f64];
        let mut grid = vec![Vec::<usize>::new(); nx * ny];
        for (i, p) in self.positions.iter().enumerate() {
            let cx = ((p[0].rem_euclid(b[0]) / cell[0]).floor() as usize).min(nx - 1);
            let cy = ((p[1].rem_euclid(b[1]) / cell[1]).floor() as usize).min(ny - 1);
            grid[cy * nx + cx].push(i);
        }

        for cy in 0..ny {
            for cx in 0..nx {
                let c = cy * nx + cx;
                for a in 0..grid[c].len() {
                    for q in (a + 1)..grid[c].len() {
                        self.visit_if_interacting(grid[c][a], grid[c][q], visit);
                    }
                }
                let mut neighbours = Vec::with_capacity(8);
                for oy in -1isize..=1 {
                    for ox in -1isize..=1 {
                        if ox == 0 && oy == 0 {
                            continue;
                        }
                        let mx = (cx as isize + ox).rem_euclid(nx as isize) as usize;
                        let my = (cy as isize + oy).rem_euclid(ny as isize) as usize;
                        let other = my * nx + mx;
                        if other > c && !neighbours.contains(&other) {
                            neighbours.push(other);
                        }
                    }
                }
                for other in neighbours {
                    for &i in &grid[c] {
                        for &j in &grid[other] {
                            self.visit_if_interacting(i, j, visit);
                        }
                    }
                }
            }
        }
    }

    pub(crate) fn refresh_accelerations(&mut self) {
        let mut result = vec![[0.0; 2]; self.n_atoms()];
        self.scan_pairs(self.force_strategy, |i, j, d, r| {
            let radial = self.rc.map_or_else(|| force(r), |rc| force_cutoff(r, rc)) / r;
            for axis in 0..2 {
                let component = radial * d[axis];
                result[i][axis] += component;
                result[j][axis] -= component;
            }
        });
        self.accelerations = result;
    }

    pub fn accelerations(&self) -> &[[f64; 2]] {
        &self.accelerations
    }

    pub fn pair_energy(&self) -> f64 {
        let mut result = 0.0;
        self.scan_pairs(self.force_strategy, |_i, _j, _d, r| {
            result += self.rc.map_or_else(|| energy(r), |rc| energy_cutoff(r, rc));
        });
        result
    }

    pub fn kinetic_energy(&self) -> f64 {
        self.velocities
            .iter()
            .map(|v| 0.5 * (v[0] * v[0] + v[1] * v[1]))
            .sum()
    }

    pub fn total_energy(&self) -> f64 {
        self.pair_energy() + self.kinetic_energy()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    fn compare(positions: Vec<[f64; 2]>, b: [f64; 2]) {
        let velocities = vec![[0.0; 2]; positions.len()];
        let mut naive = System::periodic(positions.clone(), velocities.clone(), b, 2.5);
        naive.set_force_strategy(ForceStrategy::Naive);
        let mut cells = System::periodic(positions, velocities, b, 2.5);
        cells.set_force_strategy(ForceStrategy::Cells);
        let scale = naive
            .accelerations()
            .iter()
            .flatten()
            .map(|x| x.abs())
            .fold(1.0, f64::max);
        for (a, c) in naive.accelerations().iter().flatten().zip(cells.accelerations().iter().flatten()) {
            assert!((a - c).abs() < 2e-12 * scale, "{a} != {c}");
        }
        let e_scale = naive.pair_energy().abs().max(1.0);
        assert!((naive.pair_energy() - cells.pair_energy()).abs() < 2e-12 * e_scale);
    }

    #[test]
    fn total_internal_force_vanishes() {
        let (p, b) = crate::triangular_lattice(100, 0.8).unwrap();
        let mut s = System::periodic(p, vec![[0.0; 2]; 100], b, 2.5);
        s.set_force_strategy(ForceStrategy::Cells);
        let sum = s.accelerations().iter().fold([0.0; 2], |a, x| [a[0] + x[0], a[1] + x[1]]);
        assert!(sum[0].abs() < 1e-10 && sum[1].abs() < 1e-10, "{sum:?}");
    }

    #[test]
    fn cells_match_naive_on_perturbed_lattice_and_boundaries() {
        let (mut p, b) = crate::triangular_lattice(100, 0.8).unwrap();
        let mut rng = Rng::new(2026);
        for x in &mut p {
            x[0] = (x[0] + 0.03 * (rng.uniform() - 0.5)).rem_euclid(b[0]);
            x[1] = (x[1] + 0.03 * (rng.uniform() - 0.5)).rem_euclid(b[1]);
        }
        p[0] = [0.02, b[1] / 2.0];
        p[1] = [b[0] - 0.98, b[1] / 2.0];
        compare(p, b);
    }

    #[test]
    fn cells_match_naive_at_cutoff_and_in_two_cell_box() {
        compare(
            vec![
                [0.01, 0.01],
                [2.5, 0.01],
                [2.5 + 1e-10, 2.5],
                [5.19, 5.19],
                [0.2, 5.1],
            ],
            [5.2, 5.2],
        );
    }

    #[test]
    fn minimum_image_crosses_each_rectangular_boundary() {
        let b = [10.0, 8.0];
        let mut s = System::periodic(
            vec![[0.1, 0.1], [9.9, 7.9]],
            vec![[0.0; 2]; 2],
            b,
            2.5,
        );
        s.set_force_strategy(ForceStrategy::Cells);
        assert!(s.accelerations()[0][0] > 0.0);
        assert!(s.accelerations()[0][1] > 0.0);
    }
}
