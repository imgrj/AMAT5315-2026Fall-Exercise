//! Boundary-MPS contraction of an open square Ising network (sheet Algorithms 3 and 4).
//!
//! The boundary MPS stores the weights left after absorbing rows. Each core is
//! `B[i][alpha, p, beta]` (left bond, physical down leg, right bond); absorbing a
//! row contracts the physical leg with the row tensor and grows every horizontal
//! bond from at most `chi` to at most `2 chi`. The compression then restores the
//! cap: an exactly norm-preserving right-canonicalisation followed by a
//! left-to-right truncated SVD.
//!
//! The right-to-left sweep of Algorithm 3 only needs *some* orthonormal basis of
//! the right environment; the sheet obtains it from `V^T` of an exact SVD. We use
//! the equivalent LQ factorisation (`B = L R` with orthonormal rows in `R`),
//! which is the same gauge transformation at a tenth of the cost. The truncating
//! sweep keeps the truncated SVD, so every discarded singular value is preserved.

use crate::linalg::{self, frobenius};
use crate::model::Ising;

#[derive(Clone, Debug)]
pub struct Core {
    pub l: usize,
    pub p: usize,
    pub r: usize,
    pub data: Vec<f64>,
}

impl Core {
    fn new(l: usize, p: usize, r: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), l * p * r, "core shape mismatch");
        Core { l, p, r, data }
    }

    fn entries(&self) -> usize {
        self.l * self.p * self.r
    }
}

#[derive(Clone, Debug, Default)]
pub struct RowReport {
    pub row: usize,
    pub scale: f64,
    pub log_scale: f64,
    pub ell: f64,
    pub bond_dims_before: Vec<usize>,
    pub bond_dims_after: Vec<usize>,
    pub discarded: Vec<f64>,
    pub max_discarded: f64,
    pub relative_discarded: f64,
    pub retained_singular_values: Vec<f64>,
    pub contraction_entries: u64,
    pub approximation: &'static str,
}

pub struct BoundaryResult {
    pub g: f64,
    pub log_z: f64,
    pub ell: f64,
    pub bottom_log: f64,
    pub bottom_value: f64,
    pub cores: Vec<Core>,
    pub rows: Vec<RowReport>,
}

/// Right-canonicalise one core in place and hand the triangular factor to its
/// left neighbour: `B[i] = L R` with `R` having orthonormal rows.
fn canonicalise_right(cores: &mut [Core], i: usize) {
    let (rows, cols) = (cores[i].l, cores[i].p * cores[i].r);
    let mut q = vec![0.0; rows * cols];
    let mut coefficients = vec![0.0; rows * rows];
    let mut buffer = vec![0.0; cols];
    for row in 0..rows {
        buffer.copy_from_slice(&cores[i].data[row * cols..(row + 1) * cols]);
        for pass in 0..2 {
            for j in 0..row {
                let mut dot = 0.0;
                for t in 0..cols {
                    dot += q[j * cols + t] * buffer[t];
                }
                if dot != 0.0 {
                    for t in 0..cols {
                        buffer[t] -= dot * q[j * cols + t];
                    }
                    if pass == 0 {
                        coefficients[row * rows + j] += dot;
                    }
                }
            }
        }
        let nrm = frobenius(&buffer);
        coefficients[row * rows + row] = nrm;
        if nrm > 0.0 {
            for t in 0..cols {
                q[row * cols + t] = buffer[t] / nrm;
            }
        }
    }
    // Previous core: B[i-1][alpha, p, q] = sum_gamma B[i-1][alpha, p, gamma] L[gamma, q]
    // with L[gamma, q] = coefficients[gamma][q] (q <= gamma).
    let previous = cores[i - 1].clone();
    let mut updated = vec![0.0; previous.l * previous.p * rows];
    for alpha in 0..previous.l {
        for pp in 0..previous.p {
            for gamma in 0..previous.r {
                let value = previous.data[(alpha * previous.p + pp) * previous.r + gamma];
                if value == 0.0 {
                    continue;
                }
                for qq in 0..=gamma {
                    let coefficient = coefficients[gamma * rows + qq];
                    if coefficient != 0.0 {
                        updated[(alpha * previous.p + pp) * rows + qq] += value * coefficient;
                    }
                }
            }
        }
    }
    cores[i - 1] = Core::new(previous.l, previous.p, rows, updated);
    cores[i] = Core::new(rows, cores[i].p, cores[i].r, q);
}

/// Contract one boundary core with the matching row tensor over the up leg.
/// `tensor` is row-major in `(l_dim, u, d, r_dim)`; the side legs keep their own
/// size, which is one where the row closes an open edge.
fn absorb(core: &Core, tensor: &[f64], l_dim: usize, r_dim: usize) -> Core {
    let left = core.l * l_dim;
    let right = core.r * r_dim;
    let mut data = vec![0.0; left * 2 * right];
    for alpha in 0..core.l {
        for u in 0..2 {
            for beta in 0..core.r {
                let value = core.data[(alpha * core.p + u) * core.r + beta];
                if value == 0.0 {
                    continue;
                }
                for lv in 0..l_dim {
                    for d in 0..2 {
                        for rv in 0..r_dim {
                            let entry = tensor[((lv * 2 + u) * 2 + d) * r_dim + rv];
                            if entry != 0.0 {
                                data[((alpha * l_dim + lv) * 2 + d) * right + (beta * r_dim + rv)] +=
                                    value * entry;
                            }
                        }
                    }
                }
            }
        }
    }
    Core::new(left, 2, right, data)
}

pub fn run(model: &Ising, length: usize, chi: usize) -> BoundaryResult {
    assert!(length >= 2, "L must be at least 2");
    assert!(chi >= 1, "chi must be at least 1");
    let mut cores: Vec<Core> = (0..length)
        .map(|_| Core::new(1, 2, 1, vec![model.b[0], model.b[1]]))
        .collect();
    let mut rows = Vec::with_capacity(length);
    let mut ell = 0.0;
    for y in 0..length {
        let before: Vec<usize> = cores.iter().map(|core| core.l).collect();
        let mut absorbed = Vec::with_capacity(length);
        for i in 0..length {
            let (tensor, l_dim, r_dim) = model.row_tensor(i, length);
            absorbed.push(absorb(&cores[i], &tensor, l_dim, r_dim));
        }
        let contraction_entries: u64 = absorbed.iter().map(|core| core.entries() as u64).sum();
        cores = absorbed;
        // Right-to-left canonicalisation (Algorithm 3, lines 1-6).
        for i in (1..length).rev() {
            canonicalise_right(&mut cores, i);
        }
        // Left-to-right truncation (Algorithm 3, lines 7-12).
        let mut discarded = Vec::new();
        let mut retained = Vec::new();
        for i in 0..(length - 1) {
            let row_count = cores[i].l * cores[i].p;
            let col_count = cores[i].r;
            let total: f64 = cores[i].data.iter().map(|x| x * x).sum();
            let (svd, _) = linalg::truncated_svd(&cores[i].data, row_count, col_count, chi);
            let k = svd.k;
            let kept: f64 = svd.s.iter().map(|s| s * s).sum();
            discarded.push((total - kept).max(0.0));
            if y == 0 {
                retained.extend(svd.s.iter().take(3).copied());
            }
            let left = cores[i].l;
            let physical = cores[i].p;
            cores[i] = Core::new(left, physical, k, svd.u.clone());
            let next = cores[i + 1].clone();
            assert_eq!(next.l, col_count, "bond mismatch at core {i}");
            let mut updated = vec![0.0; k * next.p * next.r];
            for qq in 0..k {
                let weight = svd.s[qq];
                if weight == 0.0 {
                    continue;
                }
                for gamma in 0..next.l {
                    let coefficient = weight * svd.v[gamma * k + qq];
                    if coefficient == 0.0 {
                        continue;
                    }
                    for pp in 0..next.p {
                        for rb in 0..next.r {
                            updated[(qq * next.p + pp) * next.r + rb] +=
                                coefficient * next.data[(gamma * next.p + pp) * next.r + rb];
                        }
                    }
                }
            }
            cores[i + 1] = Core::new(k, next.p, next.r, updated);
        }
        // Scale the whole boundary once through the last core (Algorithm 4, 7-10).
        let last = cores.last().expect("boundary has at least one core");
        let scale = frobenius(&last.data);
        assert!(scale > 0.0 && scale.is_finite(), "boundary lost its scale at row {y}");
        let mut scaled = last.data.clone();
        for value in scaled.iter_mut() {
            *value /= scale;
        }
        let keep = cores.len() - 1;
        let (l, p, r) = (cores[keep].l, cores[keep].p, cores[keep].r);
        cores[keep] = Core::new(l, p, r, scaled);
        ell += scale.ln();
        let after: Vec<usize> = cores.iter().map(|core| core.l).collect();
        let max_discarded = discarded.iter().cloned().fold(0.0f64, f64::max);
        rows.push(RowReport {
            row: y,
            scale,
            log_scale: scale.ln(),
            ell,
            bond_dims_before: before,
            bond_dims_after: after,
            discarded: discarded.clone(),
            max_discarded,
            relative_discarded: discarded.iter().sum::<f64>().sqrt(),
            retained_singular_values: retained,
            contraction_entries,
            approximation: "left-to-right truncated SVD to chi",
        });
    }
    // Bottom edge: close every physical leg with b.
    let mut vector = vec![1.0];
    for core in &cores {
        let mut next = vec![0.0; core.r];
        for alpha in 0..core.l {
            let weight = vector[alpha];
            if weight == 0.0 {
                continue;
            }
            for pp in 0..core.p {
                let combined = weight * model.b[pp];
                if combined == 0.0 {
                    continue;
                }
                for beta in 0..core.r {
                    next[beta] += combined * core.data[(alpha * core.p + pp) * core.r + beta];
                }
            }
        }
        vector = next;
    }
    let bottom = vector[0];
    assert!(bottom.is_finite(), "bottom closure is not finite");
    let log_z = ell + bottom.ln();
    let g = log_z / (length * length) as f64;
    BoundaryResult {
        g,
        log_z,
        ell,
        bottom_log: bottom.ln(),
        bottom_value: bottom,
        cores,
        rows,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_coupling_gives_log_two() {
        // Every configuration has equal weight, so g = log 2 at any size.
        let model = Ising::new(0.0);
        for length in [2usize, 3, 5, 8] {
            let result = super::run(&model, length, 8);
            assert!((result.g - 2.0f64.ln()).abs() < 1e-12, "L={length}: {}", result.g);
        }
    }

    #[test]
    fn bond_dimensions_respect_the_cap() {
        let model = Ising::new(0.44);
        let result = super::run(&model, 8, 4);
        for row in &result.rows {
            for &dim in &row.bond_dims_after {
                assert!(dim <= 4, "bond {dim} above chi");
            }
        }
    }
}
