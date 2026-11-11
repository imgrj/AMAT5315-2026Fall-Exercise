//! Square-lattice tensor renormalization group (TRG) on a periodic square (sheet Algorithm 5).
//!
//! One round normalises the representative tensor by its largest absolute entry,
//! splits it along the two diagonal cuts, and regroups four half-tensors into one
//! new site:
//!
//! ```text
//! A'[r,u,l,d] = sum_{w,a,b,g} S1[w,a,r] S2[a,b,u] S3[l,b,g] S4[d,g,w]
//! ```
//!
//! with `(S1,S3) = Split(A,(l,u),(r,d),chi)` and `(S2,S4) = Split(A,(d,l),(u,r),chi)`.
//! The final closure is `z = sum_{i,j} A[i,j,i,j]`, two independent sums.
//!
//! The two splits need not retain the same number of directions, so the legs
//! keep the dims `[r,u,l,d] = [q1,q2,q1,q2]` of the sheet's ring indices: the
//! first split leaves `q1`, the second `q2`, and both are at most `chi`.

use crate::linalg::{self, frobenius};
use crate::model::Ising;

#[derive(Clone, Debug)]
pub struct StepReport {
    pub step: usize,
    pub tensors: u64,
    pub max_abs: f64,
    pub log_scale: f64,
    pub ell: f64,
    pub source_dims: [usize; 4],
    pub bond_dims: [usize; 4],
    pub split1_singular_values: Vec<f64>,
    pub split2_singular_values: Vec<f64>,
    pub regroup_entries: u64,
    pub truncation_method: &'static str,
}

pub struct TrgResult {
    pub g: f64,
    pub log_z: f64,
    pub ell: f64,
    pub z: f64,
    pub dims: [usize; 4],
    pub tensor: Vec<f64>,
    pub steps: Vec<StepReport>,
}

/// Split along `(l,u) | (r,d)`: returns `S1[l,u,q]`, `S3[q,r,d]` and the spectrum.
fn split_lurd(a: &[f64], dims: [usize; 4], chi: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, &'static str) {
    let [dr, du, dl, dd] = dims;
    let rows = dl * du;
    let cols = dr * dd;
    assert_eq!(rows, cols, "the two diagonal cuts must be square");
    let mut matrix = vec![0.0; rows * cols];
    for rv in 0..dr {
        for u in 0..du {
            for lv in 0..dl {
                for dv in 0..dd {
                    matrix[(lv * du + u) * cols + (rv * dd + dv)] =
                        a[((rv * du + u) * dl + lv) * dd + dv];
                }
            }
        }
    }
    let (svd, stats) = linalg::truncated_svd(&matrix, rows, cols, chi);
    let k = svd.k;
    let mut s1 = vec![0.0; dl * du * k];
    let mut s3 = vec![0.0; k * dr * dd];
    for lv in 0..dl {
        for u in 0..du {
            for q in 0..k {
                s1[(lv * du + u) * k + q] = svd.s[q].sqrt() * svd.u[(lv * du + u) * k + q];
            }
        }
    }
    for q in 0..k {
        let weight = svd.s[q].sqrt();
        for rv in 0..dr {
            for dv in 0..dd {
                s3[(q * dr + rv) * dd + dv] = weight * svd.v[(rv * dd + dv) * k + q];
            }
        }
    }
    (s1, s3, svd.s, stats.method)
}

/// Split along `(d,l) | (u,r)`: returns `S2[d,l,q]`, `S4[q,u,r]` and the spectrum.
fn split_dlur(a: &[f64], dims: [usize; 4], chi: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, &'static str) {
    let [dr, du, dl, dd] = dims;
    let rows = dd * dl;
    let cols = du * dr;
    assert_eq!(rows, cols, "the two diagonal cuts must be square");
    let mut matrix = vec![0.0; rows * cols];
    for rv in 0..dr {
        for u in 0..du {
            for lv in 0..dl {
                for dv in 0..dd {
                    matrix[(dv * dl + lv) * cols + (u * dr + rv)] =
                        a[((rv * du + u) * dl + lv) * dd + dv];
                }
            }
        }
    }
    let (svd, stats) = linalg::truncated_svd(&matrix, rows, cols, chi);
    let k = svd.k;
    let mut s2 = vec![0.0; dd * dl * k];
    let mut s4 = vec![0.0; k * du * dr];
    for dv in 0..dd {
        for lv in 0..dl {
            for q in 0..k {
                s2[(dv * dl + lv) * k + q] = svd.s[q].sqrt() * svd.u[(dv * dl + lv) * k + q];
            }
        }
    }
    for q in 0..k {
        let weight = svd.s[q].sqrt();
        for u in 0..du {
            for rv in 0..dr {
                s4[(q * du + u) * dr + rv] = weight * svd.v[(u * dr + rv) * k + q];
            }
        }
    }
    (s2, s4, svd.s, stats.method)
}

/// One TRG round: the exact regrouping of the four half-tensors.
#[allow(clippy::too_many_arguments)]
fn regroup(
    s1: &[f64], s2: &[f64], s3: &[f64], s4: &[f64], dims: [usize; 4], q1: usize, q2: usize,
) -> Vec<f64> {
    let [dr, du, dl, dd] = dims;
    // P[w,b,r,u] = sum_a S1[w,a,r] S2[a,b,u]
    let mut p_tensor = vec![0.0; dl * dr * q1 * q2];
    for w in 0..dl {
        for a in 0..du {
            for rv in 0..q1 {
                let value = s1[(w * du + a) * q1 + rv];
                if value == 0.0 {
                    continue;
                }
                for b in 0..dr {
                    for u in 0..q2 {
                        p_tensor[((w * dr + b) * q1 + rv) * q2 + u] +=
                            value * s2[(a * dl + b) * q2 + u];
                    }
                }
            }
        }
    }
    // Q[l,b,d,w] = sum_g S3[l,b,g] S4[d,g,w]
    let mut q_tensor = vec![0.0; q1 * dr * q2 * dl];
    for lv in 0..q1 {
        for b in 0..dr {
            for g in 0..dd {
                let value = s3[(lv * dr + b) * dd + g];
                if value == 0.0 {
                    continue;
                }
                for dv in 0..q2 {
                    for w in 0..dl {
                        q_tensor[((lv * dr + b) * q2 + dv) * dl + w] +=
                            value * s4[(dv * du + g) * dr + w];
                    }
                }
            }
        }
    }
    // A'[(r,u),(l,d)] = sum_{w,b} P[(w,b),(r,u)] Q[(w,b),(l,d)]
    let legs = q1 * q2;
    let mut product = vec![0.0; legs * legs];
    for w in 0..dl {
        for b in 0..dr {
            let mut left = vec![0.0; legs];
            let mut right = vec![0.0; legs];
            for rv in 0..q1 {
                for u in 0..q2 {
                    left[rv * q2 + u] = p_tensor[((w * dr + b) * q1 + rv) * q2 + u];
                }
            }
            for lv in 0..q1 {
                for dv in 0..q2 {
                    right[lv * q2 + dv] = q_tensor[((lv * dr + b) * q2 + dv) * dl + w];
                }
            }
            for i in 0..legs {
                let value = left[i];
                if value == 0.0 {
                    continue;
                }
                let target = &mut product[i * legs..(i + 1) * legs];
                for j in 0..legs {
                    target[j] += value * right[j];
                }
            }
        }
    }
    // product[(r,u),(l,d)] -> A'[r,u,l,d]
    let mut out = vec![0.0; legs * legs];
    for rv in 0..q1 {
        for u in 0..q2 {
            for lv in 0..q1 {
                for dv in 0..q2 {
                    out[((rv * q2 + u) * q1 + lv) * q2 + dv] =
                        product[(rv * q2 + u) * legs + (lv * q2 + dv)];
                }
            }
        }
    }
    out
}

/// Run TRG on a periodic square. When `split_dir` is given, every step's four
/// half-tensors are written as one flat `.npy` file in the order `S1,S2,S3,S4`.
pub fn run(model: &Ising, length: usize, chi: usize, split_dir: Option<&std::path::Path>) -> TrgResult {
    assert!(length >= 2, "L must be at least 2");
    assert!(length.is_power_of_two(), "TRG needs a power-of-two side length");
    assert!(chi >= 1, "chi must be at least 1");
    let steps_log2 = (usize::BITS - 1 - length.leading_zeros()) as usize;
    let n_steps = 2 * steps_log2;
    let mut a = model.site.clone();
    let mut dims = [2usize, 2, 2, 2];
    let mut tensors = (length * length) as u64;
    let mut ell = 0.0;
    let mut reports = Vec::with_capacity(n_steps);
    for step in 1..=n_steps {
        let tensors_before = tensors;
        let max_abs = a.iter().fold(0.0f64, |acc, value| acc.max(value.abs()));
        assert!(max_abs > 0.0, "tensor vanished at step {step}");
        for value in a.iter_mut() {
            *value /= max_abs;
        }
        ell += (tensors_before as f64) * max_abs.ln();
        let source_dims = dims;
        let (s1, s3, spectrum1, method1) = split_lurd(&a, dims, chi);
        let (s2, s4, spectrum2, method2) = split_dlur(&a, dims, chi);
        let q1 = spectrum1.len();
        let q2 = spectrum2.len();
        assert!(q1 >= 1 && q2 >= 1, "a split lost every direction at step {step}");
        if let Some(directory) = split_dir {
            let mut flat = Vec::with_capacity(s1.len() + s2.len() + s3.len() + s4.len());
            flat.extend_from_slice(&s1);
            flat.extend_from_slice(&s2);
            flat.extend_from_slice(&s3);
            flat.extend_from_slice(&s4);
            let path = directory.join(format!("splits-{step}.npy"));
            linalg::write_npy(&path, &flat, &[flat.len()]).expect("split factor file");
        }
        a = regroup(&s1, &s2, &s3, &s4, dims, q1, q2);
        dims = [q1, q2, q1, q2];
        let regroup_entries = (q1 * q2 * q1 * q2) as u64;
        tensors /= 2;
        reports.push(StepReport {
            step,
            tensors: tensors_before,
            max_abs,
            log_scale: max_abs.ln(),
            ell,
            source_dims,
            bond_dims: dims,
            split1_singular_values: spectrum1,
            split2_singular_values: spectrum2,
            regroup_entries,
            truncation_method: method1,
        });
        let _ = method2;
    }
    assert_eq!(tensors, 1, "the last step must leave one tensor");
    let [dr, du, dl, dd] = dims;
    assert_eq!(dr, dl, "opposite legs must match for the closure");
    assert_eq!(du, dd, "opposite legs must match for the closure");
    let mut z = 0.0;
    for i in 0..dr {
        for j in 0..du {
            z += a[((i * du + j) * dl + i) * dd + j];
        }
    }
    assert!(z.is_finite() && z != 0.0, "closure is degenerate: {z}");
    let log_z = ell + z.abs().ln();
    let g = log_z / (length * length) as f64;
    let _ = frobenius(&a);
    TrgResult { g, log_z, ell, z, dims, tensor: a, steps: reports }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_coupling_gives_log_two() {
        let model = Ising::new(0.0);
        for length in [2usize, 4, 8] {
            let result = super::run(&model, length, 8, None);
            assert!((result.g - 2.0f64.ln()).abs() < 1e-12, "L={length}: {}", result.g);
        }
    }

    #[test]
    fn step_count_halves_the_lattice() {
        let model = Ising::new(0.44);
        let result = super::run(&model, 8, 8, None);
        assert_eq!(result.steps.len(), 6);
        assert_eq!(result.steps[0].tensors, 64);
        assert_eq!(result.steps.last().unwrap().tensors, 2);
    }

    #[test]
    fn periodic_two_by_two_matches_the_exact_value() {
        // A 2x2 torus has two parallel bonds per positive direction, so
        // Z = 2 e^{8K} + 2 e^{-8K} + 12 at any K.
        let k: f64 = 0.44;
        let exact = 2.0 * (8.0 * k).exp() + 2.0 * (-8.0 * k).exp() + 12.0;
        let model = Ising::new(k);
        let result = super::run(&model, 2, 16, None);
        assert!((result.log_z - exact.ln()).abs() < 1e-9, "{} vs {}", result.log_z, exact.ln());
        let g = exact.ln() / 4.0;
        assert!((result.g - g).abs() < 1e-9, "{} vs {g}", result.g);
    }
}
