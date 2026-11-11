//! The zero-field square-lattice Ising model as a tensor network.
//!
//! `H = -sum_{(i,j) in E} s_i s_j` with `s_i = +-1`, `J = k_B = 1`, so the
//! Boltzmann weight of one bond is `F(s,t) = exp(K s t)`. The square-root split
//! `F = W W^T` turns every bond into two factors and every spin into a site
//! tensor, exactly as the sheet's equations (14)-(18):
//!
//! ```text
//! W = [[sqrt(cosh K), sqrt(sinh K)],
//!      [sqrt(cosh K), -sqrt(sinh K)]]        rows for s = +1, -1
//! A[r,u,l,d] = sum_{s=+-1} W[s,r] W[s,u] W[s,l] W[s,d]
//! b = [1/sqrt(cosh K), 0]^T                 closes a missing bond: W b = (1,1)^T
//! ```

pub struct Ising {
    pub k: f64,
    /// `w[s*2 + a]`, rows for `s = +1, -1` and columns for the bond direction `a`.
    pub w: [f64; 4],
    /// `b[a]`.
    pub b: [f64; 2],
    /// `site[r,u,l,d]`, index `((r*2 + u)*2 + l)*2 + d`.
    pub site: Vec<f64>,
}

impl Ising {
    pub fn new(k: f64) -> Self {
        assert!(k >= 0.0, "K must be non-negative");
        let cosh = k.cosh();
        let sinh = k.sinh();
        let root_cosh = cosh.sqrt();
        let root_sinh = sinh.sqrt();
        let w = [root_cosh, root_sinh, root_cosh, -root_sinh];
        let b = [1.0 / root_cosh, 0.0];
        let mut site = vec![0.0; 16];
        for r in 0..2 {
            for u in 0..2 {
                for l in 0..2 {
                    for d in 0..2 {
                        let mut value = 0.0;
                        for s in 0..2 {
                            value += w[s * 2 + r] * w[s * 2 + u] * w[s * 2 + l] * w[s * 2 + d];
                        }
                        site[((r * 2 + u) * 2 + l) * 2 + d] = value;
                    }
                }
            }
        }
        Ising { k, w, b, site }
    }

    /// The row tensor `T[i][l,u,d,r]` with the outward side legs closed by `b`.
    /// A closed side leg keeps size one, exactly as in the sheet's equations
    /// `T[1][1,u,d,r] = sum_l b_l A[r,u,l,d]` and `T[L][l,u,d,1] = sum_r A[r,u,l,d] b_r`.
    /// Returns `(data, l_dim, r_dim)` with `data` row-major in `(l_dim, u, d, r_dim)`.
    pub fn row_tensor(&self, index: usize, length: usize) -> (Vec<f64>, usize, usize) {
        let left_edge = index == 0;
        let right_edge = index == length - 1;
        let l_dim = if left_edge { 1 } else { 2 };
        let r_dim = if right_edge { 1 } else { 2 };
        let mut t = vec![0.0; l_dim * 2 * 2 * r_dim];
        for lv in 0..l_dim {
            for u in 0..2 {
                for d in 0..2 {
                    for rv in 0..r_dim {
                        let mut value = 0.0;
                        for left in 0..2 {
                            if !left_edge && left != lv {
                                continue;
                            }
                            for right in 0..2 {
                                if !right_edge && right != rv {
                                    continue;
                                }
                                let weight = if left_edge { self.b[left] } else { 1.0 }
                                    * if right_edge { self.b[right] } else { 1.0 };
                                value += weight * self.site[((right * 2 + u) * 2 + left) * 2 + d];
                            }
                        }
                        t[((lv * 2 + u) * 2 + d) * r_dim + rv] = value;
                    }
                }
            }
        }
        (t, l_dim, r_dim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bond_split_restores_the_boltzmann_weight() {
        for k in [0.0, 0.44, 1.2] {
            let model = Ising::new(k);
            for s in 0..2 {
                for t in 0..2 {
                    let mut rebuilt = 0.0;
                    for a in 0..2 {
                        rebuilt += model.w[s * 2 + a] * model.w[t * 2 + a];
                    }
                    let sign = if s == t { 1.0 } else { -1.0 };
                    assert!((rebuilt - (k * sign).exp()).abs() < 1e-12, "K={k} s={s} t={t}");
                }
            }
        }
    }

    #[test]
    fn missing_bond_closes_with_the_b_vector() {
        for k in [0.0, 0.25, 0.44, 1.0] {
            let model = Ising::new(k);
            for s in 0..2 {
                let closed = model.w[s * 2] * model.b[0] + model.w[s * 2 + 1] * model.b[1];
                assert!((closed - 1.0).abs() < 1e-12, "K={k} s={s}: {closed}");
            }
        }
    }

    #[test]
    fn site_tensor_equals_the_four_factor_sum() {
        let k = 0.37;
        let model = Ising::new(k);
        for a in 0..16 {
            let r = a >> 3;
            let u = (a >> 2) & 1;
            let l = (a >> 1) & 1;
            let d = a & 1;
            let mut expected = 0.0;
            for s in 0..2 {
                expected +=
                    model.w[s * 2 + r] * model.w[s * 2 + u] * model.w[s * 2 + l] * model.w[s * 2 + d];
            }
            assert!((model.site[a] - expected).abs() < 1e-14);
        }
    }
}
