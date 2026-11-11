//! Dense linear algebra: exact one-sided Jacobi SVD, a Gram-matrix Lanczos
//! truncated SVD, truncated reconstruction, and a minimal NumPy `.npy` writer.
//!
//! Every matrix is row-major, so `a[i * n + j]` is row `i`, column `j`.
//! A thin SVD is stored as `M = U diag(s) V^T`; `u` is `m x k` row-major
//! (`u[i*k + q]` is row `i` of column `q`) and `v` is `n x k` row-major.

use crate::rng::Gaussian;

#[derive(Clone, Debug)]
pub struct ThinSvd {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    /// `m x k`, row-major.
    pub u: Vec<f64>,
    /// `k` singular values, non-increasing.
    pub s: Vec<f64>,
    /// `n x k`, row-major: column `q` of `V`.
    pub v: Vec<f64>,
}

impl ThinSvd {
    /// `M_chi = U[:chi] diag(s[:chi]) V[:chi]^T`, row-major `m x n`.
    pub fn reconstruct(&self, m: usize, n: usize, chi: usize) -> Vec<f64> {
        let k = chi.min(self.k);
        let mut out = vec![0.0; m * n];
        for q in 0..k {
            let weight = self.s[q];
            if weight == 0.0 {
                continue;
            }
            for i in 0..m {
                let left = self.u[i * self.k + q] * weight;
                if left == 0.0 {
                    continue;
                }
                let row = &mut out[i * n..(i + 1) * n];
                for j in 0..n {
                    row[j] += left * self.v[j * self.k + q];
                }
            }
        }
        out
    }

    /// Squared Frobenius weight carried by the directions `q >= chi`.
    pub fn discarded_weight(&self, chi: usize) -> f64 {
        self.s.iter().skip(chi).map(|s| s * s).sum()
    }
}

/// Frobenius norm of a flat array.
pub fn frobenius(values: &[f64]) -> f64 {
    values.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right.iter()).map(|(a, b)| a * b).sum()
}

/// Exact one-sided Jacobi SVD of an `m x n` matrix with `m >= n`.
/// Returns the thin SVD with `k = n` and the number of sweeps used.
fn jacobi_svd_tall(a: &[f64], m: usize, n: usize) -> (ThinSvd, usize) {
    assert!(m >= n, "jacobi_svd_tall needs m >= n");
    // Columns of A stored contiguously: column p is w[p*m .. (p+1)*m].
    let mut w = vec![0.0; n * m];
    for p in 0..n {
        for i in 0..m {
            w[p * m + i] = a[i * n + p];
        }
    }
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    let threshold = 1e-15;
    let mut sweeps = 0;
    for _ in 0..60 {
        sweeps += 1;
        let mut rotated = false;
        for p in 0..n.saturating_sub(1) {
            for q in (p + 1)..n {
                let (mut alpha, mut beta, mut gamma) = (0.0, 0.0, 0.0);
                for i in 0..m {
                    let wp = w[p * m + i];
                    let wq = w[q * m + i];
                    alpha += wp * wp;
                    beta += wq * wq;
                    gamma += wp * wq;
                }
                if gamma == 0.0 || gamma.abs() <= threshold * (alpha * beta).sqrt() {
                    continue;
                }
                rotated = true;
                let zeta = (beta - alpha) / (2.0 * gamma);
                let sign = if zeta >= 0.0 { 1.0 } else { -1.0 };
                let t = sign / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                for i in 0..m {
                    let wp = w[p * m + i];
                    let wq = w[q * m + i];
                    w[p * m + i] = c * wp - s * wq;
                    w[q * m + i] = s * wp + c * wq;
                }
                for i in 0..n {
                    let vp = v[i * n + p];
                    let vq = v[i * n + q];
                    v[i * n + p] = c * vp - s * vq;
                    v[i * n + q] = s * vp + c * vq;
                }
            }
        }
        if !rotated {
            break;
        }
    }
    let mut s = vec![0.0; n];
    for p in 0..n {
        s[p] = frobenius(&w[p * m..(p + 1) * m]);
    }
    let mut u = vec![0.0; m * n];
    for p in 0..n {
        if s[p] > 0.0 {
            for i in 0..m {
                u[i * n + p] = w[p * m + i] / s[p];
            }
        }
    }
    let mut svd = ThinSvd { m, n, k: n, u, s, v };
    canonicalise(&mut svd);
    (svd, sweeps)
}

/// Exact thin SVD of an arbitrary `m x n` matrix.
pub fn jacobi_svd(a: &[f64], m: usize, n: usize) -> ThinSvd {
    if m >= n {
        jacobi_svd_tall(a, m, n).0
    } else {
        let mut t = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                t[j * m + i] = a[i * n + j];
            }
        }
        let transposed = jacobi_svd_tall(&t, n, m).0;
        // A^T = Ut S Vt^T  =>  A = Vt S Ut^T.
        ThinSvd { m, n, k: transposed.k, u: transposed.v, s: transposed.s, v: transposed.u }
    }
}

/// Sort the singular triples into non-increasing order and fix a deterministic sign.
fn canonicalise(svd: &mut ThinSvd) {
    let (m, n, k) = (svd.m, svd.n, svd.k);
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&a, &b| svd.s[b].total_cmp(&svd.s[a]));
    let mut u = vec![0.0; m * k];
    let mut v = vec![0.0; n * k];
    let mut s = vec![0.0; k];
    for (position, &source) in order.iter().enumerate() {
        s[position] = svd.s[source];
        for i in 0..m {
            u[i * k + position] = svd.u[i * k + source];
        }
        for j in 0..n {
            v[j * k + position] = svd.v[j * k + source];
        }
    }
    for q in 0..k {
        let mut best = 0.0f64;
        let mut index = 0usize;
        for i in 0..m {
            if u[i * k + q].abs() > best {
                best = u[i * k + q].abs();
                index = i;
            }
        }
        if u[index * k + q] < 0.0 {
            for i in 0..m {
                u[i * k + q] = -u[i * k + q];
            }
            for j in 0..n {
                v[j * k + q] = -v[j * k + q];
            }
        }
    }
    svd.u = u;
    svd.v = v;
    svd.s = s;
}

/// Result metadata of a truncated SVD.
#[derive(Clone, Debug, Default)]
pub struct TruncationStats {
    pub method: &'static str,
    pub steps: usize,
    pub converged: bool,
    pub residual: f64,
}

/// Two passes of modified Gram-Schmidt against the stored columns.
fn reorthogonalise(buffer: &mut [f64], columns: &[f64], count: usize, stride: usize) {
    for _pass in 0..2 {
        for q in 0..count {
            let column = &columns[q * stride..(q + 1) * stride];
            let coefficient = dot(column, buffer);
            if coefficient != 0.0 {
                for i in 0..stride {
                    buffer[i] -= coefficient * column[i];
                }
            }
        }
    }
}

/// Eigenpairs of a symmetric tridiagonal matrix `T` (`k x k`) by an exact
/// Jacobi SVD of `T`: for symmetric `T` the singular values are `|lambda|` and
/// the singular vectors are the eigenvectors, so the sign is recovered from the
/// Rayleigh quotient.
fn tridiagonal_eigen(diagonal: &[f64], off_diagonal: &[f64], k: usize) -> (Vec<f64>, Vec<f64>) {
    let mut t = vec![0.0; k * k];
    for i in 0..k {
        t[i * k + i] = diagonal[i];
        if i + 1 < k {
            t[i * k + i + 1] = off_diagonal[i + 1];
            t[(i + 1) * k + i] = off_diagonal[i + 1];
        }
    }
    let svd = jacobi_svd(&t, k, k);
    let mut values = Vec::with_capacity(k);
    let mut vectors = vec![0.0; k * k];
    for q in 0..k {
        let mut product = vec![0.0; k];
        for i in 0..k {
            let mut acc = 0.0;
            for j in 0..k {
                acc += t[i * k + j] * svd.u[j * k + q];
            }
            product[i] = acc;
        }
        let rayleigh: f64 = (0..k).map(|i| product[i] * svd.u[i * k + q]).sum();
        let sign = if rayleigh >= 0.0 { 1.0 } else { -1.0 };
        values.push(sign * svd.s[q]);
        for i in 0..k {
            vectors[i * k + q] = sign * svd.u[i * k + q];
        }
    }
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&a, &b| values[b].total_cmp(&values[a]));
    let mut sorted_values = Vec::with_capacity(k);
    let mut sorted_vectors = vec![0.0; k * k];
    for (position, &source) in order.iter().enumerate() {
        sorted_values.push(values[source]);
        for i in 0..k {
            sorted_vectors[i * k + position] = vectors[i * k + source];
        }
    }
    (sorted_values, sorted_vectors)
}

/// Truncated SVD by a matrix-free symmetric Lanczos iteration.
///
/// The operator is `G = A^T A` (or `A A^T` when `m < n`), applied as
/// `G x = A^T (A x)`, so no `small x small` Gram matrix is ever stored: each
/// iteration costs two matrix-vector products instead of `small^3`.
/// Full reorthogonalisation keeps the basis orthonormal, and the singular values
/// are recomputed as `||A v||` so they do not inherit the squared conditioning.
/// A breakdown of the Krylov space means the invariant subspace is exhausted, so
/// its Ritz values are exact 鈥?degenerate and rank-deficient matrices included.
pub fn krylov_svd(a: &[f64], m: usize, n: usize, r: usize, tol: f64) -> (ThinSvd, TruncationStats) {
    let small = m.min(n);
    let right_side = n <= m; // G = A^T A: eigenvectors belong to the right vectors
    let max_k = (4 * r + 64).min(small).max(r.min(small)).max(1);
    let r = r.min(small);
    let mut basis = vec![0.0; (max_k + 1) * small];
    let mut diagonal = vec![0.0; max_k];
    let mut off_diagonal = vec![0.0; max_k + 1];
    let mut source = Gaussian::new(0x5EED_5EED);
    let mut buffer = vec![0.0; small];
    for value in buffer.iter_mut() {
        *value = source.next();
    }
    let initial = frobenius(&buffer);
    for i in 0..small {
        basis[i] = buffer[i] / initial;
    }
    let mut stats = TruncationStats { method: "krylov", ..Default::default() };
    let mut steps = 0usize;
    let mut work = vec![0.0; small];
    let mut image = vec![0.0; m.max(n)];
    let scale = (m * n) as f64;
    let breakdown = 1e-13 * scale.sqrt();
    for j in 0..max_k {
        // work = G q_j = A^T (A q_j), or A (A^T q_j) on the left side.
        if right_side {
            for i in 0..m {
                image[i] = dot(&a[i * n..(i + 1) * n], &basis[j * small..(j + 1) * small]);
            }
            work.iter_mut().for_each(|value| *value = 0.0);
            for i in 0..m {
                let weight = image[i];
                if weight == 0.0 {
                    continue;
                }
                let row = &a[i * n..(i + 1) * n];
                for t in 0..n {
                    work[t] += row[t] * weight;
                }
            }
        } else {
            image[..n].iter_mut().for_each(|value| *value = 0.0);
            for i in 0..m {
                let weight = basis[j * small + i];
                if weight == 0.0 {
                    continue;
                }
                let row = &a[i * n..(i + 1) * n];
                for t in 0..n {
                    image[t] += row[t] * weight;
                }
            }
            for i in 0..m {
                work[i] = dot(&a[i * n..(i + 1) * n], &image[..n]);
            }
        }
        diagonal[j] = work
            .iter()
            .enumerate()
            .map(|(i, value)| value * basis[j * small + i])
            .sum::<f64>();
        if j > 0 {
            for i in 0..small {
                work[i] -= off_diagonal[j] * basis[(j - 1) * small + i];
            }
        }
        for i in 0..small {
            work[i] -= diagonal[j] * basis[j * small + i];
        }
        reorthogonalise(&mut work, &basis, j + 1, small);
        off_diagonal[j + 1] = frobenius(&work);
        steps = j + 1;
        if off_diagonal[j + 1] <= breakdown {
            // Invariant subspace: the Ritz values from these `steps` vectors are exact.
            off_diagonal[j + 1] = 0.0;
            break;
        }
        for i in 0..small {
            basis[(j + 1) * small + i] = work[i] / off_diagonal[j + 1];
        }
        if steps >= r + 8 {
            let (values, vectors) = tridiagonal_eigen(&diagonal[..steps], &off_diagonal[..steps], steps);
            let leading = values.first().copied().unwrap_or(0.0).max(1e-300);
            let residual =
                off_diagonal[steps].abs() * vectors[(steps - 1) * steps + (r - 1).min(steps - 1)].abs();
            stats.steps = steps;
            stats.residual = residual;
            if residual <= tol * leading {
                stats.converged = true;
                break;
            }
        }
    }
    stats.steps = steps;
    let (values, vectors) = tridiagonal_eigen(&diagonal[..steps], &off_diagonal[..steps], steps);
    let leading = values.first().copied().unwrap_or(0.0).max(1e-300);
    stats.residual = off_diagonal[steps].abs() * vectors[(steps - 1) * steps + (r - 1).min(steps - 1)].abs();
    stats.converged = stats.residual <= tol * leading;
    // Eigenvectors of the Gram matrix, then the singular triples of A.
    let floor = leading * 1e-15;
    let mut candidates: Vec<usize> = (0..steps).filter(|&q| values[q] > floor).take(r).collect();
    candidates.truncate(r);
    let keep = candidates.len();
    assert!(keep > 0, "truncated SVD found no positive direction");
    let mut out_u = vec![0.0; m * keep];
    let mut out_v = vec![0.0; n * keep];
    let mut out_s = vec![0.0; keep];
    let mut vector = vec![0.0; small];
    for (slot, &q) in candidates.iter().enumerate() {
        // Gram eigenvector in the small space.
        for i in 0..small {
            let mut acc = 0.0;
            for t in 0..steps {
                acc += basis[t * small + i] * vectors[t * steps + q];
            }
            vector[i] = acc;
        }
        if right_side {
            // v is the eigenvector; u = A v / sigma.
            for i in 0..small {
                out_v[i * keep + slot] = vector[i];
            }
            let mut image = vec![0.0; m];
            for i in 0..m {
                let row = &a[i * n..(i + 1) * n];
                image[i] = dot(row, &vector);
            }
            let sigma = frobenius(&image);
            out_s[slot] = sigma;
            if sigma > 0.0 {
                for i in 0..m {
                    out_u[i * keep + slot] = image[i] / sigma;
                }
            }
        } else {
            for i in 0..small {
                out_u[i * keep + slot] = vector[i];
            }
            let mut image = vec![0.0; n];
            for i in 0..small {
                let weight = vector[i];
                if weight == 0.0 {
                    continue;
                }
                let row = &a[i * n..(i + 1) * n];
                for j in 0..n {
                    image[j] += weight * row[j];
                }
            }
            let sigma = frobenius(&image);
            out_s[slot] = sigma;
            if sigma > 0.0 {
                for j in 0..n {
                    out_v[j * keep + slot] = image[j] / sigma;
                }
            }
        }
    }
    (ThinSvd { m, n, k: keep, u: out_u, s: out_s, v: out_v }, stats)
}

/// Truncated SVD to at most `r` directions: an exact Jacobi SVD for matrices
/// with at most 16 rows or columns, Lanczos on the Gram matrix above that.
pub fn truncated_svd(a: &[f64], m: usize, n: usize, r: usize) -> (ThinSvd, TruncationStats) {
    let thin = m.min(n);
    if thin <= 16 {
        let full = jacobi_svd(a, m, n);
        let keep = r.min(full.k);
        let mut svd = ThinSvd { m, n, k: keep, u: vec![0.0; m * keep], s: full.s[..keep].to_vec(), v: vec![0.0; n * keep] };
        for q in 0..keep {
            for i in 0..m {
                svd.u[i * keep + q] = full.u[i * full.k + q];
            }
            for j in 0..n {
                svd.v[j * keep + q] = full.v[j * full.k + q];
            }
        }
        let stats = TruncationStats { method: "jacobi", steps: 0, converged: true, residual: 0.0 };
        (svd, stats)
    } else {
        krylov_svd(a, m, n, r, 1e-12)
    }
}

/// Write an `.npy` v1.0 file of `f64` values in C order.
pub fn write_npy(path: &std::path::Path, data: &[f64], shape: &[usize]) -> std::io::Result<()> {
    use std::io::Write;
    let mut header = String::from("{'descr': '<f8', 'fortran_order': False, 'shape': (");
    for (index, dimension) in shape.iter().enumerate() {
        if index > 0 {
            header.push_str(", ");
        }
        header.push_str(&dimension.to_string());
    }
    if shape.len() == 1 {
        header.push(',');
    }
    header.push_str("), }");
    let preamble = 10usize;
    let mut padded = header;
    while (preamble + padded.len() + 1) % 64 != 0 {
        padded.push(' ');
    }
    padded.push('\n');
    let mut file = std::fs::File::create(path)?;
    file.write_all(b"\x93NUMPY")?;
    file.write_all(&[1, 0])?;
    file.write_all(&(padded.len() as u16).to_le_bytes())?;
    file.write_all(padded.as_bytes())?;
    let mut bytes = Vec::with_capacity(data.len() * 8);
    for value in data {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    file.write_all(&bytes)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_matrix(m: usize, n: usize, seed: u64) -> Vec<f64> {
        let (data, _, _) = crate::rng::gaussian_matrix(m, n, seed);
        data
    }

    #[test]
    fn jacobi_reconstructs_and_orders() {
        let (m, n) = (40, 30);
        let a = sample_matrix(m, n, 7);
        let svd = jacobi_svd(&a, m, n);
        assert_eq!(svd.k, n);
        for q in 1..svd.k {
            assert!(svd.s[q - 1] >= svd.s[q]);
        }
        let rebuilt = svd.reconstruct(m, n, n);
        let error: f64 =
            a.iter().zip(rebuilt.iter()).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt();
        assert!(error < 1e-12, "reconstruction error {error}");
        for p in 0..3 {
            for q in 0..3 {
                let mut product = 0.0;
                for i in 0..m {
                    product += svd.u[i * svd.k + p] * svd.u[i * svd.k + q];
                }
                let expected = if p == q { 1.0 } else { 0.0 };
                assert!((product - expected).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn jacobi_handles_wide_matrices() {
        let (m, n) = (12, 25);
        let a = sample_matrix(m, n, 11);
        let svd = jacobi_svd(&a, m, n);
        assert_eq!(svd.k, m);
        let rebuilt = svd.reconstruct(m, n, m);
        let error: f64 =
            a.iter().zip(rebuilt.iter()).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt();
        assert!(error < 1e-12, "reconstruction error {error}");
    }

    #[test]
    fn gram_lanczos_matches_the_exact_svd() {
        let (m, n) = (300, 260);
        let a = sample_matrix(m, n, 13);
        let exact = jacobi_svd(&a, m, n);
        let (approx, stats) = krylov_svd(&a, m, n, 24, 1e-12);
        assert!(stats.converged, "lanczos did not converge: {stats:?}");
        assert_eq!(approx.k, 24);
        for q in 0..24 {
            let difference = (approx.s[q] - exact.s[q]).abs() / exact.s[0];
            assert!(difference < 1e-10, "singular value {q}: {difference}");
        }
        let rebuilt = approx.reconstruct(m, n, 24);
        let best = exact.reconstruct(m, n, 24);
        let difference: f64 =
            rebuilt.iter().zip(best.iter()).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt();
        assert!(difference < 1e-9 * exact.s[0], "truncated factors differ: {difference}");
    }

    #[test]
    fn gram_lanczos_handles_a_rank_one_matrix() {
        let (m, n) = (40, 30);
        let mut a = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                a[i * n + j] = (i as f64 + 1.0) * (j as f64 + 0.5);
            }
        }
        let exact = jacobi_svd(&a, m, n);
        let (approx, _) = krylov_svd(&a, m, n, 8, 1e-12);
        assert_eq!(approx.k, 1, "only one direction carries weight");
        assert!((approx.s[0] - exact.s[0]).abs() / exact.s[0] < 1e-12);
    }

    #[test]
    fn truncated_svd_keeps_the_strongest_directions() {
        let (m, n) = (60, 50);
        let a = sample_matrix(m, n, 17);
        let full = jacobi_svd(&a, m, n);
        for chi in [1usize, 5, 20] {
            let (svd, _) = truncated_svd(&a, m, n, chi);
            assert_eq!(svd.k, chi);
            let rebuilt = svd.reconstruct(m, n, chi);
            let measured =
                a.iter().zip(rebuilt.iter()).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt();
            let predicted = full.discarded_weight(chi).sqrt();
            assert!((measured - predicted).abs() < 1e-9, "chi {chi}: {measured} vs {predicted}");
        }
    }

    #[test]
    fn npy_header_is_readable() {
        let path = std::env::temp_dir().join("thermo-test.npy");
        write_npy(&path, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[..6], b"\x93NUMPY");
        assert!(String::from_utf8_lossy(&bytes[10..140]).contains("'shape': (2, 3)"));
        assert_eq!(bytes.len() - 10 - u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 48);
        std::fs::remove_file(&path).ok();
    }
}
