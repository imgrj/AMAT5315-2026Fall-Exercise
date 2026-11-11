//! Documented pseudo-random generator used for the Part 1 Gaussian matrix.
//!
//! SplitMix64 is the fixed-increment generator of Steele, Lea and Flood,
//! *Fast Splittable Pseudorandom Number Generators* (OOPSLA 2014):
//!
//! ```text
//! state <- state + 0x9E3779B97F4A7C15      (wrapping, 64-bit)
//! z     <- state
//! z     <- (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
//! z     <- (z ^ (z >> 27)) * 0x94D049BB133111EB
//! z     <- z ^ (z >> 31)
//! ```
//!
//! The uniform on the open interval (0,1) is `u = ((z >> 11) as f64 + 0.5) * 2^-53`,
//! and standard-normal entries come from Box-Muller pairs,
//! `r = sqrt(-2 ln u1)`, `(r cos 2 pi u2, r sin 2 pi u2)`, one pair per two uniform draws.
//! Every entry is generated in row-major order, so seed 2026 fixes the matrix exactly.

const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;
const MIX_A: u64 = 0xBF58_476D_1CE4_E5B9;
const MIX_B: u64 = 0x94D0_49BB_1331_11EB;
const SCALE: f64 = 1.0 / 9007199254740992.0; // 2^-53

/// SplitMix64 uniform source on the open interval (0,1).
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(GOLDEN);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(MIX_A);
        z = (z ^ (z >> 27)).wrapping_mul(MIX_B);
        z ^ (z >> 31)
    }

    /// Uniform in (0,1), never exactly 0 or 1.
    pub fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) * SCALE
    }
}

/// Standard-normal source built from `SplitMix64` uniforms with Box-Muller pairs.
pub struct Gaussian {
    uniforms: SplitMix64,
    spare: Option<f64>,
}

impl Gaussian {
    pub fn new(seed: u64) -> Self {
        Gaussian { uniforms: SplitMix64::new(seed), spare: None }
    }

    pub fn next(&mut self) -> f64 {
        if let Some(value) = self.spare.take() {
            return value;
        }
        let u1 = self.uniforms.uniform();
        let u2 = self.uniforms.uniform();
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = std::f64::consts::TAU * u2;
        self.spare = Some(radius * angle.sin());
        radius * angle.cos()
    }
}

/// A `rows x cols` row-major matrix of standard-normal entries.
pub fn gaussian_matrix(rows: usize, cols: usize, seed: u64) -> (Vec<f64>, f64, f64) {
    let mut source = Gaussian::new(seed);
    let mut data = vec![0.0; rows * cols];
    for value in data.iter_mut() {
        *value = source.next();
    }
    let n = (rows * cols) as f64;
    let mean = data.iter().sum::<f64>() / n;
    let variance = data.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
    (data, mean, variance.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniforms_stay_open_and_roughly_centred() {
        let mut source = SplitMix64::new(2026);
        let mut sum = 0.0;
        let draws = 100_000;
        for _ in 0..draws {
            let value = source.uniform();
            assert!(value > 0.0 && value < 1.0);
            sum += value;
        }
        let mean = sum / draws as f64;
        assert!((mean - 0.5).abs() < 0.01, "mean {mean}");
    }

    #[test]
    fn gaussian_moments_and_reproducibility() {
        let (first, mean, std) = gaussian_matrix(200, 100, 2026);
        assert!(mean.abs() < 0.02, "mean {mean}");
        assert!((std - 1.0).abs() < 0.02, "std {std}");
        let (second, _, _) = gaussian_matrix(200, 100, 2026);
        assert_eq!(first, second, "same seed must give the same matrix");
        let (other, _, _) = gaussian_matrix(200, 100, 2027);
        assert_ne!(first, other);
    }
}
