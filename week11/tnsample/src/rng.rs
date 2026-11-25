//! A documented SplitMix64 generator (Steele, Lea & Flood 2014).
//!
//! Every random decision of `tnsample` is a function of `--seed`: the bond signs
//! of a lattice magnet, the choice of contraction order among equal-cost pairs,
//! and the conditional draws of sample mode. Two streams with disjoint offsets
//! are used so that adding samples cannot change the magnet.

pub struct Rng {
    state: u64,
}

pub const STREAM_BONDS: u64 = 0x243F_6A88_85A3_08D3;
pub const STREAM_SAMPLE: u64 = 0xA409_3822_299F_31D0;

impl Rng {
    pub fn new(seed: u64, stream: u64) -> Self {
        let mut rng = Rng {
            state: seed ^ stream,
        };
        // Discard one value so that seed 0 is not degenerate.
        rng.next_u64();
        rng
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in the open interval (0, 1), 53 significant bits.
    pub fn next_f64(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        ((bits as f64) + 0.5) * (1.0 / 9_007_199_254_740_992.0)
    }

    /// Uniform in [0, 1).
    pub fn next_open(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        (bits as f64) * (1.0 / 9_007_199_254_740_992.0)
    }

    /// A categorical draw from cumulative weights; the last index takes the rest.
    pub fn draw(&mut self, cumulative: &[f64]) -> usize {
        let total = *cumulative.last().expect("empty categorical distribution");
        assert!(total > 0.0, "categorical distribution with zero total weight");
        let target = self.next_open() * total;
        let mut low = 0usize;
        let mut high = cumulative.len() - 1;
        while low < high {
            let mid = (low + high) / 2;
            if cumulative[mid] > target {
                high = mid;
            } else {
                low = mid + 1;
            }
        }
        low
    }
}
