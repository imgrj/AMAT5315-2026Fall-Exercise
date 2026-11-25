//! SplitMix64, the integer-only generator named in `branchopt.design.toml`
//! (Steele, Lea & Flood 2014). Every draw is arithmetic modulo 2^64, so the
//! instances are platform independent and exactly reproducible from the seed.

pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub const GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

    pub fn new(state: u64) -> Self {
        Self { state }
    }

    /// The generator described by the design file: `state += gamma`, then two
    /// xor-shift-multiply rounds and a final xor-shift.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(Self::GAMMA);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// An integer drawn uniformly from `{0, ..., k-1}`.
    pub fn below(&mut self, k: u64) -> u64 {
        self.next_u64() % k
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_values_of_the_reference_generator() {
        // Reference SplitMix64 output for the default seed 0.
        let mut g = SplitMix64::new(0);
        assert_eq!(g.next_u64(), 0xE220_A839_7B1D_CDAF);
        assert_eq!(g.next_u64(), 0x6E78_9E6A_A1B9_65F4);
        assert_eq!(g.next_u64(), 0x06C4_5D18_8009_454F);
    }

    #[test]
    fn below_stays_in_range() {
        let mut g = SplitMix64::new(2026);
        for _ in 0..1000 {
            assert!(g.below(22) < 22);
        }
    }
}
