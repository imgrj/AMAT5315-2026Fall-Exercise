pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn seed_from_u64(mut seed: u64) -> Self {
        let sm = |state: &mut u64| -> u64 {
            *state = state.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = *state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^ (z >> 31)
        };
        let s0 = sm(&mut seed);
        let s1 = sm(&mut seed);
        let s2 = sm(&mut seed);
        let s3 = sm(&mut seed);
        Self { s: [s0, s1, s2, s3] }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = (self.s[0].wrapping_add(self.s[3]))
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    #[inline]
    pub fn gen_range(&mut self, max: usize) -> usize {
        (self.next_u64() % (max as u64)) as usize
    }

    #[inline]
    pub fn gen_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / (1u64 << 53) as f64)
    }
}
