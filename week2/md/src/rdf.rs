//! Two-dimensional radial distribution function in a rectangular periodic box.

pub struct Rdf {
    box_size: [f64; 2],
    dr: f64,
    counts: Vec<f64>,
    frames: usize,
    n: usize,
}

impl Rdf {
    pub fn new(box_size: [f64; 2], dr: f64) -> Self {
        let r_max = 0.5 * box_size[0].min(box_size[1]);
        let bins = (r_max / dr).floor() as usize;
        Self {
            box_size,
            dr,
            counts: vec![0.0; bins],
            frames: 0,
            n: 0,
        }
    }

    pub fn add_frame(&mut self, positions: &[[f64; 2]]) {
        self.n = positions.len();
        self.frames += 1;
        for i in 0..positions.len() {
            for j in (i + 1)..positions.len() {
                let mut d = [
                    positions[i][0] - positions[j][0],
                    positions[i][1] - positions[j][1],
                ];
                d[0] -= self.box_size[0] * (d[0] / self.box_size[0]).round();
                d[1] -= self.box_size[1] * (d[1] / self.box_size[1]).round();
                let r = (d[0] * d[0] + d[1] * d[1]).sqrt();
                let bin = (r / self.dr).floor() as usize;
                if let Some(count) = self.counts.get_mut(bin) {
                    *count += 1.0;
                }
            }
        }
    }

    pub fn curve(&self) -> Vec<(f64, f64)> {
        if self.frames == 0 || self.n == 0 {
            return Vec::new();
        }
        let rho = self.n as f64 / (self.box_size[0] * self.box_size[1]);
        self.counts
            .iter()
            .enumerate()
            .map(|(bin, &count)| {
                let inner = bin as f64 * self.dr;
                let outer = inner + self.dr;
                let ring_area = std::f64::consts::PI * (outer * outer - inner * inner);
                let expected = self.frames as f64 * self.n as f64 * rho * ring_area / 2.0;
                (0.5 * (inner + outer), count / expected)
            })
            .collect()
    }

    pub fn long_range_contrast(&self) -> f64 {
        let tail: Vec<f64> = self
            .curve()
            .into_iter()
            .filter(|(r, _)| *r > 2.0)
            .map(|(_, g)| (g - 1.0) * (g - 1.0))
            .collect();
        if tail.is_empty() { 0.0 } else { (tail.iter().sum::<f64>() / tail.len() as f64).sqrt() }
    }
}

#[cfg(test)]
mod tests {
    use super::Rdf;

    #[test]
    fn known_pair_uses_rectangular_density() {
        let mut rdf = Rdf::new([10.0, 8.0], 0.5);
        rdf.add_frame(&[[1.0, 4.0], [4.0, 4.0]]);
        let curve = rdf.curve();
        let rho = 2.0 / 80.0;
        let expected = 1.0 / (2.0 * rho * std::f64::consts::PI * (3.5_f64.powi(2) - 3.0_f64.powi(2)) / 2.0);
        assert!((curve[6].1 - expected).abs() < 1e-12);
    }
}
