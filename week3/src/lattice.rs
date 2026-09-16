#[allow(dead_code)]
pub struct Lattice {
    pub l: usize,
    pub n: usize,
    pub spins: Vec<i8>,
    pub nbrs: Vec<[usize; 4]>,
    pub energy: i64,
    pub spin_sum: i64,
}

#[allow(dead_code)]
impl Lattice {
    pub fn new_all_up(l: usize) -> Self {
        let n = l * l;
        let mut nbrs = Vec::with_capacity(n);
        for y in 0..l {
            for x in 0..l {
                let right = y * l + ((x + 1) % l);
                let up = ((y + 1) % l) * l + x;
                let left = y * l + ((x + l - 1) % l);
                let down = ((y + l - 1) % l) * l + x;
                nbrs.push([right, up, left, down]);
            }
        }
        let spins = vec![1i8; n];
        let energy = -2 * (n as i64);
        let spin_sum = n as i64;
        Self {
            l,
            n,
            spins,
            nbrs,
            energy,
            spin_sum,
        }
    }

    pub fn compute_total_energy(&self) -> i64 {
        let mut e = 0i64;
        for i in 0..self.n {
            let s = self.spins[i] as i64;
            let right = self.spins[self.nbrs[i][0]] as i64;
            let up = self.spins[self.nbrs[i][1]] as i64;
            e -= s * (right + up);
        }
        e
    }

    pub fn magnetization(&self) -> f64 {
        self.spin_sum as f64 / self.n as f64
    }

    pub fn energy_per_site(&self) -> f64 {
        self.energy as f64 / self.n as f64
    }
}
