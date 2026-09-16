use crate::lattice::Lattice;
use crate::rng::Rng;

pub struct WolffState {
    tag: Vec<u32>,
    current_tag: u32,
    queue: Vec<usize>,
}

impl WolffState {
    pub fn new(n: usize) -> Self {
        Self {
            tag: vec![0; n],
            current_tag: 1,
            queue: Vec::with_capacity(n),
        }
    }

    pub fn step(
        &mut self,
        lattice: &mut Lattice,
        t: f64,
        rng: &mut Rng,
    ) -> usize {
        let n = lattice.n;
        let p_add = 1.0 - (-2.0 / t).exp();

        self.current_tag += 1;
        if self.current_tag == u32::MAX {
            self.tag.fill(0);
            self.current_tag = 1;
        }
        let this_tag = self.current_tag;

        self.queue.clear();
        let seed_site = rng.gen_range(n);
        let s0 = lattice.spins[seed_site];

        self.tag[seed_site] = this_tag;
        self.queue.push(seed_site);

        let mut head = 0;
        while head < self.queue.len() {
            let u = self.queue[head];
            head += 1;
            for &v in &lattice.nbrs[u] {
                if lattice.spins[v] == s0 && self.tag[v] != this_tag {
                    if rng.gen_f64() < p_add {
                        self.tag[v] = this_tag;
                        self.queue.push(v);
                    }
                }
            }
        }

        let cluster_size = self.queue.len();

        // Calculate energy change from boundary bonds
        let mut de = 0i64;
        for &u in &self.queue {
            for &v in &lattice.nbrs[u] {
                if self.tag[v] != this_tag {
                    de += 2 * (s0 as i64) * (lattice.spins[v] as i64);
                }
            }
        }

        // Flip all spins in cluster
        for &u in &self.queue {
            lattice.spins[u] = -s0;
        }

        lattice.energy += de;
        lattice.spin_sum -= 2 * (cluster_size as i64) * (s0 as i64);

        cluster_size
    }
}
