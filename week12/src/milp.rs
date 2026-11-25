//! The spin-glass mixed-integer linear program of the learning sheet (Eq. 5),
//! its LP relaxation, and the diet LP used as a modelling check.
//!
//! Variables: `n` spin bits `x_i in {0,1}` (relaxed to `[0,1]`) and `m` bond
//! disagreements `z_e in [0,1]`. Four inequalities per bond tie `z_e` to
//! `x_u xor x_v`, so once the spins are 0/1 every `z_e` is pinned.
//!
//! ```text
//! min  sum_e J_e (1 - 2 z_e) + sum_i h_i (1 - 2 x_i)
//! s.t. z_e <= x_u + x_v,  z_e <= 2 - x_u - x_v,
//!      z_e >= x_u - x_v,  z_e >= x_v - x_u     (each bond e = (u,v))
//! ```

use crate::instance::Instance;
use highs::{Col, Model, Row, RowProblem, Sense};
use std::num::NonZeroU32;
use std::time::Instant;

/// Column handles of one spin-glass program.
pub struct Layout {
    pub n: usize,
    pub m: usize,
    pub x: Vec<Col>,
}

pub struct Program {
    pub layout: Layout,
    /// The constant offset `C = sum_e J_e + sum_i h_i` of Eq. 6.
    pub offset: f64,
    pub model: Model,
}

/// Build the MILP (`integer_spins = true`) or its LP relaxation.
pub fn build(inst: &Instance, integer_spins: bool) -> Program {
    let n = inst.n;
    let m = inst.edges.len();
    let mut pb = RowProblem::default();

    let mut x = Vec::with_capacity(n);
    for i in 0..n {
        let cost = -2.0 * inst.h[i];
        x.push(if integer_spins {
            pb.add_integer_column(cost, 0.0..=1.0)
        } else {
            pb.add_column(cost, 0.0..=1.0)
        });
    }
    let mut z = Vec::with_capacity(m);
    for e in 0..m {
        z.push(pb.add_column(-2.0 * inst.j[e], 0.0..=1.0));
    }
    for (e, &(u, v)) in inst.edges.iter().enumerate() {
        let rows: [Vec<(Col, f64)>; 4] = [
            vec![(z[e], 1.0), (x[u], -1.0), (x[v], -1.0)],
            vec![(z[e], 1.0), (x[u], 1.0), (x[v], 1.0)],
            vec![(x[u], 1.0), (x[v], -1.0), (z[e], -1.0)],
            vec![(x[u], -1.0), (x[v], 1.0), (z[e], -1.0)],
        ];
        pb.add_row(..=0.0, rows[0].clone());
        pb.add_row(..=2.0, rows[1].clone());
        pb.add_row(..=0.0, rows[2].clone());
        pb.add_row(..=0.0, rows[3].clone());
    }

    let mut model = pb.optimise(Sense::Minimise);
    // The design file switches presolve off so every LP is solved as written,
    // and pins the solver to one thread.
    model.set_option("presolve", "off");
    model.set_option("output_flag", false);
    model.set_threads(NonZeroU32::new(1).unwrap());

    let offset = inst.j.iter().sum::<f64>() + inst.h.iter().sum::<f64>();
    Program {
        layout: Layout { n, m, x },
        offset,
        model,
    }
}

/// One solved LP relaxation.
#[derive(Clone, Debug)]
pub struct LpSol {
    /// Physical lower bound `objective + offset`.
    pub energy: f64,
    pub x: Vec<f64>,
    pub z: Vec<f64>,
    /// One dual per row, four rows per bond, in bond order.
    pub duals: Vec<f64>,
}

/// A reusable LP relaxation: bounds change from node to node, the matrix does not.
pub struct Lp {
    pub program: Program,
    lo: Vec<f64>,
    hi: Vec<f64>,
    pub solves: u64,
    pub solve_ns: u128,
}

impl Lp {
    pub fn new(inst: &Instance, integer_spins: bool) -> Self {
        let program = build(inst, integer_spins);
        let n = inst.n;
        Lp {
            program,
            lo: vec![0.0; n],
            hi: vec![1.0; n],
            solves: 0,
            solve_ns: 0,
        }
    }

    /// Solve the LP under the box `lo <= x <= hi`. `None` means infeasible.
    pub fn solve(&mut self, lo: &[f64], hi: &[f64]) -> Option<LpSol> {
        {
            let model = &mut self.program.model;
            for i in 0..lo.len() {
                if lo[i] != self.lo[i] || hi[i] != self.hi[i] {
                    model.change_column_bounds(self.program.layout.x[i], lo[i]..=hi[i]);
                    self.lo[i] = lo[i];
                    self.hi[i] = hi[i];
                }
            }
        }
        let t0 = Instant::now();
        let out = self.solve_once();
        self.solves += 1;
        self.solve_ns += t0.elapsed().as_nanos();
        out
    }

    /// Solve the current box and keep the model for the next call.
    fn solve_once(&mut self) -> Option<LpSol> {
        let model = std::mem::replace(
            &mut self.program.model,
            Model::new(highs::ColProblem::default()),
        );
        let solved = model.solve();
        let status = solved.status();
        let result = if status == highs::HighsModelStatus::Optimal {
            let objective = solved.objective_value();
            let sol = solved.get_solution();
            let n = self.program.layout.n;
            let m = self.program.layout.m;
            let cols = sol.columns();
            Some(LpSol {
                energy: objective + self.program.offset,
                x: cols[..n].to_vec(),
                z: cols[n..n + m].to_vec(),
                duals: sol.dual_rows().to_vec(),
            })
        } else {
            None
        };
        self.program.model = Model::from(solved);
        result
    }
}

/// The two-food diet LP of the sheet (Eq. 4): `min 6 x1 + 3.5 x2` subject to
/// `7 x2 >= 8`, `4 x1 + 2 x2 >= 15`, `2 x1 + x2 >= 3`, `x >= 0`.
pub fn diet() -> (f64, f64, f64) {
    let mut pb = RowProblem::default();
    let c1 = pb.add_column(6.0, 0.0..);
    let c2 = pb.add_column(3.5, 0.0..);
    pb.add_row(8.0.., [(c2, 7.0)]);
    pb.add_row(15.0.., [(c1, 4.0), (c2, 2.0)]);
    pb.add_row(3.0.., [(c1, 2.0), (c2, 1.0)]);
    let mut model = pb.optimise(Sense::Minimise);
    model.set_option("presolve", "off");
    model.set_option("output_flag", false);
    model.set_threads(NonZeroU32::new(1).unwrap());
    let solved = model.solve();
    let sol = solved.get_solution();
    let cols = sol.columns();
    (cols[0], cols[1], solved.objective_value())
}

#[allow(dead_code)]
fn unused(_: Row) {}
