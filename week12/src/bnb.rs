//! Exact Branch-and-Bound over HiGHS LP relaxations (Algorithm 1 of the sheet).
//!
//! The LP relaxation alone decides infeasibility, bound pruning and integrality;
//! the branching rule only chooses which fractional spin to split. That
//! separation is what keeps the search exact for every rule, including the
//! learned one.

use crate::instance::Instance;
use crate::milp::{Lp, LpSol};
use crate::net::Net;
use std::time::Instant;

/// Bound pruning tolerance: a node dies when `z_LP >= z_best - PRUNE_EPS`.
pub const PRUNE_EPS: f64 = 1e-6;
/// A spin counts as fractional when `FREE_EPS < x_i < 1 - FREE_EPS`.
pub const FREE_EPS: f64 = 1e-6;
/// Score floor that keeps a zero gain from erasing the other child's gain.
pub const GAIN_FLOOR: f64 = 1e-6;
/// Gain given to a child that is infeasible or pruned by bound.
pub const PRUNED_GAIN: f64 = 1e6;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Policy {
    MostFrac,
    Strong,
    Neural,
    RootNeural,
}

impl Policy {
    pub fn name(self) -> &'static str {
        match self {
            Policy::MostFrac => "most-frac",
            Policy::Strong => "strong",
            Policy::Neural => "neural",
            Policy::RootNeural => "root-neural",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "most-frac" => Some(Policy::MostFrac),
            "strong" => Some(Policy::Strong),
            "neural" => Some(Policy::Neural),
            "root-neural" => Some(Policy::RootNeural),
            _ => None,
        }
    }
}

/// The 12 permutation-invariant spin-graph features of `φ_i` (design file).
pub const N_FEATURES: usize = 12;

/// Neighbourhood structure of one spin glass, built once and shared by every
/// node of its search.
pub struct GraphIndex {
    /// Incident bond indices of every spin, in edge order.
    pub bonds: Vec<Vec<usize>>,
    /// Neighbouring spins, in edge order.
    pub nbrs: Vec<Vec<usize>>,
    pub degree: Vec<usize>,
    pub d_max: usize,
}

impl GraphIndex {
    pub fn new(inst: &Instance) -> Self {
        let n = inst.n;
        let mut bonds = vec![Vec::new(); n];
        let mut nbrs = vec![Vec::new(); n];
        for (e, &(u, v)) in inst.edges.iter().enumerate() {
            bonds[u].push(e);
            bonds[v].push(e);
            nbrs[u].push(v);
            nbrs[v].push(u);
        }
        let degree: Vec<usize> = bonds.iter().map(|b| b.len()).collect();
        let d_max = degree.iter().copied().max().unwrap_or(0);
        GraphIndex {
            bonds,
            nbrs,
            degree,
            d_max,
        }
    }
}

pub struct FeatureContext<'a> {
    pub inst: &'a Instance,
    pub graph: &'a GraphIndex,
    pub sol: &'a LpSol,
    pub spins: Vec<f64>,
    pub frac: Vec<f64>,
    /// `max(1e-9, max_r |dual_r|)`, the normalizer of the bond duals.
    pub d_dual: f64,
    pub z_node: f64,
    pub z_best: Option<f64>,
}

impl<'a> FeatureContext<'a> {
    pub fn new(
        inst: &'a Instance,
        graph: &'a GraphIndex,
        sol: &'a LpSol,
        z_node: f64,
        z_best: Option<f64>,
    ) -> Self {
        let spins: Vec<f64> = sol
            .x
            .iter()
            .map(|&x| 1.0 - 2.0 * x.clamp(0.0, 1.0))
            .collect();
        let frac: Vec<f64> = sol
            .x
            .iter()
            .map(|&x| 2.0 * x.clamp(0.0, 1.0).min(1.0 - x.clamp(0.0, 1.0)))
            .collect();
        let mut d_dual = 1e-9f64;
        for &d in &sol.duals {
            if d.abs() > d_dual {
                d_dual = d.abs();
            }
        }
        FeatureContext {
            inst,
            graph,
            sol,
            spins,
            frac,
            d_dual,
            z_node,
            z_best,
        }
    }

    /// The 12 features of every spin, in index order.
    pub fn all(&self) -> Vec<[f64; N_FEATURES]> {
        (0..self.inst.n).map(|i| self.features(i)).collect()
    }

    /// The 12 features of the listed spins, in the given order.
    pub fn subset(&self, spins: &[usize]) -> Vec<[f64; N_FEATURES]> {
        spins.iter().map(|&i| self.features(i)).collect()
    }

    pub fn features(&self, i: usize) -> [f64; N_FEATURES] {
        let inst = self.inst;
        let bonds = &self.graph.bonds[i];
        let deg = self.graph.degree[i];
        let s_i = self.spins[i];
        let d_dual = self.d_dual;

        let mut bond_bias = 0.0;
        let mut eff_field = inst.h[i];
        let mut mean_dual = 0.0;
        let mut mean_gap = 0.0;
        for &e in bonds {
            let (u, v) = inst.edges[e];
            let j = inst.j[e];
            if j > 0.0 {
                bond_bias += 1.0;
            } else if j < 0.0 {
                bond_bias -= 1.0;
            }
            let other = if u == i { v } else { u };
            eff_field += j * self.spins[other];

            let base = 4 * e;
            let mut sum4 = 0.0;
            for r in 0..4 {
                sum4 += self.sol.duals[base + r].abs();
            }
            mean_dual += sum4 / d_dual;

            let s_j = self.spins[other];
            mean_gap += (self.sol.z[e] - 0.5 * (1.0 - s_i * s_j)).abs();
        }
        let nb = bonds.len().max(1) as f64;
        mean_dual /= nb;
        mean_gap /= nb;

        let nbrs = &self.graph.nbrs[i];
        let mean_nbr_frac = if nbrs.is_empty() {
            0.0
        } else {
            nbrs.iter().map(|&j| self.frac[j]).sum::<f64>() / nbrs.len() as f64
        };

        let f = inst.h[i];
        let eff = eff_field / (deg as f64 + 1.0);
        let gap_ratio = match self.z_best {
            Some(zb) => ((zb - self.z_node) / zb.abs().max(1.0)).clamp(0.0, 2.0),
            None => 1.0,
        };

        [
            self.sol.x[i].clamp(0.0, 1.0),
            self.frac[i],
            f,
            f.abs(),
            deg as f64 / (self.graph.d_max.max(1) as f64),
            bond_bias / (deg.max(1) as f64),
            eff,
            eff.abs(),
            mean_dual,
            mean_gap,
            mean_nbr_frac,
            gap_ratio,
        ]
    }
}

#[derive(Clone)]
struct Node {
    lo: Vec<f64>,
    hi: Vec<f64>,
    depth: usize,
    fixes: Vec<(usize, f64)>,
}

impl Node {
    fn branch_label(&self) -> String {
        if self.fixes.is_empty() {
            return "root".to_string();
        }
        self.fixes
            .iter()
            .map(|(i, v)| format!("x{}={}", i, *v as i64))
            .collect::<Vec<_>>()
            .join(",")
    }
}

#[derive(Clone, Debug)]
pub struct TraceRow {
    pub node: usize,
    pub depth: usize,
    pub branch: String,
    pub status: String,
    pub bound: f64,
    pub free: usize,
    pub spin: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct CandidateRecord {
    pub spin: usize,
    pub gain0: f64,
    pub gain1: f64,
    pub score: f64,
    pub pruned_children: u32,
    pub features: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct NodeRecord {
    pub instance: usize,
    pub node: usize,
    pub depth: usize,
    pub branch: String,
    pub z_node: f64,
    pub z_best: f64,
    pub best: Vec<usize>,
    pub candidates: Vec<CandidateRecord>,
}

pub struct SearchConfig<'a> {
    pub policy: Policy,
    pub net: Option<&'a Net>,
    pub export_labels: bool,
    pub max_labels_per_instance: usize,
    pub trace: bool,
    pub instance_id: usize,
}

#[derive(Default)]
pub struct SearchOutcome {
    pub energy: f64,
    pub spins: Vec<f64>,
    pub nodes: u64,
    pub lp_solves: u64,
    pub lp_ms: f64,
    pub total_ms: f64,
    pub infer_ms: f64,
    pub first_splits: Vec<usize>,
    pub trace: Vec<TraceRow>,
    pub labels: Vec<NodeRecord>,
}

/// Which spin the rule picks, for how long features and the network ran, and —
/// for Strong Branching — every candidate's record.
struct Choice {
    spin: usize,
    records: Option<Vec<CandidateRecord>>,
    infer_ns: u128,
}

pub fn branch_and_bound(inst: &Instance, cfg: &SearchConfig) -> SearchOutcome {
    let started = Instant::now();
    let n = inst.n;
    let graph = GraphIndex::new(inst);
    let mut lp = Lp::new(inst, false);
    let mut out = SearchOutcome::default();
    let mut z_best = f64::INFINITY;
    let mut incumbent: Option<Vec<f64>> = None;
    let mut root_scores: Option<Vec<f64>> = None;
    let mut infer_ns: u128 = 0;
    let mut label_count = 0usize;

    let mut stack: Vec<Node> = vec![Node {
        lo: vec![0.0; n],
        hi: vec![1.0; n],
        depth: 0,
        fixes: Vec::new(),
    }];

    while let Some(node) = stack.pop() {
        out.nodes += 1;
        let sol = lp.solve(&node.lo, &node.hi);
        let sol = match sol {
            None => {
                if cfg.trace {
                    out.trace.push(TraceRow {
                        node: out.nodes as usize,
                        depth: node.depth,
                        branch: node.branch_label(),
                        status: "infeasible".to_string(),
                        bound: f64::NAN,
                        free: 0,
                        spin: None,
                    });
                }
                continue;
            }
            Some(s) => s,
        };

        if sol.energy >= z_best - PRUNE_EPS {
            if cfg.trace {
                out.trace.push(TraceRow {
                    node: out.nodes as usize,
                    depth: node.depth,
                    branch: node.branch_label(),
                    status: "pruned-by-bound".to_string(),
                    bound: sol.energy,
                    free: fractional(&sol.x).len(),
                    spin: None,
                });
            }
            continue;
        }

        let free = fractional(&sol.x);
        if free.is_empty() {
            z_best = sol.energy;
            incumbent = Some(sol.x.clone());
            if cfg.trace {
                out.trace.push(TraceRow {
                    node: out.nodes as usize,
                    depth: node.depth,
                    branch: node.branch_label(),
                    status: "incumbent".to_string(),
                    bound: sol.energy,
                    free: 0,
                    spin: None,
                });
            }
            continue;
        }

        // Root-cached scores are computed once, at the first split of the root.
        if cfg.policy == Policy::RootNeural && node.depth == 0 && root_scores.is_none() {
            if let Some(net) = cfg.net {
                let t0 = Instant::now();
                let ctx = FeatureContext::new(
                    inst,
                    &graph,
                    &sol,
                    sol.energy,
                    if z_best.is_finite() { Some(z_best) } else { None },
                );
                root_scores = Some(net.root_scores(&ctx.all()));
                infer_ns += t0.elapsed().as_nanos();
            }
        }

        let zb_option = if z_best.is_finite() {
            Some(z_best)
        } else {
            None
        };
        let choice = choose(
            inst,
            &graph,
            &mut lp,
            &sol,
            &node,
            &free,
            cfg,
            zb_option,
            &root_scores,
        );
        infer_ns += choice.infer_ns;
        let spin = choice.spin;
        out.first_splits.push(spin);

        if cfg.export_labels && label_count < cfg.max_labels_per_instance {
            if let Some(records) = choice.records.as_ref() {
                if records.len() >= 2 {
                    let best = records
                        .iter()
                        .map(|r| r.score)
                        .fold(f64::NEG_INFINITY, f64::max);
                    let tol = 1e-9 * best.abs().max(1.0);
                    let best_spins: Vec<usize> = records
                        .iter()
                        .filter(|r| r.score >= best - tol)
                        .map(|r| r.spin)
                        .collect();
                    out.labels.push(NodeRecord {
                        instance: cfg.instance_id,
                        node: out.nodes as usize,
                        depth: node.depth,
                        branch: node.branch_label(),
                        z_node: sol.energy,
                        z_best: if z_best.is_finite() { z_best } else { f64::NAN },
                        best: best_spins,
                        candidates: records.clone(),
                    });
                    label_count += 1;
                }
            }
        }

        if cfg.trace {
            out.trace.push(TraceRow {
                node: out.nodes as usize,
                depth: node.depth,
                branch: node.branch_label(),
                status: format!("split x{}", spin),
                bound: sol.energy,
                free: free.len(),
                spin: Some(spin),
            });
        }

        // Visit the child nearer the LP value first; at 0.5 visit x_i = 1 first.
        let xi = sol.x[spin];
        let (first, second) = if xi >= 0.5 { (0.0, 1.0) } else { (1.0, 0.0) };
        for value in [first, second] {
            let mut lo = node.lo.clone();
            let mut hi = node.hi.clone();
            lo[spin] = value;
            hi[spin] = value;
            let mut fixes = node.fixes.clone();
            fixes.push((spin, value));
            // Pushed in this order, so the nearer child is popped first.
            stack.push(Node {
                lo,
                hi,
                depth: node.depth + 1,
                fixes,
            });
        }
    }

    out.energy = z_best;
    out.spins = incumbent.unwrap_or_default();
    out.lp_solves = lp.solves;
    out.lp_ms = lp.solve_ns as f64 / 1e6;
    out.infer_ms = infer_ns as f64 / 1e6;
    out.total_ms = started.elapsed().as_secs_f64() * 1e3;
    out
}

fn fractional(x: &[f64]) -> Vec<usize> {
    (0..x.len())
        .filter(|&i| x[i] > FREE_EPS && x[i] < 1.0 - FREE_EPS)
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn choose(
    inst: &Instance,
    graph: &GraphIndex,
    lp: &mut Lp,
    sol: &LpSol,
    node: &Node,
    free: &[usize],
    cfg: &SearchConfig,
    z_best: Option<f64>,
    root_scores: &Option<Vec<f64>>,
) -> Choice {
    match cfg.policy {
        Policy::MostFrac => {
            // The edge LP is half-integral, so every candidate sits at 1/2 and
            // the tie falls back to the lowest spin index.
            let mut best = free[0];
            let mut best_d = (sol.x[free[0]] - 0.5).abs();
            for &i in &free[1..] {
                let d = (sol.x[i] - 0.5).abs();
                if d < best_d - 1e-9 {
                    best = i;
                    best_d = d;
                }
            }
            Choice {
                spin: best,
                records: None,
                infer_ns: 0,
            }
        }
        Policy::Strong => {
            let t_infer = Instant::now();
            // Features of every candidate are only needed when recording labels.
            let ctx = if cfg.export_labels {
                Some(FeatureContext::new(inst, graph, sol, sol.energy, z_best))
            } else {
                None
            };
            let infer_ns = ctx
                .as_ref()
                .map(|_| t_infer.elapsed().as_nanos())
                .unwrap_or(0);

            let mut records = Vec::with_capacity(free.len());
            let mut best_spin = free[0];
            let mut best_score = f64::NEG_INFINITY;
            for &i in free {
                let mut gains = [0.0f64; 2];
                let mut pruned = 0u32;
                for (k, value) in [0.0f64, 1.0].iter().enumerate() {
                    let mut lo = node.lo.clone();
                    let mut hi = node.hi.clone();
                    lo[i] = *value;
                    hi[i] = *value;
                    let gain = match lp.solve(&lo, &hi) {
                        None => {
                            pruned += 1;
                            PRUNED_GAIN
                        }
                        Some(child) => {
                            let pruned_by_bound =
                                z_best.map_or(false, |zb| child.energy >= zb - PRUNE_EPS);
                            if pruned_by_bound {
                                pruned += 1;
                                PRUNED_GAIN
                            } else {
                                child.energy - sol.energy
                            }
                        }
                    };
                    gains[k] = gain;
                }
                let score = gains[0].max(GAIN_FLOOR) * gains[1].max(GAIN_FLOOR);
                if score > best_score + 1e-12 {
                    best_score = score;
                    best_spin = i;
                }
                if let Some(ctx) = ctx.as_ref() {
                    records.push(CandidateRecord {
                        spin: i,
                        gain0: gains[0],
                        gain1: gains[1],
                        score,
                        pruned_children: pruned,
                        features: ctx.features(i).to_vec(),
                    });
                }
            }
            Choice {
                spin: best_spin,
                records: if cfg.export_labels { Some(records) } else { None },
                infer_ns,
            }
        }
        Policy::Neural => {
            let t0 = Instant::now();
            let net = cfg.net.expect("neural policy needs a model");
            let ctx = FeatureContext::new(inst, graph, sol, sol.energy, z_best);
            // Only the candidate spins need a score, never the whole graph.
            let feats = ctx.subset(free);
            let mut best = free[0];
            let mut best_score = net.score(&feats[0]);
            for k in 1..free.len() {
                let i = free[k];
                let s = net.score(&feats[k]);
                if s > best_score + 1e-12 {
                    best = i;
                    best_score = s;
                }
            }
            Choice {
                spin: best,
                records: None,
                infer_ns: t0.elapsed().as_nanos(),
            }
        }
        Policy::RootNeural => {
            let t0 = Instant::now();
            let r = root_scores
                .as_ref()
                .expect("root-neural needs root scores computed at the root");
            let mut best = free[0];
            let mut best_score = r[best] + 3.0 * 2.0 * sol.x[best].min(1.0 - sol.x[best]);
            for &i in &free[1..] {
                let s = r[i] + 3.0 * 2.0 * sol.x[i].min(1.0 - sol.x[i]);
                if s > best_score + 1e-12 {
                    best = i;
                    best_score = s;
                }
            }
            Choice {
                spin: best,
                records: None,
                infer_ns: t0.elapsed().as_nanos(),
            }
        }
    }
}
