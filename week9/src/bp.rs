//! Loopy belief propagation with synchronous, damped message updates.
//!
//! The factor graph is built from the model's tables with the observations
//! applied; each observed variable keeps a one-hot factor so its belief stays
//! defined. Messages and beliefs follow Algorithm 1 of the sheet; the reported
//! residual is the maximum undamped message change at the recorded state.

use std::collections::BTreeMap;

use crate::uai::{Model, Observations};

#[derive(Clone, Copy, Debug)]
pub struct Settings {
    pub damping: f64,
    pub tolerance: f64,
    pub max_sweeps: usize,
}

struct GraphFactor {
    scope: Vec<usize>,
    dims: Vec<usize>,
    entries: Vec<f64>,
}

struct FactorGraph {
    dims: Vec<usize>,
    factors: Vec<GraphFactor>,
    /// Per variable: `(factor, edge)` pairs.
    var_edges: Vec<Vec<(usize, usize)>>,
    /// Per factor: `(variable, edge, axis)` triples.
    factor_edges: Vec<Vec<(usize, usize, usize)>>,
    /// Dims of every edge's variable.
    edge_dim: Vec<usize>,
}

impl FactorGraph {
    fn build(model: &Model, observations: &Observations) -> Result<Self, String> {
        let mut observed = vec![None; model.variables];
        for (variable, state) in observations {
            if *variable >= model.variables {
                return Err(format!("observation on unknown variable {variable}"));
            }
            if *state >= model.dims[*variable] {
                return Err(format!("observation state {state} outside variable {variable}"));
            }
            observed[*variable] = Some(*state);
        }
        let mut factors: Vec<GraphFactor> = Vec::new();
        for factor in &model.factors {
            if let Some(sliced) = slice(factor, &observed) {
                factors.push(sliced);
            }
        }
        for variable in 0..model.variables {
            if let Some(state) = observed[variable] {
                let mut entries = vec![0.0; model.dims[variable]];
                entries[state] = 1.0;
                factors.push(GraphFactor {
                    scope: vec![variable],
                    dims: vec![model.dims[variable]],
                    entries,
                });
            }
        }
        let mut var_edges = vec![Vec::new(); model.variables];
        let mut factor_edges = Vec::with_capacity(factors.len());
        let mut edge_dim = Vec::new();
        for (index, factor) in factors.iter().enumerate() {
            let mut edges = Vec::with_capacity(factor.scope.len());
            for (axis, variable) in factor.scope.iter().enumerate() {
                let edge = edge_dim.len();
                edge_dim.push(model.dims[*variable]);
                var_edges[*variable].push((index, edge));
                edges.push((*variable, edge, axis));
            }
            factor_edges.push(edges);
        }
        Ok(Self {
            dims: model.dims.clone(),
            factors,
            var_edges,
            factor_edges,
            edge_dim,
        })
    }

    fn normalize(values: &mut [f64]) {
        let total: f64 = values.iter().sum();
        if total > 0.0 {
            for value in values.iter_mut() {
                *value /= total;
            }
        }
    }

    /// One synchronous, undamped sweep from the given messages.
    fn propose(&self, v2f: &[Vec<f64>], f2v: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let mut proposal_v2f = v2f.to_vec();
        for variable in 0..self.var_edges.len() {
            for (_, target) in &self.var_edges[variable] {
                let mut values = vec![1.0; self.dims[variable]];
                for (_, edge) in &self.var_edges[variable] {
                    if edge == target {
                        continue;
                    }
                    for (value, message) in values.iter_mut().zip(&f2v[*edge]) {
                        *value *= message;
                    }
                }
                Self::normalize(&mut values);
                proposal_v2f[*target] = values;
            }
        }
        let mut proposal_f2v = f2v.to_vec();
        for (index, factor) in self.factors.iter().enumerate() {
            for (target, target_edge, target_axis) in &self.factor_edges[index] {
                let mut values = vec![0.0; self.dims[*target]];
                let mut state = vec![0usize; factor.dims.len()];
                for (flat, entry) in factor.entries.iter().enumerate() {
                    let mut remainder = flat;
                    for axis in 0..factor.dims.len() {
                        state[axis] = remainder % factor.dims[axis];
                        remainder /= factor.dims[axis];
                    }
                    let mut weight = *entry;
                    for (_variable, edge, axis) in &self.factor_edges[index] {
                        if axis == target_axis {
                            continue;
                        }
                        // Algorithm 1 reads the current messages m, not the proposal.
                        weight *= v2f[*edge][state[*axis]];
                    }
                    values[state[*target_axis]] += weight;
                }
                Self::normalize(&mut values);
                proposal_f2v[*target_edge] = values;
            }
        }
        let mut beliefs = Vec::with_capacity(self.dims.len());
        for variable in 0..self.var_edges.len() {
            let mut values = vec![1.0; self.dims[variable]];
            for (_, edge) in &self.var_edges[variable] {
                for (value, message) in values.iter_mut().zip(&proposal_f2v[*edge]) {
                    *value *= message;
                }
            }
            Self::normalize(&mut values);
            beliefs.push(values);
        }
        (proposal_v2f, proposal_f2v, beliefs)
    }

    fn residual(&self, current: (&[Vec<f64>], &[Vec<f64>]), proposal: (&[Vec<f64>], &[Vec<f64>])) -> f64 {
        let mut worst = 0.0f64;
        for (old, new) in current.0.iter().zip(proposal.0) {
            for (a, b) in old.iter().zip(new) {
                worst = worst.max((a - b).abs());
            }
        }
        for (old, new) in current.1.iter().zip(proposal.1) {
            for (a, b) in old.iter().zip(new) {
                worst = worst.max((a - b).abs());
            }
        }
        worst
    }

    fn update_work(&self) -> f64 {
        let mut work = 0.0f64;
        for factor in &self.factors {
            work += factor.entries.len() as f64 * factor.scope.len() as f64;
        }
        for variable in 0..self.var_edges.len() {
            let degree = self.var_edges[variable].len();
            if degree > 1 {
                work += self.dims[variable] as f64 * (degree - 1) as f64;
            }
        }
        work
    }

    fn damp(current: &mut [Vec<f64>], proposal: &[Vec<f64>], damping: f64) {
        for (old, new) in current.iter_mut().zip(proposal) {
            for (a, b) in old.iter_mut().zip(new) {
                *a = (1.0 - damping) * *a + damping * *b;
            }
        }
    }
}

/// Slice the observed axes out of one table, keeping the selected states.
///
/// The file lists a factor's entries with the last scope axis varying fastest;
/// the sliced table is stored column-major over the kept axes.
fn slice(factor: &crate::uai::Factor, observed: &[Option<usize>]) -> Option<GraphFactor> {
    let mut scope = Vec::new();
    let mut dims = Vec::new();
    for (axis, variable) in factor.scope.iter().enumerate() {
        if observed[*variable].is_none() {
            scope.push(*variable);
            dims.push(factor.dims[axis]);
        }
    }
    let kept: usize = dims.iter().product();
    let mut entries = vec![0.0f64; kept];
    let mut state = vec![0usize; factor.dims.len()];
    let mut row_strides = vec![1usize; factor.dims.len()];
    for axis in (0..factor.dims.len().saturating_sub(1)).rev() {
        row_strides[axis] = row_strides[axis + 1] * factor.dims[axis + 1];
    }
    for (flat, entry) in entries.iter_mut().enumerate() {
        // Flat index over the kept axes, column-major (first kept axis fastest).
        let mut remainder = flat;
        for axis in 0..factor.dims.len() {
            state[axis] = match observed[factor.scope[axis]] {
                Some(value) => value,
                None => {
                    let value = remainder % factor.dims[axis];
                    remainder /= factor.dims[axis];
                    value
                }
            };
        }
        let mut row_index = 0usize;
        for axis in 0..factor.dims.len() {
            row_index += state[axis] * row_strides[axis];
        }
        *entry = factor.entries[row_index];
    }
    Some(GraphFactor {
        scope,
        dims,
        entries,
    })
}

pub struct SweepRecord {
    pub sweep: usize,
    pub residual: f64,
    pub work: f64,
    pub beliefs: BTreeMap<usize, Vec<f64>>,
}

pub struct Outcome {
    pub sweeps: Vec<SweepRecord>,
    pub final_beliefs: BTreeMap<usize, Vec<f64>>,
    pub residual: f64,
    pub converged: bool,
    pub work: f64,
    pub per_sweep_work: f64,
    pub graph_edges: usize,
    pub graph_factors: usize,
}

/// Run damped synchronous belief propagation until the residual is small or
/// the sweep limit is reached. Beliefs are recorded for every variable when
/// the model is small, and for `record` (the queried variables) otherwise.
pub fn run(
    model: &Model,
    observations: &Observations,
    settings: Settings,
    record: &[usize],
) -> Result<(Outcome, Vec<usize>), String> {
    let graph = FactorGraph::build(model, observations)?;
    let variables = model.variables;
    let record_all = variables <= 16;
    let recorded: Vec<usize> = if record_all {
        (0..variables).collect()
    } else {
        record.to_vec()
    };
    let mut v2f: Vec<Vec<f64>> = graph
        .edge_dim
        .iter()
        .map(|dim| vec![1.0 / *dim as f64; *dim])
        .collect();
    let mut f2v = v2f.clone();
    let per_sweep_work = graph.update_work();
    let mut cumulative_work = 0.0f64;
    let mut sweeps = Vec::new();
    let mut converged = false;
    let mut residual = f64::INFINITY;
    let mut beliefs: Vec<Vec<f64>> = Vec::new();
    for sweep in 1..=settings.max_sweeps {
        let (proposal_v2f, proposal_f2v, new_beliefs) = graph.propose(&v2f, &f2v);
        residual = graph.residual((&v2f, &f2v), (&proposal_v2f, &proposal_f2v));
        cumulative_work += per_sweep_work;
        beliefs = new_beliefs;
        let mut recorded_beliefs = BTreeMap::new();
        for variable in &recorded {
            recorded_beliefs.insert(*variable, beliefs[*variable].clone());
        }
        sweeps.push(SweepRecord {
            sweep,
            residual,
            work: cumulative_work,
            beliefs: recorded_beliefs,
        });
        if residual <= settings.tolerance {
            converged = true;
            break;
        }
        FactorGraph::damp(&mut v2f, &proposal_v2f, settings.damping);
        FactorGraph::damp(&mut f2v, &proposal_f2v, settings.damping);
    }
    let final_beliefs = recorded
        .iter()
        .map(|variable| (*variable, beliefs[*variable].clone()))
        .collect();
    let outcome = Outcome {
        sweeps,
        final_beliefs,
        residual,
        converged,
        work: cumulative_work,
        per_sweep_work: per_sweep_work,
        graph_edges: graph.edge_dim.len(),
        graph_factors: graph.factors.len(),
    };
    Ok((outcome, recorded))
}

