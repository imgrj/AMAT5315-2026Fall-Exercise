//! Exact inference: one contraction per query, plus full enumeration for ASIA.

use std::collections::BTreeMap;

use serde_json::json;
use tenferro_cpu::CpuEngine;

use crate::network::{build_network, contract, Array, Contracted};
use crate::plan::{self, Context, Plan, PlanTree, Search};
use crate::uai::{Factor, Model, Observations};

/// One query result before normalization: the contraction of the whole network
/// with the query labels kept open.
pub struct QueryResult {
    pub plan: Plan,
    pub log_scale: f64,
    pub array: Array,
}

impl QueryResult {
    /// Values with the removed log scale restored.
    pub fn values(&self) -> Vec<f64> {
        let scale = self.log_scale.exp();
        self.array.data.iter().map(|value| value * scale).collect()
    }

    pub fn mass(&self) -> f64 {
        self.values().first().copied().unwrap_or(0.0)
    }
}

pub struct ExactRun {
    pub mass: QueryResult,
    pub mass_tree: PlanTree,
    pub marginals: BTreeMap<usize, QueryResult>,
    pub joints: BTreeMap<(usize, usize), QueryResult>,
    pub enumeration: Option<Enumeration>,
}

pub struct Enumeration {
    pub mass: f64,
    pub marginals: BTreeMap<usize, Vec<f64>>,
    pub joints: BTreeMap<(usize, usize), Vec<f64>>,
}

pub struct Request<'a> {
    pub model: &'a Model,
    pub observations: &'a Observations,
    pub queries: &'a [usize],
    pub joints: &'a [(usize, usize)],
    pub search: &'a Search,
    /// Plan for the closed (evidence) query, when one was planned or replayed.
    pub scalar_plan: Option<&'a PlanTree>,
}

fn plan_query(
    request: &Request<'_>,
    labels: &[u32],
    engine: &mut CpuEngine,
) -> Result<(QueryResult, PlanTree), String> {
    let network = build_network(request.model, request.observations)?;
    let leaf_labels: Vec<Vec<u32>> = network.leaves.iter().map(|leaf| leaf.labels.clone()).collect();
    let context = Context {
        leaf_labels: &leaf_labels,
        dims: &request.model.dims,
        query: labels,
    };
    // The saved (or pre-planned) tree is the closed-query plan: use it only there.
    let planned = if labels.is_empty() { request.scalar_plan } else { None };
    let (tree, _settings) = match planned {
        Some(tree) => {
            plan::validate(tree, leaf_labels.len())?;
            (tree.clone(), json!(null))
        }
        None => plan::search(request.search, &leaf_labels, &request.model.dims, labels)?,
    };
    let plan = plan::analyze(&tree, &context);
    if std::env::var_os("INFER_PLAN_DEBUG").is_some() {
        eprintln!(
            "plan {labels:?}: work {:.0}, largest {}, {} steps",
            plan.work,
            plan.largest_entries,
            plan.steps.len()
        );
        for step in plan.steps.iter().take(5) {
            eprintln!(
                "  step {}: {}x{} left {:?} right {:?} -> {:?} work {:.0} entries {}",
                step.id, step.left, step.right, step.left_labels, step.right_labels,
                step.labels, step.work, step.entries
            );
        }
        for step in plan.steps.iter().rev().take(3).rev() {
            eprintln!(
                "  step {}: {}x{} left {:?} right {:?} -> {:?} work {:.0} entries {}",
                step.id, step.left, step.right, step.left_labels, step.right_labels,
                step.labels, step.work, step.entries
            );
        }
    }
    let contracted: Contracted = contract(request.model, request.observations, &plan.steps, engine)?;
    Ok((
        QueryResult {
            plan,
            log_scale: contracted.log_scale,
            array: contracted.array,
        },
        tree,
    ))
}

/// Contract the whole network for the evidence (closed query).
pub fn evidence(request: &Request<'_>, engine: &mut CpuEngine) -> Result<(QueryResult, PlanTree), String> {
    plan_query(request, &[], engine)
}

/// Keep one variable open and sum the rest.
pub fn marginal(request: &Request<'_>, variable: usize, engine: &mut CpuEngine) -> Result<QueryResult, String> {
    plan_query(request, &[variable as u32], engine).map(|(result, _)| result)
}

/// Keep two variables open: the joint marginal.
pub fn joint(
    request: &Request<'_>,
    pair: (usize, usize),
    engine: &mut CpuEngine,
) -> Result<QueryResult, String> {
    let labels = [pair.0 as u32, pair.1 as u32];
    plan_query(request, &labels, engine).map(|(result, _)| result)
}

/// Run the full exact inference, keeping the designed queries in the output.
pub fn run(request: &Request<'_>, engine: &mut CpuEngine) -> Result<ExactRun, String> {
    let (mass, mass_tree) = evidence(request, engine)?;
    let mut marginals = BTreeMap::new();
    for variable in request.queries {
        marginals.insert(*variable, marginal(request, *variable, engine)?);
    }
    let mut joints = BTreeMap::new();
    for pair in request.joints {
        if pair.0 == pair.1 {
            return Err(format!("joint query {:?} repeats a variable", pair));
        }
        joints.insert(*pair, joint(request, *pair, engine)?);
    }
    let enumeration = if request.model.variables <= ENUMERATION_LIMIT {
        Some(enumerate(request.model, request.observations, request.queries, request.joints)?)
    } else {
        None
    };
    Ok(ExactRun {
        mass,
        mass_tree,
        marginals,
        joints,
        enumeration,
    })
}

/// Joint enumeration stays cheap only for the teaching model.
pub const ENUMERATION_LIMIT: usize = 20;

/// Sum every joint row that agrees with the observations (the sheet's 256 rows).
pub fn enumerate(
    model: &Model,
    observations: &[(usize, usize)],
    queries: &[usize],
    joints: &[(usize, usize)],
) -> Result<Enumeration, String> {
    let states = model.dims.iter().product::<usize>();
    let mut mass = 0.0f64;
    let mut marginals: BTreeMap<usize, Vec<f64>> = queries
        .iter()
        .map(|variable| (*variable, vec![0.0; model.dims[*variable]]))
        .collect();
    let mut joint_values: BTreeMap<(usize, usize), Vec<f64>> = joints
        .iter()
        .map(|pair| {
            (
                *pair,
                vec![0.0; model.dims[pair.0] * model.dims[pair.1]],
            )
        })
        .collect();
    let mut assignment = vec![0usize; model.variables];
    for flat in 0..states {
        let mut remainder = flat;
        for variable in (0..model.variables).rev() {
            assignment[variable] = remainder % model.dims[variable];
            remainder /= model.dims[variable];
        }
        if observations
            .iter()
            .any(|(variable, state)| assignment[*variable] != *state)
        {
            continue;
        }
        let mut probability = 1.0f64;
        for factor in &model.factors {
            probability *= entry(factor, &assignment);
        }
        mass += probability;
        for (variable, values) in marginals.iter_mut() {
            values[assignment[*variable]] += probability;
        }
        for (pair, values) in joint_values.iter_mut() {
            values[assignment[pair.0] * model.dims[pair.1] + assignment[pair.1]] += probability;
        }
    }
    if mass == 0.0 {
        return Err("the observations have zero probability under the model".into());
    }
    Ok(Enumeration {
        mass,
        marginals: marginals
            .into_iter()
            .map(|(variable, values)| (variable, values.into_iter().map(|value| value / mass).collect()))
            .collect(),
        joints: joint_values
            .into_iter()
            .map(|(pair, values)| {
                (
                    pair,
                    values.into_iter().map(|value| value / mass).collect(),
                )
            })
            .collect(),
    })
}

fn entry(factor: &Factor, assignment: &[usize]) -> f64 {
    let mut flat = 0usize;
    for (axis, variable) in factor.scope.iter().enumerate() {
        flat = flat * factor.dims[axis] + assignment[*variable];
    }
    factor.entries[flat]
}

/// Normalize a one-variable marginal.
pub fn normalize(values: &[f64]) -> Vec<f64> {
    let total: f64 = values.iter().sum();
    if total == 0.0 {
        return values.to_vec();
    }
    values.iter().map(|value| value / total).collect()
}

/// Normalize a two-variable joint and return it as nested rows.
///
/// The contraction keeps the labels in order, so the first label varies fastest
/// in the column-major buffer: entry `(row, column)` sits at `row + left * column`.
pub fn normalize_joint(values: &[f64], left: usize, right: usize) -> Vec<Vec<f64>> {
    let total: f64 = values.iter().sum();
    (0..left)
        .map(|row| {
            (0..right)
                .map(|column| values[row + left * column] / total)
                .collect()
        })
        .collect()
}

/// joint minus the product of its marginals: the dependence the joint keeps.
pub fn dependence(
    values: &[f64],
    left: usize,
    right: usize,
    rows: &[f64],
    columns: &[f64],
) -> Vec<Vec<f64>> {
    let total: f64 = values.iter().sum();
    (0..left)
        .map(|row| {
            (0..right)
                .map(|column| values[row + left * column] / total - rows[row] * columns[column])
                .collect()
        })
        .collect()
}




