//! Contraction plans: binary pair steps, their costs, and the order searches.

use std::collections::HashMap;

use omeco::{GreedyMethod, Initializer, ScoreFunction, TreeSA};
use serde_json::{json, Value};

/// A binary contraction tree over the network's leaves.
#[derive(Clone, Debug, PartialEq)]
pub enum PlanTree {
    Leaf(usize),
    Pair(Box<PlanTree>, Box<PlanTree>),
}

/// One pair step of an analysed plan.
#[derive(Clone, Debug)]
pub struct Step {
    pub id: usize,
    pub left: usize,
    pub right: usize,
    pub left_labels: Vec<u32>,
    pub right_labels: Vec<u32>,
    pub labels: Vec<u32>,
    pub work: f64,
    pub entries: usize,
}

#[derive(Clone, Debug)]
pub struct Plan {
    pub steps: Vec<Step>,
    pub work: f64,
    pub largest_entries: usize,
}

pub struct Context<'a> {
    pub leaf_labels: &'a [Vec<u32>],
    pub dims: &'a [usize],
    pub query: &'a [u32],
}

impl Context<'_> {
    pub fn size(&self, labels: &[u32]) -> usize {
        labels.iter().map(|label| self.dims[*label as usize]).product()
    }

    pub fn factor_entries(&self) -> usize {
        self.leaf_labels
            .iter()
            .map(|labels| self.size(labels))
            .max()
            .unwrap_or(1)
    }
}

struct Walked {
    id: usize,
    labels: Vec<u32>,
    counts: Vec<u32>,
    work: f64,
    largest: usize,
    steps: Vec<Step>,
}

fn merge_counts(mut left: Vec<u32>, right: &[u32]) -> Vec<u32> {
    for (value, other) in left.iter_mut().zip(right) {
        *value += *other;
    }
    left
}

fn walk(tree: &PlanTree, context: &Context<'_>, totals: &[u32], next_id: &mut usize) -> Walked {
    match tree {
        PlanTree::Leaf(index) => {
            let labels = context.leaf_labels[*index].clone();
            let mut counts = vec![0u32; totals.len()];
            for label in &labels {
                counts[*label as usize] = 1;
            }
            Walked {
                id: *index,
                labels,
                counts,
                work: 0.0,
                largest: context.size(&context.leaf_labels[*index]),
                steps: Vec::new(),
            }
        }
        PlanTree::Pair(left, right) => {
            let mut child_left = walk(left, context, totals, next_id);
            let mut child_right = walk(right, context, totals, next_id);
            let left_labels = child_left.labels.clone();
            let right_labels = child_right.labels.clone();
            let counts = merge_counts(child_left.counts, &child_right.counts);
            let mut carried: Vec<u32> = Vec::new();
            for label in 0..counts.len() {                if counts[label] == 0 {
                    continue;
                }
                let needed_outside = counts[label] < totals[label];
                let requested = context.query.contains(&(label as u32));
                if needed_outside || requested {
                    carried.push(label as u32);
                }
            }
            let step_work = context.size(&union(&left_labels, &right_labels)) as f64;
            let entries = context.size(&carried);
            let id = *next_id;
            *next_id += 1;
            let step = Step {
                id,
                left: child_left.id,
                right: child_right.id,
                left_labels,
                right_labels,
                labels: carried.clone(),
                work: step_work,
                entries,
            };
            let mut steps = child_left.steps.drain(..).collect::<Vec<_>>();
            steps.append(&mut child_right.steps);
            steps.push(step);
            Walked {
                id,
                labels: carried,
                counts,
                work: child_left.work + child_right.work + step_work,
                largest: child_left.largest.max(child_right.largest).max(entries),
                steps,
            }
        }
    }
}

fn union(left: &[u32], right: &[u32]) -> Vec<u32> {
    let mut labels = left.to_vec();
    for label in right {
        if !labels.contains(label) {
            labels.push(*label);
        }
    }
    labels
}

pub fn analyze(tree: &PlanTree, context: &Context<'_>) -> Plan {
    let mut totals = vec![0u32; context.dims.len()];
    for labels in context.leaf_labels {
        for label in labels {
            totals[*label as usize] += 1;
        }
    }
    let mut next_id = context.leaf_labels.len();
    let walked = walk(tree, context, &totals, &mut next_id);
    Plan {
        steps: walked.steps,
        work: walked.work,
        largest_entries: walked.largest.max(context.factor_entries()),
    }
}

/// SplitMix64, so a random plan is reproducible from its seed alone.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn below(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }
}

pub fn random_tree(leaves: usize, seed: u64) -> PlanTree {
    let mut rng = Rng::new(seed);
    let mut nodes: Vec<PlanTree> = (0..leaves).map(PlanTree::Leaf).collect();
    while nodes.len() > 1 {
        let first = rng.below(nodes.len());
        let mut second = rng.below(nodes.len() - 1);
        if second >= first {
            second += 1;
        }
        let (low, high) = if first < second {
            (first, second)
        } else {
            (second, first)
        };
        let right = nodes.swap_remove(high);
        let left = nodes.swap_remove(low);
        nodes.push(PlanTree::Pair(Box::new(left), Box::new(right)));
    }
    nodes.pop().expect("a network has at least one leaf")
}

fn from_nested(tree: &omeco::NestedEinsum<u32>) -> Result<PlanTree, String> {
    match tree {
        omeco::NestedEinsum::Leaf { tensor_index } => Ok(PlanTree::Leaf(*tensor_index)),
        omeco::NestedEinsum::Node { args, .. } => {
            let mut plan = from_nested(
                args.first()
                    .ok_or_else(|| "empty contraction node".to_string())?,
            )?;
            for argument in &args[1..] {
                plan = PlanTree::Pair(Box::new(plan), Box::new(from_nested(argument)?));
            }
            Ok(plan)
        }
    }
}

/// Settings of the simulated-annealing search, recorded beside its plan.
pub fn treesa_settings() -> (TreeSA, Value) {
    let betas: Vec<f64> = (0..30).map(|step| 0.01 + 0.5 * step as f64).collect();
    let settings = TreeSA::new(
        betas,
        2,
        10,
        Initializer::Greedy,
        ScoreFunction::time_optimized(),
    );
    let record = json!({
        "betas": settings.betas,
        "ntrials": settings.ntrials,
        "niters": settings.niters,
        "initializer": "greedy",
        "score": "time_optimized (tc_weight=1, sc_weight=0, rw_weight=0, sc_target=inf)",
        "seed": "omeco seeds trial t with 42 + t",
    });
    (settings, record)
}

pub enum Search {
    Greedy,
    Exhaustive,
    Treesa,
    Random(u64),
}

/// Exhaustive search over pair orders: the smallest possible work for this cost
/// model. Limited to 16 factors (3^16 subset partitions), which covers ASIA.
pub fn exhaustive_tree(
    leaves: &[Vec<u32>],
    dims: &[usize],
    query: &[u32],
) -> Result<PlanTree, String> {
    let count = leaves.len();
    if count > 16 {
        return Err(format!(
            "exhaustive search supports at most 16 factors, this network has {count}; use greedy or treesa"
        ));
    }
    let masks = 1usize << count;
    let mut union_inside: Vec<Vec<u32>> = vec![Vec::new(); masks];
    for mask in 1..masks {
        let mut labels: Vec<u32> = Vec::new();
        for (index, leaf) in leaves.iter().enumerate() {
            if mask >> index & 1 == 1 {
                for label in leaf {
                    if !labels.contains(label) {
                        labels.push(*label);
                    }
                }
            }
        }
        labels.sort_unstable();
        union_inside[mask] = labels;
    }
    let mut carried: Vec<Vec<u32>> = vec![Vec::new(); masks];
    for mask in 1..masks {
        if mask & (mask - 1) == 0 {
            carried[mask] = union_inside[mask].clone();
            continue;
        }
        let outside = union_inside[masks - 1 - mask].clone();
        carried[mask] = union_inside[mask]
            .iter()
            .copied()
            .filter(|label| outside.contains(label) || query.contains(label))
            .collect();
    }
    let size = |labels: &[u32]| -> f64 {
        labels
            .iter()
            .map(|label| dims[*label as usize] as f64)
            .product()
    };
    let mut cost = vec![f64::INFINITY; masks];
    let mut choice: Vec<Option<(usize, usize)>> = vec![None; masks];
    for index in 0..count {
        cost[1 << index] = 0.0;
    }
    for mask in 1..masks {
        if mask & (mask - 1) == 0 {
            continue;
        }
        let low = mask & mask.wrapping_neg();
        let mut subset = (mask - 1) & mask;
        while subset > 0 {
            if subset & low != 0 {
                let other = mask ^ subset;
                if other != 0 {
                    let step = size(&union(&carried[subset], &carried[other]));
                    let total = cost[subset] + cost[other] + step;
                    if total < cost[mask] - 1e-9 {
                        cost[mask] = total;
                        choice[mask] = Some((subset, other));
                    }
                }
            }
            subset = (subset - 1) & mask;
        }
    }
    fn build(mask: usize, choice: &[Option<(usize, usize)>]) -> PlanTree {
        match choice[mask] {
            None => PlanTree::Leaf(mask.trailing_zeros() as usize),
            Some((left, right)) => PlanTree::Pair(
                Box::new(build(left, choice)),
                Box::new(build(right, choice)),
            ),
        }
    }
    Ok(build(masks - 1, &choice))
}

/// Plan a query with the requested search; `settings` is filled for TreeSA.
pub fn search(
    method: &Search,
    leaves: &[Vec<u32>],
    dims: &[usize],
    query: &[u32],
) -> Result<(PlanTree, Value), String> {
    match method {
        Search::Random(seed) => Ok((random_tree(leaves.len(), *seed), json!({ "seed": seed }))),
        Search::Exhaustive => Ok((
            exhaustive_tree(leaves, dims, query)?,
            json!({ "method": "dynamic program over factor subsets, at most 16 factors" }),
        )),
        Search::Greedy | Search::Treesa => {
            let code = omeco::EinCode::new(leaves.to_vec(), query.to_vec());
            let sizes: HashMap<u32, usize> = dims
                .iter()
                .enumerate()
                .map(|(label, size)| (label as u32, *size))
                .collect();
            let (nested, settings) = match method {
                Search::Greedy => (
                    omeco::optimize_code(&code, &sizes, &GreedyMethod::default()),
                    json!({ "alpha": 0.0, "temperature": 0.0 }),
                ),
                Search::Treesa => {
                    let (settings, record) = treesa_settings();
                    (omeco::optimize_code(&code, &sizes, &settings), record)
                }
                _ => unreachable!(),
            };
            let nested = nested.ok_or_else(|| "the order search returned no plan".to_string())?;
            Ok((from_nested(&nested)?, settings))
        }
    }
}

pub fn tree_to_json(tree: &PlanTree) -> Value {
    match tree {
        PlanTree::Leaf(index) => json!({ "leaf": index }),
        PlanTree::Pair(left, right) => {
            json!({ "left": tree_to_json(left), "right": tree_to_json(right) })
        }
    }
}

pub fn tree_from_json(value: &Value, leaves: usize) -> Result<PlanTree, String> {
    if let Some(index) = value.get("leaf") {
        let index = index.as_u64().ok_or("leaf index is not an integer")? as usize;
        if index >= leaves {
            return Err(format!("plan leaf {index} outside 0..{leaves}"));
        }
        return Ok(PlanTree::Leaf(index));
    }
    let left = value.get("left").ok_or("plan node has no left child")?;
    let right = value.get("right").ok_or("plan node has no right child")?;
    Ok(PlanTree::Pair(
        Box::new(tree_from_json(left, leaves)?),
        Box::new(tree_from_json(right, leaves)?),
    ))
}

pub fn steps_to_json(plan: &Plan) -> Value {
    Value::Array(
        plan.steps
            .iter()
            .map(|step| {
                json!({
                    "id": step.id,
                    "left": step.left,
                    "right": step.right,
                    "left_labels": step.left_labels,
                    "right_labels": step.right_labels,
                    "labels": step.labels,
                    "work": step.work,
                    "entries": step.entries,
                })
            })
            .collect(),
    )
}

/// Verify that a plan uses every leaf exactly once and returns one root.
pub fn validate(tree: &PlanTree, leaves: usize) -> Result<(), String> {
    fn walk(tree: &PlanTree, seen: &mut Vec<usize>, depth: usize, limit: usize) -> Result<usize, String> {
        match tree {
            PlanTree::Leaf(index) => {
                if *index >= seen.len() {
                    return Err(format!("leaf {index} outside the network"));
                }
                seen[*index] += 1;
                Ok(1)
            }
            PlanTree::Pair(left, right) => {
                if depth > limit {
                    return Err("plan is deeper than the network".into());
                }
                Ok(walk(left, seen, depth + 1, limit)? + walk(right, seen, depth + 1, limit)?)
            }
        }
    }
    let mut seen = vec![0usize; leaves];
    let count = walk(tree, &mut seen, 0, leaves)?;
    if count != leaves {
        return Err(format!("plan covers {count} of {leaves} leaves"));
    }
    if let Some((index, times)) = seen.iter().enumerate().find(|(_, times)| **times != 1) {
        return Err(format!("leaf {index} appears {times} times"));
    }
    Ok(())
}


