//! `infer`: exact tensor-network inference, contraction-order planning and
//! loopy belief propagation for UAI models.

mod bp;
mod checks;
mod exact;
mod network;
mod plan;
mod uai;

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use serde_json::{json, Map, Value};
use tenferro_cpu::CpuEngine;

use crate::exact::{ExactRun, QueryResult, Request};
use crate::plan::{Plan, PlanTree, Search};
use crate::uai::{Model, Observations};

const USAGE: &str = "\
infer --model <path> [--evidence prior|AD|ADX] --out <dir> [--query ids]
      [--order greedy|random|exhaustive|treesa] [--seed n]
      [--plan <order.json> | --analyze <order.json>]
      [--method exact|bp --damping x --tolerance t --max-sweeps n]

Exact inference fixes the observed indices, sums the unknown ones and keeps the
queried ones open. Contraction orders only change the cost. With --method bp the
same factors feed damped synchronous loopy belief propagation.";

struct Args {
    model: Option<PathBuf>,
    evidence: Option<String>,
    out: Option<PathBuf>,
    query: Option<String>,
    order: Option<String>,
    seed: Option<u64>,
    plan: Option<PathBuf>,
    analyze: Option<PathBuf>,
    method: String,
    damping: Option<f64>,
    tolerance: Option<f64>,
    max_sweeps: Option<usize>,
}

fn main() -> ExitCode {
    match run_cli() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("infer: {message}");
            ExitCode::FAILURE
        }
    }
}

fn run_cli() -> Result<(), String> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.is_empty() || raw.iter().any(|argument| argument == "--help" || argument == "-h") {
        println!("{USAGE}");
        return Ok(());
    }
    let args = parse_args(&raw)?;
    validate(&args)?;
    let model = Model::read(&args.model.clone().expect("validated")).map_err(|error| error.to_string())?;
    let out = args.out.clone().expect("validated");
    fs::create_dir_all(&out).map_err(|error| format!("cannot create {}: {error}", out.display()))?;

    let (label, observations) = resolve_observations(&model, args.evidence.as_deref())?;
    let queries = resolve_queries(&model, args.query.as_deref())?;
    let joints: Vec<(usize, usize)> = if model.variables <= exact::ENUMERATION_LIMIT {
        let mut pairs = Vec::new();
        for (index, first) in queries.iter().enumerate() {
            for second in &queries[index + 1..] {
                pairs.push((*first, *second));
            }
        }
        pairs
    } else {
        Vec::new()
    };
    let search = match args.order.as_deref() {
        Some("random") => Search::Random(args.seed.expect("validated")),
        Some("exhaustive") => Search::Exhaustive,
        Some("treesa") => Search::Treesa,
        _ => Search::Greedy,
    };

    let network = network::build_network(&model, &observations)?;
    let leaf_labels: Vec<Vec<u32>> = network.leaves.iter().map(|leaf| leaf.labels.clone()).collect();
    let leaves = leaf_labels.len();

    if let Some(path) = &args.analyze {
        return analyze_plan(&model, &leaf_labels, &out, path);
    }

    let mut replayed = false;
    let (scalar_plan, settings) = match &args.plan {
        Some(path) => {
            replayed = true;
            let tree = read_plan(path, leaves)?;
            (
                tree,
                json!({
                    "source": path.display().to_string(),
                    "source_sha256": uai::sha256_file(path),
                }),
            )
        }
        None => {
            let (tree, settings) = plan::search(&search, &leaf_labels, &model.dims, &[])?;
            (tree, settings)
        }
    };
    let method = if replayed {
        "replay".to_string()
    } else {
        search_name(&search).to_string()
    };
    let request = Request {
        model: &model,
        observations: &observations,
        queries: &queries,
        joints: &joints,
        search: &search,
        scalar_plan: Some(&scalar_plan),
    };
    let context = RunContext {
        args: &args,
        method,
        seed: match search {
            Search::Random(seed) => Some(seed),
            _ => None,
        },
        settings,
        leaves,
        label,
        plan_path: args.plan.clone(),
    };
    match args.method.as_str() {
        "bp" => run_bp(&context, &request, &out),
        _ => run_exact(&context, &request, &out),
    }
}

struct RunContext<'a> {
    args: &'a Args,
    method: String,
    seed: Option<u64>,
    settings: Value,
    leaves: usize,
    label: String,
    plan_path: Option<PathBuf>,
}

fn parse_args(raw: &[String]) -> Result<Args, String> {
    let mut args = Args {
        model: None,
        evidence: None,
        out: None,
        query: None,
        order: None,
        seed: None,
        plan: None,
        analyze: None,
        method: "exact".to_string(),
        damping: None,
        tolerance: None,
        max_sweeps: None,
    };
    let mut index = 0usize;
    while index < raw.len() {
        let flag = raw[index].clone();
        match flag.as_str() {
            "--model" => args.model = Some(PathBuf::from(next(raw, &mut index, &flag)?)),
            "--evidence" => args.evidence = Some(next(raw, &mut index, &flag)?),
            "--out" => args.out = Some(PathBuf::from(next(raw, &mut index, &flag)?)),
            "--query" => args.query = Some(next(raw, &mut index, &flag)?),
            "--order" => args.order = Some(next(raw, &mut index, &flag)?),
            "--seed" => {
                args.seed = Some(
                    next(raw, &mut index, &flag)?
                        .parse()
                        .map_err(|_| "--seed needs an integer".to_string())?,
                )
            }
            "--plan" => args.plan = Some(PathBuf::from(next(raw, &mut index, &flag)?)),
            "--analyze" => args.analyze = Some(PathBuf::from(next(raw, &mut index, &flag)?)),
            "--method" => args.method = next(raw, &mut index, &flag)?,
            "--damping" => {
                args.damping = Some(
                    next(raw, &mut index, &flag)?
                        .parse()
                        .map_err(|_| "--damping needs a number".to_string())?,
                )
            }
            "--tolerance" => {
                args.tolerance = Some(
                    next(raw, &mut index, &flag)?
                        .parse()
                        .map_err(|_| "--tolerance needs a number".to_string())?,
                )
            }
            "--max-sweeps" => {
                args.max_sweeps = Some(
                    next(raw, &mut index, &flag)?
                        .parse()
                        .map_err(|_| "--max-sweeps needs an integer".to_string())?,
                )
            }
            other => return Err(format!("unknown argument {other}")),
        }
        index += 1;
    }
    Ok(args)
}

fn next(raw: &[String], index: &mut usize, flag: &str) -> Result<String, String> {
    *index += 1;
    raw.get(*index)
        .cloned()
        .ok_or_else(|| format!("{flag} needs a value"))
}

fn validate(args: &Args) -> Result<(), String> {
    if args.model.is_none() {
        return Err("--model is required".into());
    }
    if args.out.is_none() {
        return Err("--out is required".into());
    }
    let exclusive = [args.order.is_some(), args.plan.is_some(), args.analyze.is_some()];
    if exclusive.iter().filter(|given| **given).count() > 1 {
        return Err("--order, --plan and --analyze are mutually exclusive".into());
    }
    if let Some(order) = &args.order {
        if !["greedy", "random", "exhaustive", "treesa"].contains(&order.as_str()) {
            return Err(format!("unknown order {order}"));
        }
    }
    match args.order.as_deref() {
        Some("random") if args.seed.is_none() => return Err("--order random needs --seed".into()),
        Some("random") => {}
        _ if args.seed.is_some() => return Err("--seed is only valid with --order random".into()),
        _ => {}
    }
    if args.method != "exact" && args.method != "bp" {
        return Err(format!("unknown method {}", args.method));
    }
    let bp_flags = args.damping.is_some() || args.tolerance.is_some() || args.max_sweeps.is_some();
    if args.method == "bp" {
        let damping = args
            .damping
            .ok_or("--method bp needs --damping, --tolerance and --max-sweeps")?;
        let tolerance = args
            .tolerance
            .ok_or("--method bp needs --damping, --tolerance and --max-sweeps")?;
        args.max_sweeps
            .ok_or("--method bp needs --damping, --tolerance and --max-sweeps")?;
        if !(0.0..=1.0).contains(&damping) || damping == 0.0 {
            return Err("--damping must be in (0, 1]".into());
        }
        if tolerance <= 0.0 {
            return Err("--tolerance must be positive".into());
        }
        if args.max_sweeps == Some(0) {
            return Err("--max-sweeps must be at least one".into());
        }
    } else if bp_flags {
        return Err("--damping, --tolerance and --max-sweeps belong to --method bp".into());
    }
    if args.analyze.is_some() && args.method == "bp" {
        return Err("--analyze only recounts a plan; it does not run belief propagation".into());
    }
    Ok(())
}

fn resolve_observations(model: &Model, evidence: Option<&str>) -> Result<(String, Observations), String> {
    match evidence {
        Some("prior") => Ok(("prior".into(), Vec::new())),
        Some(label @ ("AD" | "ADX")) => {
            if model.variables != 8 {
                return Err(format!(
                    "--evidence AD/ADX names the ASIA cases; this model has {} variables",
                    model.variables
                ));
            }
            let mut observations = vec![(0usize, 0usize), (7, 0)];
            if label == "ADX" {
                observations.push((6, 0));
            }
            Ok((label.to_string(), observations))
        }
        Some(other) => Err(format!("unknown evidence {other}; use prior, AD or ADX")),
        None => match model.sibling_evidence() {
            Some(path) => {
                let observations = uai::read_evidence(&path).map_err(|error| error.to_string())?;
                Ok((path.display().to_string(), observations))
            }
            None => Ok(("prior".into(), Vec::new())),
        },
    }
}

fn resolve_queries(model: &Model, query: Option<&str>) -> Result<Vec<usize>, String> {
    match query {
        Some(list) => {
            let mut queries = Vec::new();
            for token in list.split(',') {
                let variable: usize = token
                    .trim()
                    .parse()
                    .map_err(|_| format!("query {token:?} is not a variable ID"))?;
                if variable >= model.variables {
                    return Err(format!("query {variable} outside 0..{}", model.variables));
                }
                if !queries.contains(&variable) {
                    queries.push(variable);
                }
            }
            if queries.is_empty() {
                return Err("--query lists no variable".into());
            }
            Ok(queries)
        }
        None if model.variables <= 16 => Ok((0..model.variables).collect()),
        None => Err(format!(
            "this model has {} variables; --query is required above 16",
            model.variables
        )),
    }
}

fn read_plan(path: &Path, leaves: usize) -> Result<PlanTree, String> {
    let text = fs::read_to_string(path).map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    let value: Value = serde_json::from_str(&text).map_err(|error| format!("{}: {error}", path.display()))?;
    let tree = value
        .get("tree")
        .ok_or_else(|| format!("{} has no plan tree", path.display()))?;
    let tree = plan::tree_from_json(tree, leaves)?;
    plan::validate(&tree, leaves)?;
    Ok(tree)
}

fn analyze_plan(model: &Model, leaf_labels: &[Vec<u32>], out: &Path, path: &Path) -> Result<(), String> {
    let tree = read_plan(path, leaf_labels.len())?;
    let context = plan::Context {
        leaf_labels,
        dims: &model.dims,
        query: &[],
    };
    let analysis = plan::analyze(&tree, &context);
    write_json(&out.join("complexity.json"), &complexity_json(&analysis, "analysis"))?;
    println!(
        "plan {}: work = {} terms, largest tensor = {} entries, {} steps",
        path.display(),
        analysis.work,
        analysis.largest_entries,
        analysis.steps.len()
    );
    for step in &analysis.steps {
        println!(
            "  step {:>3}: {} x {} -> {:?}  work {}  entries {}",
            step.id, step.left, step.right, step.labels, step.work, step.entries
        );
    }
    Ok(())
}

fn complexity_json(analysis: &Plan, method: &str) -> Value {
    json!({
        "method": method,
        "query": { "labels": [] },
        "work": analysis.work,
        "largest_entries": analysis.largest_entries,
        "steps": plan::steps_to_json(analysis),
    })
}

fn order_json(context: &RunContext<'_>, analysis: &Plan, tree: &PlanTree) -> Value {
    let mut value = json!({
        "method": context.method,
        "seed": context.seed,
        "leaves": context.leaves,
        "query": { "labels": [] },
        "settings": context.settings,
        "work": analysis.work,
        "largest_entries": analysis.largest_entries,
        "steps": plan::steps_to_json(analysis),
        "tree": plan::tree_to_json(tree),
    });
    if let Some(path) = &context.plan_path {
        value["source"] = json!(path.display().to_string());
        value["source_sha256"] = json!(uai::sha256_file(path));
    }
    value
}

fn model_json(model: &Model) -> Value {
    json!({
        "path": model.path.display().to_string(),
        "sha256": model.sha256(),
        "variables": model.variables,
        "dims": model.dims,
        "factors": model.factors.iter().map(|factor| json!({
            "scope": factor.scope,
            "dims": factor.dims,
            "entries": factor.entries,
        })).collect::<Vec<_>>(),
    })
}

fn observations_json(observations: &Observations) -> Value {
    let mut map = Map::new();
    let mut sorted = observations.to_vec();
    sorted.sort_unstable();
    for (variable, state) in sorted {
        map.insert(variable.to_string(), json!(state));
    }
    Value::Object(map)
}

fn marginal_values(result: &QueryResult) -> Vec<f64> {
    exact::normalize(&result.values())
}

fn max_abs(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .fold(0.0f64, |worst, (a, b)| worst.max((a - b).abs()))
}

fn search_name(search: &Search) -> &'static str {
    match search {
        Search::Greedy => "greedy",
        Search::Exhaustive => "exhaustive",
        Search::Treesa => "treesa",
        Search::Random(_) => "random",
    }
}

fn run_exact(context: &RunContext<'_>, request: &Request<'_>, out: &Path) -> Result<(), String> {
    let mut engine = CpuEngine::serial();
    let check_values = checks::run(&mut engine)?;
    let started = Instant::now();
    let result = exact::run(request, &mut engine)?;
    let runtime = started.elapsed().as_secs_f64();
    let mass_value = result.mass.mass();
    let log10_mass = mass_value.log10();

    let queries = request.queries.to_vec();
    let mut marginals: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
    let mut raw_marginals = Map::new();
    for variable in &queries {
        let query = &result.marginals[variable];
        marginals.insert(*variable, marginal_values(query));
        raw_marginals.insert(variable.to_string(), json!(query.values()));
    }
    let mut joint_map = Map::new();
    let mut dependence_map = Map::new();
    for (pair, query) in &result.joints {
        let left = request.model.dims[pair.0];
        let right = request.model.dims[pair.1];
        joint_map.insert(
            format!("{},{}", pair.0, pair.1),
            json!(exact::normalize_joint(&query.values(), left, right)),
        );
        dependence_map.insert(
            format!("{},{}", pair.0, pair.1),
            json!(exact::dependence(
                &query.values(),
                left,
                right,
                &marginals[&pair.0],
                &marginals[&pair.1],
            )),
        );
    }

    let (max_marginal_error, mass_relative_error) = match &result.enumeration {
        Some(enumeration) => {
            let mut worst = 0.0f64;
            for variable in &queries {
                worst = worst.max(max_abs(&marginals[variable], &enumeration.marginals[variable]));
            }
            for (pair, query) in &result.joints {
                let left = request.model.dims[pair.0];
                let right = request.model.dims[pair.1];
                let joint = exact::normalize_joint(&query.values(), left, right);
                let expected = &enumeration.joints[pair];
                for row in 0..left {
                    for column in 0..right {
                        worst = worst.max((joint[row][column] - expected[row * right + column]).abs());
                    }
                }
            }
            (
                worst,
                (mass_value - enumeration.mass).abs() / enumeration.mass.abs(),
            )
        }
        None => (f64::NAN, f64::NAN),
    };

    let joints_work: f64 = result.joints.values().map(|query| query.plan.work).sum();
    let queries_work: f64 = result.marginals.values().map(|query| query.plan.work).sum();
    write_json(&out.join("order.json"), &order_json(context, &result.mass.plan, &result.mass_tree))?;
    write_json(
        &out.join("complexity.json"),
        &complexity_json(&result.mass.plan, &context.method),
    )?;

    let mut contraction = json!({
        "log_scale": result.mass.log_scale,
        "mass": mass_value,
        "log10_mass": log10_mass,
        "node_marginals": marginal_map(&marginals),
        "node_joints": raw_marginals,
        "joints": joint_map,
        "dependence": dependence_map,
        "max_marginal_error": max_marginal_error,
        "mass_relative_error": mass_relative_error,
        "marginals_work": queries_work,
        "joints_work": joints_work,
    });
    if result.joints.is_empty() {
        contraction["joints"] = Value::Null;
        contraction["dependence"] = Value::Null;
    }
    let reference = reference_block(request.model, Some(mass_value), &marginals);
    let mut run = json!({
        "command": std::env::args().skip(1).collect::<Vec<_>>(),
        "model": model_json(request.model),
        "evidence": { "label": context.label, "observations": observations_json(request.observations) },
        "method": "exact",
        "ordering": {
            "method": context.method,
            "seed": context.seed,
            "work": result.mass.plan.work,
            "largest_entries": result.mass.plan.largest_entries,
            "plan": "scalar",
        },
        "queries": queries,
        "contraction": contraction,
        "enumeration": result.enumeration.as_ref().map(enumeration_json),
        "checks": check_values,
        "runtime_seconds": runtime,
    });
    if let Some(reference) = reference {
        run["reference"] = reference;
    }
    write_json(&out.join("run.json"), &run)?;

    println!(
        "evidence {0}: mass = {1:.17e}  log10 = {2:.15}",
        context.label, mass_value, log10_mass
    );
    for (variable, values) in &marginals {
        println!("  p({variable} = 0) = {:.12}", values.first().copied().unwrap_or(f64::NAN));
    }
    if max_marginal_error.is_finite() {
        println!(
            "  max |marginal - enumeration| = {max_marginal_error:.3e}; relative evidence error = {mass_relative_error:.3e}"
        );
    }
    println!(
        "  plan work = {:.0} terms, largest tensor = {} entries, runtime {runtime:.2} s",
        result.mass.plan.work, result.mass.plan.largest_entries
    );
    Ok(())
}

fn marginal_map(marginals: &BTreeMap<usize, Vec<f64>>) -> Value {
    let mut map = Map::new();
    for (variable, values) in marginals {
        map.insert(variable.to_string(), json!(values));
    }
    Value::Object(map)
}

fn enumeration_json(enumeration: &exact::Enumeration) -> Value {
    let mut joint_map = Map::new();
    for (pair, values) in &enumeration.joints {
        joint_map.insert(format!("{},{}", pair.0, pair.1), json!(values));
    }
    json!({
        "mass": enumeration.mass,
        "node_marginals": marginal_map(&enumeration.marginals),
        "joints": joint_map,
    })
}

fn reference_block(
    model: &Model,
    mass: Option<f64>,
    marginals: &BTreeMap<usize, Vec<f64>>,
) -> Option<Value> {
    let pr_path = model.sibling(".PR");
    let mar_path = model.sibling(".MAR");
    if pr_path.is_none() && mar_path.is_none() {
        return None;
    }
    let mut block = Map::new();
    if let Some(path) = pr_path {
        if let Ok(published) = uai::read_pr(&path) {
            let ours = mass.map(f64::log10);
            block.insert(
                "pr".into(),
                json!({
                    "path": path.display().to_string(),
                    "published": published,
                    "log10_mass": ours,
                    "difference": ours.map(|value| (value - published).abs()),
                    "tolerance": 1e-4,
                }),
            );
        }
    }
    if let Some(path) = mar_path {
        if let Ok(published) = uai::read_mar(&path) {
            let mut per_variable = Map::new();
            let mut worst = 0.0f64;
            for (variable, values) in marginals {
                if let Some(expected) = published.get(*variable) {
                    let difference = max_abs(values, expected);
                    worst = worst.max(difference);
                    per_variable.insert(variable.to_string(), json!(difference));
                }
            }
            block.insert(
                "mar".into(),
                json!({
                    "path": path.display().to_string(),
                    "max_abs_difference": worst,
                    "per_variable": per_variable,
                    "tolerance": 1e-5,
                }),
            );
        }
    }
    Some(Value::Object(block))
}

fn run_bp(context: &RunContext<'_>, request: &Request<'_>, out: &Path) -> Result<(), String> {
    let mut engine = CpuEngine::serial();
    let started = Instant::now();
    let result: ExactRun = exact::run(request, &mut engine)?;
    let exact_marginals: BTreeMap<usize, Vec<f64>> = request
        .queries
        .iter()
        .map(|variable| (*variable, marginal_values(&result.marginals[variable])))
        .collect();
    let settings = bp::Settings {
        damping: context.args.damping.expect("validated"),
        tolerance: context.args.tolerance.expect("validated"),
        max_sweeps: context.args.max_sweeps.expect("validated"),
    };
    let (outcome, recorded) = bp::run(request.model, request.observations, settings, request.queries)?;
    let runtime = started.elapsed().as_secs_f64();

    let mut node_errors = Map::new();
    let mut worst = 0.0f64;
    for variable in &recorded {
        let error = max_abs(&outcome.final_beliefs[variable], &exact_marginals[variable]);
        worst = worst.max(error);
        node_errors.insert(variable.to_string(), json!(error));
    }
    let sweeps: Vec<Value> = outcome
        .sweeps
        .iter()
        .map(|record| {
            let error = record.beliefs.iter().fold(0.0f64, |worst, (variable, values)| {
                worst.max(max_abs(values, &exact_marginals[variable]))
            });
            let mut beliefs = Map::new();
            for (variable, values) in &record.beliefs {
                beliefs.insert(variable.to_string(), json!(values));
            }
            json!({
                "sweep": record.sweep,
                "residual": record.residual,
                "cumulative_work": record.work,
                "max_node_error": error,
                "beliefs": Value::Object(beliefs),
            })
        })
        .collect();
    let mut final_beliefs = Map::new();
    for (variable, values) in &outcome.final_beliefs {
        final_beliefs.insert(variable.to_string(), json!(values));
    }
    let reference = reference_block(request.model, Some(result.mass.mass()), &exact_marginals);
    let mut run = json!({
        "command": std::env::args().skip(1).collect::<Vec<_>>(),
        "model": model_json(request.model),
        "evidence": { "label": context.label, "observations": observations_json(request.observations) },
        "method": "bp",
        "settings": {
            "damping": settings.damping,
            "tolerance": settings.tolerance,
            "max_sweeps": settings.max_sweeps,
        },
        "ordering": {
            "method": context.method,
            "work": result.mass.plan.work,
            "largest_entries": result.mass.plan.largest_entries,
            "plan": "scalar",
        },
        "queries": request.queries,
        "graph": {
            "factors": outcome.graph_factors,
            "edges": outcome.graph_edges,
            "update_work_per_sweep": outcome.per_sweep_work,
        },
        "sweeps": sweeps,
        "final": {
            "sweeps": outcome.sweeps.len(),
            "converged": outcome.converged,
            "residual": outcome.residual,
            "cumulative_work": outcome.work,
            "beliefs": Value::Object(final_beliefs),
        },
        "exact": {
            "method": context.method,
            "log10_mass": result.mass.mass().log10(),
            "node_marginals": marginal_map(&exact_marginals),
        },
        "bp": {
            "max_node_error": worst,
            "node_errors": node_errors,
        },
        "runtime_seconds": runtime,
    });
    if let Some(reference) = reference {
        run["reference"] = reference;
    }
    write_json(&out.join("run.json"), &run)?;
    println!(
        "bp {0}: {1} sweeps, residual {2:.3e}, converged {3}, max marginal error {4:.6}",
        context.label,
        outcome.sweeps.len(),
        outcome.residual,
        outcome.converged,
        worst
    );
    println!(
        "  exact log10 evidence {:.15}, update work per sweep {:.0}",
        result.mass.mass().log10(),
        outcome.per_sweep_work
    );
    Ok(())
}

fn write_json(path: &Path, value: &Value) -> Result<(), String> {
    let text = serde_json::to_string_pretty(value).map_err(|error| error.to_string())?;
    fs::write(path, format!("{text}\n")).map_err(|error| format!("cannot write {}: {error}", path.display()))
}
