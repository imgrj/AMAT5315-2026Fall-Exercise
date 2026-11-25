//! `tnsample`: the Week 11 tool, built from `tnsample.design.toml`.
//!
//! ```text
//! tnsample --mode gadgets --seed 2026 --out runs/gadgets
//! tnsample --mode plan   --order greedy --input runs/pred/factor_15.bundle.json --seed 2026 --out runs/plan
//! tnsample --mode count  --l 10 --p-af 0 --beta 2.0 --seed 2026 --out runs/count_p0
//! tnsample --mode count  --input runs/pred/factor_15.bundle.json --beta 2.0 --seed 2026 --out runs/count_f15
//! tnsample --mode sample --input runs/pred/factor_15.bundle.json --beta 2.0 --samples 2000 --seed 2026 --out runs/sample_f15
//! tnsample --mode count  --n 8 --seed 2026 --out runs/count_nq8
//! tnsample --mode sample --n 6 --samples 200 --seed 2026 --out runs/sample_nq6
//! ```
//!
//! Nothing is defaulted: a missing or unknown flag is an error. Every run prints
//! `mode metric value runtime_ms` rows and writes `<out>/run.json`.

use tnsample::{gates, net, rng};

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use serde_json::{json, Value};

const SCHEMA: &str = "tnsample/1";

struct Args {
    mode: String,
    seed: u64,
    out: PathBuf,
    l: Option<usize>,
    p_af: Option<f64>,
    beta: Option<f64>,
    input: Option<PathBuf>,
    order: Option<net::Order>,
    samples: Option<usize>,
    n: Option<usize>,
}

fn parse_args() -> Result<Args, String> {
    let mut argv = std::env::args().skip(1);
    let mut mode = None;
    let mut seed = None;
    let mut out = None;
    let mut l = None;
    let mut p_af = None;
    let mut beta = None;
    let mut input = None;
    let mut order = None;
    let mut samples = None;
    let mut n = None;
    while let Some(flag) = argv.next() {
        let slot = match flag.as_str() {
            "--mode" | "--seed" | "--out" | "--l" | "--p-af" | "--beta" | "--input" | "--order"
            | "--samples" | "--n" => flag,
            other => return Err(format!("unknown flag {other}")),
        };
        let value = argv
            .next()
            .ok_or_else(|| format!("flag {slot} needs a value"))?;
        match slot.as_str() {
            "--mode" => mode = Some(value),
            "--seed" => seed = Some(value.parse::<u64>().map_err(|_| "--seed must be an integer")?),
            "--out" => out = Some(PathBuf::from(value)),
            "--l" => l = Some(value.parse::<usize>().map_err(|_| "--l must be an integer")?),
            "--p-af" => p_af = Some(value.parse::<f64>().map_err(|_| "--p-af must be a number")?),
            "--beta" => beta = Some(value.parse::<f64>().map_err(|_| "--beta must be a number")?),
            "--input" => input = Some(PathBuf::from(value)),
            "--order" => {
                order = Some(
                    net::Order::parse(&value)
                        .ok_or_else(|| format!("--order must be greedy or treesa, not {value}"))?,
                )
            }
            "--samples" => {
                samples = Some(
                    value
                        .parse::<usize>()
                        .map_err(|_| "--samples must be an integer")?,
                )
            }
            "--n" => n = Some(value.parse::<usize>().map_err(|_| "--n must be an integer")?),
            _ => unreachable!(),
        }
    }
    let mode = mode.ok_or("--mode is required (gadgets, plan, count or sample)")?;
    if !matches!(mode.as_str(), "gadgets" | "plan" | "count" | "sample") {
        return Err(format!("unknown mode {mode}"));
    }
    let args = Args {
        mode,
        seed: seed.ok_or("--seed is required")?,
        out: out.ok_or("--out is required")?,
        l,
        p_af,
        beta,
        input,
        order,
        samples,
        n,
    };
    let has_lattice = args.l.is_some() || args.p_af.is_some();
    if args.l.is_some() != args.p_af.is_some() {
        return Err("--l and --p-af go together".into());
    }
    let descriptions = [has_lattice, args.input.is_some(), args.n.is_some()]
        .iter()
        .filter(|&&x| x)
        .count();
    if descriptions > 1 {
        return Err("give exactly one of --input, --l/--p-af, or --n".into());
    }
    if args.mode == "gadgets" {
        if descriptions > 0 || args.beta.is_some() || args.samples.is_some() {
            return Err("--mode gadgets takes no problem flags".into());
        }
    } else {
        if descriptions == 0 {
            return Err(format!("--mode {} needs --input, --l/--p-af or --n", args.mode));
        }
        if args.n.is_some() {
            if args.beta.is_some() {
                return Err("--n takes no --beta: the queen rules have no temperature".into());
            }
        } else if args.beta.is_none() && args.mode != "plan" {
            return Err(format!("--mode {} needs --beta for a magnet", args.mode));
        }
        if args.mode == "sample" && args.samples.is_none() {
            return Err("--mode sample needs --samples".into());
        }
        if args.mode != "sample" && args.samples.is_some() {
            return Err("--samples belongs to --mode sample".into());
        }
    }
    Ok(args)
}

fn number(x: f64) -> Value {
    if x.fract() == 0.0 && x.abs() < 1.0e15 {
        json!(x.round() as i64)
    } else {
        json!(x)
    }
}

fn fmt(x: f64) -> String {
    if x.fract() == 0.0 && x.abs() < 1.0e15 {
        format!("{}", x.round() as i64)
    } else {
        format!("{x}")
    }
}

struct Reporter {
    rows: Vec<(String, String)>,
    start: Instant,
}

impl Reporter {
    fn new() -> Self {
        Reporter {
            rows: Vec::new(),
            start: Instant::now(),
        }
    }

    fn push(&mut self, metric: impl Into<String>, value: impl Into<String>) {
        self.rows.push((metric.into(), value.into()));
    }

    fn print(&self, mode: &str) {
        println!("mode\tmetric\tvalue\truntime_ms");
        let elapsed = self.start.elapsed().as_secs_f64() * 1000.0;
        for (metric, value) in &self.rows {
            println!("{mode}\t{metric}\t{value}\t{elapsed:.3}");
        }
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("tnsample: {message}");
            ExitCode::from(2u8)
        }
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let start = Instant::now();
    let mut report = Reporter::new();
    let value = match args.mode.as_str() {
        "gadgets" => run_gadgets(&args, &mut report)?,
        "plan" => run_plan(&args, &mut report)?,
        "count" => run_count(&args, &mut report)?,
        "sample" => run_sample(&args, &mut report)?,
        _ => unreachable!(),
    };
    let runtime_ms = start.elapsed().as_secs_f64() * 1000.0;
    let mut record = value;
    record["schema"] = json!(SCHEMA);
    record["mode"] = json!(args.mode);
    record["seed"] = json!(args.seed);
    record["runtime_ms"] = json!(runtime_ms);
    std::fs::create_dir_all(&args.out)
        .map_err(|error| format!("cannot create {}: {error}", args.out.display()))?;
    let path = args.out.join("run.json");
    std::fs::write(&path, serde_json::to_string_pretty(&record).unwrap())
        .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
    report.print(&args.mode);
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 2: --mode gadgets
// ---------------------------------------------------------------------------

fn run_gadgets(args: &Args, report: &mut Reporter) -> Result<Value, String> {
    let _ = args;
    let mut gates_value = serde_json::Map::new();
    let mut floors = serde_json::Map::new();
    for gate in gates::GATES.iter() {
        let k = gate.bits.len();
        let mut rows = Vec::new();
        let mut floor = f64::INFINITY;
        let mut forbidden_lowest = f64::INFINITY;
        for flat in 0..(1usize << k) {
            let bits: Vec<usize> = (0..k).map(|p| (flat >> (k - 1 - p)) & 1).collect();
            let spins: Vec<f64> = bits.iter().map(|&b| gates::spin(b)).collect();
            let (energy, helper) = match gate.helper {
                None => (gates::energy(gate.name, &spins, None), None),
                Some(_) => {
                    let low = gates::energy(gate.name, &spins, Some(-1.0));
                    let high = gates::energy(gate.name, &spins, Some(1.0));
                    if high < low {
                        (high, Some(1))
                    } else {
                        (low, Some(0))
                    }
                }
            };
            floor = floor.min(energy);
            if !gates::allowed(gate.name, &bits) {
                forbidden_lowest = forbidden_lowest.min(energy);
            }
            report.push(format!("{}({})", gate.name, joins(&bits)), fmt(energy));
            rows.push((bits, energy, helper));
        }
        let gap = forbidden_lowest - floor;
        if gap < 2.0 {
            return Err(format!(
                "{}: a forbidden row sits only {gap} above the floor",
                gate.name
            ));
        }
        report.push(format!("{}.floor", gate.name), fmt(floor));
        let mut gate_map = serde_json::Map::new();
        gate_map.insert("bits".into(), json!(gate.bits));
        gate_map.insert(
            "rows".into(),
            Value::Array(
                rows.iter()
                    .map(|(bits, energy, helper)| {
                        json!({
                            "bits": bits,
                            "energy": number(*energy),
                            "helper": helper,
                            "allowed": gates::allowed(gate.name, bits),
                        })
                    })
                    .collect(),
            ),
        );
        gate_map.insert("floor".into(), number(floor));
        gate_map.insert("forbidden_lowest".into(), number(forbidden_lowest));
        gate_map.insert("helper".into(), json!(gate.helper));
        gates_value.insert(gate.name.to_string(), Value::Object(gate_map));
        floors.insert(gate.name.to_string(), number(floor));
    }

    let mut circuit_rows = Vec::new();
    let mut circuit_floor = f64::INFINITY;
    for flat in 0..16usize {
        let bits: Vec<usize> = (0..4).map(|p| (flat >> (3 - p)) & 1).collect();
        let s: Vec<f64> = bits.iter().map(|&b| gates::spin(b)).collect();
        let energy = gates::circuit(s[0], s[1], s[2], s[3]);
        circuit_floor = circuit_floor.min(energy);
        circuit_rows.push((bits, energy));
    }
    let at_floor: Vec<(Vec<usize>, f64)> = circuit_rows
        .iter()
        .filter(|(_, energy)| *energy == circuit_floor)
        .cloned()
        .collect();
    if at_floor.len() != 3 {
        return Err(format!(
            "the AND-NOT-SET1 circuit reaches its floor on {} states, not 3",
            at_floor.len()
        ));
    }
    for (bits, energy) in &circuit_rows {
        report.push(format!("circuit({})", joins(bits)), fmt(*energy));
    }
    report.push("circuit.floor", fmt(circuit_floor));
    report.push("circuit.states_at_floor", at_floor.len().to_string());

    Ok(json!({
        "parameters": {},
        "gates": Value::Object(gates_value),
        "circuit": {
            "bits": ["a", "b", "t", "out"],
            "rows": circuit_rows.iter().map(|(bits, energy)| json!({
                "bits": bits,
                "energy": number(*energy),
                "allowed": gates::circuit_allowed(bits),
            })).collect::<Vec<_>>(),
            "floor": number(circuit_floor),
            "states_at_floor": at_floor.len(),
            "inputs_at_floor": at_floor.iter()
                .map(|(bits, _)| json!([bits[0], bits[1]]))
                .collect::<Vec<_>>(),
        },
        "summary": {
            "floors": floors,
            "circuit_floor": number(circuit_floor),
            "circuit_states_at_floor": at_floor.len(),
        },
    }))
}

fn joins(bits: &[usize]) -> String {
    bits.iter()
        .map(|b| b.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

// ---------------------------------------------------------------------------
// Parts 3 and 4: --mode plan, count, sample
// ---------------------------------------------------------------------------

enum Problem {
    Magnet(net::Magnet),
    Queens(net::Queens),
}

fn load_problem(args: &Args) -> Result<Problem, String> {
    if let Some(path) = &args.input {
        let text = std::fs::read_to_string(path)
            .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
        let value: Value = serde_json::from_str(&text)
            .map_err(|error| format!("{} is not JSON: {error}", path.display()))?;
        Ok(Problem::Magnet(net::Magnet::from_bundle(&value)?))
    } else if let (Some(l), Some(p_af)) = (args.l, args.p_af) {
        Ok(Problem::Magnet(net::Magnet::open_lattice(l, p_af, args.seed)))
    } else if let Some(n) = args.n {
        Ok(Problem::Queens(net::Queens::build(n)))
    } else {
        Err("no problem".into())
    }
}

fn describe(args: &Args) -> Value {
    if let Some(path) = &args.input {
        json!({"source": "input", "input": path.display().to_string()})
    } else if let (Some(l), Some(p_af)) = (args.l, args.p_af) {
        json!({"source": "lattice", "l": l, "p_af": p_af})
    } else {
        json!({"source": "nqueens", "n": args.n})
    }
}

fn order_of(args: &Args) -> net::Order {
    args.order.unwrap_or(net::Order::Treesa)
}

/// A magnet's Boltzmann network does not depend on `beta` in its structure, so
/// `--mode plan`, which never looks at a weight, builds it at `beta = 1`.
const PLAN_BETA: f64 = 1.0;

fn magnet_network(magnet: &net::Magnet, beta: f64) -> (net::Network, Vec<net::TTensor>) {
    magnet.networks(beta)
}

fn run_plan(args: &Args, report: &mut Reporter) -> Result<Value, String> {
    let problem = load_problem(args)?;
    let order = order_of(args);
    let (plan, spins, bonds, fields, tensors) = match &problem {
        Problem::Magnet(magnet) => {
            let (network, _) = magnet_network(magnet, PLAN_BETA);
            let plan = net::plan(&network, order);
            (
                plan,
                magnet.nspins,
                magnet.bonds.len(),
                magnet.fields.iter().filter(|&&h| h != 0.0).count(),
                network.terms.len(),
            )
        }
        Problem::Queens(queens) => {
            let plan = net::plan(&queens.net, order);
            (
                plan,
                queens.n * queens.n,
                0,
                0,
                queens.net.terms.len(),
            )
        }
    };
    report.push("order", order.name());
    report.push("spins", spins.to_string());
    report.push("bonds", bonds.to_string());
    report.push("tensors", tensors.to_string());
    report.push("variables", plan.nvars.to_string());
    report.push("tc", format!("{:.6}", plan.tc));
    report.push("sc", format!("{:.6}", plan.sc));
    report.push("max_entries", plan.max_entries.to_string());
    let mut parameters = describe(args);
    parameters["order"] = json!(order.name());
    Ok(json!({
        "parameters": parameters,
        "network": {
            "spins": spins,
            "bonds": bonds,
            "fields": fields,
            "tensors": tensors,
            "variables": plan.nvars,
        },
        "results": {
            "order": order.name(),
            "plan_hash": net::tree_hash(&plan.tree),
            "tc": plan.tc,
            "sc": plan.sc,
            "rwc": plan.rwc,
            "max_entries": plan.max_entries,
            "planned_beta": PLAN_BETA,
        },
    }))
}

fn run_count(args: &Args, report: &mut Reporter) -> Result<Value, String> {
    let problem = load_problem(args)?;
    let order = order_of(args);
    let mut parameters = describe(args);
    parameters["order"] = json!(order.name());
    match problem {
        Problem::Queens(queens) => {
            let plan = net::plan(&queens.net, order);
            let stored = net::contract_sparse(&queens.net, &plan.tree);
            let boards = *stored[plan.tree.root].map.get(&0).unwrap_or(&0.0);
            let max_nonzero = stored.iter().map(|t| t.map.len()).max().unwrap_or(0);
            let total_nonzero: usize = stored.iter().map(|t| t.map.len()).sum();
            report.push("n", queens.n.to_string());
            report.push("tensors", queens.net.terms.len().to_string());
            report.push("variables", plan.nvars.to_string());
            report.push("tc", format!("{:.6}", plan.tc));
            report.push("sc", format!("{:.6}", plan.sc));
            report.push("boards", fmt(boards));
            Ok(json!({
                "parameters": parameters,
                "network": {
                    "n": queens.n,
                    "tensors": queens.net.terms.len(),
                    "variables": plan.nvars,
                    "source": "N-queens carry network, one bit per square",
                },
                "results": {
                    "boards": number(boards),
                    "plan_hash": net::tree_hash(&plan.tree),
                    "tc": plan.tc,
                    "sc": plan.sc,
                    "max_entries": plan.max_entries,
                    "largest_nonzero_intermediate": max_nonzero,
                    "total_nonzero_intermediate": total_nonzero,
                    "sparse": true,
                },
            }))
        }
        Problem::Magnet(magnet) => {
            let beta = args.beta.expect("--beta checked");
            let (network, tropical) = magnet.networks(beta);
            let plan = net::plan(&network, order);
            let (stored, log_z, log_scale) = net::contract_dense(&network, &plan.tree);
            let (e_min, g0) = net::contract_tropical(&tropical, &plan.tree, &network.dims);
            let log_excess = log_z - (g0 as f64).ln() + beta * e_min;
            // How many entries each intermediate of the planned tree holds; the lab
            // page replays the contraction from exactly these numbers.
            let node_entries: Vec<usize> = stored.iter().map(|tensor| tensor.data.len()).collect();
            let node_kinds: Vec<&str> = plan
                .tree
                .nodes
                .iter()
                .map(|node| match node {
                    net::Node::Leaf { .. } => "term",
                    net::Node::Pair { .. } => "contract",
                })
                .collect();
            let mut network_json = json!({
                "spins": magnet.nspins,
                "bonds": magnet.bonds.len(),
                "fields": magnet.fields.iter().filter(|&&h| h != 0.0).count(),
                "tensors": network.terms.len(),
                "variables": plan.nvars,
                "source": magnet.source,
            });
            // A lattice is generated from the seed, so its bonds are part of the
            // evidence; a bundle is already saved under runs/pred/.
            if args.input.is_none() {
                network_json["bond_list"] = json!(magnet
                    .bonds
                    .iter()
                    .map(|&(i, j, k)| json!([i, j, k]))
                    .collect::<Vec<_>>());
                network_json["field_list"] = json!(magnet.fields);
            }
            report.push("spins", magnet.nspins.to_string());
            report.push("tensors", network.terms.len().to_string());
            report.push("bonds", magnet.bonds.len().to_string());
            report.push("variables", plan.nvars.to_string());
            report.push("tc", format!("{:.6}", plan.tc));
            report.push("sc", format!("{:.6}", plan.sc));
            report.push("e_min", format!("{e_min:.12}"));
            report.push("g0", g0.to_string());
            report.push("log_z", format!("{log_z:.12}"));
            Ok(json!({
                "parameters": parameters,
                "network": network_json,
                "results": {
                    "beta": beta,
                    "plan_hash": net::tree_hash(&plan.tree),
                    "tc": plan.tc,
                    "sc": plan.sc,
                    "max_entries": plan.max_entries,
                    "e_min": number(e_min),
                    "g0": g0,
                    "log_z": log_z,
                    "log_scale_sum": log_scale,
                    "root_entry": stored[plan.tree.root].data[0],
                    "log_excess_above_ground": log_excess,
                    "node_entries": node_entries,
                    "node_kinds": node_kinds,
                },
            }))
        }
    }
}

fn run_sample(args: &Args, report: &mut Reporter) -> Result<Value, String> {
    let problem = load_problem(args)?;
    let order = order_of(args);
    let samples = args.samples.expect("--samples checked");
    let mut parameters = describe(args);
    parameters["order"] = json!(order.name());
    parameters["samples"] = json!(samples);
    match problem {
        Problem::Queens(queens) => {
            let plan = net::plan(&queens.net, order);
            let stored = net::contract_sparse(&queens.net, &plan.tree);
            let boards = *stored[plan.tree.root].map.get(&0).unwrap_or(&0.0);
            let mut rng = rng::Rng::new(args.seed, rng::STREAM_SAMPLE);
            let mut drawn: Vec<Vec<usize>> = Vec::with_capacity(samples);
            let mut frequencies: BTreeMap<Vec<usize>, usize> = BTreeMap::new();
            for _ in 0..samples {
                let values =
                    net::sample_sparse(&plan.tree, &stored, queens.net.dims.len(), &mut rng);
                let board = queens
                    .board(&values)
                    .ok_or("a sampled assignment is not a board with one queen per row")?;
                if !net::valid_queens(&board) {
                    return Err(format!("a sampled board attacks itself: {board:?}"));
                }
                *frequencies.entry(board.clone()).or_insert(0) += 1;
                drawn.push(board);
            }
            let distinct = frequencies.len();
            report.push("n", queens.n.to_string());
            report.push("boards_total", fmt(boards));
            report.push("samples", samples.to_string());
            report.push("distinct", distinct.to_string());
            Ok(json!({
                "parameters": parameters,
                "network": {
                    "n": queens.n,
                    "tensors": queens.net.terms.len(),
                    "variables": plan.nvars,
                },
                "results": {
                    "boards_total": number(boards),
                    "plan_hash": net::tree_hash(&plan.tree),
                    "tc": plan.tc,
                    "sc": plan.sc,
                    "max_entries": plan.max_entries,
                    "samples": samples,
                    "distinct": distinct,
                    "frequencies": frequencies.iter().map(|(board, count)| json!({
                        "board": board,
                        "count": count,
                    })).collect::<Vec<_>>(),
                },
                "boards": drawn,
            }))
        }
        Problem::Magnet(magnet) => {
            let beta = args.beta.expect("--beta checked");
            let (network, tropical) = magnet.networks(beta);
            let plan = net::plan(&network, order);
            let (stored, log_z, _) = net::contract_dense(&network, &plan.tree);
            let (e_min, g0) = net::contract_tropical(&tropical, &plan.tree, &network.dims);
            let mut rng = rng::Rng::new(args.seed, rng::STREAM_SAMPLE);
            let mut energies = Vec::with_capacity(samples);
            let mut bits_all = Vec::with_capacity(samples);
            let mut histogram: BTreeMap<i64, usize> = BTreeMap::new();
            let mut lowest = f64::INFINITY;
            let mut lowest_bits: Vec<usize> = Vec::new();
            let mut at_ground = 0usize;
            for _ in 0..samples {
                let values = net::sample_dense(&plan.tree, &stored, network.dims.len(), &mut rng);
                let energy = magnet.energy(&values);
                if energy < lowest {
                    lowest = energy;
                    lowest_bits = values.clone();
                }
                if (energy - e_min).abs() < 1.0e-9 {
                    at_ground += 1;
                }
                let key = (energy * 1.0e6).round() as i64;
                *histogram.entry(key).or_insert(0) += 1;
                energies.push(energy);
                bits_all.push(values.iter().map(|b| b.to_string()).collect::<String>());
            }
            let p0 = ((g0 as f64).ln() - beta * e_min - log_z).exp();
            let predicted = samples as f64 * p0;
            let spread = (samples as f64 * p0 * (1.0 - p0)).sqrt();
            let mut network_json = json!({
                "spins": magnet.nspins,
                "bonds": magnet.bonds.len(),
                "tensors": network.terms.len(),
                "variables": plan.nvars,
                "source": magnet.source,
            });
            if args.input.is_none() {
                network_json["bond_list"] = json!(magnet
                    .bonds
                    .iter()
                    .map(|&(i, j, k)| json!([i, j, k]))
                    .collect::<Vec<_>>());
                network_json["field_list"] = json!(magnet.fields);
            }
            report.push("spins", magnet.nspins.to_string());
            report.push("samples", samples.to_string());
            report.push("e_min", format!("{e_min:.12}"));
            report.push("g0", g0.to_string());
            report.push("log_z", format!("{log_z:.12}"));
            report.push("p0", format!("{p0:.9}"));
            report.push("predicted_ground_draws", format!("{predicted:.6}"));
            report.push("ground_draws", at_ground.to_string());
            report.push("binomial_spread", format!("{spread:.6}"));
            Ok(json!({
                "parameters": parameters,
                "network": network_json,
                "results": {
                    "beta": beta,
                    "plan_hash": net::tree_hash(&plan.tree),
                    "tc": plan.tc,
                    "sc": plan.sc,
                    "max_entries": plan.max_entries,
                    "e_min": number(e_min),
                    "g0": g0,
                    "log_z": log_z,
                    "p0": p0,
                    "samples": samples,
                    "predicted_ground_draws": predicted,
                    "binomial_spread": spread,
                    "ground_draws": at_ground,
                    "lowest_sample": number(lowest),
                    "histogram": histogram.iter().map(|(key, count)| {
                        let energy = *key as f64 / 1.0e6;
                        json!({"energy": energy, "count": count})
                    }).collect::<Vec<_>>(),
                },
                "sample_energies": energies,
                "sample_bits": bits_all,
                "lowest_energy_bits": lowest_bits.iter().map(|b| b.to_string()).collect::<String>(),
            }))
        }
    }
}

