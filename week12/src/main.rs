//! `branchopt` — spin glasses as mixed-integer linear programs, exact
//! Branch-and-Bound with a cheap and a strong branching rule, and a learned
//! rule that imitates the strong one.
//!
//! Modes (see `branchopt.design.toml`):
//!   --mode gap   LP bound vs exact MILP optimum, integrality gap, rounding excess
//!   --mode bnb   exact Branch-and-Bound, `--policy most-frac | strong`
//!   --mode eval  most-frac, strong, neural and root-neural on unseen instances

mod bnb;
mod instance;
mod milp;
mod net;
mod rng;

use bnb::{branch_and_bound, Policy, SearchConfig};
use instance::Instance;
use milp::Lp;
use serde_json::{json, Value};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

struct Args {
    mode: String,
    spins: usize,
    edges: usize,
    count: usize,
    seed: u64,
    out: PathBuf,
    policy: Option<String>,
    model: Option<PathBuf>,
    export_labels: bool,
}

impl Default for Args {
    fn default() -> Self {
        Args {
            mode: "gap".to_string(),
            spins: 22,
            edges: 52,
            count: 16,
            seed: 2026,
            out: PathBuf::from("artifacts/gap"),
            policy: None,
            model: None,
            export_labels: false,
        }
    }
}

fn parse_args() -> Args {
    let mut a = Args::default();
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        let key = argv[i].as_str();
        let next = |i: &mut usize| -> String {
            *i += 1;
            argv.get(*i).cloned().unwrap_or_else(|| {
                panic!("missing value for {}", argv[*i - 1]);
            })
        };
        match key {
            "--mode" => a.mode = next(&mut i),
            "--spins" => a.spins = next(&mut i).parse().expect("spins"),
            "--edges" => a.edges = next(&mut i).parse().expect("edges"),
            "--count" => a.count = next(&mut i).parse().expect("count"),
            "--seed" => a.seed = next(&mut i).parse().expect("seed"),
            "--out" => a.out = PathBuf::from(next(&mut i)),
            "--policy" => a.policy = Some(next(&mut i)),
            "--model" => a.model = Some(PathBuf::from(next(&mut i))),
            "--export-labels" => a.export_labels = true,
            other => panic!("unknown argument {other}"),
        }
        i += 1;
    }
    a
}

struct Report {
    rows: Vec<(String, String, String, f64)>,
    start: Instant,
}

impl Report {
    fn new() -> Self {
        Report {
            rows: Vec::new(),
            start: Instant::now(),
        }
    }

    fn row(&mut self, mode: &str, metric: &str, value: impl ToString) {
        let ms = self.start.elapsed().as_secs_f64() * 1e3;
        self.rows
            .push((mode.to_string(), metric.to_string(), value.to_string(), ms));
    }

    fn num(&mut self, mode: &str, metric: &str, v: f64) {
        self.row(mode, metric, format!("{v:.6}"));
    }

    fn sci(&mut self, mode: &str, metric: &str, v: f64) {
        self.row(mode, metric, format!("{v:.3e}"));
    }

    fn print(&self) {
        println!("mode\tmetric\tvalue\truntime_ms");
        for (m, k, v, ms) in &self.rows {
            println!("{m}\t{k}\t{v}\t{ms:.1}");
        }
    }

    fn write_json(&self, path: &Path, payload: Value) {
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir).expect("create out dir");
        }
        let text = serde_json::to_string_pretty(&payload).expect("serialize");
        fs::write(path, text).expect("write run.json");
    }
}

/// The exact ground state from HiGHS's own mixed-integer solver.
fn oracle(inst: &Instance) -> (f64, Vec<f64>) {
    let mut lp = Lp::new(inst, true);
    let lo = vec![0.0; inst.n];
    let hi = vec![1.0; inst.n];
    match lp.solve(&lo, &hi) {
        Some(s) => (s.energy, s.x),
        None => (f64::NAN, vec![f64::NAN; inst.n]),
    }
}

/// The exact ground-state energy from HiGHS's own mixed-integer solver.
fn oracle_energy(inst: &Instance) -> f64 {
    oracle(inst).0
}

fn main() {
    let args = parse_args();
    match args.mode.as_str() {
        "gap" => run_gap(&args),
        "bnb" => run_bnb(&args),
        "eval" => run_eval(&args),
        other => panic!("unknown mode {other}"),
    }
}

// ---------------------------------------------------------------- mode: gap

fn run_gap(args: &Args) {
    let mut rep = Report::new();
    fs::create_dir_all(&args.out).expect("create out dir");

    // The opening four-spin square.
    let square = instance::square();
    let mut lp = Lp::new(&square, false);
    let n = square.n;
    let sol = lp
        .solve(&vec![0.0; n], &vec![1.0; n])
        .expect("square LP is feasible");
    let square_lp = sol.energy;
    let square_bits = sol.x.clone();
    let (square_exact, square_exact_bits) = oracle(&square);
    let rounded = square.nearest_rounding(&square_bits);
    let square_rounded = square.energy_from_bits(&rounded);

    rep.num("gap", "square_lp_bound", square_lp);
    rep.num("gap", "square_exact_energy", square_exact);
    rep.num("gap", "square_rounded_energy", square_rounded);
    rep.num("gap", "square_rounded_excess", square_rounded - square_exact);
    rep.row(
        "gap",
        "square_lp_spins",
        square_bits
            .iter()
            .map(|v| format!("{v:.6}"))
            .collect::<Vec<_>>()
            .join(","),
    );
    rep.row(
        "gap",
        "square_exact_spins",
        square_exact_bits
            .iter()
            .map(|v| format!("{}", *v as i64))
            .collect::<Vec<_>>()
            .join(","),
    );
    rep.row(
        "gap",
        "square_rounded_spins",
        rounded
            .iter()
            .map(|v| format!("{}", *v as i64))
            .collect::<Vec<_>>()
            .join(","),
    );

    // The two-food diet LP.
    let (d1, d2, dcost) = milp::diet();
    rep.num("gap", "diet_cost", dcost);
    rep.num("gap", "diet_x1", d1);
    rep.num("gap", "diet_x2", d2);

    // The seeded benchmark.
    let mut instances = Vec::new();
    let mut lps = Vec::new();
    let mut rels = Vec::new();
    let mut exacts = Vec::new();
    let mut gaps = Vec::new();
    let mut fracs = Vec::new();
    let mut rounds = Vec::new();
    let mut excesses = Vec::new();
    let t_inst = Instant::now();

    for id in 0..args.count {
        let inst = instance::generate(args.seed, id, args.spins, args.edges);
        let mut lp = Lp::new(&inst, false);
        let sol = lp
            .solve(&vec![0.0; inst.n], &vec![1.0; inst.n])
            .expect("relaxation is feasible");
        let lp_bound = sol.energy;
        let frac = sol
            .x
            .iter()
            .filter(|&&x| x > 1e-6 && x < 1.0 - 1e-6)
            .count();
        let exact = oracle_energy(&inst);
        let rounded = inst.energy_from_bits(&inst.nearest_rounding(&sol.x));
        let excess = rounded - exact;
        instances.push(json!({
            "id": inst.id,
            "n": inst.n,
            "m": inst.m,
            "edges": inst.edges,
            "couplings": inst.j,
            "fields": inst.h,
            "lp_bound": lp_bound,
            "exact_energy": exact,
            "integrality_gap": exact - lp_bound,
            "relative_gap": if exact.abs() > 0.0 { (exact - lp_bound) / exact.abs() } else { 0.0 },
            "fractional_spins": frac,
            "rounded_energy": rounded,
            "rounding_excess": excess,
            "lp_spins": sol.x.clone(),
            "lp_bonds": sol.z.clone(),
        }));
        lps.push(lp_bound);
        rels.push(if exact.abs() > 0.0 {
            (exact - lp_bound) / exact.abs()
        } else {
            0.0
        });
        exacts.push(exact);
        gaps.push(exact - lp_bound);
        fracs.push(frac as f64);
        rounds.push(rounded);
        excesses.push(excess);
    }
    let inst_ms = t_inst.elapsed().as_secs_f64() * 1e3;

    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let suboptimal = excesses.iter().filter(|&&e| e > 1e-9).count();
    let (worst_i, worst_e) = excesses
        .iter()
        .enumerate()
        .fold((0usize, f64::NEG_INFINITY), |acc, (i, &e)| {
            if e > acc.1 {
                (i, e)
            } else {
                acc
            }
        });

    rep.num("gap", "instances", args.count as f64);
    rep.num("gap", "mean_lp_bound", mean(&lps));
    rep.num("gap", "mean_exact_energy", mean(&exacts));
    rep.num("gap", "mean_integrality_gap", mean(&gaps));
    rep.num("gap", "mean_relative_gap", mean(&rels));
    rep.num("gap", "mean_fractional_spins", mean(&fracs));
    rep.num("gap", "mean_rounded_energy", mean(&rounds));
    rep.num("gap", "mean_rounding_excess", mean(&excesses));
    rep.row("gap", "suboptimal_roundings", suboptimal);
    rep.row("gap", "largest_rounding_excess_instance", worst_i);
    rep.num("gap", "largest_rounding_excess", worst_e);
    rep.num("gap", "instance_benchmark_ms", inst_ms);

    let payload = json!({
        "schema": "branchopt-gap-v1",
        "mode": "gap",
        "seed": args.seed,
        "parameters": {
            "spins": args.spins,
            "edges": args.edges,
            "count": args.count,
        },
        "command": format!(
            "branchopt --mode gap --spins {} --edges {} --count {} --seed {} --out {}",
            args.spins, args.edges, args.count, args.seed, args.out.display()),
        "solver": { "name": "HiGHS via the highs Rust crate", "presolve": false, "threads": 1 },
        "square": {
            "couplings": square.j,
            "edges": square.edges,
            "lp_bound": square_lp,
            "lp_spins": square_bits,
            "exact_energy": square_exact,
            "exact_spins": square_exact_bits,
            "rounded_spins": rounded,
            "rounded_energy": square_rounded,
            "rounded_excess": square_rounded - square_exact,
        },
        "diet": { "x1": d1, "x2": d2, "cost": dcost },
        "summary": {
            "mean_lp_bound": mean(&lps),
            "mean_exact_energy": mean(&exacts),
            "mean_integrality_gap": mean(&gaps),
            "mean_relative_gap": mean(&rels),
            "mean_fractional_spins": mean(&fracs),
            "mean_rounded_energy": mean(&rounds),
            "mean_rounding_excess": mean(&excesses),
            "suboptimal_roundings": suboptimal,
            "largest_rounding_excess": worst_e,
            "largest_rounding_excess_instance": worst_i,
            "runtime_ms": inst_ms,
        },
        "instances": instances,
    });
    rep.write_json(&args.out.join("run.json"), payload);
    rep.print();
}

// ---------------------------------------------------------------- mode: bnb

fn run_bnb(args: &Args) {
    let mut rep = Report::new();
    fs::create_dir_all(&args.out).expect("create out dir");
    let policy = Policy::parse(args.policy.as_deref().unwrap_or("most-frac"))
        .expect("--policy must be most-frac, strong, neural or root-neural");
    if policy == Policy::Neural || policy == Policy::RootNeural {
        panic!("neural policies need --mode eval with --model");
    }

    // The five-node search of the frustrated tetrahedron, always traced with the
    // cheap rule so the printed table is identical in every run.
    let tetra = instance::tetrahedron();
    let tetra_out = branch_and_bound(
        &tetra,
        &SearchConfig {
            policy: Policy::MostFrac,
            net: None,
            export_labels: false,
            max_labels_per_instance: 0,
            trace: true,
            instance_id: 0,
        },
    );
    rep.num("bnb", "tetrahedron_energy", tetra_out.energy);
    rep.row("bnb", "tetrahedron_nodes", tetra_out.nodes);
    rep.row("bnb", "tetrahedron_lp_solves", tetra_out.lp_solves);
    for t in &tetra_out.trace {
        rep.row(
            "bnb",
            &format!("tetrahedron_node{}", t.node),
            format!(
                "{} | {} | {} | bound {:.6} | free {}",
                t.branch,
                t.status,
                if t.bound.is_finite() {
                    "lp"
                } else {
                    "infeasible"
                },
                t.bound,
                t.free
            ),
        );
    }

    let mut per_instance = Vec::new();
    let mut nodes = Vec::new();
    let mut solves = Vec::new();
    let mut times = Vec::new();
    let mut errors: Vec<f64> = Vec::new();
    let mut labels: Vec<bnb::NodeRecord> = Vec::new();
    let mut energies = Vec::new();

    for id in 0..args.count {
        let inst = instance::generate(args.seed, id, args.spins, args.edges);
        let oracle = oracle_energy(&inst);
        let out = branch_and_bound(
            &inst,
            &SearchConfig {
                policy,
                net: None,
                export_labels: args.export_labels,
                max_labels_per_instance: 80,
                trace: false,
                instance_id: id,
            },
        );
        let err = (out.energy - oracle).abs();
        errors.push(err);
        nodes.push(out.nodes as f64);
        solves.push(out.lp_solves as f64);
        times.push(out.total_ms);
        energies.push(out.energy);
        per_instance.push(json!({
            "id": id,
            "n": inst.n,
            "m": inst.m,
            "energy": out.energy,
            "highs_energy": oracle,
            "error": err,
            "nodes": out.nodes,
            "lp_solves": out.lp_solves,
            "elapsed_ms": out.total_ms,
            "lp_ms": out.lp_ms,
            "first_splits": out.first_splits,
        }));
        labels.extend(out.labels);
    }

    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let max_err = errors.iter().copied().fold(0.0f64, f64::max);
    let pname = policy.name();

    rep.row("bnb", &format!("{pname}_instances"), args.count);
    rep.num("bnb", &format!("{pname}_mean_nodes"), mean(&nodes));
    rep.num("bnb", &format!("{pname}_mean_lp_solves"), mean(&solves));
    rep.num("bnb", &format!("{pname}_mean_solve_ms"), mean(&times));
    rep.sci("bnb", &format!("{pname}_largest_error"), max_err);
    rep.num(
        "bnb",
        &format!("{pname}_lp_solves_per_node"),
        mean(&solves) / mean(&nodes),
    );

    if args.export_labels {
        let train: Vec<&bnb::NodeRecord> = labels.iter().filter(|l| l.instance < 36).collect();
        let val: Vec<&bnb::NodeRecord> = labels.iter().filter(|l| l.instance >= 36).collect();
        rep.row("bnb", "label_nodes", labels.len());
        rep.row("bnb", "label_train_nodes", train.len());
        rep.row("bnb", "label_validation_nodes", val.len());
        let mut train_inst: Vec<usize> = train.iter().map(|l| l.instance).collect();
        train_inst.sort_unstable();
        train_inst.dedup();
        let mut val_inst: Vec<usize> = val.iter().map(|l| l.instance).collect();
        val_inst.sort_unstable();
        val_inst.dedup();
        rep.row("bnb", "label_train_instances", train_inst.len());
        rep.row("bnb", "label_validation_instances", val_inst.len());

        let dataset = json!({
            "schema": "branchopt-labels-v1",
            "seed": args.seed,
            "spins": args.spins,
            "edges": args.edges,
            "count": args.count,
            "train_instances": train_inst,
            "validation_instances": val_inst,
            "policy": "strong",
            "nodes": labels.iter().map(|l| json!({
                "instance": l.instance,
                "node": l.node,
                "depth": l.depth,
                "branch": l.branch,
                "z_node": l.z_node,
                "z_best": l.z_best,
                "best": l.best,
                "candidates": l.candidates.iter().map(|c| json!({
                    "spin": c.spin,
                    "gain0": c.gain0,
                    "gain1": c.gain1,
                    "score": c.score,
                    "pruned_children": c.pruned_children,
                    "features": c.features,
                })).collect::<Vec<_>>(),
            })).collect::<Vec<_>>(),
        });
        fs::write(
            args.out.join("dataset.json"),
            serde_json::to_string(&dataset).expect("serialize labels"),
        )
        .expect("write dataset.json");
    }

    let payload = json!({
        "schema": "branchopt-bnb-v1",
        "mode": "bnb",
        "seed": args.seed,
        "policy": pname,
        "parameters": {
            "spins": args.spins,
            "edges": args.edges,
            "count": args.count,
            "export_labels": args.export_labels,
        },
        "command": format!(
            "branchopt --mode bnb --spins {} --edges {} --count {} --policy {} --seed {} --out {}{}",
            args.spins, args.edges, args.count, pname, args.seed, args.out.display(),
            if args.export_labels { " --export-labels" } else { "" }),
        "solver": { "name": "HiGHS via the highs Rust crate", "presolve": false, "threads": 1 },
        "prune_epsilon": bnb::PRUNE_EPS,
        "tetrahedron": {
            "energy": tetra_out.energy,
            "nodes": tetra_out.nodes,
            "lp_solves": tetra_out.lp_solves,
            "trace": tetra_out.trace.iter().map(|t| json!({
                "node": t.node,
                "depth": t.depth,
                "branch": t.branch,
                "status": t.status,
                "bound": if t.bound.is_finite() { Some(t.bound) } else { None },
                "free": t.free,
                "spin": t.spin,
            })).collect::<Vec<_>>(),
        },
        "summary": {
            "mean_nodes": mean(&nodes),
            "mean_lp_solves": mean(&solves),
            "mean_elapsed_ms": mean(&times),
            "largest_error": max_err,
            "min_energy": energies.iter().copied().fold(f64::INFINITY, f64::min),
            "max_energy": energies.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        },
        "instances": per_instance,
    });
    rep.write_json(&args.out.join("run.json"), payload);
    rep.print();
}

// --------------------------------------------------------------- mode: eval

struct PolicyRun {
    name: &'static str,
    energy: Vec<f64>,
    nodes: Vec<f64>,
    solves: Vec<f64>,
    total_ms: Vec<f64>,
    lp_ms: Vec<f64>,
    infer_ms: Vec<f64>,
    first_splits: Vec<Vec<usize>>,
}

fn run_eval(args: &Args) {
    let mut rep = Report::new();
    fs::create_dir_all(&args.out).expect("create out dir");
    let model_path = args.model.as_ref().expect("--model is required in eval mode");
    let text = fs::read_to_string(model_path).expect("read model.json");
    let net = net::Net::from_json(&text).expect("parse model.json");

    let (probe_p, probe_l) = net.probe_differences();
    rep.row("eval", "probes", net.probes.len());
    rep.row(
        "eval",
        "probe_nodes",
        net.probes
            .iter()
            .map(|p| format!("{}:{}", p.instance, p.node))
            .collect::<Vec<_>>()
            .join(","),
    );
    rep.sci("eval", "probe_probability_max_diff", probe_p);
    rep.sci("eval", "probe_logit_max_diff", probe_l);

    let oracle: Vec<f64> = (0..args.count)
        .map(|id| oracle_energy(&instance::generate(args.seed, id, args.spins, args.edges)))
        .collect();

    let policies = [Policy::MostFrac, Policy::Strong, Policy::Neural, Policy::RootNeural];
    let mut runs: Vec<PolicyRun> = Vec::new();

    for policy in policies {
        let mut r = PolicyRun {
            name: policy.name(),
            energy: Vec::new(),
            nodes: Vec::new(),
            solves: Vec::new(),
            total_ms: Vec::new(),
            lp_ms: Vec::new(),
            infer_ms: Vec::new(),
            first_splits: Vec::new(),
        };
        for id in 0..args.count {
            let inst = instance::generate(args.seed, id, args.spins, args.edges);
            let out = branch_and_bound(
                &inst,
                &SearchConfig {
                    policy,
                    net: Some(&net),
                    export_labels: false,
                    max_labels_per_instance: 0,
                    trace: false,
                    instance_id: id,
                },
            );
            r.energy.push(out.energy);
            r.nodes.push(out.nodes as f64);
            r.solves.push(out.lp_solves as f64);
            r.total_ms.push(out.total_ms);
            r.lp_ms.push(out.lp_ms);
            r.infer_ms.push(out.infer_ms);
            r.first_splits.push(out.first_splits);
        }
        runs.push(r);
    }

    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let mut max_err = 0.0f64;
    for r in &runs {
        for (i, &e) in r.energy.iter().enumerate() {
            max_err = max_err.max((e - oracle[i]).abs());
        }
        rep.num("eval", &format!("{}_mean_nodes", r.name), mean(&r.nodes));
        rep.num("eval", &format!("{}_mean_total_ms", r.name), mean(&r.total_ms));
        rep.num("eval", &format!("{}_mean_lp_ms", r.name), mean(&r.lp_ms));
        rep.num("eval", &format!("{}_mean_infer_ms", r.name), mean(&r.infer_ms));
        rep.num("eval", &format!("{}_mean_lp_solves", r.name), mean(&r.solves));
    }
    rep.sci("eval", "largest_highs_error", max_err);

    let most = runs.iter().find(|r| r.name == "most-frac").unwrap();
    let neural = runs.iter().find(|r| r.name == "neural").unwrap();
    let lp_equal_most = runs
        .iter()
        .filter(|r| r.name != "strong")
        .all(|r| r.solves.iter().zip(r.nodes.iter()).all(|(s, n)| (s - n).abs() < 0.5));
    rep.row("eval", "lp_solves_equal_nodes_for_non_strong", lp_equal_most);

    // Where does the network save the most nodes over most-frac?
    let mut best_i = 0usize;
    let mut best_saving = f64::NEG_INFINITY;
    for i in 0..args.count {
        let saving = most.nodes[i] - neural.nodes[i];
        if saving > best_saving {
            best_saving = saving;
            best_i = i;
        }
    }
    rep.row("eval", "largest_neural_saving_instance", best_i);
    rep.num("eval", "largest_neural_saving_nodes", best_saving);
    rep.row(
        "eval",
        "most_frac_first_splits",
        most.first_splits[best_i]
            .iter()
            .take(2)
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join(","),
    );
    rep.row(
        "eval",
        "neural_first_splits",
        neural.first_splits[best_i]
            .iter()
            .take(2)
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join(","),
    );
    rep.num("eval", "most_frac_instance_nodes", most.nodes[best_i]);
    rep.num("eval", "neural_instance_nodes", neural.nodes[best_i]);
    rep.num("eval", "most_frac_instance_ms", most.total_ms[best_i]);
    rep.num("eval", "neural_instance_ms", neural.total_ms[best_i]);

    let instances: Vec<Value> = (0..args.count)
        .map(|i| {
            json!({
                "id": i,
                "highs_energy": oracle[i],
                "policies": runs.iter().map(|r| (r.name.to_string(), json!({
                    "energy": r.energy[i],
                    "error": (r.energy[i] - oracle[i]).abs(),
                    "nodes": r.nodes[i],
                    "lp_solves": r.solves[i],
                    "total_ms": r.total_ms[i],
                    "lp_ms": r.lp_ms[i],
                    "infer_ms": r.infer_ms[i],
                    "first_splits": r.first_splits[i],
                }))).collect::<serde_json::Map<String, Value>>(),
            })
        })
        .collect();

    let payload = json!({
        "schema": "branchopt-eval-v1",
        "mode": "eval",
        "seed": args.seed,
        "parameters": {
            "spins": args.spins,
            "edges": args.edges,
            "count": args.count,
            "model": model_path.display().to_string(),
        },
        "command": format!(
            "branchopt --mode eval --spins {} --edges {} --count {} --seed {} --model {} --out {}",
            args.spins, args.edges, args.count, args.seed, model_path.display(), args.out.display()),
        "solver": { "name": "HiGHS via the highs Rust crate", "presolve": false, "threads": 1 },
        "model": {
            "best_epoch": net.best_epoch,
            "best_val_ce": net.best_val_ce,
            "probe_probability_max_diff": probe_p,
            "probe_logit_max_diff": probe_l,
        },
        "summary": {
            "largest_highs_error": max_err,
            "lp_solves_equal_nodes_for_non_strong": lp_equal_most,
            "largest_neural_saving_instance": best_i,
            "policies": runs.iter().map(|r| (r.name.to_string(), json!({
                "mean_nodes": mean(&r.nodes),
                "mean_total_ms": mean(&r.total_ms),
                "mean_lp_ms": mean(&r.lp_ms),
                "mean_infer_ms": mean(&r.infer_ms),
                "mean_lp_solves": mean(&r.solves),
            }))).collect::<serde_json::Map<String, Value>>(),
        },
        "instances": instances,
    });
    rep.write_json(&args.out.join("run.json"), payload);
    rep.print();
}
