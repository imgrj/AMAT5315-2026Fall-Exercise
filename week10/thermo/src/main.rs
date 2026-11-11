//! `thermo`: truncated SVD evidence, boundary-MPS and TRG partition functions of
//! the zero-field square-lattice Ising model.
//!
//! See `../thermo.design.toml` for the argument and output contract.

mod boundary;
mod linalg;
mod model;
mod rng;
mod trg;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde_json::{json, Value};

use crate::model::Ising;

const USAGE: &str = "usage:
  thermo --method boundary --l <side> --k <K> --chi <chi> --out <dir>
  thermo --method trg --l <side> --k <K> --chi <chi> --out <dir>
  thermo --method svd --l <side> --seed <seed> --ranks <r1,r2,...> --out <dir>";

struct Arguments {
    values: BTreeMap<String, String>,
}

impl Arguments {
    fn parse(raw: &[String]) -> Result<Self, String> {
        let mut values = BTreeMap::new();
        let mut index = 0;
        while index < raw.len() {
            let token = &raw[index];
            let Some(name) = token.strip_prefix("--") else {
                return Err(format!("unexpected argument {token:?}\n{USAGE}"));
            };
            if !matches!(name, "method" | "l" | "k" | "chi" | "seed" | "ranks" | "out") {
                return Err(format!("unknown flag --{name}\n{USAGE}"));
            }
            let value = raw
                .get(index + 1)
                .ok_or_else(|| format!("--{name} needs a value\n{USAGE}"))?
                .clone();
            if values.insert(name.to_string(), value).is_some() {
                return Err(format!("--{name} given twice"));
            }
            index += 2;
        }
        Ok(Arguments { values })
    }

    fn required(&self, name: &str) -> Result<&str, String> {
        self.values
            .get(name)
            .map(String::as_str)
            .ok_or_else(|| format!("missing required --{name}\n{USAGE}"))
    }

    fn number(&self, name: &str) -> Result<f64, String> {
        let raw = self.required(name)?;
        raw.parse::<f64>().map_err(|_| format!("--{name} must be a number, got {raw:?}"))
    }

    fn integer(&self, name: &str) -> Result<usize, String> {
        let raw = self.required(name)?;
        raw.parse::<usize>().map_err(|_| format!("--{name} must be an integer, got {raw:?}"))
    }

    fn reject(&self, names: &[&str], method: &str) -> Result<(), String> {
        for name in names {
            if self.values.contains_key(*name) {
                return Err(format!("--{name} does not apply to --method {method}\n{USAGE}"));
            }
        }
        Ok(())
    }
}

fn write_json(path: &Path, value: &Value) -> Result<(), String> {
    let text = serde_json::to_string_pretty(value).map_err(|error| error.to_string())?;
    std::fs::write(path, format!("{text}\n")).map_err(|error| format!("{}: {error}", path.display()))
}

fn prepare(path: &Path) -> Result<(), String> {
    std::fs::create_dir_all(path).map_err(|error| format!("{}: {error}", path.display()))
}

fn model_json(model: &Ising) -> Value {
    json!({
        "hamiltonian": "H=-sum_{(i,j) in E} s_i s_j",
        "spin_values": [1, -1],
        "coupling_j": 1.0,
        "boltzmann_constant": 1.0,
        "field": 0.0,
        "k": model.k,
        "temperature": if model.k > 0.0 { json!(1.0 / model.k) } else { Value::Null },
        "boltzmann_weight": "exp(K s_i s_j)",
        "w_tensor": [[model.w[0], model.w[1]], [model.w[2], model.w[3]]],
        "w_rows": "s = +1 then s = -1; columns are the bond direction a",
        "b_vector": [model.b[0], model.b[1]],
        "site_tensor": model.site,
        "site_tensor_axes": "[r,u,l,d] and A[r,u,l,d]=sum_s W[s,r]W[s,u]W[s,l]W[s,d]",
    })
}

fn main() {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.is_empty() || raw.iter().any(|token| token == "--help" || token == "-h") {
        println!("{USAGE}");
        return;
    }
    if let Err(message) = dispatch(&raw) {
        eprintln!("thermo: {message}");
        std::process::exit(2);
    }
}

fn dispatch(raw: &[String]) -> Result<(), String> {
    let arguments = Arguments::parse(raw)?;
    let method = arguments.required("method")?.to_string();
    let out = PathBuf::from(arguments.required("out")?);
    match method.as_str() {
        "boundary" => run_boundary(&arguments, &out),
        "trg" => run_trg(&arguments, &out),
        "svd" => run_svd(&arguments, &out),
        other => Err(format!("unknown --method {other:?}; use boundary, trg or svd\n{USAGE}")),
    }
}

fn chemistry(arguments: &Arguments) -> Result<(usize, f64, usize), String> {
    let length = arguments.integer("l")?;
    let k = arguments.number("k")?;
    let chi = arguments.integer("chi")?;
    if length < 2 {
        return Err("--l must be at least 2".into());
    }
    if !(k.is_finite() && k >= 0.0) {
        return Err("--k must be finite and non-negative".into());
    }
    if chi < 1 {
        return Err("--chi must be at least 1".into());
    }
    arguments.reject(&["seed", "ranks"], "boundary/trg")?;
    Ok((length, k, chi))
}

fn run_boundary(arguments: &Arguments, out: &Path) -> Result<(), String> {
    let (length, k, chi) = chemistry(arguments)?;
    prepare(out)?;
    let model = Ising::new(k);
    let start = Instant::now();
    let result = boundary::run(&model, length, chi);
    let solver_time = start.elapsed().as_secs_f64();
    let cores_file = out.join("cores.json");
    let dims: Vec<[usize; 3]> = result.cores.iter().map(|core| [core.l, core.p, core.r]).collect();
    let data: Vec<f64> = result.cores.iter().flat_map(|core| core.data.iter().copied()).collect();
    write_json(
        &cores_file,
        &json!({
            "dims": dims,
            "data": data,
            "bottom_vector": [result.bottom_value],
            "scales": result.rows.iter().map(|row| row.scale).collect::<Vec<_>>(),
            "ell": result.ell,
            "bottom_log": result.bottom_log,
            "layout": "core data row-major in [left, physical, right] with the physical leg the down leg",
        }),
    )?;
    let rows: Vec<Value> = result
        .rows
        .iter()
        .map(|row| {
            json!({
                "row": row.row + 1,
                "scale": row.scale,
                "log_scale": row.log_scale,
                "ell": row.ell,
                "bond_dims_before": row.bond_dims_before,
                "bond_dims_after": row.bond_dims_after,
                "discarded_weight": row.discarded,
                "max_discarded_weight": row.max_discarded,
                "discarded_norm": row.relative_discarded,
                "retained_singular_values": row.retained_singular_values,
                "contraction_entries": row.contraction_entries,
                "approximation": row.approximation,
            })
        })
        .collect();
    let report = json!({
        "tool": "thermo 0.1.0",
        "method": "boundary",
        "geometry": {
            "boundary": "open",
            "l": length,
            "n": length * length,
            "bonds": 2 * length * (length - 1),
            "boundary_condition": "open edges; no bond crosses the border",
        },
        "settings": {"method": "boundary", "l": length, "k": k, "chi": chi, "storage": "left-to-right truncated SVD"},
        "model": model_json(&model),
        "log_z": result.log_z,
        "g": result.g,
        "normalisation": {"ell": result.ell, "bottom_log": result.bottom_log, "bottom_value": result.bottom_value},
        "solver_time_s": solver_time,
        "rows": rows,
        "cores_file": "cores.json",
        "reproduction": "g = (ell + log(bottom_value)) / L^2; recompute bottom_value from cores.json and the model's b vector",
    });
    write_json(&out.join("run.json"), &report)?;
    println!(
        "boundary L={length} K={k} chi={chi} open g={:.12} logZ={:.9} t={:.3}s",
        result.g, result.log_z, solver_time
    );
    Ok(())
}

fn run_trg(arguments: &Arguments, out: &Path) -> Result<(), String> {
    let (length, k, chi) = chemistry(arguments)?;
    if !length.is_power_of_two() {
        return Err(format!("--l {length} is not a power of two; TRG needs L = 2^(n/2)"));
    }
    prepare(out)?;
    let model = Ising::new(k);
    let start = Instant::now();
    let result = trg::run(&model, length, chi, Some(out));
    let solver_time = start.elapsed().as_secs_f64();
    let final_file = out.join("final.npy");
    let [dr, du, dl, dd] = result.dims;
    linalg::write_npy(&final_file, &result.tensor, &[dr, du, dl, dd])
        .map_err(|error| format!("{}: {error}", final_file.display()))?;
    let mut steps = Vec::with_capacity(result.steps.len());
    for step in &result.steps {
        let [sr, su, sl, sd] = step.source_dims;
        let [br, bu, _, _] = step.bond_dims;
        steps.push(json!({
            "step": step.step,
            "tensors": step.tensors,
            "max_abs": step.max_abs,
            "log_scale": step.log_scale,
            "ell": step.ell,
            "source_dims": step.source_dims,
            "bond_dims": step.bond_dims,
            "split1_singular_values": step.split1_singular_values,
            "split2_singular_values": step.split2_singular_values,
            "regroup_entries": step.regroup_entries,
            "truncation_method": step.truncation_method,
            "split_factor_file": format!("splits-{}.npy", step.step),
            "split_factor_shapes": [[sl, su, br], [sd, sl, bu], [br, sr, sd], [bu, su, sr]],
            "split_factor_order": "S1,S2,S3,S4 concatenated in a flat array",
        }));
    }
    let report = json!({
        "tool": "thermo 0.1.0",
        "method": "trg",
        "geometry": {
            "boundary": "periodic",
            "l": length,
            "n": length * length,
            "bonds": 2 * length * length,
            "boundary_condition": "periodic; opposite edges are joined into a torus",
            "steps": result.steps.len(),
        },
        "settings": {"method": "trg", "l": length, "k": k, "chi": chi, "normalisation": "max absolute entry"},
        "model": model_json(&model),
        "log_z": result.log_z,
        "g": result.g,
        "normalisation": {"ell": result.ell, "z": result.z, "closure": "sum_ij A[i,j,i,j]"},
        "solver_time_s": solver_time,
        "steps": steps,
        "final_tensor_file": "final.npy",
        "final_tensor_shape": result.dims,
        "final_tensor_axes": "[r,u,l,d]",
    });
    write_json(&out.join("run.json"), &report)?;
    println!(
        "trg L={length} K={k} chi={chi} periodic g={:.12} logZ={:.9} t={:.3}s",
        result.g, result.log_z, solver_time
    );
    Ok(())
}

fn run_svd(arguments: &Arguments, out: &Path) -> Result<(), String> {
    let side = arguments.integer("l")?;
    let seed = arguments.integer("seed")? as u64;
    let ranks_raw = arguments.required("ranks")?.to_string();
    arguments.reject(&["k", "chi"], "svd")?;
    if side < 1 {
        return Err("--l must be at least 1".into());
    }
    let mut ranks: Vec<usize> = Vec::new();
    for token in ranks_raw.split(',') {
        let rank = token
            .trim()
            .parse::<usize>()
            .map_err(|_| format!("--ranks must be comma-separated integers, got {token:?}"))?;
        if rank < 1 || rank > side {
            return Err(format!("--ranks {rank} must lie in 1..={side}"));
        }
        ranks.push(rank);
    }
    ranks.sort_unstable();
    ranks.dedup();
    prepare(out)?;
    let start = Instant::now();
    let (matrix, mean, std) = rng::gaussian_matrix(side, side, seed);
    let generated = start.elapsed().as_secs_f64();
    let svd_start = Instant::now();
    let svd = linalg::jacobi_svd(&matrix, side, side);
    let svd_time = svd_start.elapsed().as_secs_f64();
    let total_weight: f64 = svd.s.iter().map(|s| s * s).sum();
    let norm = total_weight.sqrt();
    let matrix_file = out.join("matrix.npy");
    linalg::write_npy(&matrix_file, &matrix, &[side, side])
        .map_err(|error| format!("{}: {error}", matrix_file.display()))?;
    let values_file = out.join("singular-values.npy");
    linalg::write_npy(&values_file, &svd.s, &[svd.k])
        .map_err(|error| format!("{}: {error}", values_file.display()))?;
    let mut records = Vec::new();
    for &rank in &ranks {
        let rebuilt = svd.reconstruct(side, side, rank);
        let difference: Vec<f64> =
            matrix.iter().zip(rebuilt.iter()).map(|(x, y)| x - y).collect();
        let measured = linalg::frobenius(&difference) / norm;
        let prediction = (svd.discarded_weight(rank) / total_weight).sqrt();
        let file = format!("recon-{rank}.npy");
        linalg::write_npy(&out.join(&file), &rebuilt, &[side, side])
            .map_err(|error| format!("{}: {error}", out.join(&file).display()))?;
        let saved = side * rank + rank * side;
        records.push(json!({
            "rank": rank,
            "file": file,
            "relative_frobenius_error": measured,
            "discarded_spectrum_prediction": prediction,
            "difference": (measured - prediction).abs(),
            "saved_entries": saved,
            "full_entries": side * side,
            "storage_ratio": saved as f64 / (side * side) as f64,
            "retained_weight_fraction": 1.0 - svd.discarded_weight(rank) / total_weight,
        }));
    }
    let threshold = 0.1;
    let mut crossing = side;
    let mut crossing_error = 0.0;
    for rank in 1..=side {
        let error = (svd.discarded_weight(rank) / total_weight).sqrt();
        if error <= threshold {
            crossing = rank;
            crossing_error = error;
            break;
        }
    }
    let storage_rank = (side * side + 2 * side - 1) / (2 * side);
    let report = json!({
        "tool": "thermo 0.1.0",
        "method": "svd",
        "geometry": {"m": side, "n": side, "entries": side * side},
        "generator": {
            "distribution": "independent standard-normal entries",
            "uniform": "SplitMix64, u = ((z >> 11) + 0.5) * 2^-53 in (0,1)",
            "gaussian": "Box-Muller pairs, r = sqrt(-2 ln u1), (r cos 2 pi u2, r sin 2 pi u2)",
            "fill_order": "row-major",
            "seed": seed,
        },
        "settings": {"method": "svd", "l": side, "seed": seed, "ranks": ranks},
        "sample_mean": mean,
        "sample_standard_deviation": std,
        "singular_value_sum": svd.s.iter().sum::<f64>(),
        "frobenius_norm": norm,
        "matrix_file": "matrix.npy",
        "singular_values_file": "singular-values.npy",
        "solver_time_s": svd_time,
        "generation_time_s": generated,
        "reconstructions": records,
        "storage_crossover": {
            "rank": storage_rank,
            "formula": "(m+n) chi = m n  =>  chi = m n / (m + n)",
            "saved_at_crossover": storage_rank * 2 * side,
            "full_entries": side * side,
        },
        "ten_percent_crossing": {
            "rank": crossing,
            "relative_error": crossing_error,
            "storage_ratio": (side + side) as f64 * crossing as f64 / (side * side) as f64,
        },
        "full_rank_error": (svd.discarded_weight(side) / total_weight).sqrt(),
    });
    write_json(&out.join("run.json"), &report)?;
    println!(
        "svd m=n={side} seed={seed} ranks={ranks:?} 10% crossing chi={crossing} t={:.3}s",
        svd_time
    );
    Ok(())
}
