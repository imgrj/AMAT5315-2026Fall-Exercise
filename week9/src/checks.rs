//! The sheet's library checks: matrix identities, the lb network, Ising ring.
//!
//! Every pair of methods is compared with a 1e-12 absolute tolerance, and both
//! results are printed so a misread input cannot hide behind agreement.

use serde_json::{json, Value};
use tenferro_cpu::CpuEngine;

use crate::network::{einsum, Array};

const TOLERANCE: f64 = 1e-12;

fn flip(spin: usize) -> f64 {
    if spin == 0 {
        -1.0
    } else {
        1.0
    }
}

fn max_difference(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .fold(0.0f64, |worst, (a, b)| worst.max((a - b).abs()))
}

fn flatten(value: &Value) -> Vec<f64> {
    let mut out = Vec::new();
    fn visit(value: &Value, out: &mut Vec<f64>) {
        match value {
            Value::Array(items) => items.iter().for_each(|item| visit(item, out)),
            Value::Number(number) => out.push(number.as_f64().unwrap_or(f64::NAN)),
            _ => {}
        }
    }
    visit(value, &mut out);
    out
}

fn direct_matrix_product(a: &[[f64; 2]; 2], b: &[[f64; 2]; 2]) -> [[f64; 2]; 2] {
    let mut out = [[0.0f64; 2]; 2];
    for (row, values) in out.iter_mut().enumerate() {
        for (column, value) in values.iter_mut().enumerate() {
            let mut sum = 0.0;
            for inner in 0..2 {
                sum += a[row][inner] * b[inner][column];
            }
            *value = sum;
        }
    }
    out
}

fn matrix_check(engine: &mut CpuEngine) -> Result<Value, String> {
    let a = [[1.0f64, 2.0], [3.0, 4.0]];
    let b = [[2.0f64, 0.0], [1.0, 2.0]];
    let a_array = Array::from_row_major(vec![2, 2], a.iter().flatten().copied().collect());
    let b_array = Array::from_row_major(vec![2, 2], b.iter().flatten().copied().collect());

    let labels_a: &[u32] = &[0, 1];
    let labels_b: &[u32] = &[1, 2];
    let product = einsum(
        engine,
        &[(labels_a, a_array.clone()), (labels_b, b_array.clone())],
        &[0, 2],
    )?;
    let product_direct = direct_matrix_product(&a, &b);
    let product_value = product.to_nested();
    let direct_value = json!(product_direct);
    let product_error = max_difference(&flatten(&product_value), &flatten(&direct_value));

    // tr(AB) joins i and k: sum over i and j of A[i,j] B[j,i].
    let trace = einsum(
        engine,
        &[(labels_a, a_array.clone()), (&[1, 0], b_array.clone())],
        &[],
    )?;
    let trace_value = trace.data.first().copied().unwrap_or(f64::NAN);
    let trace_direct: f64 = (0..2).map(|i| (0..2).map(|j| a[i][j] * b[j][i]).sum::<f64>()).sum();

    // Independent sums: every leg gets its own all-ones vector.
    let total = einsum(
        engine,
        &[(labels_a, a_array), (labels_b, b_array)],
        &[],
    )?;
    let total_value = total.data.first().copied().unwrap_or(f64::NAN);
    let total_direct: f64 = product_direct.iter().flatten().sum();

    Ok(json!({
        "a": a, "b": b,
        "product": product_value, "product_direct": direct_value,
        "product_error": product_error,
        "trace": trace_value, "trace_direct": trace_direct,
        "trace_error": (trace_value - trace_direct).abs(),
        "total": total_value, "total_direct": total_direct,
        "total_error": (total_value - total_direct).abs(),
        "tolerance": TOLERANCE,
    }))
}

fn lung_bronchitis_check(engine: &mut CpuEngine) -> Result<Value, String> {
    // Table axes follow the UAI scopes: p(s), p(l|s), p(b|s).
    let p_s = Array::from_row_major(vec![2], vec![0.5, 0.5]);
    let p_ls = Array::from_row_major(vec![2, 2], vec![0.1, 0.01, 0.9, 0.99]);
    let p_bs = Array::from_row_major(vec![2, 2], vec![0.6, 0.3, 0.4, 0.7]);
    let s: &[u32] = &[0];
    let ls: &[u32] = &[1, 0];
    let bs: &[u32] = &[2, 0];
    let joint = einsum(
        engine,
        &[(s, p_s), (ls, p_ls), (bs, p_bs)],
        &[1, 2],
    )?;
    let mut direct = vec![vec![0.0f64; 2]; 2];
    for (l, row) in direct.iter_mut().enumerate() {
        for (b, value) in row.iter_mut().enumerate() {
            let mut sum = 0.0;
            for state in 0..2 {
                sum += 0.5 * p_ls_entries(state, l) * p_bs_entries(state, b);
            }
            *value = sum;
        }
    }
    let joint_value = joint.to_nested();
    let direct_value = json!(direct);
    let error = max_difference(&flatten(&joint_value), &flatten(&direct_value));
    let normalization: f64 = joint.sum();
    Ok(json!({
        "p_s": [0.5, 0.5],
        "p_l_s": [[0.1, 0.01], [0.9, 0.99]],
        "p_b_s": [[0.6, 0.3], [0.4, 0.7]],
        "joint": joint_value,
        "joint_direct": direct_value,
        "max_abs_error": error,
        "normalization": normalization,
        "labels": ["L", "B"],
    }))
}

fn p_ls_entries(state: usize, l: usize) -> f64 {
    // p(l|s) is stored with scope (L, S), last axis s fastest.
    [[0.1f64, 0.01], [0.9, 0.99]][l][state]
}

fn p_bs_entries(state: usize, b: usize) -> f64 {
    [[0.6f64, 0.3], [0.4, 0.7]][b][state]
}

fn ising_check(engine: &mut CpuEngine) -> Result<Value, String> {
    let beta = 0.5f64;
    let coupling = 1.0f64;
    let bond = |left: usize, right: usize| {
        (beta * coupling * flip(left) * flip(right)).exp()
    };
    let k12 = Array::from_row_major(
        vec![2, 2],
        (0..4).map(|index| bond(index / 2, index % 2)).collect(),
    );
    let k23 = k12.clone();
    let k31 = k12.clone();
    let contracted = einsum(
        engine,
        &[(&[0, 1], k12), (&[1, 2], k23), (&[2, 0], k31)],
        &[],
    )?;
    let value = contracted.data.first().copied().unwrap_or(f64::NAN);
    let mut enumeration = 0.0f64;
    for s1 in 0..2 {
        for s2 in 0..2 {
            for s3 in 0..2 {
                enumeration += (beta * coupling
                    * (flip(s1) * flip(s2) + flip(s2) * flip(s3) + flip(s3) * flip(s1)))
                    .exp();
            }
        }
    }
    let closed_form = 2.0 * (1.5f64 * beta * coupling).exp() + 6.0 * (-0.5f64 * beta * coupling).exp();
    Ok(json!({
        "beta": beta, "coupling": coupling, "weights": 8,
        "contraction": value,
        "enumeration": enumeration,
        "closed_form": closed_form,
        "max_abs_error": max_difference(&[value], &[enumeration]),
        "closed_form_error": max_difference(&[value], &[closed_form]),
    }))
}

/// Print and return the three checks.
pub fn run(engine: &mut CpuEngine) -> Result<Value, String> {
    let matrix = matrix_check(engine)?;
    let lung = lung_bronchitis_check(engine)?;
    let ising = ising_check(engine)?;
    println!("tensor checks (tolerance {TOLERANCE:e})");
    println!(
        "  AB = {} via einsum, {} via direct sums, error {:.3e}",
        matrix["product"], matrix["product_direct"], matrix["product_error"].as_f64().unwrap_or(f64::NAN)
    );
    println!(
        "  tr(AB) = {:.17}, direct {:.17}, error {:.3e}",
        matrix["trace"].as_f64().unwrap_or(f64::NAN),
        matrix["trace_direct"].as_f64().unwrap_or(f64::NAN),
        matrix["trace_error"].as_f64().unwrap_or(f64::NAN)
    );
    println!(
        "  p(l,b) = {} via einsum, error {:.3e}, normalisation {:.17}",
        lung["joint"], lung["max_abs_error"].as_f64().unwrap_or(f64::NAN),
        lung["normalization"].as_f64().unwrap_or(f64::NAN)
    );
    println!(
        "  Ising Z = {:.17} via contraction, {:.17} via enumeration, error {:.3e}",
        ising["contraction"].as_f64().unwrap_or(f64::NAN),
        ising["enumeration"].as_f64().unwrap_or(f64::NAN),
        ising["max_abs_error"].as_f64().unwrap_or(f64::NAN)
    );
    Ok(json!({ "matrix": matrix, "lung_bronchitis": lung, "ising": ising, "tolerance": TOLERANCE }))
}
