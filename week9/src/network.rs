//! Tensor storage and pairwise contraction on the CPU.
//!
//! Tensors use tenferro's column-major layout (the leftmost axis varies
//! fastest). Factors are stored in their declared scope order, and the
//! observations enter as one-hot cap tensors, exactly as drawn in the sheet.

use tenferro_cpu::CpuEngine;
use tenferro_einsum::{EinsumSubscripts, TensorEinsumExt};
use tenferro_tensor::Tensor;

use crate::plan::Step;
use crate::uai::Model;

/// A dense array in column-major order.
#[derive(Clone, Debug)]
pub struct Array {
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
}

impl Array {
    pub fn max_abs(&self) -> f64 {
        self.data.iter().fold(0.0f64, |acc, value| acc.max(value.abs()))
    }

    pub fn sum(&self) -> f64 {
        self.data.iter().sum()
    }

    /// Divide by the largest magnitude and report the removed scale.
    pub fn rescale(&mut self) -> f64 {
        let largest = self.max_abs();
        if largest > 0.0 && largest.is_finite() {
            for value in &mut self.data {
                *value /= largest;
            }
            largest
        } else {
            1.0
        }
    }

    fn to_tensor(self) -> Result<Tensor, String> {
        Tensor::from_vec_col_major(self.shape, self.data).map_err(|error| error.to_string())
    }

    fn from_tensor(tensor: Tensor) -> Result<Self, String> {
        let (shape, data) = tensor
            .into_vec_col_major::<f64>()
            .map_err(|error| error.to_string())?;
        Ok(Self { shape, data })
    }

    /// Build from a row-major buffer (the last axis varies fastest), as printed
    /// in the sheet's examples.
    pub fn from_row_major(shape: Vec<usize>, data: Vec<f64>) -> Self {
        let rank = shape.len();
        let mut column_strides = vec![1usize; rank];
        for axis in 1..rank {
            column_strides[axis] = column_strides[axis - 1] * shape[axis - 1];
        }
        let mut row_strides = vec![1usize; rank];
        for axis in (0..rank.saturating_sub(1)).rev() {
            row_strides[axis] = row_strides[axis + 1] * shape[axis + 1];
        }
        let mut column_major = vec![0.0; data.len()];
        for (row_index, value) in data.iter().enumerate() {
            let mut column_index = 0usize;
            for axis in 0..rank {
                let state = (row_index / row_strides[axis]) % shape[axis];
                column_index += state * column_strides[axis];
            }
            column_major[column_index] = *value;
        }
        Self {
            shape,
            data: column_major,
        }
    }

    /// Nested JSON arrays in axis order (first axis outermost).
    pub fn to_nested(&self) -> serde_json::Value {
        fn build(shape: &[usize], strides: &[usize], data: &[f64], offset: usize) -> serde_json::Value {
            if shape.is_empty() {
                return serde_json::json!(data[offset]);
            }
            let values: Vec<serde_json::Value> = (0..shape[0])
                .map(|index| build(&shape[1..], &strides[1..], data, offset + index * strides[0]))
                .collect();
            serde_json::Value::Array(values)
        }
        let mut strides = Vec::with_capacity(self.shape.len());
        let mut stride = 1usize;
        for dim in &self.shape {
            strides.push(stride);
            stride *= *dim;
        }
        build(&self.shape, &strides, &self.data, 0)
    }
}

/// One leaf of a network: ordered labels plus the dense table.
pub struct Leaf {
    pub labels: Vec<u32>,
    pub array: Array,
}

/// Leaves of a network with the observations attached as caps.
pub struct Network {
    pub leaves: Vec<Leaf>,
    /// Product of factors whose scope is empty.
    pub scalar: f64,
}

pub fn build_network(model: &Model, observations: &[(usize, usize)]) -> Result<Network, String> {
    let mut leaves = Vec::with_capacity(model.factors.len() + observations.len());
    let mut scalar = 1.0f64;
    for factor in &model.factors {
        let labels: Vec<u32> = factor.scope.iter().map(|variable| *variable as u32).collect();
        if labels.is_empty() {
            scalar *= factor.entries.first().copied().unwrap_or(1.0);
            continue;
        }
        // The file lists entries with the last scope axis varying fastest.
        leaves.push(Leaf {
            labels,
            array: Array::from_row_major(factor.dims.clone(), factor.entries.clone()),
        });
    }
    for (variable, state) in observations {
        let dim = model.dims[*variable];
        let mut data = vec![0.0; dim];
        if *state < dim {
            data[*state] = 1.0;
        }
        leaves.push(Leaf {
            labels: vec![*variable as u32],
            array: Array {
                shape: vec![dim],
                data,
            },
        });
    }
    Ok(Network { leaves, scalar })
}

pub struct Contracted {
    pub log_scale: f64,
    pub array: Array,
}

/// Contract the network along the planned pairwise steps.
///
/// Every intermediate is rescaled by its largest magnitude and the removed
/// logarithms accumulate, so small (or huge) benchmarking weights keep full
/// relative precision.
pub fn contract(
    model: &Model,
    observations: &[(usize, usize)],
    steps: &[Step],
    engine: &mut CpuEngine,
) -> Result<Contracted, String> {
    let network = build_network(model, observations)?;
    let mut nodes: Vec<Option<Array>> = network
        .leaves
        .into_iter()
        .map(|leaf| Some(leaf.array))
        .collect();
    let mut log_scale = network.scalar.ln();
    for step in steps {
        let left = take(&mut nodes, step.left)?;
        let right = take(&mut nodes, step.right)?;
        let subscripts = EinsumSubscripts::new(
            &[step.left_labels.as_slice(), step.right_labels.as_slice()],
            &step.labels,
        );
        let left_tensor = left.to_tensor()?;
        let right_tensor = right.to_tensor()?;
        let product = [&left_tensor, &right_tensor]
            .einsum_subscripts(&subscripts, engine)
            .map_err(|error| error.to_string())?;
        let mut array = Array::from_tensor(product)?;
        let scale = array.rescale();
        log_scale += scale.ln();
        nodes.push(Some(array));
    }
    let mut remaining: Vec<Array> = nodes.into_iter().flatten().collect();
    if remaining.is_empty() {
        return Err("the plan produced no intermediate tensor".into());
    }
    let final_array = if remaining.len() == 1 {
        remaining.pop().unwrap()
    } else {
        // A plan that leaves several tensors: multiply them together.
        let mut product = remaining.pop().unwrap();
        for other in remaining {
            let labels_other: Vec<u32> = (0..other.shape.len()).map(|axis| axis as u32).collect();
            let labels_product: Vec<u32> = (0..product.shape.len())
                .map(|axis| axis as u32 + 1000)
                .collect();
            let subscripts = EinsumSubscripts::new(
                &[labels_product.as_slice(), labels_other.as_slice()],
                &[labels_product.clone(), labels_other.clone()].concat(),
            );
            let product_tensor = product.to_tensor()?;
            let other_tensor = other.to_tensor()?;
            let out = [&product_tensor, &other_tensor]
                .einsum_subscripts(&subscripts, engine)
                .map_err(|error| error.to_string())?;
            product = Array::from_tensor(out)?;
        }
        let scale = product.rescale();
        log_scale += scale.ln();
        product
    };
    let _ = model;
    Ok(Contracted {
        log_scale,
        array: final_array,
    })
}

fn take(nodes: &mut Vec<Option<Array>>, index: usize) -> Result<Array, String> {
    nodes
        .get_mut(index)
        .ok_or_else(|| format!("plan refers to node {index} outside the network"))?
        .take()
        .ok_or_else(|| format!("plan reuses node {index}"))
}

/// Contract explicit labelled arrays; used by the sheet's small examples.
pub fn einsum(
    engine: &mut CpuEngine,
    operands: &[(&[u32], Array)],
    output: &[u32],
) -> Result<Array, String> {
    let tensors: Vec<Tensor> = operands
        .iter()
        .map(|(_, array)| array.clone().to_tensor())
        .collect::<Result<Vec<_>, _>>()?;
    let inputs: Vec<&[u32]> = operands.iter().map(|(labels, _)| *labels).collect();
    let subscripts = EinsumSubscripts::new(&inputs, output);
    let references: Vec<&Tensor> = tensors.iter().collect();
    let product = references
        .einsum_subscripts(&subscripts, engine)
        .map_err(|error| error.to_string())?;
    Array::from_tensor(product)
}
