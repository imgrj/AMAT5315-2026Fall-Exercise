//! Tensor networks, `omeco` contraction planning, tropical (min-plus) contraction
//! and conditional sampling from one stored contraction.
//!
//! Every problem of the Week 11 sheet becomes a network whose indices all have
//! dimension 2: spin variables `s_i = +-1` for the magnets, one bit per chessboard
//! square and one bit per carry for N-queens. The network is contracted in the
//! binary order chosen by `omeco`.
//!
//! Three arithmetic regimes share one traversal:
//!
//! * [`contract_dense`] multiplies Boltzmann weights `exp(-beta E)` and divides every
//!   intermediate by its largest entry, accumulating the logs of those scales, so
//!   `log Z` never overflows;
//! * [`contract_tropical`] stores `(energy, count)` pairs, adds energies on a product
//!   and takes the minimum (adding the counts on a tie) on a sum, returning
//!   `(E_min, g_0)` exactly;
//! * [`contract_sparse`] stores only the nonzero entries, which keeps the 0/1 rule
//!   tensors of N-queens to a few seconds.

use std::collections::{BTreeMap, HashMap};

use omeco::{contraction_complexity, optimize_code, EinCode, GreedyMethod, NestedEinsum, TreeSA};

pub type Labels = Vec<usize>;

fn strides(dims: &[usize]) -> Vec<usize> {
    let mut out = vec![0usize; dims.len()];
    let mut acc = 1usize;
    for i in (0..dims.len()).rev() {
        out[i] = acc;
        acc *= dims[i];
    }
    out
}

fn product(dims: &[usize]) -> usize {
    dims.iter().product::<usize>().max(1)
}

/// A dense tensor with its index labels; dimensions come from the network.
#[derive(Clone, Debug)]
pub struct DTensor {
    pub labels: Labels,
    pub data: Vec<f64>,
}

impl DTensor {
    pub fn new(labels: Labels, data: Vec<f64>) -> Self {
        DTensor { labels, data }
    }
}

/// One term of the network, in row-major order over its (unsorted) labels.
#[derive(Clone, Debug)]
pub struct Term {
    pub labels: Labels,
    pub values: Vec<f64>,
}

pub struct Network {
    pub dims: Vec<usize>,
    pub terms: Vec<Term>,
}

impl Network {
    pub fn new(nvars: usize) -> Self {
        Network {
            dims: vec![2; nvars],
            terms: Vec::new(),
        }
    }

    pub fn push(&mut self, labels: Labels, values: Vec<f64>) {
        let expected = product(&labels.iter().map(|&l| self.dims[l]).collect::<Vec<_>>());
        assert_eq!(expected, values.len(), "term {labels:?} has the wrong number of entries");
        self.terms.push(Term { labels, values });
    }

    fn term_labels(&self) -> Vec<Labels> {
        self.terms.iter().map(|t| t.labels.clone()).collect()
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Order {
    Greedy,
    Treesa,
}

impl Order {
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "greedy" => Some(Order::Greedy),
            "treesa" => Some(Order::Treesa),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Order::Greedy => "greedy",
            Order::Treesa => "treesa",
        }
    }
}

/// A flattened binary contraction tree in post-order: children precede parents.
pub enum Node {
    Leaf {
        term: usize,
        labels: Labels,
    },
    Pair {
        left: usize,
        right: usize,
        out: Labels,
        in0: Labels,
        in1: Labels,
    },
}

pub struct Tree {
    pub nodes: Vec<Node>,
    pub root: usize,
}

pub struct Plan {
    pub tree: Tree,
    pub nvars: usize,
    pub tc: f64,
    pub sc: f64,
    pub rwc: f64,
    pub max_entries: usize,
}

fn fnv(hash: &mut u64, value: u64) {
    *hash ^= value;
    *hash = hash.wrapping_mul(0x1000_0000_01B3);
}

/// A stable fingerprint of the planned tree, so a run can say which plan it used.
/// `omeco`'s annealer breaks ties with its own randomness, so two `treesa` runs can
/// differ here even when `tc` and `sc` agree.
pub fn tree_hash(tree: &Tree) -> String {
    let mut hash = 0xCBF2_9CE4_8422_2325u64;
    fnv(&mut hash, tree.nodes.len() as u64);
    fnv(&mut hash, tree.root as u64);
    for node in &tree.nodes {
        match node {
            Node::Leaf { term, labels } => {
                fnv(&mut hash, 0);
                fnv(&mut hash, *term as u64);
                for &label in labels {
                    fnv(&mut hash, label as u64);
                }
            }
            Node::Pair {
                left,
                right,
                out,
                in0,
                in1,
            } => {
                fnv(&mut hash, 1);
                fnv(&mut hash, *left as u64);
                fnv(&mut hash, *right as u64);
                for labels in [out, in0, in1] {
                    fnv(&mut hash, labels.len() as u64);
                    for &label in labels {
                        fnv(&mut hash, label as u64);
                    }
                }
            }
        }
    }
    format!("{hash:016x}")
}

/// Plan a contraction with `omeco`, without contracting any tensor.
pub fn plan(net: &Network, order: Order) -> Plan {
    let inputs = net.term_labels();
    let code = EinCode::new(inputs.clone(), Vec::<usize>::new());
    let sizes: HashMap<usize, usize> = net.dims.iter().enumerate().map(|(i, &d)| (i, d)).collect();
    let nested = match order {
        Order::Greedy => optimize_code(&code, &sizes, &GreedyMethod::default()),
        Order::Treesa => optimize_code(&code, &sizes, &TreeSA::fast().with_ntrials(4)),
    }
    .expect("omeco returned no contraction order");
    let complexity = contraction_complexity(&nested, &sizes, &inputs);
    let mut nodes = Vec::new();
    let root = flatten(&nested, &inputs, &mut nodes);
    Plan {
        tree: Tree { nodes, root },
        nvars: net.dims.len(),
        tc: complexity.tc,
        sc: complexity.sc,
        rwc: complexity.rwc,
        max_entries: 2f64.powf(complexity.sc).round() as usize,
    }
}

fn flatten(node: &NestedEinsum<usize>, inputs: &[Labels], nodes: &mut Vec<Node>) -> usize {
    match node {
        NestedEinsum::Leaf { tensor_index } => {
            let index = nodes.len();
            nodes.push(Node::Leaf {
                term: *tensor_index,
                labels: inputs[*tensor_index].clone(),
            });
            index
        }
        NestedEinsum::Node { args, eins } => {
            assert_eq!(args.len(), 2, "the planner must return a binary tree");
            let left = flatten(&args[0], inputs, nodes);
            let right = flatten(&args[1], inputs, nodes);
            let index = nodes.len();
            nodes.push(Node::Pair {
                left,
                right,
                out: eins.iy.clone(),
                in0: eins.ixs[0].clone(),
                in1: eins.ixs[1].clone(),
            });
            index
        }
    }
}

/// Iterate over every combined index, handing out the flat offsets into `a`, `b`
/// and the output. `ax`, `bx`, `ox` are the per-axis strides, zero when the axis
/// is absent from that tensor.
fn for_each_combined(
    cdims: &[usize],
    ax: &[usize],
    bx: &[usize],
    ox: &[usize],
    mut visit: impl FnMut(usize, usize, usize),
) {
    let n = cdims.len();
    if n == 0 {
        visit(0, 0, 0);
        return;
    }
    let mut idx = vec![0usize; n];
    let (mut a_off, mut b_off, mut o_off) = (0usize, 0usize, 0usize);
    'outer: loop {
        visit(a_off, b_off, o_off);
        let mut p = n;
        loop {
            if p == 0 {
                break 'outer;
            }
            p -= 1;
            idx[p] += 1;
            a_off += ax[p];
            b_off += bx[p];
            o_off += ox[p];
            if idx[p] < cdims[p] {
                continue 'outer;
            }
            idx[p] = 0;
            a_off -= ax[p] * cdims[p];
            b_off -= bx[p] * cdims[p];
            o_off -= ox[p] * cdims[p];
        }
    }
}

/// The combined index order and the per-axis strides of a pairwise contraction.
fn layout(
    a_labels: &[usize],
    b_labels: &[usize],
    out: &[usize],
    dims: &[usize],
) -> (Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>) {
    let mut combined: Vec<usize> = a_labels.to_vec();
    let mut cdims: Vec<usize> = a_labels.iter().map(|&l| dims[l]).collect();
    for &l in b_labels {
        if !combined.contains(&l) {
            combined.push(l);
            cdims.push(dims[l]);
        }
    }
    let a_strides = strides(&a_labels.iter().map(|&l| dims[l]).collect::<Vec<_>>());
    let b_strides = strides(&b_labels.iter().map(|&l| dims[l]).collect::<Vec<_>>());
    let odims: Vec<usize> = out.iter().map(|&l| dims[l]).collect();
    let o_strides = strides(&odims);
    let mut ax = vec![0usize; combined.len()];
    let mut bx = vec![0usize; combined.len()];
    let mut ox = vec![0usize; combined.len()];
    for (p, &l) in combined.iter().enumerate() {
        if let Some(k) = a_labels.iter().position(|&x| x == l) {
            ax[p] = a_strides[k];
        }
        if let Some(k) = b_labels.iter().position(|&x| x == l) {
            bx[p] = b_strides[k];
        }
        if let Some(k) = out.iter().position(|&x| x == l) {
            ox[p] = o_strides[k];
        }
    }
    (cdims, ax, bx, ox)
}

fn dense_contract(a: &DTensor, b: &DTensor, out: &[usize], dims: &[usize]) -> DTensor {
    let (cdims, ax, bx, ox) = layout(&a.labels, &b.labels, out, dims);
    let odims: Vec<usize> = out.iter().map(|&l| dims[l]).collect();
    let mut data = vec![0.0f64; product(&odims)];
    for_each_combined(&cdims, &ax, &bx, &ox, |ia, ib, io| {
        data[io] += a.data[ia] * b.data[ib];
    });
    DTensor::new(out.to_vec(), data)
}

/// A tensor of `(energy, count)` pairs for min-plus contraction.
#[derive(Clone, Debug)]
pub struct TTensor {
    pub labels: Labels,
    pub energy: Vec<f64>,
    pub count: Vec<u64>,
}

fn tropical_contract(a: &TTensor, b: &TTensor, out: &[usize], dims: &[usize]) -> TTensor {
    let (cdims, ax, bx, ox) = layout(&a.labels, &b.labels, out, dims);
    let odims: Vec<usize> = out.iter().map(|&l| dims[l]).collect();
    let size = product(&odims);
    let mut energy = vec![f64::INFINITY; size];
    let mut count = vec![0u64; size];
    for_each_combined(&cdims, &ax, &bx, &ox, |ia, ib, io| {
        let e = a.energy[ia] + b.energy[ib];
        let g = a.count[ia].saturating_mul(b.count[ib]);
        if e < energy[io] {
            energy[io] = e;
            count[io] = g;
        }
    });
    TTensor {
        labels: out.to_vec(),
        energy,
        count,
    }
}

/// A sparse tensor whose keys are bit masks over the sorted `labels`.
#[derive(Clone, Debug)]
pub struct STensor {
    pub labels: Labels,
    pub map: BTreeMap<u64, f64>,
}

impl STensor {
    fn from_term(term: &Term) -> Self {
        let mut labels = term.labels.clone();
        labels.sort_unstable();
        let mut map = BTreeMap::new();
        // `term.values` is row-major over `term.labels`, which may be unsorted.
        let term_strides = strides(&term.labels.iter().map(|_| 2).collect::<Vec<_>>());
        let size = term.values.len();
        for flat in 0..size {
            let value = term.values[flat];
            if value == 0.0 {
                continue;
            }
            let mut key = 0u64;
            for (p, &l) in labels.iter().enumerate() {
                let src = term.labels.iter().position(|&x| x == l).unwrap();
                let bit = (flat / term_strides[src]) % 2;
                key |= (bit as u64) << p;
            }
            map.insert(key, value);
        }
        STensor { labels, map }
    }
}

/// Compress the bits of `shared` out of a key laid out over `labels`.
fn compress(key: u64, shared_pos: &[Option<usize>]) -> u64 {
    let mut out = 0u64;
    for (p, position) in shared_pos.iter().enumerate() {
        if let Some(pos) = position {
            out |= ((key >> pos) & 1) << p;
        }
    }
    out
}

fn sparse_contract(a: &STensor, b: &STensor, out: &[usize]) -> STensor {
    let mut out_labels = out.to_vec();
    out_labels.sort_unstable();
    out_labels.dedup();
    let shared: Labels = a.labels.iter().filter(|l| b.labels.contains(l)).cloned().collect();
    let a_shared: Vec<Option<usize>> = shared
        .iter()
        .map(|l| a.labels.iter().position(|x| x == l))
        .collect();
    let b_shared: Vec<Option<usize>> = shared
        .iter()
        .map(|l| b.labels.iter().position(|x| x == l))
        .collect();
    // For every output label: where its bit lives in a or in b.
    let out_source: Vec<(usize, Option<usize>, Option<usize>)> = out_labels
        .iter()
        .enumerate()
        .map(|(p, l)| {
            (
                p,
                a.labels.iter().position(|x| x == l),
                b.labels.iter().position(|x| x == l),
            )
        })
        .collect();
    let mut grouped: HashMap<u64, Vec<(u64, f64)>> = HashMap::new();
    for (&key, &value) in &b.map {
        if value == 0.0 {
            continue;
        }
        let sk = compress(key, &b_shared);
        grouped.entry(sk).or_default().push((key, value));
    }
    let mut result: BTreeMap<u64, f64> = BTreeMap::new();
    for (&key, &value) in &a.map {
        if value == 0.0 {
            continue;
        }
        let sk = compress(key, &a_shared);
        let Some(list) = grouped.get(&sk) else { continue };
        for &(b_key, b_value) in list {
            let mut out_key = 0u64;
            for &(p, pa, pb) in &out_source {
                let bit = match (pa, pb) {
                    (Some(pos), _) => (key >> pos) & 1,
                    (None, Some(pos)) => (b_key >> pos) & 1,
                    (None, None) => continue,
                };
                out_key |= bit << p;
            }
            *result.entry(out_key).or_insert(0.0) += value * b_value;
        }
    }
    STensor {
        labels: out_labels,
        map: result,
    }
}

/// Divide a tensor by its largest entry and return `log` of that scale.
fn rescale(data: &mut [f64]) -> f64 {
    let max = data.iter().cloned().fold(0.0f64, |m, v| m.max(v.abs()));
    if max > 0.0 && max != 1.0 {
        for value in data.iter_mut() {
            *value /= max;
        }
        max.ln()
    } else {
        0.0
    }
}

/// Contract the Boltzmann network, keeping every intermediate for the down pass.
/// Returns the stored tensors, `log Z`, and the sum of the `log` scales.
pub fn contract_dense(net: &Network, tree: &Tree) -> (Vec<DTensor>, f64, f64) {
    let mut stored: Vec<DTensor> = Vec::with_capacity(tree.nodes.len());
    let mut log_scale = 0.0;
    for node in &tree.nodes {
        match node {
            Node::Leaf { term, labels } => {
                let mut data = net.terms[*term].values.clone();
                log_scale += rescale(&mut data);
                stored.push(DTensor::new(labels.clone(), data));
            }
            Node::Pair {
                left,
                right,
                out,
                ..
            } => {
                let mut tensor = dense_contract(&stored[*left], &stored[*right], out, &net.dims);
                log_scale += rescale(&mut tensor.data);
                stored.push(tensor);
            }
        }
    }
    let root = &stored[tree.root];
    assert_eq!(root.labels.len(), 0, "the root must be a scalar");
    let log_z = log_scale + root.data[0].ln();
    (stored, log_z, log_scale)
}

/// Contract in min-plus arithmetic; returns `(E_min, g_0)`.
pub fn contract_tropical(terms: &[TTensor], tree: &Tree, dims: &[usize]) -> (f64, u64) {
    let mut stored: Vec<TTensor> = Vec::with_capacity(tree.nodes.len());
    for node in &tree.nodes {
        match node {
            Node::Leaf { term, .. } => stored.push(terms[*term].clone()),
            Node::Pair {
                left,
                right,
                out,
                ..
            } => {
                stored.push(tropical_contract(&stored[*left], &stored[*right], out, dims));
            }
        }
    }
    let root = &stored[tree.root];
    assert_eq!(root.labels.len(), 0, "the root must be a scalar");
    (root.energy[0], root.count[0])
}

/// Contract only the nonzero entries; one sparse tensor per node.
pub fn contract_sparse(net: &Network, tree: &Tree) -> Vec<STensor> {
    let mut stored: Vec<STensor> = Vec::with_capacity(tree.nodes.len());
    for node in &tree.nodes {
        match node {
            Node::Leaf { term, .. } => stored.push(STensor::from_term(&net.terms[*term])),
            Node::Pair {
                left,
                right,
                out,
                ..
            } => stored.push(sparse_contract(&stored[*left], &stored[*right], out)),
        }
    }
    stored
}

/// The summed labels of a node, in the canonical (ascending) order.
fn summed_labels(out: &[usize], in0: &[usize], in1: &[usize]) -> Labels {
    let mut summed: Labels = in0
        .iter()
        .chain(in1.iter())
        .filter(|l| !out.contains(l))
        .cloned()
        .collect();
    summed.sort_unstable();
    summed.dedup();
    summed
}

/// Draw one configuration by walking the stored tree from the root down.
pub fn sample_dense(
    tree: &Tree,
    stored: &[DTensor],
    nvars: usize,
    rng: &mut crate::rng::Rng,
) -> Vec<usize> {
    let mut values = vec![0usize; nvars];
    down_dense(tree, tree.root, stored, &mut values, rng);
    values
}

fn down_dense(
    tree: &Tree,
    index: usize,
    stored: &[DTensor],
    values: &mut [usize],
    rng: &mut crate::rng::Rng,
) {
    match &tree.nodes[index] {
        Node::Leaf { .. } => {}
        Node::Pair {
            left,
            right,
            out,
            in0,
            in1,
        } => {
            let summed = summed_labels(out, in0, in1);
            let size = 1usize << summed.len();
            let a = &stored[*left];
            let b = &stored[*right];
            let a_strides = strides(&a.labels.iter().map(|_| 2).collect::<Vec<_>>());
            let b_strides = strides(&b.labels.iter().map(|_| 2).collect::<Vec<_>>());
            let a_plan: Vec<(usize, Option<usize>, Option<usize>)> = a
                .labels
                .iter()
                .enumerate()
                .map(|(p, l)| {
                    (
                        a_strides[p],
                        out.iter().position(|x| x == l),
                        summed.iter().position(|x| x == l),
                    )
                })
                .collect();
            let b_plan: Vec<(usize, Option<usize>, Option<usize>)> = b
                .labels
                .iter()
                .enumerate()
                .map(|(p, l)| {
                    (
                        b_strides[p],
                        out.iter().position(|x| x == l),
                        summed.iter().position(|x| x == l),
                    )
                })
                .collect();
            let mut cumulative = Vec::with_capacity(size);
            let mut total = 0.0;
            for assignment in 0..size {
                let mut ia = 0usize;
                for &(stride, out_pos, sum_pos) in &a_plan {
                    let bit = match (out_pos, sum_pos) {
                        (Some(p), _) => values[out[p]],
                        (None, Some(p)) => (assignment >> p) & 1,
                        (None, None) => 0,
                    };
                    ia += bit * stride;
                }
                let mut ib = 0usize;
                for &(stride, out_pos, sum_pos) in &b_plan {
                    let bit = match (out_pos, sum_pos) {
                        (Some(p), _) => values[out[p]],
                        (None, Some(p)) => (assignment >> p) & 1,
                        (None, None) => 0,
                    };
                    ib += bit * stride;
                }
                total += a.data[ia] * b.data[ib];
                cumulative.push(total);
            }
            let pick = rng.draw(&cumulative);
            for (p, &label) in summed.iter().enumerate() {
                values[label] = (pick >> p) & 1;
            }
            down_dense(tree, *left, stored, values, rng);
            down_dense(tree, *right, stored, values, rng);
        }
    }
}

/// Draw one configuration from the sparse tree (uniform over the counted objects).
pub fn sample_sparse(
    tree: &Tree,
    stored: &[STensor],
    nvars: usize,
    rng: &mut crate::rng::Rng,
) -> Vec<usize> {
    let mut values = vec![0usize; nvars];
    down_sparse(tree, tree.root, stored, &mut values, rng);
    values
}

fn down_sparse(
    tree: &Tree,
    index: usize,
    stored: &[STensor],
    values: &mut [usize],
    rng: &mut crate::rng::Rng,
) {
    match &tree.nodes[index] {
        Node::Leaf { .. } => {}
        Node::Pair {
            left,
            right,
            out,
            in0,
            in1,
        } => {
            let summed = summed_labels(out, in0, in1);
            let a = &stored[*left];
            let b = &stored[*right];
            // Index b by the labels it shares with a, and drop entries that
            // contradict the values already drawn above this node.
            let shared: Labels = a.labels.iter().filter(|l| b.labels.contains(l)).cloned().collect();
            let a_shared: Vec<Option<usize>> = shared
                .iter()
                .map(|l| a.labels.iter().position(|x| x == l))
                .collect();
            let b_shared: Vec<Option<usize>> = shared
                .iter()
                .map(|l| b.labels.iter().position(|x| x == l))
                .collect();
            let fixed_ok = |labels: &[usize], key: u64| -> bool {
                labels.iter().enumerate().all(|(position, &label)| {
                    match out.iter().position(|x| *x == label) {
                        Some(p) => ((key >> position) & 1) as usize == values[out[p]],
                        None => true,
                    }
                })
            };
            let mut grouped: HashMap<u64, Vec<(u64, f64)>> = HashMap::new();
            for (&key, &value) in &b.map {
                if value == 0.0 || !fixed_ok(&b.labels, key) {
                    continue;
                }
                grouped
                    .entry(compress(key, &b_shared))
                    .or_default()
                    .push((key, value));
            }
            let out_source: Vec<(usize, Option<usize>, Option<usize>)> = summed
                .iter()
                .enumerate()
                .map(|(p, l)| {
                    (
                        p,
                        a.labels.iter().position(|x| x == l),
                        b.labels.iter().position(|x| x == l),
                    )
                })
                .collect();
            let mut distribution: BTreeMap<u64, f64> = BTreeMap::new();
            for (&key, &value) in &a.map {
                if value == 0.0 || !fixed_ok(&a.labels, key) {
                    continue;
                }
                let Some(list) = grouped.get(&compress(key, &a_shared)) else {
                    continue;
                };
                for &(b_key, b_value) in list {
                    let mut summed_key = 0u64;
                    for &(p, pa, pb) in &out_source {
                        let bit = match (pa, pb) {
                            (Some(pos), _) => (key >> pos) & 1,
                            (None, Some(pos)) => (b_key >> pos) & 1,
                            (None, None) => continue,
                        };
                        summed_key |= bit << p;
                    }
                    *distribution.entry(summed_key).or_insert(0.0) += value * b_value;
                }
            }
            let mut cumulative = Vec::with_capacity(distribution.len());
            let mut total = 0.0;
            for &weight in distribution.values() {
                total += weight;
                cumulative.push(total);
            }
            assert!(total > 0.0, "sample rejected: no state reaches this node");
            let pick = rng.draw(&cumulative);
            let summed_key = *distribution.keys().nth(pick).expect("categorical index");
            for (p, &label) in summed.iter().enumerate() {
                values[label] = ((summed_key >> p) & 1) as usize;
            }
            down_sparse(tree, *left, stored, values, rng);
            down_sparse(tree, *right, stored, values, rng);
        }
    }
}

// ---------------------------------------------------------------------------
// The sheet's two kinds of network
// ---------------------------------------------------------------------------

/// A magnet in the sheet's lattice convention `H = -sum J_ij s_i s_j`, rewritten as
/// `E = sum K_ij s_i s_j + sum h_i s_i` with `K_ij = -J_ij` and no fields, which is
/// also the `pred` bundle convention, so all arithmetic below is one convention.
pub struct Magnet {
    pub nspins: usize,
    pub bonds: Vec<(usize, usize, f64)>,
    pub fields: Vec<f64>,
    pub source: String,
}

impl Magnet {
    /// An open `l x l` square; each bond is `J = -1` with probability `p_af`.
    /// Bonds are drawn in row-major site order, the horizontal one first, one
    /// uniform per bond from the documented splitmix64 stream.
    pub fn open_lattice(l: usize, p_af: f64, seed: u64) -> Self {
        use crate::rng::{Rng, STREAM_BONDS};
        let mut rng = Rng::new(seed, STREAM_BONDS);
        let mut bonds = Vec::new();
        for r in 0..l {
            for c in 0..l {
                let i = r * l + c;
                if c + 1 < l {
                    let j_ij = if rng.next_f64() < p_af { -1.0 } else { 1.0 };
                    bonds.push((i, r * l + c + 1, -j_ij));
                }
                if r + 1 < l {
                    let j_ij = if rng.next_f64() < p_af { -1.0 } else { 1.0 };
                    bonds.push((i, (r + 1) * l + c, -j_ij));
                }
            }
        }
        Magnet {
            nspins: l * l,
            bonds,
            fields: vec![0.0; l * l],
            source: format!("open {l}x{l} lattice, p_af = {p_af}, seed = {seed}, H = -sum J s s"),
        }
    }

    /// A `pred` SpinGlass bundle: `E = sum_i<j K_ij s_i s_j + sum_i h_i s_i`.
    pub fn from_bundle(value: &serde_json::Value) -> Result<Self, String> {
        let data = value
            .get("target")
            .and_then(|t| t.get("data"))
            .ok_or("bundle has no target data")?;
        let couplings = data
            .get("couplings")
            .and_then(|c| c.as_array())
            .ok_or("bundle has no couplings")?;
        let fields = data
            .get("fields")
            .and_then(|f| f.as_array())
            .ok_or("bundle has no fields")?;
        let graph = data.get("graph").ok_or("bundle has no graph")?;
        let edges = graph
            .get("edges")
            .and_then(|e| e.as_array())
            .ok_or("bundle graph has no edges")?;
        if edges.len() != couplings.len() {
            return Err("bundle couplings and edges differ in length".into());
        }
        let nspins = graph
            .get("num_vertices")
            .and_then(|n| n.as_u64())
            .ok_or("bundle graph has no num_vertices")? as usize;
        let mut bonds = Vec::with_capacity(edges.len());
        for (edge, coupling) in edges.iter().zip(couplings.iter()) {
            let pair = edge.as_array().ok_or("bundle edge is not a pair")?;
            let i = pair[0].as_u64().ok_or("edge end is not an integer")? as usize;
            let j = pair[1].as_u64().ok_or("edge end is not an integer")? as usize;
            let k = coupling.as_f64().ok_or("coupling is not a number")?;
            if k != 0.0 {
                bonds.push((i, j, k));
            }
        }
        let mut h = Vec::with_capacity(fields.len());
        for field in fields {
            h.push(field.as_f64().ok_or("field is not a number")?);
        }
        if h.len() != nspins {
            return Err("bundle field count differs from the spin count".into());
        }
        Ok(Magnet {
            nspins,
            bonds,
            fields: h,
            source: "pred SpinGlass bundle, E = +sum K s s + sum h s".to_string(),
        })
    }

    /// `E = sum K_ij s_i s_j + sum h_i s_i` of a bit assignment (`x = 1` is `s = +1`).
    pub fn energy(&self, bits: &[usize]) -> f64 {
        let mut e = 0.0;
        for &(i, j, k) in &self.bonds {
            let si = 2.0 * bits[i] as f64 - 1.0;
            let sj = 2.0 * bits[j] as f64 - 1.0;
            e += k * si * sj;
        }
        for (i, &h) in self.fields.iter().enumerate() {
            e += h * (2.0 * bits[i] as f64 - 1.0);
        }
        e
    }

    /// The Boltzmann network at `beta` together with the same terms as
    /// `(energy, count)` pairs, built in one pass so the two agree term by term:
    /// a bond contributes `K_ij s_i s_j` and a nonzero field contributes `h_i s_i`.
    pub fn networks(&self, beta: f64) -> (Network, Vec<TTensor>) {
        let mut net = Network::new(self.nspins);
        let mut tropical: Vec<TTensor> = Vec::new();
        for &(i, j, k) in &self.bonds {
            let mut values = vec![0.0; 4];
            let mut energy = vec![0.0; 4];
            for a in 0..2usize {
                for b in 0..2usize {
                    let sa = 2.0 * a as f64 - 1.0;
                    let sb = 2.0 * b as f64 - 1.0;
                    values[a * 2 + b] = (-beta * k * sa * sb).exp();
                    energy[a * 2 + b] = k * sa * sb;
                }
            }
            net.push(vec![i, j], values);
            tropical.push(TTensor {
                labels: vec![i, j],
                energy,
                count: vec![1; 4],
            });
        }
        for (i, &h) in self.fields.iter().enumerate() {
            if h == 0.0 {
                continue;
            }
            net.push(vec![i], vec![(-beta * h * -1.0).exp(), (-beta * h).exp()]);
            tropical.push(TTensor {
                labels: vec![i],
                energy: vec![-h, h],
                count: vec![1; 2],
            });
        }
        (net, tropical)
    }
}

// Carry-tensor rules: a carry tensor is 1 at (0,0,0), (0,1,1), (1,0,1) and 0 elsewhere.
const FULL_ROWS: [[usize; 3]; 3] = [[0, 0, 0], [0, 1, 1], [1, 0, 1]];
const START_ROWS: [[usize; 2]; 2] = [[0, 0], [1, 1]];
const END_ROWS: [[usize; 2]; 2] = [[0, 0], [1, 1]];

/// Build one 0/1 rule tensor: `order` gives the variable of every column of the
/// row set, so the tensor is 1 exactly on the rows listed and 0 elsewhere. Entries
/// are laid out row-major over the sorted labels, the first label being the most
/// significant index.
fn rule_tensor(order: &[usize], rows: &[Vec<usize>]) -> (Labels, Vec<f64>) {
    let mut labels = order.to_vec();
    labels.sort_unstable();
    let mut values = vec![0.0; 1usize << order.len()];
    for row in rows {
        let mut key = 0usize;
        for (p, &label) in labels.iter().enumerate() {
            let source = order.iter().position(|&x| x == label).expect("label not in row");
            key |= row[source] << (order.len() - 1 - p);
        }
        values[key] = 1.0;
    }
    (labels, values)
}

fn rows3(rows: &[[usize; 3]; 3]) -> Vec<Vec<usize>> {
    rows.iter().map(|r| r.to_vec()).collect()
}

fn rows2(rows: &[[usize; 2]; 2]) -> Vec<Vec<usize>> {
    rows.iter().map(|r| r.to_vec()).collect()
}

/// The N-queens carry network: one bit per square, four carry tensors per square.
pub struct Queens {
    pub n: usize,
    pub net: Network,
    pub square_bits: Vec<usize>,
}

impl Queens {
    pub fn build(n: usize) -> Self {
        assert!(n >= 2, "the carry construction needs a board of at least 2 x 2");
        let mut nvars = n * n;
        let mut terms: Vec<(Labels, Vec<f64>)> = Vec::new();
        let square_bits: Vec<usize> = (0..n * n).collect();
        let new_var = |nvars: &mut usize| -> usize {
            let v = *nvars;
            *nvars += 1;
            v
        };

        // A full carry tensor over (c_in, x, c_out), the first and last rank-2
        // variants that pin an end carry, and the single-square variant.
        let full = |cin: usize, x: usize, cout: usize| {
            rule_tensor(&[cin, x, cout], &rows3(&FULL_ROWS))
        };
        let start = |x: usize, cout: usize| rule_tensor(&[x, cout], &rows2(&START_ROWS));
        let end = |cin: usize, x: usize| rule_tensor(&[cin, x], &rows2(&END_ROWS));

        for r in 0..n {
            let mut carry: Option<usize> = None;
            for c in 0..n {
                let x = square_bits[r * n + c];
                let last = c + 1 == n;
                if c == 0 && last {
                    terms.push((vec![x], vec![0.0, 1.0]));
                    continue;
                }
                let cout = if last { None } else { Some(new_var(&mut nvars)) };
                if c == 0 {
                    terms.push(start(x, cout.expect("start of a longer row")));
                } else if last {
                    terms.push(end(carry.expect("carry into the last square"), x));
                } else {
                    terms.push(full(carry.expect("carry into the square"), x, cout.unwrap()));
                }
                carry = cout;
            }
            let _ = carry;
        }

        for c in 0..n {
            let mut carry: Option<usize> = None;
            for r in 0..n {
                let x = square_bits[r * n + c];
                let last = r + 1 == n;
                if r == 0 && last {
                    terms.push((vec![x], vec![0.0, 1.0]));
                    continue;
                }
                let cout = if last { None } else { Some(new_var(&mut nvars)) };
                if r == 0 {
                    terms.push(start(x, cout.expect("start of a longer column")));
                } else if last {
                    terms.push(end(carry.expect("carry into the last square"), x));
                } else {
                    terms.push(full(carry.expect("carry into the square"), x, cout.unwrap()));
                }
                carry = cout;
            }
            let _ = carry;
        }

        // Diagonals: carry 0 at the start, free at the end (the last c_out is summed).
        for dir in 0..2 {
            for line in diagonal_lines(n, dir) {
                if line.len() == 1 {
                    let (r, c) = line[0];
                    let x = square_bits[r * n + c];
                    let cout = new_var(&mut nvars);
                    terms.push(start(x, cout));
                    continue;
                }
                let mut carry: Option<usize> = None;
                for (k, &(r, c)) in line.iter().enumerate() {
                    let x = square_bits[r * n + c];
                    let cout = new_var(&mut nvars);
                    if k == 0 {
                        terms.push(start(x, cout));
                    } else {
                        terms.push(full(carry.expect("carry into the diagonal square"), x, cout));
                    }
                    carry = Some(cout);
                }
            }
        }

        let mut net = Network::new(nvars);
        for (labels, values) in terms {
            net.push(labels, values);
        }
        Queens {
            n,
            net,
            square_bits,
        }
    }

    /// The column of the queen in every row, from a bit assignment of the squares.
    pub fn board(&self, values: &[usize]) -> Option<Vec<usize>> {
        let mut board = Vec::with_capacity(self.n);
        for r in 0..self.n {
            let mut column = None;
            for c in 0..self.n {
                if values[self.square_bits[r * self.n + c]] == 1 {
                    if column.is_some() {
                        return None;
                    }
                    column = Some(c);
                }
            }
            board.push(column?);
        }
        Some(board)
    }
}

fn diagonal_lines(n: usize, dir: usize) -> Vec<Vec<(usize, usize)>> {    let mut lines = Vec::new();
    if dir == 0 {
        for start in -(n as isize - 1)..(n as isize) {
            let mut line = Vec::new();
            for r in 0..n {
                let c = r as isize + start;
                if (0..n as isize).contains(&c) {
                    line.push((r, c as usize));
                }
            }
            lines.push(line);
        }
    } else {
        for start in 0..(2 * n - 1) {
            let mut line = Vec::new();
            for r in 0..n {
                let c = start as isize - r as isize;
                if (0..n as isize).contains(&c) {
                    line.push((r, c as usize));
                }
            }
            lines.push(line);
        }
    }
    lines
}

/// The queen rules on a board given as the queen's column in every row: one
/// queen per row (by construction), one per column, at most one per diagonal.
pub fn valid_queens(board: &[usize]) -> bool {
    let n = board.len();
    let mut columns = vec![false; n];
    let mut diagonals = std::collections::HashSet::new();
    let mut anti = std::collections::HashSet::new();
    for (row, &column) in board.iter().enumerate() {
        if column >= n || columns[column] {
            return false;
        }
        columns[column] = true;
        let d = row as isize - column as isize;
        let a = row as isize + column as isize;
        if !diagonals.insert(d) || !anti.insert(a) {
            return false;
        }
    }
    true
}

