//! Development checks that do not need the CLI: sparse and dense contraction must
//! agree, the documented generator must be reproducible, and the draw machinery
//! must be a pure function of the seed.

use tnsample::net::{contract_dense, contract_sparse, plan, Network, Order, Queens};
use tnsample::rng::{Rng, STREAM_SAMPLE};

#[test]
fn sparse_matches_dense_and_counts_queens() {
    let known = [0usize, 1, 0, 0, 2, 10, 4];
    for n in 2..=6usize {
        let queens = Queens::build(n);
        let p = plan(&queens.net, Order::Treesa);
        let stored = contract_sparse(&queens.net, &p.tree);
        let sparse = *stored[p.tree.root].map.get(&0).unwrap_or(&0.0);
        let (_, log_z, _) = contract_dense(&queens.net, &p.tree);
        let dense = log_z.exp();
        assert!((sparse - dense).abs() < 1e-6, "n = {n}: {sparse} vs {dense}");
        assert_eq!(sparse, known[n] as f64, "n = {n} count");
    }
}

#[test]
fn every_sampled_six_queens_board_is_a_solution() {
    let queens = Queens::build(6);
    let p = plan(&queens.net, Order::Greedy);
    let stored = contract_sparse(&queens.net, &p.tree);
    let mut rng = Rng::new(2026, STREAM_SAMPLE);
    for _ in 0..64 {
        let values = tnsample::net::sample_sparse(&p.tree, &stored, queens.net.dims.len(), &mut rng);
        let board = queens.board(&values).expect("one queen per row");
        assert!(tnsample::net::valid_queens(&board), "{board:?}");
    }
}

#[test]
fn the_generator_is_a_pure_function_of_the_seed() {
    let mut first = Rng::new(2026, STREAM_SAMPLE);
    let mut second = Rng::new(2026, STREAM_SAMPLE);
    let mut other = Rng::new(2027, STREAM_SAMPLE);
    let a: Vec<u64> = (0..8).map(|_| first.next_u64()).collect();
    let b: Vec<u64> = (0..8).map(|_| second.next_u64()).collect();
    let c: Vec<u64> = (0..8).map(|_| other.next_u64()).collect();
    assert_eq!(a, b);
    assert_ne!(a, c);
    // The uniforms are inside (0, 1) and the categorical draw follows the weights.
    let mut rng = Rng::new(2026, STREAM_SAMPLE);
    for _ in 0..1000 {
        let value = rng.next_f64();
        assert!(value > 0.0 && value < 1.0, "{value}");
    }
    let mut first = Rng::new(7, STREAM_SAMPLE);
    let mut second = Rng::new(7, STREAM_SAMPLE);
    let cumulative = [0.25, 0.5, 1.0];
    for _ in 0..100 {
        assert_eq!(first.draw(&cumulative), second.draw(&cumulative));
    }
}

#[test]
fn an_empty_network_contracts_to_one() {
    // A single constant tensor is the identity of the product.
    let mut net = Network::new(0);
    net.push(vec![], vec![1.0]);
    let p = plan(&net, Order::Greedy);
    let stored = contract_sparse(&net, &p.tree);
    assert_eq!(*stored[p.tree.root].map.get(&0).unwrap(), 1.0);
}
