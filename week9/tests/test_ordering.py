"""Contraction orders change the cost, never the probability."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

import reference
from conftest import CASES

TOLERANCE = 1e-10


def recount(steps, model):
    """Independent recount of work and largest tensor from the saved step list."""
    dims = list(model.dims)
    largest = max(int(np.prod(factor.dims)) for factor in model.factors)
    work = 0.0
    for step in steps:
        left_labels = frozenset(step['left_labels'])
        right_labels = frozenset(step['right_labels'])
        step_work = reference.pair_work(left_labels, right_labels, dims)
        assert step_work == pytest.approx(step['work'], rel=1e-12), step
        work += step_work
        entries = int(np.prod([dims[l] for l in sorted(step['labels'])])) if step['labels'] else 1
        assert entries == step['entries'], step
        largest = max(largest, entries)
    return work, largest


@pytest.mark.parametrize('method,seed', [('greedy', None), ('random', 42), ('exhaustive', None)])
def test_plan_keeps_the_answer(method, seed, infer, asia_model):
    out = reference.ROOT / 'artifacts' / 'runs' / method
    args = ['--model', 'asia.uai', '--evidence', 'AD', '--order', method, '--out', str(out)]
    if seed is not None:
        args += ['--seed', str(seed)]
    infer(*args)
    run = json.loads((out / 'run.json').read_text())
    order = json.loads((out / 'order.json').read_text())
    complexity = json.loads((out / 'complexity.json').read_text())

    expected = reference.enumerate_query(asia_model, CASES['AD'])
    assert run['contraction']['mass'] == pytest.approx(expected.mass, rel=TOLERANCE)
    assert order['method'] == method
    if seed is None:
        assert order.get('seed') is None
    else:
        assert order['seed'] == seed
    assert len(order['steps']) == len(asia_model.factors) - 1
    assert order['leaves'] == len(asia_model.factors)

    work, largest = recount(order['steps'], asia_model)
    assert complexity['work'] == pytest.approx(work, rel=1e-12)
    assert complexity['largest_entries'] == largest
    assert complexity['query']['labels'] == []


def test_random_and_greedy_share_the_probability_at_different_costs(infer):
    costs = {}
    for method in ('random', 'greedy', 'exhaustive'):
        out = reference.ROOT / 'artifacts' / 'runs' / method
        infer('--model', 'asia.uai', '--evidence', 'AD', '--order', method,
              *(['--seed', '42'] if method == 'random' else []), '--out', str(out))
        costs[method] = json.loads((out / 'complexity.json').read_text())
        run = json.loads((out / 'run.json').read_text())
        assert run['contraction']['mass_relative_error'] <= TOLERANCE
    assert costs['greedy']['work'] <= costs['random']['work']
    assert costs['random']['work'] != costs['greedy']['work']
    assert costs['greedy']['largest_entries'] <= costs['random']['largest_entries']
    assert costs['exhaustive']['work'] <= costs['greedy']['work']
    for method in costs:
        assert costs[method]['largest_entries'] >= 8


@pytest.mark.parametrize('case,optimum', [('prior', 44), ('AD', 50), ('ADX', 52)])
def test_exhaustive_optimum_matches_the_cost_model(case, optimum, infer, asia_model):
    scopes = [f.scope for f in asia_model.factors] + [(v,) for v in sorted(CASES[case])]
    assert reference.optimal_work(scopes, asia_model.dims) == optimum
    out = reference.ROOT / 'artifacts' / 'runs' / f'exhaustive-{case}'
    infer('--model', 'asia.uai', '--evidence', case, '--order', 'exhaustive', '--out', str(out))
    complexity = json.loads((out / 'complexity.json').read_text())
    assert complexity['work'] == optimum


def test_replay_reproduces_the_plan(infer, asia_model):
    greedy = reference.ROOT / 'artifacts' / 'runs' / 'greedy'
    replay = reference.ROOT / 'artifacts' / 'runs' / 'replay'
    infer('--model', 'asia.uai', '--evidence', 'AD', '--plan', str(greedy / 'order.json'),
          '--out', str(replay))
    replay_run = json.loads((replay / 'run.json').read_text())
    greedy_run = json.loads((greedy / 'run.json').read_text())
    assert replay_run['contraction']['mass'] == pytest.approx(greedy_run['contraction']['mass'], rel=1e-14)
    replayed = json.loads((replay / 'order.json').read_text())
    source = greedy / 'order.json'
    assert replayed['method'] == 'replay'
    assert replayed['source_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert replayed['steps'] == json.loads(source.read_text())['steps']


def test_analysis_writes_costs_without_inference(infer, asia_model):
    random_out = reference.ROOT / 'artifacts' / 'runs' / 'random'
    analysis = reference.ROOT / 'artifacts' / 'runs' / 'analysis'
    infer('--model', 'asia.uai', '--evidence', 'AD', '--analyze', str(random_out / 'order.json'),
          '--out', str(analysis))
    complexity = json.loads((analysis / 'complexity.json').read_text())
    steps = json.loads((random_out / 'order.json').read_text())['steps']
    work, largest = recount(steps, asia_model)
    assert complexity['work'] == pytest.approx(work, rel=1e-12)
    assert complexity['largest_entries'] == largest
    assert not (analysis / 'run.json').exists(), 'analysis must not run inference'


def test_seed_changes_the_random_plan(infer):
    plans = []
    for seed in (1, 2):
        out = reference.ROOT / 'artifacts' / 'runs' / f'random-{seed}'
        infer('--model', 'asia.uai', '--evidence', 'AD', '--order', 'random',
              '--seed', str(seed), '--out', str(out))
        plans.append(json.loads((out / 'order.json').read_text())['steps'])
    assert plans[0] != plans[1]
