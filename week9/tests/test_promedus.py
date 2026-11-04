"""Promedus_14: exact contraction at scale and an unconverged BP answer."""
from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

import reference
from conftest import ROOT

MODEL = ROOT / 'promedus' / 'Promedus_14.uai'
QUERIES = (59, 102, 139)
PR = -7.806746304582435
MAR_REFERENCE = {59: 0.1975882582, 102: 0.1497534323, 139: 0.6175220184}


@pytest.fixture(scope='session')
def prom_model() -> reference.Model:
    return reference.read_uai(MODEL)


@pytest.fixture(scope='session')
def exact_runs(infer) -> dict[str, dict]:
    runs = {}
    for method in ('greedy', 'treesa'):
        out = ROOT / 'artifacts' / 'promedus' / method
        infer('--model', str(MODEL), '--query', ','.join(map(str, QUERIES)),
              '--order', method, '--out', str(out))
        runs[method] = json.loads((out / 'run.json').read_text())
    return runs


@pytest.fixture(scope='session')
def bp_run(infer) -> dict:
    out = ROOT / 'artifacts' / 'promedus' / 'bp'
    infer('--model', str(MODEL), '--query', ','.join(map(str, QUERIES)),
          '--order', 'treesa', '--method', 'bp', '--damping', '0.5',
          '--tolerance', '1e-10', '--max-sweeps', '1000', '--out', str(out))
    return json.loads((out / 'run.json').read_text())


def test_model_loads_with_the_nine_observations(exact_runs, prom_model):
    run = exact_runs['greedy']
    assert prom_model.variables == 414
    assert run['model']['variables'] == 414
    observations = reference.read_evid(MODEL.with_name(MODEL.name + '.evid'))
    assert observations == {41: 1, 47: 1, 26: 1, 29: 1, 28: 1, 14: 1, 123: 1, 4: 1, 147: 1}
    assert run['evidence']['observations'] == {str(k): v for k, v in observations.items()}


def test_exact_evidence_matches_the_published_pr(exact_runs):
    pr = reference.read_pr(MODEL.with_name(MODEL.name + '.PR'))
    assert pr == pytest.approx(-7.80675)
    for method, run in exact_runs.items():
        log10 = run['contraction']['log10_mass']
        assert log10 == pytest.approx(PR, abs=1e-9), (method, log10)
        assert abs(log10 - pr) < 1e-4, (method, log10, pr)
        assert run['contraction']['mass'] == pytest.approx(10.0 ** PR, rel=1e-9)


def test_exact_marginals_match_the_published_mar(exact_runs):
    mar = reference.read_mar(MODEL.with_name(MODEL.name + '.MAR'))
    for variable in QUERIES:
        assert mar[variable][1] == pytest.approx(MAR_REFERENCE[variable], abs=1e-5)
    for method, run in exact_runs.items():
        for variable in QUERIES:
            got = run['contraction']['node_marginals'][str(variable)][1]
            assert got == pytest.approx(MAR_REFERENCE[variable], abs=1e-6), (method, variable, got)
            assert got == pytest.approx(mar[variable][1], abs=1e-5), (method, variable, got)


def test_the_two_exact_plans_agree_and_treesa_is_cheaper(exact_runs):
    greedy, treesa = exact_runs['greedy'], exact_runs['treesa']
    assert greedy['contraction']['log10_mass'] == pytest.approx(treesa['contraction']['log10_mass'],
                                                               abs=1e-10)
    for variable in QUERIES:
        assert greedy['contraction']['node_marginals'][str(variable)] == pytest.approx(
            treesa['contraction']['node_marginals'][str(variable)], abs=1e-10)
    assert treesa['ordering']['work'] < greedy['ordering']['work']
    for run in (greedy, treesa):
        assert run['ordering']['plan'] == 'scalar'
        assert run['ordering']['largest_entries'] >= 2
        assert run['runtime_seconds'] > 0


def test_belief_propagation_is_unconverged_and_inaccurate(bp_run):
    final = bp_run['final']
    assert final['sweeps'] == 1000
    assert final['converged'] is False
    assert final['residual'] > 1e-10
    assert bp_run['bp']['max_node_error'] > 0.1
    for variable in QUERIES:
        assert bp_run['bp']['node_errors'][str(variable)] > 0.05


def test_without_observations_the_partition_sum_moves(infer, exact_runs):
    """Negative control: a lost evidence file must fail the PR comparison."""
    directory = ROOT / 'artifacts' / 'promedus-noevid'
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(MODEL, directory / 'Promedus_14.uai')
    out = ROOT / 'artifacts' / 'promedus' / 'noevid'
    infer('--model', str(directory / 'Promedus_14.uai'), '--query', '59',
          '--order', 'treesa', '--out', str(out))
    run = json.loads((out / 'run.json').read_text())
    assert run['evidence']['observations'] == {}
    assert abs(run['contraction']['log10_mass'] - PR) > 1e-2
    assert abs(exact_runs['treesa']['contraction']['log10_mass'] - PR) < 1e-4
