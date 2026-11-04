"""Loopy belief propagation: convergence is not accuracy."""
from __future__ import annotations

import json

import numpy as np
import pytest

import reference
from conftest import CASES

DAMPING = 0.5
TOLERANCE = 1e-10
MAX_SWEEPS = 1000


@pytest.fixture(scope='session')
def bp_runs(infer) -> dict[str, dict]:
    runs = {}
    for case in CASES:
        out = reference.ROOT / 'artifacts' / f'bp-{case}'
        infer('--model', 'asia.uai', '--evidence', case, '--method', 'bp',
              '--damping', str(DAMPING), '--tolerance', str(TOLERANCE),
              '--max-sweeps', str(MAX_SWEEPS), '--out', str(out))
        runs[case] = json.loads((out / 'run.json').read_text())
    return runs


def test_settings_are_recorded(bp_runs):
    for case, run in bp_runs.items():
        assert run['method'] == 'bp'
        assert run['settings'] == {'damping': DAMPING, 'tolerance': TOLERANCE, 'max_sweeps': MAX_SWEEPS}
        assert run['evidence']['observations'] == {str(k): v for k, v in CASES[case].items()}


def test_all_cases_converge(bp_runs):
    for case, run in bp_runs.items():
        final = run['final']
        assert final['converged'] is True, case
        assert final['residual'] <= TOLERANCE, (case, final['residual'])
        assert 3 <= final['sweeps'] <= MAX_SWEEPS, (case, final['sweeps'])
        assert len(run['sweeps']) == final['sweeps']


def test_residual_and_work_are_traced(bp_runs):
    for case, run in bp_runs.items():
        sweeps = run['sweeps']
        residuals = [s['residual'] for s in sweeps]
        work = [s['cumulative_work'] for s in sweeps]
        assert residuals[0] > residuals[-1], case
        assert all(b >= a for a, b in zip(work, work[1:])), case
        assert work[-1] == pytest.approx(run['final']['cumulative_work'])
        for record in sweeps:
            for variable, values in record['beliefs'].items():
                assert np.asarray(values).sum() == pytest.approx(1.0, abs=1e-12), (case, variable)


def test_adx_is_converged_but_inaccurate(bp_runs):
    run = bp_runs['ADX']
    final = run['final']
    assert final['residual'] < 5e-10
    error = run['bp']['max_node_error']
    assert 0.005 < error < 0.02, error
    assert error == pytest.approx(0.0106, abs=3e-3)
    assert run['exact']['method'] == 'greedy'
    assert len(run['bp']['node_errors']) == 8


def test_converged_loopy_beliefs_are_reproducible(infer, bp_runs):
    out = reference.ROOT / 'artifacts' / 'bp-ADX-repeat'
    infer('--model', 'asia.uai', '--evidence', 'ADX', '--method', 'bp',
          '--damping', str(DAMPING), '--tolerance', str(TOLERANCE),
          '--max-sweeps', str(MAX_SWEEPS), '--out', str(out))
    repeat = json.loads((out / 'run.json').read_text())
    for variable, values in bp_runs['ADX']['final']['beliefs'].items():
        np.testing.assert_allclose(values, repeat['final']['beliefs'][variable], rtol=0, atol=1e-15)


def test_bp_is_exact_on_a_tree(infer, tree_model):
    """A nonuniform tree separates the algorithm from the approximation error."""
    exact_out = reference.ROOT / 'artifacts' / 'tree' / 'exact'
    bp_out = reference.ROOT / 'artifacts' / 'tree' / 'bp'
    infer('--model', str(tree_model), '--out', str(exact_out))
    infer('--model', str(tree_model), '--method', 'bp', '--damping', '0.5',
          '--tolerance', '1e-14', '--max-sweeps', '500', '--out', str(bp_out))
    exact = json.loads((exact_out / 'run.json').read_text())
    bp = json.loads((bp_out / 'run.json').read_text())

    model = reference.read_uai(tree_model)
    expected = reference.enumerate_query(model, {})
    assert exact['contraction']['mass'] == pytest.approx(expected.mass, rel=1e-12)
    assert bp['final']['converged'] is True
    for variable in range(model.variables):
        want = expected.marginals[variable]
        np.testing.assert_allclose(exact['contraction']['node_marginals'][str(variable)], want,
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(bp['final']['beliefs'][str(variable)], want, rtol=0, atol=1e-12)
    assert bp['bp']['max_node_error'] <= 1e-12
