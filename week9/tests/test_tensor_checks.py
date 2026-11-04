"""Library checks printed by the inference build: matrices, the lb network, Ising."""
from __future__ import annotations

import numpy as np
import pytest

import reference


def check_arrays(actual, expected, atol=1e-12):
    np.testing.assert_allclose(np.asarray(actual, dtype=float), np.asarray(expected, dtype=float),
                               rtol=0, atol=atol)


def test_matrix_product_and_trace(asia):
    checks = asia['prior']['checks']
    matrix = checks['matrix']
    check_arrays(matrix['product'], reference.matrix_product())
    check_arrays(matrix['product_direct'], reference.matrix_product())
    check_arrays(matrix['product'][0], [4.0, 4.0])
    check_arrays(matrix['product'][1], [10.0, 8.0])
    assert matrix['trace'] == pytest.approx(reference.matrix_trace(), abs=1e-12)
    assert matrix['trace_direct'] == pytest.approx(12.0, abs=1e-12)
    assert matrix['total'] == pytest.approx(26.0, abs=1e-12)
    assert matrix['total_direct'] == pytest.approx(26.0, abs=1e-12)
    for key in ('product_error', 'trace_error', 'total_error'):
        assert matrix[key] <= 1e-12, (key, matrix[key])
    assert matrix['tolerance'] == pytest.approx(1e-12)


def test_lung_bronchitis_network(asia):
    example = asia['prior']['checks']['lung_bronchitis']
    expected = reference.lung_bronchitis_example()
    check_arrays(example['joint'], expected, atol=1e-15)
    check_arrays(example['joint_direct'], expected, atol=1e-15)
    check_arrays(expected, [[0.0315, 0.0235], [0.4185, 0.5265]], atol=1e-15)
    assert example['max_abs_error'] <= 1e-15
    assert example['normalization'] == pytest.approx(1.0, abs=1e-15)


def test_ising_ring_matches_enumeration(asia):
    ising = asia['prior']['checks']['ising']
    assert ising['beta'] == 0.5 and ising['coupling'] == 1.0
    assert ising['contraction'] == pytest.approx(reference.ising_ring(), rel=0, abs=1e-12)
    assert ising['contraction'] == pytest.approx(12.60256209895193, abs=1e-12)
    assert ising['enumeration'] == pytest.approx(ising['contraction'], abs=1e-12)
    assert ising['max_abs_error'] <= 1e-12
    assert ising['weights'] == 8


def test_checks_are_identical_across_evidence_cases(asia):
    priors = asia['prior']['checks']
    for case in ('AD', 'ADX'):
        other = asia[case]['checks']
        assert other['matrix'] == priors['matrix']
        assert other['ising'] == priors['ising']
