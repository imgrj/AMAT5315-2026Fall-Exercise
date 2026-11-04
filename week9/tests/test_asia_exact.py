"""Exact ASIA inference: enumeration and contraction must agree to 1e-10."""
from __future__ import annotations

import numpy as np
import pytest

import reference
from conftest import CASES

TOLERANCE = 1e-10


def marginal(run, variable):
    return np.asarray(run['contraction']['node_marginals'][str(variable)])


def joint(run, key='1,3'):
    return np.asarray(run['contraction']['joints'][key])


@pytest.mark.parametrize('case', list(CASES))
def test_contraction_matches_enumeration(case, asia, asia_model):
    run = asia[case]
    expected = reference.enumerate_query(asia_model, CASES[case], joints=[(1, 3)])
    assert run['evidence']['label'] == case
    assert run['contraction']['mass'] == pytest.approx(expected.mass, rel=TOLERANCE)
    assert run['enumeration']['mass'] == pytest.approx(expected.mass, rel=1e-15)
    assert run['contraction']['mass_relative_error'] <= TOLERANCE
    assert run['contraction']['max_marginal_error'] <= TOLERANCE
    for variable in range(asia_model.variables):
        got = marginal(run, variable)
        assert got.sum() == pytest.approx(1.0, abs=TOLERANCE)
        np.testing.assert_allclose(got, expected.marginals[variable], rtol=0, atol=TOLERANCE)
        np.testing.assert_allclose(got, run['enumeration']['node_marginals'][str(variable)],
                                   rtol=0, atol=TOLERANCE)
    np.testing.assert_allclose(joint(run), expected.joints[(1, 3)], rtol=0, atol=TOLERANCE)
    assert joint(run).sum() == pytest.approx(1.0, abs=TOLERANCE)


def test_evidence_masses(asia):
    assert asia['prior']['contraction']['mass'] == pytest.approx(1.0, rel=1e-12)
    assert asia['AD']['contraction']['mass'] == pytest.approx(0.004501375, rel=1e-12)
    assert asia['ADX']['contraction']['mass'] == pytest.approx(0.00098822675, rel=1e-12)


def test_disease_probabilities(asia):
    prior = asia['prior']['contraction']['node_marginals']
    assert prior['1'][0] == pytest.approx(0.0104, abs=1e-12)
    assert prior['6'][0] == pytest.approx(0.11029004, abs=1e-12)
    ad = asia['AD']['contraction']['node_marginals']
    assert ad['1'][0] == pytest.approx(0.0877509649829, abs=1e-10)
    assert ad['3'][0] == pytest.approx(0.0995251450946, abs=1e-10)
    early = asia['ADX']['contraction']['node_marginals']
    assert early['1'][0] == pytest.approx(0.391711720008, abs=1e-10)
    assert early['3'][0] == pytest.approx(0.444270507755, abs=1e-10)
    assert early['4'][0] == pytest.approx(0.628821775974, abs=1e-10)
    # The three observed variables are certain.
    for variable in (0, 6, 7):
        assert early[str(variable)] == pytest.approx([1.0, 0.0], abs=1e-12)


def test_joint_keeps_the_dependence(asia):
    run = asia['ADX']
    values = joint(run)
    assert values[0, 0] == pytest.approx(0.02221353, abs=1e-8)
    product = np.outer(marginal(run, 1), marginal(run, 3))
    assert product[0, 0] == pytest.approx(0.17402596, abs=1e-8)
    dependence = np.asarray(run['contraction']['dependence']['1,3'])
    np.testing.assert_allclose(dependence, values - product, rtol=0, atol=1e-12)
    assert dependence[0, 0] == pytest.approx(-0.15181244, abs=1e-8)
    assert abs(dependence).max() > 0.1
    ad = asia['AD']
    assert abs(np.asarray(ad['contraction']['dependence']['1,3'])).max() > 1e-3


def test_added_xray_raises_tuberculosis_and_lung_cancer(asia):
    ad = asia['AD']['contraction']['node_marginals']
    adx = asia['ADX']['contraction']['node_marginals']
    assert adx['1'][0] > ad['1'][0]
    assert adx['3'][0] > ad['3'][0]
    assert adx['4'][0] < ad['4'][0]


def test_evidence_is_the_denominator(asia, asia_model):
    """Open T values divided by the closed network give the conditioned marginal."""
    run = asia['ADX']
    unscaled = np.asarray(run['contraction']['node_joints']['1'])
    assert unscaled.sum() == pytest.approx(run['contraction']['mass'], rel=1e-12)
    np.testing.assert_allclose(unscaled / run['contraction']['mass'], marginal(run, 1),
                               rtol=0, atol=1e-12)
