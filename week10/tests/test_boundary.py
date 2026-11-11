"""Part 2: the boundary-MPS contraction of open squares.

Ground truth is the full sum over `2^16` spin configurations of the same open
geometry, plus an independent NumPy implementation of the sheet's Algorithm 4.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

import ising_reference as reference

from conftest import ARTIFACTS

COUPLING = 0.44
STENCIL = 0.0025


def test_the_recorded_model_is_the_sheet_definition(thermo):
    out = ARTIFACTS / 'tests' / 'model'
    record = thermo.run('boundary', out, l=4, k=COUPLING, chi=8)
    model = record['model']
    assert model['k'] == COUPLING
    assert abs(model['temperature'] - 1.0 / COUPLING) < 1e-12
    assert model['field'] == 0.0
    assert model['coupling_j'] == 1.0
    cosh = math.cosh(COUPLING)
    sinh = math.sinh(COUPLING)
    expected_w = [[math.sqrt(cosh), math.sqrt(sinh)],
                  [math.sqrt(cosh), -math.sqrt(sinh)]]
    assert np.allclose(model['w_tensor'], expected_w, atol=1e-14)
    assert np.allclose(model['b_vector'], [1.0 / math.sqrt(cosh), 0.0], atol=1e-14)
    site = np.array(model['site_tensor']).reshape(2, 2, 2, 2)
    w = np.array(expected_w)
    expected = np.einsum('sr,su,sl,sd->ruld', w, w, w, w)
    assert np.allclose(site, expected, atol=1e-14)
    closure = np.einsum('sa,a->s', w, np.array(model['b_vector']))
    assert np.allclose(closure, [1.0, 1.0], atol=1e-14)


def test_open_four_by_four_matches_enumeration(thermo):
    for k in (0.0, COUPLING, 0.7):
        out = ARTIFACTS / 'tests' / f'open4-k{k}'
        record = thermo.run('boundary', out, l=4, k=k, chi=8)
        truth = reference.enumerate_square(4, k, 'open')
        assert abs(record['g'] - truth['g']) <= 1e-8, (k, record['g'], truth['g'])
        assert record['geometry']['boundary'] == 'open'
        assert record['geometry']['bonds'] == truth['bonds']


def test_specific_heat_matches_the_energy_variance(thermo):
    values = {}
    for k in (COUPLING - STENCIL, COUPLING, COUPLING + STENCIL):
        out = ARTIFACTS / 'tests' / f'open4-cv-{k:.4f}'
        values[k] = thermo.run('boundary', out, l=4, k=k, chi=8)['g']
    second = (values[COUPLING + STENCIL] - 2.0 * values[COUPLING]
              + values[COUPLING - STENCIL]) / STENCIL ** 2
    finite_difference = COUPLING ** 2 * second
    truth = reference.enumerate_square(4, COUPLING, 'open')
    assert abs(finite_difference - truth['specific_heat']) <= 1e-3


def test_zero_coupling_gives_log_two(thermo):
    for length in (2, 3, 4, 8):
        out = ARTIFACTS / 'tests' / f'zero-{length}'
        record = thermo.run('boundary', out, l=length, k=0.0, chi=4)
        assert abs(record['g'] - math.log(2.0)) < 1e-12
        if length <= 4:
            truth = reference.enumerate_square(length, 0.0, 'open')
            assert abs(record['log_z'] - truth['log_z']) <= 1e-10


def test_bond_dimensions_respect_the_cap(thermo):
    out = ARTIFACTS / 'tests' / 'cap'
    record = thermo.run('boundary', out, l=8, k=COUPLING, chi=3)
    for row in record['rows']:
        for dimension in row['bond_dims_after']:
            assert dimension <= 3, row
    assert len(record['rows']) == 8
    assert record['rows'][0]['contraction_entries'] > 0


def test_saved_cores_reproduce_g(thermo):
    out = ARTIFACTS / 'tests' / 'cores'
    record = thermo.run('boundary', out, l=8, k=COUPLING, chi=6)
    cores = json.loads((out / 'cores.json').read_text())
    close = np.array(record['model']['b_vector'])
    vector = np.array([1.0])
    offset = 0
    for left, physical, right in cores['dims']:
        size = left * physical * right
        core = np.array(cores['data'][offset:offset + size]).reshape(left, physical, right)
        vector = vector @ np.tensordot(core, close, axes=([1], [0]))
        offset += size
    ell = sum(math.log(scale) for scale in cores['scales'])
    log_z = ell + math.log(float(vector[0]))
    assert abs(log_z - record['log_z']) <= 1e-12
    assert abs(log_z / 64.0 - record['g']) <= 1e-13


def test_matches_the_independent_numpy_reference(thermo):
    for length, chi in ((4, 8), (6, 8), (8, 16)):
        out = ARTIFACTS / 'tests' / f'ref-{length}-{chi}'
        record = thermo.run('boundary', out, l=length, k=COUPLING, chi=chi)
        independent = reference.boundary_mps(length, COUPLING, chi)
        assert abs(record['g'] - independent['g']) <= 1e-10, (length, chi)


def test_more_directions_reduce_the_error(thermo):
    values = {}
    for chi in (1, 2, 4, 8, 32):
        out = ARTIFACTS / 'tests' / f'chi-{chi}'
        values[chi] = thermo.run('boundary', out, l=8, k=COUPLING, chi=chi)['g']
    errors = [abs(values[chi] - values[32]) for chi in (1, 2, 4, 8)]
    assert errors == sorted(errors, reverse=True), errors
    assert abs(values[8] - values[32]) < 1e-3


@pytest.mark.parametrize('length,chi', [(2, 16)])
def test_small_square_is_exact_at_a_generous_chi(thermo, length, chi):
    out = ARTIFACTS / 'tests' / 'tiny'
    record = thermo.run('boundary', out, l=length, k=COUPLING, chi=chi)
    truth = reference.enumerate_square(length, COUPLING, 'open')
    assert abs(record['g'] - truth['g']) <= 1e-10
