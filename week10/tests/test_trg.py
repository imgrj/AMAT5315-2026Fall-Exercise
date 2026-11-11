"""Part 3: TRG on periodic squares.

Ground truth is a full sum over the periodic spin configurations, plus an
independent NumPy implementation of the sheet's Algorithm 5.
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


def test_two_by_two_torus_is_exact(thermo):
    out = ARTIFACTS / 'tests' / 'trg-2'
    record = thermo.run('trg', out, l=2, k=COUPLING, chi=16)
    truth = reference.enumerate_square(2, COUPLING, 'periodic')
    assert abs(record['g'] - truth['g']) <= 1e-8
    assert record['geometry']['boundary'] == 'periodic'
    assert record['geometry']['bonds'] == truth['bonds'] == 8
    assert abs(record['log_z'] - truth['log_z']) <= 1e-8


def test_two_by_two_specific_heat(thermo):
    values = {}
    for k in (COUPLING - STENCIL, COUPLING, COUPLING + STENCIL):
        out = ARTIFACTS / 'tests' / f'trg-2-cv-{k:.4f}'
        values[k] = thermo.run('trg', out, l=2, k=k, chi=16)['g']
    second = (values[COUPLING + STENCIL] - 2.0 * values[COUPLING]
              + values[COUPLING - STENCIL]) / STENCIL ** 2
    finite_difference = COUPLING ** 2 * second
    truth = reference.enumerate_square(2, COUPLING, 'periodic')
    assert abs(finite_difference - truth['specific_heat']) <= 1e-3


def test_four_by_four_truncation_is_small(thermo):
    out = ARTIFACTS / 'tests' / 'trg-4'
    record = thermo.run('trg', out, l=4, k=COUPLING, chi=8)
    truth = reference.enumerate_square(4, COUPLING, 'periodic')
    difference = abs(record['g'] - truth['g'])
    assert difference <= 2e-3, difference
    assert record['geometry']['bonds'] == truth['bonds'] == 32


def test_zero_coupling_gives_log_two(thermo):
    for length in (2, 4, 8):
        out = ARTIFACTS / 'tests' / f'trg-zero-{length}'
        record = thermo.run('trg', out, l=length, k=0.0, chi=8)
        assert abs(record['g'] - math.log(2.0)) < 1e-12


def test_matches_the_independent_numpy_reference(thermo):
    for length, chi in ((4, 4), (4, 8), (8, 8)):
        out = ARTIFACTS / 'tests' / f'trg-ref-{length}-{chi}'
        record = thermo.run('trg', out, l=length, k=COUPLING, chi=chi)
        independent = reference.numpy_trg(length, COUPLING, chi)
        assert abs(record['g'] - independent['g']) <= 1e-10, (length, chi)


def test_steps_halve_the_tensor_count(thermo):
    out = ARTIFACTS / 'tests' / 'trg-steps'
    record = thermo.run('trg', out, l=64, k=COUPLING, chi=16)
    steps = record['steps']
    assert len(steps) == 12  # n = 2 log2(64)
    assert steps[0]['tensors'] == 64 * 64
    assert steps[-1]['tensors'] == 2
    assert [step['tensors'] for step in steps] == [4096 // 2 ** index for index in range(12)]
    for step in steps:
        dims = step['bond_dims']
        assert dims[0] == dims[2] and dims[1] == dims[3], dims
        assert max(dims) <= 16, dims
        assert len(step['split1_singular_values']) == step['bond_dims'][0]
        assert len(step['split2_singular_values']) == step['bond_dims'][1]
        assert step['regroup_entries'] == dims[0] * dims[1] * dims[2] * dims[3]
    scales = [step['max_abs'] for step in steps]
    assert all(scale > 0.0 for scale in scales)


def test_saved_evidence_reproduces_log_z(thermo):
    out = ARTIFACTS / 'tests' / 'trg-evidence'
    record = thermo.run('trg', out, l=16, k=COUPLING, chi=8)
    ell = 0.0
    for step in record['steps']:
        ell += step['tensors'] * math.log(step['max_abs'])
    assert abs(ell - record['normalisation']['ell']) <= 1e-9
    shape = record['final_tensor_shape']
    tensor = np.load(out / record['final_tensor_file']).reshape(shape)
    closure = float(np.einsum('ijij->', tensor))
    assert abs(closure - record['normalisation']['z']) <= 1e-9 * abs(closure)
    log_z = ell + math.log(abs(closure))
    assert abs(log_z - record['log_z']) <= 1e-9


def test_saved_split_factors_rebuild_the_first_round(thermo):
    """The saved halves must reconstruct the split matrices they came from."""
    out = ARTIFACTS / 'tests' / 'splits'
    record = thermo.run('trg', out, l=8, k=COUPLING, chi=8)
    step = record['steps'][0]
    flat = np.load(out / step['split_factor_file'])
    shapes = [tuple(shape) for shape in step['split_factor_shapes']]
    assert sum(int(np.prod(shape)) for shape in shapes) == flat.size
    blocks = np.split(flat, np.cumsum([int(np.prod(shape)) for shape in shapes])[:-1])
    s1, s2, s3, s4 = [block.reshape(shape) for block, shape in zip(blocks, shapes)]
    site = np.array(record['model']['site_tensor']).reshape(2, 2, 2, 2)
    scaled = site / step['max_abs']
    # First split: rows (l,u), columns (r,d); S1 and S3 share the retained index.
    first = np.transpose(scaled, (2, 1, 0, 3)).reshape(4, 4)
    k1 = s1.shape[2]
    assert np.allclose(first, s1.reshape(4, k1) @ s3.reshape(k1, 4), atol=1e-12)
    u, singular, _ = np.linalg.svd(first, full_matrices=False)
    assert np.allclose(singular[:k1], step['split1_singular_values'], rtol=1e-10)
    # Second split: rows (d,l), columns (u,r).
    second = np.transpose(scaled, (3, 2, 1, 0)).reshape(4, 4)
    k2 = s2.shape[2]
    assert np.allclose(second, s2.reshape(4, k2) @ s4.reshape(k2, 4), atol=1e-12)
    _, second_singular, _ = np.linalg.svd(second, full_matrices=False)
    assert np.allclose(second_singular[:k2], step['split2_singular_values'], rtol=1e-10)
    assert k1 == s1.shape[2] == s3.shape[0]
    assert k2 == s2.shape[2] == s4.shape[0]


def test_regrouping_the_saved_halves_matches_the_next_step(thermo):
    """Regrouping step 1's halves must reproduce the tensor step 2 splits."""
    out = ARTIFACTS / 'tests' / 'splits-regroup'
    record = thermo.run('trg', out, l=8, k=COUPLING, chi=8)

    def factors(step):
        flat = np.load(out / step['split_factor_file'])
        shapes = [tuple(shape) for shape in step['split_factor_shapes']]
        sizes = [int(np.prod(shape)) for shape in shapes]
        blocks = np.split(flat, np.cumsum(sizes)[:-1])
        return [block.reshape(shape) for block, shape in zip(blocks, shapes)]

    s1, s2, s3, s4 = factors(record['steps'][0])
    product = np.einsum('war,abu,lbg,dgw->ruld', s1, s2, s3, s4)
    second = record['steps'][1]
    assert product.shape == tuple(second['source_dims'])
    assert np.isclose(float(np.abs(product).max()), second['max_abs'], rtol=1e-10)
    scaled = product / second['max_abs']
    rows = scaled.shape[2] * scaled.shape[1]
    cols = scaled.shape[0] * scaled.shape[3]
    first_cut = np.transpose(scaled, (2, 1, 0, 3)).reshape(rows, cols)
    _, singular1, _ = np.linalg.svd(first_cut, full_matrices=False)
    assert np.allclose(singular1[:len(second['split1_singular_values'])],
                       second['split1_singular_values'], rtol=1e-9, atol=1e-12)
    second_cut = np.transpose(scaled, (3, 2, 1, 0)).reshape(rows, cols)
    _, singular2, _ = np.linalg.svd(second_cut, full_matrices=False)
    assert np.allclose(singular2[:len(second['split2_singular_values'])],
                       second['split2_singular_values'], rtol=1e-9, atol=1e-12)


def test_rejects_non_power_of_two(thermo):
    completed = thermo('--method', 'trg', '--l', '12', '--k', '0.44', '--chi', '8',
                       '--out', str(ARTIFACTS / 'tests' / 'bad-trg'))
    assert completed.returncode != 0
    assert 'not a power of two' in completed.stderr


def test_ising_arguments_are_required(thermo):
    completed = thermo('--method', 'trg', '--l', '8', '--chi', '8',
                       '--out', str(ARTIFACTS / 'tests' / 'bad-trg'))
    assert completed.returncode != 0
    assert 'missing required --k' in completed.stderr
    completed = thermo('--method', 'boundary', '--l', '8', '--k', '-1', '--chi', '8',
                       '--out', str(ARTIFACTS / 'tests' / 'bad-trg'))
    assert completed.returncode != 0
    assert 'non-negative' in completed.stderr


@pytest.mark.parametrize('length,tolerance', [(2, 1e-9), (8, 1e-3)])
def test_a_larger_chi_settles_the_answer(thermo, length, tolerance):
    values = []
    for chi in (4, 16):
        out = ARTIFACTS / 'tests' / f'trg-stable-{length}-{chi}'
        values.append(thermo.run('trg', out, l=length, k=COUPLING, chi=chi)['g'])
    assert abs(values[0] - values[1]) < tolerance, values
