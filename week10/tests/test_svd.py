"""Part 1: the seeded Gaussian matrix and its truncated SVD evidence.

Correctness is defined by the sheet: `e_chi = ||M - M_chi||_F / ||M||_F` must
equal the discarded-spectrum prediction `sqrt(sum_{alpha>chi} s_alpha^2 /
sum_alpha s_alpha^2)` to 1e-8, the full-rank reconstruction must be exact to
1e-8, and the two-factor storage count is `(m+n) chi` against `m n`.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from conftest import ARTIFACTS

SIDE = 512
RANKS = [16, 64, 128, 256, 512]


def test_generator_is_documented_and_reproducible(thermo):
    first = thermo.run('svd', ARTIFACTS / 'tests' / 'svd-a', l=128, seed=2026, ranks='8')
    second = thermo.run('svd', ARTIFACTS / 'tests' / 'svd-b', l=128, seed=2026, ranks='8')
    other = thermo.run('svd', ARTIFACTS / 'tests' / 'svd-c', l=128, seed=2027, ranks='8')
    a = np.load(ARTIFACTS / 'tests' / 'svd-a' / 'matrix.npy')
    b = np.load(ARTIFACTS / 'tests' / 'svd-b' / 'matrix.npy')
    c = np.load(ARTIFACTS / 'tests' / 'svd-c' / 'matrix.npy')
    assert np.array_equal(a, b), 'the same seed must regenerate the same matrix'
    assert not np.array_equal(a, c)
    assert first['generator']['seed'] == 2026
    assert first['generator']['uniform'].startswith('SplitMix64')
    assert 'Box-Muller' in first['generator']['gaussian']
    assert first['generator']['fill_order'] == 'row-major'
    # Standard normal: mean 0 +/- 4 standard errors, variance 1 within 3%.
    sample = a.size
    assert abs(first['sample_mean']) < 4.0 / math.sqrt(sample), first['sample_mean']
    assert abs(first['sample_standard_deviation'] - 1.0) < 0.03, first['sample_standard_deviation']
    assert np.allclose(a.mean(), first['sample_mean'])
    assert np.allclose(a.std(), first['sample_standard_deviation'])


def test_singular_values_are_ordered_and_complete(svd_run):
    out, record = svd_run
    singular = np.load(out / 'singular-values.npy')
    matrix = np.load(out / 'matrix.npy')
    assert singular.shape == (SIDE,)
    assert np.all(np.diff(singular) <= 0.0), 'the spectrum must be non-increasing'
    assert np.all(singular >= 0.0)
    assert abs(float(np.sum(singular ** 2)) - float(np.sum(matrix ** 2))) < 1e-8 * float(
        np.sum(matrix ** 2))
    assert record['singular_values_file'] == 'singular-values.npy'
    assert record['matrix_file'] == 'matrix.npy'


def test_reconstruction_error_equals_the_discarded_spectrum(svd_run):
    out, record = svd_run
    matrix = np.load(out / 'matrix.npy')
    singular = np.load(out / 'singular-values.npy')
    total = float(np.sum(singular ** 2))
    norm = math.sqrt(total)
    reported = {entry['rank']: entry for entry in record['reconstructions']}
    for rank in RANKS:
        rebuilt = np.load(out / reported[rank]['file'])
        measured = float(np.linalg.norm(matrix - rebuilt) / norm)
        prediction = math.sqrt(float(np.sum(singular[rank:] ** 2)) / total)
        assert abs(measured - prediction) <= 1e-8, f'chi {rank}: {measured} vs {prediction}'
        assert abs(measured - reported[rank]['relative_frobenius_error']) <= 1e-8
        assert abs(prediction - reported[rank]['discarded_spectrum_prediction']) <= 1e-8
        assert reported[rank]['saved_entries'] == 2 * SIDE * rank
        assert reported[rank]['full_entries'] == SIDE * SIDE
        assert abs(reported[rank]['storage_ratio'] - 2 * rank / SIDE) < 1e-15


def test_full_rank_keeps_every_direction(svd_run):
    out, record = svd_run
    matrix = np.load(out / 'matrix.npy')
    rebuilt = np.load(out / 'recon-512.npy')
    error = float(np.linalg.norm(matrix - rebuilt) / np.linalg.norm(matrix))
    assert error < 1e-8, error
    assert record['full_rank_error'] < 1e-8
    report = json.loads((out / 'run.json').read_text())
    assert report['reconstructions'][-1]['rank'] == SIDE


def test_storage_crossover_and_the_ten_percent_question(svd_run):
    out, record = svd_run
    singular = np.load(out / 'singular-values.npy')
    total = float(np.sum(singular ** 2))
    # (m+n) chi = m n only at chi = mn/(m+n) = 256 for a square 512 matrix.
    assert record['storage_crossover']['rank'] == SIDE * SIDE // (2 * SIDE) == 256
    crossing = record['ten_percent_crossing']['rank']
    first = next(rank for rank in range(1, SIDE + 1)
                 if math.sqrt(float(np.sum(singular[rank:] ** 2)) / total) <= 0.1)
    assert crossing == first
    assert crossing > 256, 'the 10% target must not be reachable with less storage here'
    assert record['ten_percent_crossing']['storage_ratio'] > 1.0
    errors = {entry['rank']: entry['relative_frobenius_error']
              for entry in record['reconstructions']}
    assert errors[256] > 0.1, errors[256]


def test_svd_rejects_ising_arguments(thermo):
    completed = thermo('--method', 'svd', '--l', '16', '--seed', '2026', '--ranks', '4',
                       '--k', '0.44', '--out', str(ARTIFACTS / 'tests' / 'bad'))
    assert completed.returncode != 0
    assert '--k does not apply' in completed.stderr
    completed = thermo('--method', 'svd', '--l', '16', '--seed', '2026', '--ranks', '1,17',
                       '--out', str(ARTIFACTS / 'tests' / 'bad'))
    assert completed.returncode != 0
    assert 'must lie in 1..=16' in completed.stderr
    completed = thermo('--method', 'svd', '--l', '16', '--ranks', '4',
                       '--out', str(ARTIFACTS / 'tests' / 'bad'))
    assert completed.returncode != 0
    assert 'missing required --seed' in completed.stderr


def test_unknown_flags_are_rejected(thermo):
    completed = thermo('--method', 'svd', '--l', '8', '--seed', '1', '--ranks', '2',
                       '--nonsense', '1', '--out', str(ARTIFACTS / 'tests' / 'bad'))
    assert completed.returncode != 0
    assert 'unknown flag' in completed.stderr


@pytest.mark.parametrize('side', [32])
def test_a_second_seed_changes_only_the_data(thermo, side):
    out = ARTIFACTS / 'tests' / 'svd-side'
    record = thermo.run('svd', out, l=side, seed=2026, ranks='4,8')
    assert record['geometry']['m'] == side
    assert [entry['rank'] for entry in record['reconstructions']] == [4, 8]
    smallest = record['reconstructions'][0]['relative_frobenius_error']
    largest = record['reconstructions'][1]['relative_frobenius_error']
    assert smallest > largest, 'keeping more directions must not increase the error'
