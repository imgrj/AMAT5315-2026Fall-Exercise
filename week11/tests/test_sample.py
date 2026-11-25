"""Part 3: conditional sampling from one stored contraction.

The stated run draws 2000 states of the 104-spin magnet at beta = 2 and is checked
against Boltzmann theory, `2000 g_0 exp(-beta E_min) / Z`; a small exactly
enumerable magnet is checked bin by bin against its exact distribution.
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess

import pytest

import reference

from conftest import ARTIFACTS, ROOT


def test_factor_15_sampling_follows_boltzmann_theory(tnsample, bundle):
    out = ARTIFACTS / 'tests' / 'sample-f15'
    record = tnsample.run('sample', out, input=bundle, beta=2.0, samples=2000, order='treesa')
    results = record['results']
    assert results['samples'] == 2000
    p0 = reference.p_ground(results['g0'], float(results['e_min']), 2.0, float(results['log_z']))
    assert results['p0'] == pytest.approx(p0, abs=1e-12)
    predicted = 2000.0 * p0
    assert results['predicted_ground_draws'] == pytest.approx(predicted, abs=1e-6)
    spread = reference.binomial_spread(2000, p0)
    assert results['binomial_spread'] == pytest.approx(spread, abs=1e-9)
    # Our seed's count lies inside four binomial standard deviations of theory.
    assert abs(results['ground_draws'] - predicted) <= 4.0 * spread
    assert abs(results['ground_draws'] / 2000.0 - p0) < 0.1
    histogram = results['histogram']
    assert sum(bin_['count'] for bin_ in histogram) == 2000
    assert min(bin_['energy'] for bin_ in histogram) == -299
    assert len(record['sample_energies']) == 2000
    assert len(record['sample_bits']) == 2000
    assert results['histogram'][0]['energy'] == float(results['e_min'])


def test_every_sampled_state_has_its_recorded_energy(tnsample, bundle):
    out = ARTIFACTS / 'tests' / 'sample-f15-energies'
    record = tnsample.run('sample', out, input=bundle, beta=2.0, samples=50, order='greedy')
    magnet = reference.bundle_magnet(bundle)
    for bits, energy in zip(record['sample_bits'][:10], record['sample_energies'][:10]):
        assignment = [int(bit) for bit in bits]
        spins = [2 * b - 1 for b in assignment]
        expected = sum(k * spins[i] * spins[j] for i, j, k in magnet['bonds'])
        expected += sum(h * spins[i] for i, h in enumerate(magnet['fields']) if h != 0.0)
        assert energy == pytest.approx(expected, abs=1e-9)
        assert energy >= float(record['results']['e_min'])


def test_small_magnet_samples_match_the_exact_distribution(tnsample):
    out = ARTIFACTS / 'tests' / 'sample-small'
    record = tnsample.run('sample', out, l=3, p_af=0.5, beta=0.7, samples=8000, order='greedy')
    network = record['network']
    magnet = {
        'nspins': network['spins'],
        'bonds': [(int(i), int(j), float(k)) for i, j, k in network['bond_list']],
        'fields': [float(h) for h in network['field_list']],
    }
    beta = 0.7
    energies = reference.magnet_energies(magnet)
    weights = {}
    for energy in energies:
        weights[energy] = weights.get(energy, 0.0) + math.exp(-beta * energy)
    total = sum(weights.values())
    counted: dict[float, int] = {}
    for energy in record['sample_energies']:
        counted[energy] = counted.get(energy, 0) + 1
    assert set(counted) <= set(weights)
    dof = 0
    chi2 = 0.0
    for energy, weight in weights.items():
        expected = 8000 * weight / total
        if expected < 5.0:
            continue
        observed = counted.get(energy, 0)
        chi2 += (observed - expected) ** 2 / expected
        dof += 1
    assert dof >= 3
    # A chi-square with `dof` degrees of freedom stays under mean + 5*sqrt(2 dof) here.
    assert chi2 <= dof + 5.0 * math.sqrt(2.0 * dof), (chi2, dof)


def test_sampling_repeats_its_statistics(tnsample, bundle):
    """Two runs of the same command must agree on the physics.

    `omeco` breaks equal-cost ties through hash iteration order, so the planned tree
    is one sample of the planner and the exact draw sequence follows that tree; the
    Boltzmann law it samples does not depend on the plan.
    """
    runs = [tnsample.run('sample', ARTIFACTS / 'tests' / f'sample-repeat-{index}',
                         input=bundle, beta=2.0, samples=400, order='greedy')
            for index in range(2)]
    for record in runs:
        assert record['results']['plan_hash']
        assert float(record['results']['e_min']) == -299
        assert record['results']['g0'] == 1
        assert float(record['results']['log_z']) == pytest.approx(598.3179, abs=1e-4)
        p0 = record['results']['p0']
        spread = reference.binomial_spread(400, p0)
        assert abs(record['results']['ground_draws'] - 400 * p0) <= 4.0 * spread


def test_treesa_annealing_is_recorded_even_though_it_breaks_ties_at_random(tnsample, bundle):
    """The plan is one sample of omeco's annealer; the physics must not move.

    Any plan contracts the same network, so `E_min`, `g_0` and `log Z` agree to
    machine precision while the recorded plan fingerprint may differ.
    """
    runs = [tnsample.run('sample', ARTIFACTS / 'tests' / f'sample-treesa-{index}',
                         input=bundle, beta=2.0, samples=400, order='treesa')
            for index in range(2)]
    for record in runs:
        assert record['results']['plan_hash']
        assert record['results']['e_min'] == runs[0]['results']['e_min']
        assert record['results']['g0'] == runs[0]['results']['g0']
        assert abs(record['results']['log_z'] - runs[0]['results']['log_z']) < 1e-9
    p0 = runs[0]['results']['p0']
    spread = reference.binomial_spread(400, p0)
    for record in runs:
        assert abs(record['results']['ground_draws'] - 400 * p0) <= 4.0 * spread


@pytest.mark.skipif(shutil.which('pred') is None, reason='the pred CLI is not installed')
def test_pred_extract_decodes_the_best_sample(tnsample, bundle):
    out = ARTIFACTS / 'tests' / 'sample-f15-extract'
    record = tnsample.run('sample', out, input=bundle, beta=2.0, samples=500, order='treesa')
    bits = record['lowest_energy_bits']
    assert len(bits) == record['network']['spins']
    completed = subprocess.run(
        ['pred', 'extract', str(bundle), '--config', ','.join(bits), '--json'],
        cwd=ROOT, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    answer = json.loads(completed.stdout)
    assert answer['solution'] == [1, 1, 1, 0, 1]
    assert answer['evaluation'] == 'Min(0)'
    assert answer['intermediate']['evaluation'] == 'Min(-299)'
