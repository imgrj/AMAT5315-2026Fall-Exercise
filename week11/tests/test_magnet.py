"""Part 3: exact counting of spin glasses by rescaled and by min-plus contraction."""
from __future__ import annotations

import math

import pytest

import reference

from conftest import ARTIFACTS


def magnet_of(record: dict) -> dict:
    network = record['network']
    return {
        'nspins': network['spins'],
        'bonds': [(int(i), int(j), float(k)) for i, j, k in network['bond_list']],
        'fields': [float(h) for h in network['field_list']],
    }


def test_lattice_bonds_reproduce_the_documented_generator(tnsample):
    out = ARTIFACTS / 'tests' / 'lattice-bonds'
    record = tnsample.run('count', out, l=3, p_af=0.5, beta=1.0, order='greedy')
    expected = reference.lattice_bonds(3, 0.5, 2026)
    got = [(int(i), int(j), float(k)) for i, j, k in record['network']['bond_list']]
    assert got == expected
    assert record['network']['field_list'] == [0.0] * 9
    assert record['network']['fields'] == 0


@pytest.mark.parametrize('length', [3, 4])
@pytest.mark.parametrize('p_af', [0.0, 0.5, 1.0])
def test_exact_enumeration_agrees_on_small_lattices(tnsample, length, p_af):
    for beta in (0.5, 2.0):
        out = ARTIFACTS / 'tests' / f'lattice-{length}-{p_af}-{beta}'
        record = tnsample.run('count', out, l=length, p_af=p_af, beta=beta, order='treesa')
        magnet = magnet_of(record)
        energies = reference.magnet_energies(magnet)
        e_min = float(energies.min())
        g0 = int((energies == e_min).sum())
        log_z = float(-beta * energies.min()) + math.log(
            float(sum(math.exp(-beta * (e - e_min)) for e in energies)))
        assert float(record['results']['e_min']) == pytest.approx(e_min, abs=1e-9)
        assert record['results']['g0'] == g0
        assert float(record['results']['log_z']) == pytest.approx(log_z, abs=1e-9)


def test_ten_by_ten_ferromagnet_is_fully_satisfied(tnsample):
    out = ARTIFACTS / 'tests' / 'count-p0'
    record = tnsample.run('count', out, l=10, p_af=0.0, beta=2.0, order='treesa')
    results = record['results']
    assert record['network']['spins'] == 100
    assert record['network']['bonds'] == 2 * 10 * 9
    assert float(results['e_min']) == -2 * 10 * 9
    assert results['g0'] == 2
    # Every bond is satisfied, so the two aligned states are the whole ground space.
    assert float(results['log_z']) >= math.log(2.0) + 2.0 * 180.0
    assert float(results['log_excess_above_ground']) > 0.0


def test_factor_15_matches_the_sheet(tnsample, bundle):
    out = ARTIFACTS / 'tests' / 'count-f15'
    record = tnsample.run('count', out, input=bundle, beta=2.0, order='treesa')
    results = record['results']
    assert record['network']['spins'] == 104
    assert record['network']['bonds'] == 178
    assert float(results['e_min']) == -299
    assert results['g0'] == 1
    assert results['sc'] == 5
    assert results['max_entries'] == 32
    assert float(results['log_z']) == pytest.approx(598.3179, abs=1e-4)


def test_factor_15_log_z_from_an_independent_numpy_contraction(tnsample, bundle):
    out = ARTIFACTS / 'tests' / 'count-f15-numpy'
    record = tnsample.run('count', out, input=bundle, beta=2.0, order='greedy')
    magnet = reference.bundle_magnet(bundle)
    independent = reference.log_partition(magnet, 2.0)
    assert float(record['results']['log_z']) == pytest.approx(independent, abs=1e-9)


def test_factor_15_ground_state_from_low_temperature_partition_sums(bundle):
    magnet = reference.bundle_magnet(bundle)
    e_min, g0 = reference.ground_state(magnet, beta=10.0)
    assert e_min == pytest.approx(-299.0, abs=1e-6)
    assert g0 == 1


def test_a_larger_beta_keeps_the_same_ground_state(tnsample, bundle):
    out = ARTIFACTS / 'tests' / 'count-f15-beta10'
    record = tnsample.run('count', out, input=bundle, beta=10.0, order='treesa')
    assert float(record['results']['e_min']) == -299
    assert record['results']['g0'] == 1
    assert float(record['results']['log_z']) > float(598.3179) * 4.0
