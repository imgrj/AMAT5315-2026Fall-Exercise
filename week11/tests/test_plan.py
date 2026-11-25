"""Part 3: planning only, without contracting, and how the cost scales with k.

The sheet's table is the reference: `14k^2 + 8k` spins, a `treesa` space complexity
near `2k + 1` (the multiplier's diagonal seam) and a `greedy` order that runs into
memory. `greedy` breaks equal-cost ties at random, so only its scale is asserted.
"""
from __future__ import annotations

import shutil
import subprocess

import pytest

from conftest import ARTIFACTS, ROOT

BITS = [2, 3, 4, 5, 6, 7, 8]
SPINS = {k: 14 * k * k + 8 * k for k in BITS}


def make_bundle(k: int) -> tuple[object, object]:
    folder = ARTIFACTS / 'tests' / 'pred-scale'
    folder.mkdir(parents=True, exist_ok=True)
    target = (2 ** k - 1) ** 2
    source = folder / f'scale_{k}.json'
    bundle = folder / f'scale_{k}.bundle.json'
    if not bundle.exists():
        for command in (
            ['pred', 'create', 'Factoring', '--target', str(target), '--m', str(k),
             '--n', str(k), '-o', str(source)],
            ['pred', 'reduce', str(source), '--to', 'SpinGlass', '-o', str(bundle)],
        ):
            completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            assert completed.returncode == 0, completed.stderr
    return source, bundle


def test_planning_the_stated_factor_15_bundle(tnsample, bundle):
    treesa = tnsample.run('plan', ARTIFACTS / 'tests' / 'plan-f15-treesa',
                          input=bundle, order='treesa')
    greedy = tnsample.run('plan', ARTIFACTS / 'tests' / 'plan-f15-greedy',
                          input=bundle, order='greedy')
    assert treesa['network']['spins'] == 104
    assert treesa['results']['sc'] == 5
    assert treesa['results']['max_entries'] == 32
    # greedy breaks equal-cost ties at random (the sheet says so), so only its
    # scale and its ordering against treesa are asserted.
    assert greedy['results']['sc'] >= treesa['results']['sc']
    assert greedy['results']['max_entries'] == 2 ** greedy['results']['sc']
    for record in (treesa, greedy):
        assert 'log_z' not in record['results'], 'plan must not contract'
    assert treesa['results']['tc'] < greedy['results']['tc']


def test_the_default_order_is_treesa(tnsample, bundle):
    out = ARTIFACTS / 'tests' / 'plan-default'
    out.mkdir(parents=True, exist_ok=True)
    completed = tnsample('--mode', 'plan', '--input', str(bundle), '--out', str(out),
                         '--seed', '2026')
    assert completed.returncode == 0, completed.stderr
    import json
    record = json.loads((out / 'run.json').read_text(encoding='utf-8'))
    assert record['results']['order'] == 'treesa'


@pytest.mark.skipif(shutil.which('pred') is None, reason='the pred CLI is not installed')
@pytest.mark.slow
def test_contraction_cost_scales_with_the_factor_bit_length(tnsample):
    rows = {}
    for k in BITS:
        _, bundle_path = make_bundle(k)
        for order in ('greedy', 'treesa'):
            out = ARTIFACTS / 'tests' / f'plan-k{k}-{order}'
            record = tnsample.run('plan', out, input=bundle_path, order=order)
            assert record['network']['spins'] == SPINS[k], k
            rows[(k, order)] = record['results']
    # The multiplier's wire count, not just the bond count, sets the growth.
    treesa_sc = [rows[(k, 'treesa')]['sc'] for k in BITS]
    assert treesa_sc[0] == 4
    for earlier, later in zip(treesa_sc, treesa_sc[1:]):
        assert earlier <= later, treesa_sc
    for k in BITS[1:]:
        # treesa finds the multiplier's ~2k + 1 diagonal seam; omeco's annealer is
        # stochastic, so the assertion is the seam's linear law, not one table.
        assert rows[(k, 'treesa')]['sc'] <= 2 * k + 3, (k, rows[(k, 'treesa')]['sc'])
        assert rows[(k, 'treesa')]['sc'] <= rows[(k, 'greedy')]['sc'], k
    assert rows[(3, 'treesa')]['sc'] <= 2 * 3 + 3
    growth = rows[(8, 'treesa')]['sc'] - rows[(3, 'treesa')]['sc']
    assert 10 <= growth <= 14, growth
    # greedy pays for its speed with memory: it is already around 2^19 at k = 5\
    # (the sheet's greedy run reaches 23 there; greedy's ties are random).
    assert rows[(5, 'greedy')]['sc'] >= 18
    assert rows[(8, 'greedy')]['sc'] > rows[(8, 'treesa')]['sc'] + 20
    # Time grows by about two orders of magnitude per extra bit of each factor.
    steps = [rows[(k, 'treesa')]['tc'] - rows[(k - 1, 'treesa')]['tc'] for k in BITS[1:]]
    assert all(1.5 < step < 3.0 for step in steps), steps


def test_planning_an_n_queens_board(tnsample):
    for n in (4, 5, 6):
        record = tnsample.run('plan', ARTIFACTS / 'tests' / f'plan-nq{n}', n=n, order='treesa')
        assert record['network']['tensors'] == 4 * n * n
        assert record['network']['variables'] == 5 * n * n - 2 * n
        assert record['results']['sc'] >= 8
