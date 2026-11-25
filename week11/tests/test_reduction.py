"""Part 1: the factoring round trip through the `pred` reduction toolkit.

The transcript is produced by `python scripts/run_pred.py`, which runs the sheet's
commands; this file reads it and checks the three answers the sheet prints.
"""
from __future__ import annotations

import json

import pytest

from conftest import ROOT

TRANSCRIPT = ROOT / 'runs' / 'pred' / 'transcript.json'


@pytest.fixture(scope='module')
def transcript() -> dict:
    if not TRANSCRIPT.exists():
        pytest.skip(f'{TRANSCRIPT} is missing: run python scripts/run_pred.py')
    return json.loads(TRANSCRIPT.read_text(encoding='utf-8'))


def test_every_stated_command_succeeded(transcript):
    assert transcript['pred_version'] == 'pred 0.6.0'
    for entry in transcript['commands']:
        assert entry['returncode'] == 0, entry['command']


def test_the_route_goes_through_a_circuit(transcript):
    assert transcript['path'] == ['Factoring', 'CircuitSAT', 'SpinGlass']


def test_fifteen_factors_into_three_times_five(transcript):
    answer = transcript['answers']['15']
    assert answer['bits'] == [1, 1, 1, 0, 1]
    assert answer['factors'] == [3, 5]
    assert answer['product'] == 15
    assert answer['evaluation'] == 'Min(0)'
    assert answer['spin_glass_evaluation'] == 'Min(-299)'
    assert answer['spin_glass_spins'] == 104
    assert answer['solver'] == 'ilp'
    assert transcript['evaluate_15']['result'] == 'Min(0)'


def test_six_factors_into_two_times_three(transcript):
    answer = transcript['answers']['6']
    assert answer['bits'] == [0, 1, 1, 1]
    assert answer['factors'] == [2, 3]
    assert answer['evaluation'] == 'Min(0)'
    assert answer['spin_glass_spins'] == 14 * 2 * 2 + 8 * 2


def test_seven_has_no_two_bit_factor_pair_and_the_check_exposes_it(transcript):
    answer = transcript['answers']['7']
    assert answer['product'] != 7
    assert answer['error'] == abs(answer['product'] - 7)
    assert answer['evaluation'] == f"Min({answer['error']})"
    assert answer['error'] > 0
    # Seven is prime: neither 2 x 2 pair can reach it, and the magnet still finds
    # its lowest state. The reference run decodes a different wrong product than
    # the sheet's example (1 x 3), which the sheet allows for.
    assert (answer['factors'][0], answer['factors'][1]) != (1, 3)
