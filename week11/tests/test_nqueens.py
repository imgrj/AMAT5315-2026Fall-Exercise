"""Part 4: counting and sampling N-queens boards from the carry tensor network."""
from __future__ import annotations

import pytest

import reference

from conftest import ARTIFACTS

SIZES = [4, 5, 6, 7, 8, 9, 10]


@pytest.fixture(scope='session')
def counts(tnsample) -> dict[int, dict]:
    records = {}
    for n in SIZES:
        records[n] = tnsample.run('count', ARTIFACTS / 'tests' / f'count-nq{n}', n=n, order='treesa')
    return records


def test_every_count_matches_oeis_a000170(counts):
    for n in SIZES:
        result = counts[n]['results']
        assert int(result['boards']) == reference.KNOWN_QUEENS[n], n
        assert counts[n]['network']['tensors'] == 4 * n * n
        assert result['sparse'] is True
    # The sheet promises that sparse contraction keeps every N <= 10 to seconds.
    assert counts[10]['runtime_ms'] < 60_000, counts[10]['runtime_ms']


def test_the_contraction_order_does_not_change_the_count(tnsample):
    for n in (4, 5, 6):
        greedy = tnsample.run('count', ARTIFACTS / 'tests' / f'count-nq{n}-greedy',
                              n=n, order='greedy')
        treesa = tnsample.run('count', ARTIFACTS / 'tests' / f'count-nq{n}-treesa',
                              n=n, order='treesa')
        assert int(greedy['results']['boards']) == int(treesa['results']['boards'])
        assert int(greedy['results']['boards']) == reference.KNOWN_QUEENS[n]


def test_brute_force_agrees_for_small_boards():
    for n in (4, 5, 6, 7):
        assert len(reference.enumerate_boards(n)) == reference.KNOWN_QUEENS[n]


@pytest.fixture(scope='session')
def sample_six(tnsample) -> dict:
    return tnsample.run('sample', ARTIFACTS / 'tests' / 'sample-nq6', n=6, samples=200,
                        order='treesa')


def test_two_hundred_six_queens_boards_obey_the_rules_and_are_uniform(sample_six):
    boards = sample_six['boards']
    assert len(boards) == 200
    assert all(reference.valid_board(board) for board in boards)
    assert sample_six['results']['boards_total'] == 4
    frequencies = {tuple(entry['board']): entry['count'] for entry in sample_six['results']['frequencies']}
    assert len(frequencies) == 4
    assert sum(frequencies.values()) == 200
    # Four equally likely boards: the sheet's reference run lands within +-6 of 50,
    # which is one binomial sigma (6.1). We test the distribution itself: a
    # chi-square with three degrees of freedom stays below its 99.9% point, 16.27.
    expected = 200 / 4
    chi2 = sum((count - expected) ** 2 / expected for count in frequencies.values())
    assert chi2 < 16.27, (chi2, frequencies)
    assert min(frequencies.values()) >= 200 / 4 - 4 * 6.12


def test_fifty_boards_of_eight_queens(tnsample):
    record = tnsample.run('sample', ARTIFACTS / 'tests' / 'sample-nq8', n=8, samples=200,
                          order='treesa')
    boards = record['boards']
    assert len(boards) == 200
    assert all(reference.valid_board(board) for board in boards)
    assert record['results']['boards_total'] == 92
    assert len(set(map(tuple, boards))) == record['results']['distinct']
    # Uniform draws out of 92 boards visit 92 (1 - (91/92)^200) ~= 82 of them.
    assert abs(record['results']['distinct']
               - reference.uniform_bookkeeping(92, 200)) <= 8, record['results']['distinct']



def test_sampling_repeats_its_statistics(tnsample):
    # omeco breaks equal-cost ties through hash iteration order, so the plan (and
    # therefore the exact draw sequence) is one sample of the planner; every plan
    # must still produce the same uniform law.
    runs = [tnsample.run('sample', ARTIFACTS / 'tests' / f'sample-nq6-{tag}', n=6,
                          samples=200, order='greedy') for tag in ('a', 'b')]
    for record in runs:
        assert len({tuple(board) for board in record['boards']}) == 4
        counts = [entry['count'] for entry in record['results']['frequencies']]
        chi2 = sum((count - 50.0) ** 2 / 50.0 for count in counts)
        assert chi2 < 16.27, (chi2, counts)
