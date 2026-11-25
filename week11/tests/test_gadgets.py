"""Part 2: the seven gate gadgets and the three-gate circuit.

The truth tables are checked against the sheet's own expressions, written out
independently in `scripts/reference.py`, and against the hand arithmetic printed
on pages 9 and 10 of the learning sheet.
"""
from __future__ import annotations

import pytest

import reference

from conftest import ARTIFACTS

FLOORS = {"AND": -3, "OR": -3, "NOT": -1, "COPY": -1, "SET0": -1, "SET1": -1, "XOR": -4}


def row_energies(record: dict, gate: str) -> dict[tuple[int, ...], float]:
    return {tuple(row["bits"]): float(row["energy"]) for row in record["gates"][gate]["rows"]}


def test_every_gate_row_matches_the_sheet_expression(gadgets_run):
    for name, (bits, table) in reference.gates().items():
        rows = row_energies(gadgets_run, name)
        assert set(rows) == set(table), name
        for row, energy in table.items():
            assert rows[row] == pytest.approx(energy), (name, row)


def test_allowed_rows_sit_on_their_floor_and_forbidden_rows_do_not(gadgets_run):
    for name, (_, table) in reference.gates().items():
        rows = row_energies(gadgets_run, name)
        floor = min(rows.values())
        assert floor == FLOORS[name], name
        for row, energy in rows.items():
            if reference.gate_allowed(name, row):
                assert energy == floor, (name, row, energy)
            else:
                assert energy >= floor + 2, (name, row, energy)
        assert float(gadgets_run['gates'][name]['floor']) == floor


def test_xor_needs_its_helper_and_obeys_the_penalty_formula(gadgets_run):
    gate = gadgets_run['gates']['XOR']
    assert gate['helper'] == 'w'
    # H_XOR + 4 = 2 (x + y - z - 2 r)^2 at the recorded helper r.
    for row in gate['rows']:
        a, b, z = row['bits']
        w = row['helper']
        assert w in (0, 1)
        assert float(row['energy']) + 4.0 == pytest.approx(2.0 * (a + b - z - 2 * w) ** 2)
    # The helper is essential: minimizing is not the same as fixing w.
    assert gate['floor'] == -4
    assert float(gate['forbidden_lowest']) == -2


def test_hand_arithmetic_of_the_sheet(gadgets_run):
    # Page 9: with (a,b,z) = (1,1,1) every spin is +1 and H_AND = -3.
    assert row_energies(gadgets_run, 'AND')[(1, 1, 1)] == -3
    # Page 9: the COPY and NOT ladders of Figure 8.
    assert row_energies(gadgets_run, 'NOT') == {(0, 0): 1, (0, 1): -1, (1, 0): -1, (1, 1): 1}
    assert row_energies(gadgets_run, 'COPY') == {(0, 0): -1, (0, 1): 1, (1, 0): 1, (1, 1): -1}


def test_the_three_gate_circuit_reaches_its_floor_three_times(gadgets_run):
    circuit = gadgets_run['circuit']
    rows = {tuple(row['bits']): float(row['energy']) for row in circuit['rows']}
    expected = {(a, b, t, o): reference.circuit_energy(a, b, t, o)
                for a in (0, 1) for b in (0, 1) for t in (0, 1) for o in (0, 1)}
    assert set(rows) == set(expected)
    for bits, energy in expected.items():
        assert rows[bits] == pytest.approx(energy), bits
    assert float(circuit['floor']) == -5
    at_floor = sorted(bits for bits, energy in rows.items() if energy == -5)
    assert at_floor == [(0, 0, 0, 1), (0, 1, 0, 1), (1, 0, 0, 1)]
    assert circuit['states_at_floor'] == 3
    # Page 6: an inconsistent shared wire (a,b) = (1,0) with t = 1 lands at +1.
    assert rows[(1, 0, 1, 1)] == 1
    # Figure 10's table: the row (1,1,1,0) is at -3.
    assert rows[(1, 1, 1, 0)] == -3


def test_the_run_records_its_schema_and_seed(gadgets_run):
    assert gadgets_run['schema'] == 'tnsample/1'
    assert gadgets_run['mode'] == 'gadgets'
    assert gadgets_run['seed'] == 2026
    assert set(gadgets_run['summary']['floors']) == set(FLOORS)
    assert float(gadgets_run['summary']['circuit_floor']) == -5


def test_the_stdout_table_has_the_designed_header(tnsample):
    out = ARTIFACTS / 'tests' / 'gadgets-stdout'
    completed = tnsample('--mode', 'gadgets', '--seed', '2026', '--out', str(out))
    assert completed.returncode == 0, completed.stderr
    lines = completed.stdout.strip().splitlines()
    assert lines[0] == 'mode\tmetric\tvalue\truntime_ms'
    assert all(line.startswith('gadgets\t') for line in lines[1:])
    assert any(line.split('\t')[1] == 'AND.floor' for line in lines[1:])
