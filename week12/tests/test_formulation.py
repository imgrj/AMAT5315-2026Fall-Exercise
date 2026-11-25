"""Part 1: the generator, the mixed-integer formulation, its relaxation and the
diet LP, checked against enumeration and against the saved evidence.

The specification is `branchopt.design.toml` and Section 3 of the learning sheet:
`x_i = (1 - s_i)/2`, `z_ij = x_i xor x_j`, four inequalities per bond, and the
linear objective of Eq. 6.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import build_instance, brute_force_ground_state, energy, milp_energy


def test_generator_matches_the_design_file(small_run):
    """Edges, couplings and fields of a saved run match an independent draw."""
    for inst in small_run["instances"]:
        edges, couplings, fields = build_instance(
            small_run["seed"], inst["id"], inst["n"], inst["m"]
        )
        assert [tuple(e) for e in inst["edges"]] == edges
        assert inst["couplings"] == couplings
        assert inst["fields"] == fields


def test_milp_optimum_equals_full_enumeration(small_run):
    """HiGHS's integer optimum is the lowest energy over all 2^n spin states."""
    for inst in small_run["instances"]:
        edges, couplings, fields = build_instance(
            small_run["seed"], inst["id"], inst["n"], inst["m"]
        )
        best, spins = brute_force_ground_state(edges, couplings, fields)
        assert inst["exact_energy"] == pytest.approx(best, abs=1e-9)
        # the same spins read through the 0/1 convention
        bits = (1.0 - spins) / 2.0
        assert energy(edges, couplings, fields, 1.0 - 2.0 * bits) == pytest.approx(
            inst["exact_energy"], abs=1e-9
        )


def test_relaxation_bound_is_minus_the_bond_count(gap):
    """At x = 1/2 every bond takes its preferred value, so z_LP = -m."""
    for inst in gap["instances"]:
        assert inst["lp_bound"] == pytest.approx(-float(inst["m"]), abs=1e-9)
        assert inst["fractional_spins"] == inst["n"]
        assert all(xi == pytest.approx(0.5, abs=1e-7) for xi in inst["lp_spins"])


def test_saved_lp_solution_is_feasible_and_attains_its_bound(gap):
    """The four bond inequalities hold, and Eq. 6 reproduces the reported bound."""
    for inst in gap["instances"]:
        x = np.asarray(inst["lp_spins"], dtype=float)
        z = np.asarray(inst["lp_bonds"], dtype=float)
        for (u, v), ze in zip(inst["edges"], z):
            assert ze <= x[u] + x[v] + 1e-9
            assert ze <= 2.0 - x[u] - x[v] + 1e-9
            assert ze >= x[u] - x[v] - 1e-9
            assert ze >= x[v] - x[u] - 1e-9
            assert -1e-12 <= ze <= 1.0 + 1e-12
        assert milp_energy(inst, x, z) == pytest.approx(inst["lp_bound"], abs=1e-9)
        assert inst["lp_bound"] <= inst["exact_energy"] + 1e-9
        assert inst["integrality_gap"] == pytest.approx(
            inst["exact_energy"] - inst["lp_bound"], abs=1e-12
        )


def test_rounding_reproduces_the_saved_excess(gap):
    """Nearest rounding gives a legal spin state whose energy matches the record."""
    for inst in gap["instances"]:
        x = np.asarray(inst["lp_spins"], dtype=float)
        bits = (x >= 0.5).astype(float)
        spins = 1.0 - 2.0 * bits
        edges = [tuple(e) for e in inst["edges"]]
        value = energy(edges, inst["couplings"], inst["fields"], spins)
        assert value == pytest.approx(inst["rounded_energy"], abs=1e-9)
        assert inst["rounding_excess"] == pytest.approx(
            inst["rounded_energy"] - inst["exact_energy"], abs=1e-12
        )
        # The rounded state is a legal spin assignment of the MILP.
        z = np.array([abs(bits[u] - bits[v]) for u, v in edges])
        assert milp_energy(inst, bits, z) == pytest.approx(value, abs=1e-9)


def test_opening_square(gap):
    """LP bound -4 at x = 1/2, exact optimum -2, nearest rounding +2."""
    square = gap["square"]
    assert square["lp_bound"] == pytest.approx(-4.0, abs=1e-9)
    assert square["lp_spins"] == pytest.approx([0.5] * 4, abs=1e-9)
    assert square["exact_energy"] == pytest.approx(-2.0, abs=1e-9)
    assert square["rounded_energy"] == pytest.approx(2.0, abs=1e-9)
    assert square["rounded_excess"] == pytest.approx(4.0, abs=1e-9)


def test_square_enumeration_has_a_frustrated_optimum():
    """No state reaches -4: the four couplings cannot all be satisfied."""
    edges = [(0, 1), (0, 3), (1, 2), (2, 3)]
    couplings = [1.0, -1.0, 1.0, 1.0]
    fields = [0.0] * 4
    best, _ = brute_force_ground_state(edges, couplings, fields)
    assert best == pytest.approx(-2.0)
    values = []
    states = 1 - 2 * ((np.arange(16)[:, None] >> np.arange(4)[None, :]) & 1)
    for spins in states:
        values.append(energy(edges, couplings, fields, spins))
    assert min(values) == pytest.approx(-2.0)
    assert max(values) == pytest.approx(2.0)


def test_diet_optimum_is_the_sheet_value(gap):
    """min 6 x1 + 3.5 x2 at (89/28, 8/7) costs 323/14."""
    diet = gap["diet"]
    assert diet["cost"] == pytest.approx(323.0 / 14.0, abs=1e-9)
    assert diet["x1"] == pytest.approx(89.0 / 28.0, abs=1e-9)
    assert diet["x2"] == pytest.approx(8.0 / 7.0, abs=1e-9)
    assert 7.0 * diet["x2"] >= 8.0 - 1e-9
    assert 4.0 * diet["x1"] + 2.0 * diet["x2"] >= 15.0 - 1e-9
    assert 2.0 * diet["x1"] + diet["x2"] >= 3.0 - 1e-9
