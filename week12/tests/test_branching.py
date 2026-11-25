"""Part 2: the exact Branch-and-Bound search, checked on the frustrated
tetrahedron and on the 48 saved spin glasses.

The specification is `benchmark.bnb` in `branchopt.design.toml`: depth-first, no
incumbent at the start, prune infeasible LPs and bounds `>= z_best - 1e-6`, split
a free spin into `x_i = 0` and `x_i = 1`, and visit the child nearer the LP value
first (at 0.5, `x_i = 1` first).
"""
from __future__ import annotations

import pytest


def test_tetrahedron_visits_five_nodes(bnb_most):
    """The pruned search of K4 with h_0 = +1: 5 nodes instead of 7."""
    tetra = bnb_most["tetrahedron"]
    assert tetra["nodes"] == 5
    assert tetra["lp_solves"] == 5
    assert tetra["energy"] == pytest.approx(-3.0, abs=1e-9)

    trace = tetra["trace"]
    bounds = [row["bound"] for row in trace]
    assert bounds == pytest.approx([-6.0, -4.0, -3.0, -3.0, -2.0], abs=1e-9)
    assert [row["branch"] for row in trace] == ["root", "x0=1", "x0=1,x1=1", "x0=1,x1=0", "x0=0"]
    assert trace[0]["status"] == "split x0"
    assert trace[1]["status"] == "split x1"
    assert trace[2]["status"] == "incumbent"
    assert trace[3]["status"] == "pruned-by-bound"
    assert trace[4]["status"] == "pruned-by-bound"
    # The root splits on spin 0, whose LP value is exactly 1/2.
    assert trace[0]["free"] == 4


def test_every_rule_matches_the_highs_optimum(bnb_most, bnb_strong):
    """Both rules return the same energy as HiGHS on all 48 instances."""
    for run in (bnb_most, bnb_strong):
        for inst in run["instances"]:
            assert inst["error"] < 1e-9, (run["policy"], inst["id"], inst["error"])
            assert inst["energy"] == pytest.approx(inst["highs_energy"], abs=1e-9)


def test_most_frac_costs_one_lp_per_node(bnb_most):
    """The cheap rule never probes, so LP solves equal explored nodes."""
    for inst in bnb_most["instances"]:
        assert inst["lp_solves"] == inst["nodes"]


def test_strong_branching_spends_trial_lps(bnb_strong):
    """Strong Branching solves two trial LPs per candidate at every node."""
    for inst in bnb_strong["instances"]:
        assert inst["lp_solves"] > inst["nodes"]
        assert inst["lp_solves"] >= 2 * (inst["nodes"] - 1) + 1
        assert inst["lp_solves"] / inst["nodes"] > 2.0


def test_strong_branching_shrinks_the_tree(bnb_most, bnb_strong):
    """A smaller tree at a much higher LP cost per node, as the sheet reports."""
    most_nodes = bnb_most["summary"]["mean_nodes"]
    strong_nodes = bnb_strong["summary"]["mean_nodes"]
    most_lp = bnb_most["summary"]["mean_lp_solves"]
    strong_lp = bnb_strong["summary"]["mean_lp_solves"]
    assert strong_nodes < most_nodes
    assert most_nodes / strong_nodes > 2.0
    assert strong_lp > 4.0 * most_lp
    assert strong_lp / strong_nodes > 10.0


def test_node_counts_are_consistent_across_rules(bnb_most, bnb_strong):
    """48 instances each, at least one explored node each."""
    for run in (bnb_most, bnb_strong):
        assert len(run["instances"]) == 48
        assert all(inst["nodes"] >= 1 for inst in run["instances"])
        assert run["prune_epsilon"] == pytest.approx(1e-6)


def test_energies_are_at_most_the_rounded_upper_bounds(bnb_most):
    """Every explored leaf is a legal spin state, so it bounds the optimum."""
    for inst in bnb_most["instances"]:
        assert inst["energy"] <= inst["highs_energy"] + 1e-9
