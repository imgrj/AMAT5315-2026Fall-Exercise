"""Part 1 figure: the LP lower bound against the exact ground state, and the
integrality gap beside the naive rounding excess of the same 16 spin glasses.

Run from week12/:  python scripts/plot_gap.py   ->   artifacts/gap.png
"""
from __future__ import annotations

import numpy as np

import _common as common

import matplotlib.pyplot as plt


def main() -> None:
    common.style()
    run = common.ARTIFACTS / "gap"
    record = common.read_json(run / "run.json")
    instances = record["instances"]
    summary = record["summary"]

    ids = np.array([entry["id"] for entry in instances])
    lp = np.array([entry["lp_bound"] for entry in instances])
    exact = np.array([entry["exact_energy"] for entry in instances])
    gap = np.array([entry["integrality_gap"] for entry in instances])
    excess = np.array([entry["rounding_excess"] for entry in instances])

    figure, (bounds, misses) = plt.subplots(1, 2, figsize=(11.4, 4.3))

    width = 0.4
    bounds.bar(ids - width / 2, lp, width, color=common.COLORS["lp"], label="LP lower bound $z_{LP}$")
    bounds.bar(ids + width / 2, exact, width, color=common.COLORS["exact"],
               label="exact ground state $H^*$ (HiGHS MILP)")
    bounds.axhline(0, color="#3b4a63", linewidth=0.8)
    bounds.set_xlabel("spin-glass instance (22 spins, 52 bonds, seed 2026)")
    bounds.set_ylabel("energy")
    bounds.set_title("(a) Relaxation bound against the exact optimum")
    bounds.set_xticks(ids)
    bounds.grid(axis="y", alpha=0.35)
    bounds.legend(fontsize=8.5, loc="lower left", framealpha=0.95)
    bounds.text(
        0.03,
        0.96,
        f"mean $z_{{LP}}$ = {summary['mean_lp_bound']:.2f}   "
        f"mean $H^*$ = {summary['mean_exact_energy']:.2f}",
        transform=bounds.transAxes,
        fontsize=8.5,
        va="top",
        color=common.COLORS["ink"],
        bbox={"facecolor": "#0f172a", "edgecolor": "none", "alpha": 0.9},
    )

    misses.bar(ids - width / 2, gap, width, color=common.COLORS["gap"],
               label="integrality gap $H^* - z_{LP}$")
    misses.bar(ids + width / 2, excess, width, color=common.COLORS["excess"],
               label="rounding excess $H(\\tilde s) - H^*$")
    worst = summary["largest_rounding_excess_instance"]
    misses.annotate(
        f"instance {worst}: excess {summary['largest_rounding_excess']:.0f}",
        xy=(worst + width / 2, summary["largest_rounding_excess"]),
        xytext=(0.30, 0.99),
        textcoords="axes fraction",
        va="top",
        ha="left",
        fontsize=8.5,
        color=common.COLORS["ink"],
        bbox={"facecolor": "#0f172a", "edgecolor": "none", "alpha": 0.9},
        arrowprops={"arrowstyle": "->", "color": common.COLORS["muted"], "lw": 0.9},
    )
    misses.set_xlabel("spin-glass instance")
    misses.set_ylabel("energy above the optimum")
    misses.set_title("(b) What the relaxation hides, and what rounding costs")
    misses.set_xticks(ids)
    misses.grid(axis="y", alpha=0.35)
    misses.set_ylim(0, 70)
    misses.legend(fontsize=8.5, loc="upper right")
    misses.text(
        0.03,
        0.66,
        f"mean gap {summary['mean_integrality_gap']:.2f} "
        f"({100 * summary['mean_relative_gap']:.1f}% of $|H^*|$)\n"
        f"naive rounding suboptimal on {summary['suboptimal_roundings']}/"
        f"{len(instances)} instances\n"
        f"mean excess {summary['mean_rounding_excess']:.2f}",
        transform=misses.transAxes,
        fontsize=8.5,
        color=common.COLORS["ink"],
        bbox={"facecolor": "#0f172a", "edgecolor": "none", "alpha": 0.9},
    )

    common.save(figure, common.ARTIFACTS / "gap.png")
    print(f"square: LP {record['square']['lp_bound']:.6f}, exact "
          f"{record['square']['exact_energy']:.6f}, rounded "
          f"{record['square']['rounded_energy']:.6f}")
    print(f"diet cost {record['diet']['cost']:.6f}")


if __name__ == "__main__":
    main()
