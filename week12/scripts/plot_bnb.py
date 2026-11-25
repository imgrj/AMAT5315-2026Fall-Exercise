"""Part 2 figure: explored nodes and LP solves per instance for the cheap and the
strong branching rule, each curve sorted independently.

Run from week12/:  python scripts/plot_bnb.py   ->   artifacts/bnb.png
"""
from __future__ import annotations

import numpy as np

import _common as common

import matplotlib.pyplot as plt


def main() -> None:
    common.style()
    most = common.read_json(common.ARTIFACTS / "bnb_most" / "run.json")
    strong = common.read_json(common.ARTIFACTS / "bnb_strong" / "run.json")

    nodes_most = np.array(common.sorted_nodes_bnb(most))
    nodes_strong = np.array(common.sorted_nodes_bnb(strong))
    lp_most = np.array(sorted(entry["lp_solves"] for entry in most["instances"]))
    lp_strong = np.array(sorted(entry["lp_solves"] for entry in strong["instances"]))
    index = np.arange(1, len(nodes_most) + 1)

    figure, (nodes, solves) = plt.subplots(1, 2, figsize=(11.4, 4.3))

    nodes.step(index, nodes_most, where="mid", color=common.COLORS["most-frac"],
               linewidth=1.6, label="most-frac")
    nodes.step(index, nodes_strong, where="mid", color=common.COLORS["strong"],
               linewidth=1.6, label="Strong Branching")
    nodes.set_yscale("log")
    nodes.set_xlabel("instances, each curve sorted from smallest to largest")
    nodes.set_ylabel("explored nodes")
    nodes.set_title("(a) Tree size: the same 48 spin glasses, 22 spins / 52 bonds")
    nodes.grid(alpha=0.35, which="both")
    nodes.legend(fontsize=8.5, loc="upper left")
    m_most = most["summary"]["mean_nodes"]
    m_strong = strong["summary"]["mean_nodes"]
    nodes.text(
        0.97,
        0.05,
        f"mean {m_most:.2f} -> {m_strong:.2f} nodes\n({m_most / m_strong:.2f}$\\times$ smaller trees)",
        transform=nodes.transAxes,
        fontsize=8.5,
        va="bottom",
        ha="right",
        color=common.COLORS["ink"],
        bbox={"facecolor": "#0f172a", "edgecolor": "none", "alpha": 0.9},
    )

    solves.step(index, lp_most, where="mid", color=common.COLORS["most-frac"],
                linewidth=1.6, label="most-frac (1 LP per node)")
    solves.step(index, lp_strong, where="mid", color=common.COLORS["strong"],
                linewidth=1.6, label="Strong Branching (trial LPs per node)")
    solves.set_yscale("log")
    solves.set_xlabel("instances, each curve sorted independently")
    solves.set_ylabel("LP solves")
    solves.set_title("(b) Cost: every candidate spin costs two trial LPs")
    solves.grid(alpha=0.35, which="both")
    solves.legend(fontsize=8.5, loc="upper left")
    solves.text(
        0.97,
        0.05,
        f"mean {most['summary']['mean_lp_solves']:.2f} -> "
        f"{strong['summary']['mean_lp_solves']:.2f} LP solves\n"
        f"{strong['summary']['mean_lp_solves'] / m_strong:.1f} LP solves per node\n"
        f"mean time {most['summary']['mean_elapsed_ms']:.2f} ms -> "
        f"{strong['summary']['mean_elapsed_ms']:.2f} ms",
        transform=solves.transAxes,
        fontsize=8.5,
        va="bottom",
        ha="right",
        color=common.COLORS["ink"],
        bbox={"facecolor": "#0f172a", "edgecolor": "none", "alpha": 0.9},
    )

    common.save(figure, common.ARTIFACTS / "bnb.png")
    print(
        f"most-frac {m_most:.2f} nodes / {most['summary']['mean_lp_solves']:.2f} LPs / "
        f"{most['summary']['mean_elapsed_ms']:.2f} ms; strong {m_strong:.2f} nodes / "
        f"{strong['summary']['mean_lp_solves']:.2f} LPs / "
        f"{strong['summary']['mean_elapsed_ms']:.2f} ms"
    )
    print(
        f"largest error against HiGHS: {max(most['summary']['largest_error'], strong['summary']['largest_error']):.2e}"
    )


if __name__ == "__main__":
    main()
