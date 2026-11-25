"""Part 4 figure: size generalisation of the four rules and the wall-clock split
between LP solves and inference overhead on the larger spin glasses.

Run from week12/:  python scripts/plot_eval.py   ->   artifacts/eval.png
"""
from __future__ import annotations

import numpy as np

import _common as common

import matplotlib.pyplot as plt

POLICIES = ("most-frac", "root-neural", "neural", "strong")


def table(run: dict) -> None:
    print(f"\n{run['command']}")
    print(f"{'rule':<12}{'nodes':>9}{'total ms':>11}{'LP ms':>9}{'infer ms':>10}{'LP solves':>11}")
    for policy in POLICIES:
        p = run["summary"]["policies"][policy]
        print(
            f"{policy:<12}{p['mean_nodes']:>9.2f}{p['mean_total_ms']:>11.2f}"
            f"{p['mean_lp_ms']:>9.2f}{p['mean_infer_ms']:>10.3f}{p['mean_lp_solves']:>11.2f}"
        )
    print(
        f"largest error against HiGHS {run['summary']['largest_highs_error']:.2e}; "
        f"LP solves equal nodes for every non-Strong rule: "
        f"{run['summary']['lp_solves_equal_nodes_for_non_strong']}"
    )


def main() -> None:
    common.style()
    small = common.read_json(common.ARTIFACTS / "eval_test" / "run.json")
    large = common.read_json(common.ARTIFACTS / "eval_transfer" / "run.json")

    figure, (generalisation, timing) = plt.subplots(1, 2, figsize=(11.6, 4.4))

    width = 0.2
    positions = np.arange(2)
    for k, policy in enumerate(POLICIES):
        values = [
            small["summary"]["policies"][policy]["mean_nodes"],
            large["summary"]["policies"][policy]["mean_nodes"],
        ]
        offsets = positions + (k - 1.5) * width
        bars = generalisation.bar(offsets, values, width, color=common.COLORS[policy],
                                 label=policy)
        for bar, value in zip(bars, values):
            generalisation.text(bar.get_x() + bar.get_width() / 2, value * 1.04,
                                f"{value:.1f}", ha="center", va="bottom", fontsize=7.8,
                                color=common.COLORS["ink"])
    generalisation.set_xticks(positions)
    generalisation.set_xticklabels([
        "16 unseen 22-spin\n52 bonds (seed 3026)",
        "8 larger 30-spin\n72 bonds (seed 4026)",
    ])
    generalisation.set_yscale("log")
    generalisation.set_ylabel("mean explored nodes")
    generalisation.set_title("(a) The 22-spin network transfers to 30 spins")
    generalisation.grid(axis="y", alpha=0.35, which="both")
    generalisation.legend(fontsize=8, loc="upper left", ncol=2)

    lp = [large["summary"]["policies"][p]["mean_lp_ms"] for p in POLICIES]
    infer = [large["summary"]["policies"][p]["mean_infer_ms"] for p in POLICIES]
    x = np.arange(len(POLICIES))
    timing.bar(x, lp, 0.55, color=common.COLORS["lp"], label="LP solves + tree")
    timing.bar(x, infer, 0.55, bottom=lp, color=common.COLORS["neural"],
               label="feature extraction + network")
    totals = [a + b for a, b in zip(lp, infer)]
    for k, policy in enumerate(POLICIES):
        timing.text(k, totals[k] * 1.02, f"{totals[k]:.1f} ms", ha="center", va="bottom",
                    fontsize=8.2, color=common.COLORS["ink"])
    timing.set_ylim(0, max(totals) * 1.25)
    timing.set_xticks(x)
    timing.set_xticklabels(POLICIES)
    timing.set_ylabel("mean wall time per search (ms)")
    timing.set_title("(b) Where the time goes on 30-spin, 72-bond graphs")
    timing.grid(axis="y", alpha=0.35)
    timing.legend(fontsize=8, loc="upper left")
    timing.text(
        0.97,
        0.62,
        f"inference stays under {max(infer):.1f} ms\n"
        f"Strong Branching's trial LPs cost {lp[3]:.0f} ms",
        transform=timing.transAxes,
        fontsize=8.5,
        ha="right",
        va="top",
        color=common.COLORS["ink"],
        bbox={"facecolor": "#0f172a", "edgecolor": "none", "alpha": 0.9},
    )

    common.save(figure, common.ARTIFACTS / "eval.png")
    table(small)
    table(large)


if __name__ == "__main__":
    main()
