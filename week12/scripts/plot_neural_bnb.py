"""Part 3 figure: the imitation-learning curves and the explored nodes of the
learned rule beside most-frac and Strong Branching on 16 unseen spin glasses.

Run from week12/:  python scripts/plot_neural_bnb.py   ->   artifacts/neural_bnb.png
"""
from __future__ import annotations

import numpy as np

import _common as common

import matplotlib.pyplot as plt


def main() -> None:
    common.style()
    model = common.read_json(common.ARTIFACTS / "model.json")
    run = common.read_json(common.ARTIFACTS / "eval_test" / "run.json")

    history = model["history"]
    epochs = np.array(history["epoch"])
    train = np.array(history["train_ce"])
    val = np.array(history["val_ce"])

    figure, (curves, nodes) = plt.subplots(1, 2, figsize=(11.4, 4.3))

    curves.plot(epochs, train, color=common.COLORS["neural"], linewidth=1.5,
                label="training cross-entropy")
    curves.plot(epochs, val, color=common.COLORS["strong"], linewidth=1.5,
                label="validation cross-entropy")
    best = model["best_epoch"]
    curves.axvline(best, color=common.COLORS["root-neural"], linestyle=":", linewidth=1.1)
    curves.annotate(
        f"kept epoch {best}\nvalidation {model['best_val_ce']:.3f}",
        xy=(best, model["best_val_ce"]),
        xytext=(0.48, 0.86),
        textcoords="axes fraction",
        va="top",
        fontsize=8.5,
        color=common.COLORS["ink"],
        bbox={"facecolor": "#0f172a", "edgecolor": "none", "alpha": 0.9},
        arrowprops={"arrowstyle": "->", "color": common.COLORS["muted"], "lw": 0.9},
    )
    curves.set_xlabel("epoch")
    curves.set_ylabel("cross-entropy against Strong Branching (nats)")
    curves.set_title("(a) Imitating the strong rule")
    curves.grid(alpha=0.35)
    curves.legend(fontsize=8.5, loc="upper right")
    curves.text(
        0.03,
        0.58,
        f"{model['train']['nodes']} training nodes / {len(model['train']['instances'])} spin glasses\n"
        f"{model['validation']['nodes']} validation nodes / {len(model['validation']['instances'])} spin glasses\n"
        f"instance overlap {model['overlap']}",
        transform=curves.transAxes,
        fontsize=8.5,
        va="top",
        color=common.COLORS["ink"],
        bbox={"facecolor": "#0f172a", "edgecolor": "none", "alpha": 0.9},
    )

    index = np.arange(1, len(run["instances"]) + 1)
    for policy in ("most-frac", "neural", "strong"):
        values = np.array(common.sorted_nodes(run, policy))
        nodes.step(index, values, where="mid", color=common.COLORS[policy],
                   linewidth=1.6, label=policy)
    nodes.set_yscale("log")
    nodes.set_xlabel("16 unseen 22-spin, 52-bond spin glasses (seed 3026), sorted")
    nodes.set_ylabel("explored nodes")
    nodes.set_title("(b) Nodes explored on instances the network never saw")
    nodes.grid(alpha=0.35, which="both")
    nodes.legend(fontsize=8.5, loc="upper left")
    summary = run["summary"]
    nodes.text(
        0.97,
        0.05,
        f"mean most-frac {summary['policies']['most-frac']['mean_nodes']:.1f}   "
        f"neural {summary['policies']['neural']['mean_nodes']:.1f}   "
        f"strong {summary['policies']['strong']['mean_nodes']:.1f}",
        transform=nodes.transAxes,
        fontsize=8.5,
        va="bottom",
        ha="right",
        color=common.COLORS["ink"],
        bbox={"facecolor": "#0f172a", "edgecolor": "none", "alpha": 0.9},
    )

    common.save(figure, common.ARTIFACTS / "neural_bnb.png")
    print(f"probe probability difference {run['model']['probe_probability_max_diff']:.2e}")
    print(f"largest HiGHS error {summary['largest_highs_error']:.2e}")
    print(f"LP solves equal nodes (non-Strong): {summary['lp_solves_equal_nodes_for_non_strong']}")


if __name__ == "__main__":
    main()
