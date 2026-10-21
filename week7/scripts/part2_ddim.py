"""Part 2: deterministic DDIM sampling (Algorithm 3).

Loads the Part 1 noise predictor, checks that DDIM is deterministic while DDPM
is not, and that both still reproduce the four clusters.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import _common
from _common import ARTIFACTS, COLORS, dump_json, save_figure, scatter_cloud, setup_matplotlib, style_axes

sys.path.insert(0, str(_common.ROOT))
from src import diffusion, metrics, mixture, model, schedule

SEED = 2027
N = 2000


def main() -> int:
    setup_matplotlib()
    _common.ensure_dirs()
    (ARTIFACTS / "part2").mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS / "part1" / "stage_25000.npz"
    if not path.exists():
        raise SystemExit(f"missing {path}; run scripts/part1_diffusion.py first")
    sched = schedule.linear_schedule()
    params = model.load_params(path)
    eps_learned = jax.jit(model.noise_predictor(params))

    def eps_exact(x, t):
        return mixture.exact_conditional_noise(x, float(sched.abars[t]))

    print("=== Part 2: deterministic DDIM ===")
    levels = diffusion.even_levels(1000)
    key = jax.random.PRNGKey(SEED)
    key, k1, k2 = jax.random.split(key, 3)
    starts = jax.random.normal(key, (N, 2))

    ddim_a = np.asarray(diffusion.ddim_sample(eps_learned, starts, sched, levels))
    ddim_b = np.asarray(diffusion.ddim_sample(eps_learned, starts, sched, levels))
    ddim_diff = float(np.max(np.abs(ddim_a - ddim_b)))
    ddpm_a = np.asarray(diffusion.ddpm_sample(eps_learned, starts, sched, levels, key=k1))
    ddpm_b = np.asarray(diffusion.ddpm_sample(eps_learned, starts, sched, levels, key=k2))
    same_cluster = float(np.mean(mixture.nearest_component(ddpm_a) == mixture.nearest_component(ddpm_b)))
    ddim_stats = metrics.cluster_stats(ddim_a)
    ddpm_stats = metrics.cluster_stats(ddpm_a)
    ddim_exact = np.asarray(diffusion.ddim_sample(eps_exact, starts, sched, levels))
    ddim_exact_stats = metrics.cluster_stats(ddim_exact)
    exact_stats = metrics.cluster_stats(np.asarray(mixture.sample(jax.random.fold_in(key, 7), N)))
    gap = float(np.nanmax(np.abs(ddim_stats["spreads"] - ddim_exact_stats["spreads"])))
    print(f"DDIM max |difference| over 1000 levels: {ddim_diff:.3e}")
    print(f"DDPM runs ending in the same cluster: {same_cluster:.3f} (chance 0.25)")
    print(f"DDIM populations {np.round(ddim_stats['populations'], 4)}")
    print(f"widths learned/exact-noise/exact-draw max: "
          f"{np.nanmax(ddim_stats['spreads']):.4f} / {np.nanmax(ddim_exact_stats['spreads']):.4f} / "
          f"{np.nanmax(exact_stats['spreads']):.4f}")

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.3))
    trail = starts[:5]
    trail_path = diffusion.ddim_sample_path(eps_learned, trail, sched, levels)
    for j in range(5):
        pts = np.stack([np.asarray(trail_path[lvl][j]) for lvl in range(1000, -1, -50)])
        axes[0].plot(pts[:, 0], pts[:, 1], lw=1.0, color=COLORS["learned"])
        axes[0].scatter(pts[-1, 0], pts[-1, 1], s=24, color=COLORS["learned"], zorder=4)
        axes[0].scatter(pts[0, 0], pts[0, 1], s=20, facecolors="none", edgecolors="#333333", zorder=4)
    axes[0].set_title(f"DDIM trajectories\nmax two-run difference = {ddim_diff:.0e}")
    style_axes(axes[0])
    scatter_cloud(axes[1], ddim_a, COLORS["learned"], size=3)
    axes[1].set_title(f"DDIM samples (1000 levels)\npopulations {ddim_stats['populations'].min():.3f}-{ddim_stats['populations'].max():.3f}")
    style_axes(axes[1])
    scatter_cloud(axes[2], ddpm_a, COLORS["aux"], size=3)
    axes[2].set_title(f"DDPM samples (1000 levels)\nsame-cluster rate {same_cluster:.2f}")
    style_axes(axes[2])
    scatter_cloud(axes[3], np.asarray(mixture.sample(jax.random.fold_in(key, 5), N)), COLORS["exact"], size=3)
    axes[3].set_title("exact target samples")
    style_axes(axes[3])
    for ax in axes[1:]:
        for center in mixture.CENTERS:
            ax.add_patch(plt.Circle(center, 2 * mixture.STD, fill=False, ls=":", lw=0.8, color="#666666"))
    fig.suptitle("One starting noise gives one deterministic DDIM sample; DDPM does not, yet both reproduce the target", y=1.03)
    save_figure(fig, "part2-determinism.png")

    payload = {
        "seed": SEED,
        "determinism": {
            "n": N,
            "ddim_max_abs_diff": ddim_diff,
            "ddpm_same_cluster_fraction": same_cluster,
            "ddim_populations": ddim_stats["populations"].tolist(),
            "ddim_spreads": np.nan_to_num(ddim_stats["spreads"]).tolist(),
            "ddim_centres": np.nan_to_num(ddim_stats["centres"]).tolist(),
            "ddpm_populations": ddpm_stats["populations"].tolist(),
            "ddpm_spreads": np.nan_to_num(ddpm_stats["spreads"]).tolist(),
            "ddpm_centres": np.nan_to_num(ddpm_stats["centres"]).tolist(),
            "ddim_exact_spreads": np.nan_to_num(ddim_exact_stats["spreads"]).tolist(),
            "exact_sample_spreads": np.nan_to_num(exact_stats["spreads"]).tolist(),
            "constants": {
                "exact_learned_width_gap": gap,
                "reference_band": [0.29, 0.32],
                "note": "the sheet's Part 1 allowance for the same quantity at 2000 samples is 0.03",
            },
        },
    }
    dump_json("part2_metrics.json", payload)
    np.savez_compressed(ARTIFACTS / "part2" / "ddim_samples.npz", ddim=ddim_a, ddpm=ddpm_a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
