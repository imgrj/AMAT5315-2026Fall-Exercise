"""Part 2: fewer-step sampling with DDIM and flow matching.

Loads the Part 1 noise predictor, checks that DDIM is deterministic while DDPM
is not, trains a velocity network from scratch by flow matching, compares the
learned field with the exact marginal velocity, and sweeps DDPM, DDIM and Euler
flow matching over 1..100 steps (and 1000 levels for DPPM/DDIM).
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
from src import diffusion, flow, metrics, mixture, model, schedule, train

SEED = 20260708
UPDATES = 25000
BATCH = 512
LR = 1e-3
N_DETERMINISM = 2000
N_SWEEP = 4000
STEPS = [1, 2, 5, 10, 20, 50, 100]
S_VALUES = [0.1, 0.5, 0.9]


def load_final_predictor():
    path = ARTIFACTS / "part1" / "stage_25000.npz"
    if not path.exists():
        raise SystemExit(f"missing {path}; run scripts/part1_diffusion.py first")
    return model.load_params(path)


def make_samplers(sched, params):
    eps_learned = jax.jit(model.noise_predictor(params))

    def eps_exact(x, t):
        return mixture.exact_conditional_noise(x, float(sched.abars[t]))

    def velocity_exact(x, s):
        return flow.exact_marginal_velocity(x, float(s))

    def ddim(eps_fn, start, levels):
        return np.asarray(diffusion.ddim_sample(eps_fn, start, sched, levels))

    def ddpm(eps_fn, start, levels, key):
        return np.asarray(diffusion.ddpm_sample(eps_fn, start, sched, levels, key=key))

    return eps_learned, eps_exact, velocity_exact, ddim, ddpm


def main() -> int:
    setup_matplotlib()
    _common.ensure_dirs()
    (ARTIFACTS / "part2").mkdir(parents=True, exist_ok=True)
    sched = schedule.linear_schedule()
    params = load_final_predictor()
    eps_learned, eps_exact, velocity_exact, ddim, ddpm = make_samplers(sched, params)
    key = jax.random.PRNGKey(SEED)
    print("=== Part 2: fewer-step sampling ===")

    # ----------------------------------------------------------- determinism
    levels_full = diffusion.even_levels(1000)
    starts = jax.random.normal(key, (N_DETERMINISM, 2))
    ddim_a = ddim(eps_learned, starts, levels_full)
    ddim_b = ddim(eps_learned, starts, levels_full)
    ddim_diff = float(np.max(np.abs(ddim_a - ddim_b)))
    key, k1, k2 = jax.random.split(key, 3)
    ddpm_a = ddpm(eps_learned, starts, levels_full, k1)
    ddpm_b = ddpm(eps_learned, starts, levels_full, k2)
    labels_a = mixture.nearest_component(ddpm_a)
    labels_b = mixture.nearest_component(ddpm_b)
    same_cluster = float(np.mean(labels_a == labels_b))
    ddim_stats = metrics.cluster_stats(ddim_a)
    ddpm_stats = metrics.cluster_stats(ddpm_a)
    # controls: the same starting noise with exact conditional-mean noise, and an
    # independent exact draw of the same size, isolate finite-sample variation.
    ddim_exact = ddim(eps_exact, starts, levels_full)
    ddim_exact_stats = metrics.cluster_stats(ddim_exact)
    exact_stats = metrics.cluster_stats(np.asarray(mixture.sample(jax.random.fold_in(key, 7), N_DETERMINISM)))
    exact_learned_width_gap = float(
        np.nanmax(np.abs(ddim_stats["spreads"] - ddim_exact_stats["spreads"]))
    )
    print(f"exact-noise DDIM widths max {np.nanmax(ddim_exact_stats['spreads']):.4f}; "
          f"exact target widths max {np.nanmax(exact_stats['spreads']):.4f}; "
          f"learned vs exact-noise gap {exact_learned_width_gap:.4f}")
    print(f"DDIM max |difference| over 1000 levels: {ddim_diff:.3e}")
    print(f"DDPM runs ending in the same cluster: {same_cluster:.3f} (chance 0.25)")
    print("DDIM populations", ddim_stats["populations"], "spreads", ddim_stats["spreads"].ravel())

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.3))
    # trajectories of the first five starting points
    trail_starts = starts[:5]
    path_a = diffusion.ddim_sample_path(eps_learned, trail_starts, sched, levels_full)
    for j in range(5):
        pts = np.stack([np.asarray(path_a[lvl][j]) for lvl in range(1000, -1, -50)])
        axes[0].plot(pts[:, 0], pts[:, 1], lw=1.0, color=COLORS["learned"])
        axes[0].scatter(pts[-1, 0], pts[-1, 1], s=24, color=COLORS["learned"], zorder=4)
        axes[0].scatter(pts[0, 0], pts[0, 1], s=20, facecolors="none", edgecolors="#333333", zorder=4)
    axes[0].set_title(f"DDIM trajectories\nmax two-run difference = {ddim_diff:.0e}")
    style_axes(axes[0])
    scatter_cloud(axes[1], ddim_a, COLORS["learned"], size=3)
    axes[1].set_title(f"DDIM samples (1000 levels)\npop spread {ddim_stats['populations'].min():.3f}-{ddim_stats['populations'].max():.3f}")
    style_axes(axes[1])
    scatter_cloud(axes[2], ddpm_a, COLORS["aux"], size=3)
    axes[2].set_title(f"DDPM samples (1000 levels)\nsame-cluster rate {same_cluster:.2f}")
    style_axes(axes[2])
    scatter_cloud(axes[3], np.asarray(mixture.sample(jax.random.fold_in(key, 5), N_DETERMINISM)), COLORS["exact"], size=3)
    axes[3].set_title("exact target samples")
    style_axes(axes[3])
    for ax in axes[1:]:
        for center in mixture.CENTERS:
            ax.add_patch(plt.Circle(center, 2 * mixture.STD, fill=False, ls=":", lw=0.8, color="#666666"))
    fig.suptitle("One starting noise gives one deterministic DDIM sample; DDPM does not, yet both reproduce the target", y=1.03)
    save_figure(fig, "part2-determinism.png")

    # ------------------------------------------------------ velocity training
    key, train_key = jax.random.split(key)
    velocity_path = ARTIFACTS / "part2" / "velocity_final.npz"
    history_path = ARTIFACTS / "part2" / "velocity_history.json"
    if "--reuse-velocity" in sys.argv and velocity_path.exists():
        print("reusing the saved velocity network")
        velocity_params = model.load_params(velocity_path)
        history = json.loads(history_path.read_text(encoding="utf-8"))
    else:
        print(f"training the velocity network for {UPDATES} updates ...")
        stages, history = train.train_velocity_network(train_key, updates=UPDATES, batch=BATCH, lr=LR)
        velocity_params = stages[UPDATES]
        model.save_params(velocity_path, velocity_params)
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    velocity_learned = jax.jit(model.velocity_predictor(velocity_params))
    print(f"velocity training finished in {history['seconds']:.1f}s, final loss {history['loss'][-1]:.4f}")

    # relative RMS over the region the field has to transport: exact path samples
    field_key = jax.random.fold_in(key, 21)
    field_clean = mixture.sample(field_key, 4000)
    field_noise = jax.random.normal(jax.random.fold_in(field_key, 1), field_clean.shape)
    velocity_rel, velocity_rel_grid, velocity_rel_wide, velocity_panels = [], [], [], []
    for s in S_VALUES:
        samples = (1.0 - s) * np.asarray(field_clean) + s * np.asarray(field_noise)
        exact = np.asarray(velocity_exact(samples, s))
        learned = np.asarray(velocity_learned(jnp.asarray(samples), s))
        rel = flow.velocity_relative_rms(learned, exact)
        radius = float(np.quantile(np.linalg.norm(samples, axis=-1), 0.995))
        grid_1d = np.linspace(-radius, radius, 17)
        gx, gy = np.meshgrid(grid_1d, grid_1d)
        grid = np.stack([gx.ravel(), gy.ravel()], axis=-1)
        inside = np.linalg.norm(grid, axis=-1) <= radius
        grid_rel = flow.velocity_relative_rms(
            np.asarray(velocity_learned(jnp.asarray(grid[inside]), s)),
            np.asarray(velocity_exact(grid[inside], s)),
        )
        wide_1d = np.linspace(-6.0, 6.0, 17)
        wx, wy = np.meshgrid(wide_1d, wide_1d)
        wide = np.stack([wx.ravel(), wy.ravel()], axis=-1)
        wide_rel = flow.velocity_relative_rms(
            np.asarray(velocity_learned(jnp.asarray(wide), s)),
            np.asarray(velocity_exact(wide, s)),
        )
        velocity_rel.append(rel)
        velocity_rel_grid.append(grid_rel)
        velocity_rel_wide.append(wide_rel)
        velocity_panels.append((s, gx, gy, grid, inside, learned, exact, rel))
        print(f"velocity s={s}: relative RMS {rel:.4f} (samples), {grid_rel:.4f} (in-support grid), {wide_rel:.4f} (wide grid)")

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.9))
    for ax, (s, gx, gy, grid, inside, learned, exact, rel) in zip(axes, velocity_panels):
        vec_learned = np.asarray(velocity_learned(jnp.asarray(grid), s))
        vec_exact = np.asarray(velocity_exact(grid, s))
        spacing = float(gx[0, 1] - gx[0, 0])
        norms = np.linalg.norm(vec_exact, axis=-1)
        cap = float(np.quantile(norms, 0.90))
        factor = 0.55 * spacing / cap
        for vec, color, label, width in (
            (vec_exact, "#777777", "exact $v$", 0.007),
            (vec_learned, COLORS["learned"], "learned $v_\\theta$", 0.0035),
        ):
            magnitudes = np.linalg.norm(vec, axis=-1)
            clipped = vec * (np.minimum(magnitudes, cap) / (magnitudes + 1e-12))[:, None]
            ax.quiver(gx.ravel(), gy.ravel(), (clipped * factor)[:, 0], (clipped * factor)[:, 1],
                      color=color, angles="xy", scale_units="xy", scale=1.0, width=width, alpha=0.9,
                      label=label, pivot="tail")
        limit = float(abs(gx).max())
        style_axes(ax, (-limit, limit), (-limit, limit))
        ax.set_title(f"$s={s}$: relative RMS {rel:.3f}", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("The learned velocity overlaps the exact marginal velocity (arrows show $dx_s/ds$, lengths clipped)", y=1.02)
    save_figure(fig, "part2-velocity.png")

    # ---------------------------------------------------------------- sweep
    print("sweeping sampler step counts ...")
    key, *ref_keys = jax.random.split(key, 3)
    reference = np.asarray(mixture.sample(ref_keys[0], N_SWEEP))
    null = metrics.null_scale(ref_keys[1], size=N_SWEEP, pairs=20)
    print(f"null scale (95% quantile, 20 pairs of {N_SWEEP}): {null['q95']:.5f}")

    def run_ddpm(eps_fn, steps, k):
        start = jax.random.normal(k, (N_SWEEP, 2))
        return np.asarray(diffusion.ddpm_sample(eps_fn, start, sched, diffusion.even_levels(steps), key=jax.random.fold_in(k, 1)))

    def run_ddim(eps_fn, steps, k):
        start = jax.random.normal(k, (N_SWEEP, 2))
        return np.asarray(diffusion.ddim_sample(eps_fn, start, sched, diffusion.even_levels(steps)))

    def run_flow(velocity_fn, steps, k):
        start = jax.random.normal(k, (N_SWEEP, 2))
        return np.asarray(flow.euler_sample(velocity_fn, start, steps))

    samplers = {
        "ddpm": lambda steps, k: run_ddpm(eps_learned, steps, k),
        "ddim": lambda steps, k: run_ddim(eps_learned, steps, k),
        "ddpm_exact": lambda steps, k: run_ddpm(eps_exact, steps, k),
        "ddim_exact": lambda steps, k: run_ddim(eps_exact, steps, k),
    }
    results = {"steps": STEPS, "null_quantile_95": null["q95"], "null_values": null["values"].tolist()}
    run_keys = jax.random.split(key, 64)
    run_key_iter = iter(run_keys)
    for name, sampler in samplers.items():
        energy, frac, width, centre = [], [], [], []
        for steps in STEPS:
            samples = sampler(steps, next(run_key_iter))
            stats = metrics.cluster_stats(samples)
            energy.append(metrics.energy_distance(samples, reference))
            frac.append(float(np.max(np.abs(stats["populations"] - 0.25))))
            width.append(float(np.max(np.abs(stats["spreads"] - mixture.STD))))
            centre.append(float(np.nanmax(np.abs(stats["centres"] - mixture.CENTERS))))
        results[name] = {"energy": energy, "fraction_error": frac, "width_error": width, "centre_error": centre}
        print(name, "energy", [f"{e:.4f}" for e in energy])

    # DPPM / DDIM at 1000 levels
    for name, sampler in (("ddpm", lambda steps, k: run_ddpm(eps_learned, steps, k)),
                          ("ddim", lambda steps, k: run_ddim(eps_learned, steps, k))):
        samples = sampler(1000, next(run_key_iter))
        stats = metrics.cluster_stats(samples)
        results[name]["energy"].append(metrics.energy_distance(samples, reference))
        results[name]["fraction_error"].append(float(np.max(np.abs(stats["populations"] - 0.25))))
        results[name]["width_error"].append(float(np.max(np.abs(stats["spreads"] - mixture.STD))))
        results[name]["centre_error"].append(float(np.nanmax(np.abs(stats["centres"] - mixture.CENTERS))))

    # flow comparisons
    for name, velocity_fn in (("flow", velocity_learned), ("flow_exact", velocity_exact)):
        energy, frac, width, centre = [], [], [], []
        for steps in STEPS:
            samples = run_flow(velocity_fn, steps, next(run_key_iter))
            stats = metrics.cluster_stats(samples)
            energy.append(metrics.energy_distance(samples, reference))
            frac.append(float(np.max(np.abs(stats["populations"] - 0.25))))
            width.append(float(np.max(np.abs(stats["spreads"] - mixture.STD))))
            centre.append(float(np.nanmax(np.abs(stats["centres"] - mixture.CENTERS))))
        results[name] = {"energy": energy, "fraction_error": frac, "width_error": width, "centre_error": centre}
    print("flow energy", [f"{e:.4f}" for e in results["flow"]["energy"]])
    print("ddim energy", [f"{e:.4f}" for e in results["ddim"]["energy"]])

    # --------------------------------------------------------------- figure
    style = {
        "ddpm": (COLORS["aux"], "o", "DDPM (learned)"),
        "ddim": (COLORS["learned"], "s", "DDIM (learned)"),
        "flow": (COLORS["accent"], "^", "flow matching (learned)"),
        "ddpm_exact": (COLORS["aux"], "o", "DDPM (exact noise)"),
        "ddim_exact": (COLORS["learned"], "s", "DDIM (exact noise)"),
        "flow_exact": (COLORS["accent"], "^", "flow (exact velocity)"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    steps_arr = np.asarray(STEPS)
    for ax, key_name, ylabel, title in (
        (axes[0], "energy", "energy distance to 4000 exact samples", "energy distance"),
        (axes[1], "fraction_error", "largest cluster-fraction error", "cluster fractions"),
        (axes[2], "width_error", "largest per-coordinate width error", "cluster widths"),
    ):
        if key_name == "energy":
            ax.axhspan(0, null["q95"], color="#bbbbbb", alpha=0.35, label=f"null 95% quantile ({null['q95']:.4f})")
        if key_name == "fraction_error":
            se = float(np.sqrt(0.25 * 0.75 / N_SWEEP))
            ax.axhspan(0, 2 * se, color="#bbbbbb", alpha=0.35, label=f"2 binomial SE ({2 * se:.4f})")
        for name in ("ddpm", "ddim", "flow", "ddpm_exact", "ddim_exact", "flow_exact"):
            color, marker, label = style[name]
            values = np.asarray(results[name][key_name])[: len(STEPS)]
            dashed = name.endswith("_exact")
            ax.plot(steps_arr, values, marker=marker, ms=4, lw=1.1, color=color, ls="--" if dashed else "-",
                    alpha=0.75 if dashed else 1.0, label=label)
            if name in ("ddpm", "ddim"):
                ax.scatter([1000], [results[name][key_name][-1]], marker=marker, s=26, color=color)
        ax.axvline(20, color="#444444", ls=":", lw=1.0)
        ax.set_xscale("log")
        ax.set_xlabel("number of network evaluations (steps)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_yscale("log")
        ax.legend(fontsize=6.5, loc="upper right")
    fig.suptitle("Flow matching and DDIM reach the target in tens of steps; DDPM needs about fifty", y=1.02)
    save_figure(fig, "part2-steps.png")

    payload = {
        "seed": SEED,
        "determinism": {
            "n": N_DETERMINISM,
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
                "exact_learned_width_gap": exact_learned_width_gap,
                "reference_band": [0.29, 0.32],
                "note": "the sheet's Part 1 allowance for the same quantity at 2000 samples is 0.03",
            },
        },
        "velocity": {
            "s": S_VALUES,
            "relative_rms": velocity_rel,
            "relative_rms_grid": velocity_rel_grid,
            "relative_rms_wide_grid": velocity_rel_wide,
            "n_samples": 4000,
            "evaluation": "relative RMS over 4000 exact path samples; grid values use a 17x17 grid over the sample support and a 17x17 grid on [-6, 6]",
            "training": {
                "updates": UPDATES,
                "batch_size": BATCH,
                "learning_rate": LR,
                "learning_rate_schedule": "cosine decay to zero",
                "optimizer": "Adam",
                "final_loss": float(history["loss"][-1]),
                "seconds": history["seconds"],
                "loss_history": history,
            },
            "exact_velocity_at_s1_note": "v(x, 1) = x for the zero-mean mixture",
        },
        "sweep": results,
    }
    dump_json("part2_metrics.json", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())







