"""Part 3: flow matching and the cost of fewer steps (Algorithms 4 and 5).

Trains the velocity network from scratch with ``flow.toml``, compares it with the
mixture's exact marginal velocity, and sweeps DDPM, DDIM and Euler flow matching
over 1, 2, 5, 10, 20, 50 and 100 steps plus 1000 levels for DDPM and DDIM.
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
from _common import ARTIFACTS, COLORS, dump_json, save_figure, setup_matplotlib, style_axes

sys.path.insert(0, str(_common.ROOT))
from src import diffusion, flow, metrics, mixture, model, schedule, train

SEED = 2029
UPDATES = 25000
BATCH = 512
LR = 1e-3
N_SWEEP = 4000
STEPS = [1, 2, 5, 10, 20, 50, 100]
S_VALUES = [0.1, 0.5, 0.9]


def main() -> int:
    setup_matplotlib()
    _common.ensure_dirs()
    (ARTIFACTS / "part3").mkdir(parents=True, exist_ok=True)
    noise_path = ARTIFACTS / "part1" / "stage_25000.npz"
    if not noise_path.exists():
        raise SystemExit(f"missing {noise_path}; run scripts/part1_diffusion.py first")
    sched = schedule.linear_schedule()
    noise_params = model.load_params(noise_path)
    eps_learned = jax.jit(model.noise_predictor(noise_params))

    def eps_exact(x, t):
        return mixture.exact_conditional_noise(x, float(sched.abars[t]))

    print("=== Part 3: flow matching and the step count ===")
    key = jax.random.PRNGKey(SEED)
    velocity_path = ARTIFACTS / "part3" / "velocity_final.npz"
    history_path = ARTIFACTS / "part3" / "velocity_history.json"
    key, train_key = jax.random.split(key)
    if "--reuse-velocity" in sys.argv and velocity_path.exists():
        print("reusing the saved velocity network")
        velocity_params = model.load_params(velocity_path)
        history = json.loads(history_path.read_text(encoding="utf-8"))
    else:
        print(f"training the velocity network for {UPDATES} updates (flow.toml, seed {SEED}) ...")
        stages, history = train.train_velocity_network(train_key, updates=UPDATES, batch=BATCH, lr=LR)
        velocity_params = stages[UPDATES]
        model.save_params(velocity_path, velocity_params)
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    velocity_learned = jax.jit(model.velocity_predictor(velocity_params))
    print(f"velocity training finished in {history['seconds']:.1f}s, final loss {history['loss'][-1]:.4f}")

    def velocity_exact(x, s):
        return flow.exact_marginal_velocity(x, float(s))

    # ------------------------------------------------------------ velocity field
    field_key = jax.random.fold_in(key, 21)
    field_clean = mixture.sample(field_key, N_SWEEP)
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
        wide_rel = flow.velocity_relative_rms(
            np.asarray(velocity_learned(jnp.asarray(np.stack([wx.ravel(), wy.ravel()], -1)), s)),
            np.asarray(velocity_exact(np.stack([wx.ravel(), wy.ravel()], -1), s)),
        )
        velocity_rel.append(rel)
        velocity_rel_grid.append(grid_rel)
        velocity_rel_wide.append(wide_rel)
        velocity_panels.append((s, gx, gy, grid, learned, exact, rel))
        print(f"velocity s={s}: relative RMS {rel:.4f} (samples), {grid_rel:.4f} (in-support grid), {wide_rel:.4f} (wide grid)")

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.9))
    for ax, (s, gx, gy, grid, learned, exact, rel) in zip(axes, velocity_panels):
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
    save_figure(fig, "part3-velocity.png")

    # ------------------------------------------------------------------- sweep
    print("sweeping sampler step counts ...")
    key, *ref_keys = jax.random.split(key, 3)
    reference = np.asarray(mixture.sample(ref_keys[0], N_SWEEP))
    null = metrics.null_scale(ref_keys[1], size=N_SWEEP, pairs=20)
    print(f"null scale (95% quantile, 20 pairs of {N_SWEEP}): {null['q95']:.5f}")

    def run_ddpm(eps_fn, steps, k):
        start = jax.random.normal(k, (N_SWEEP, 2))
        return np.asarray(diffusion.ddpm_sample(eps_fn, start, sched, diffusion.even_levels(steps),
                                                key=jax.random.fold_in(k, 1)))

    def run_ddim(eps_fn, steps, k):
        start = jax.random.normal(k, (N_SWEEP, 2))
        return np.asarray(diffusion.ddim_sample(eps_fn, start, sched, diffusion.even_levels(steps)))

    def run_flow(velocity_fn, steps, k):
        start = jax.random.normal(k, (N_SWEEP, 2))
        return np.asarray(flow.euler_sample(velocity_fn, start, steps))

    results = {"steps": STEPS, "null_quantile_95": null["q95"], "null_values": null["values"].tolist()}
    run_keys = jax.random.split(key, 64)
    run_key_iter = iter(run_keys)
    samplers = {
        "ddpm": lambda steps, k: run_ddpm(eps_learned, steps, k),
        "ddim": lambda steps, k: run_ddim(eps_learned, steps, k),
        "ddpm_exact": lambda steps, k: run_ddpm(eps_exact, steps, k),
        "ddim_exact": lambda steps, k: run_ddim(eps_exact, steps, k),
    }
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

    for name, sampler in (("ddpm", lambda steps, k: run_ddpm(eps_learned, steps, k)),
                          ("ddim", lambda steps, k: run_ddim(eps_learned, steps, k))):
        samples = sampler(1000, next(run_key_iter))
        stats = metrics.cluster_stats(samples)
        results[name]["energy"].append(metrics.energy_distance(samples, reference))
        results[name]["fraction_error"].append(float(np.max(np.abs(stats["populations"] - 0.25))))
        results[name]["width_error"].append(float(np.max(np.abs(stats["spreads"] - mixture.STD))))
        results[name]["centre_error"].append(float(np.nanmax(np.abs(stats["centres"] - mixture.CENTERS))))

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
    print("ddpm energy", [f"{e:.4f}" for e in results["ddpm"]["energy"]])

    style_map = {
        "ddpm": (COLORS["aux"], "o", "DDPM (learned)"),
        "ddim": (COLORS["learned"], "s", "DDIM (learned)"),
        "flow": (COLORS["accent"], "^", "flow matching (learned)"),
        "ddpm_exact": (COLORS["aux"], "o", "DDPM (exact noise)"),
        "ddim_exact": (COLORS["learned"], "s", "DDIM (exact noise)"),
        "flow_exact": (COLORS["accent"], "^", "flow (exact velocity)"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    steps_arr = np.asarray(STEPS)
    for ax, series_key, ylabel, title in (
        (axes[0], "energy", "energy distance to 4000 exact samples", "energy distance"),
        (axes[1], "fraction_error", "largest cluster-fraction error", "cluster fractions"),
        (axes[2], "width_error", "largest per-coordinate width error", "cluster widths"),
    ):
        if series_key == "energy":
            ax.axhspan(0, null["q95"], color="#bbbbbb", alpha=0.35, label=f"null 95% quantile ({null['q95']:.4f})")
        if series_key == "fraction_error":
            se = float(np.sqrt(0.25 * 0.75 / N_SWEEP))
            ax.axhspan(0, 2 * se, color="#bbbbbb", alpha=0.35, label=f"2 binomial SE ({2 * se:.4f})")
        for name in ("ddpm", "ddim", "flow", "ddpm_exact", "ddim_exact", "flow_exact"):
            color, marker, label = style_map[name]
            values = np.asarray(results[name][series_key])[: len(STEPS)]
            dashed = name.endswith("_exact")
            ax.plot(steps_arr, values, marker=marker, ms=4, lw=1.1, color=color, ls="--" if dashed else "-",
                    alpha=0.75 if dashed else 1.0, label=label)
            if name in ("ddpm", "ddim"):
                ax.scatter([1000], [results[name][series_key][-1]], marker=marker, s=26, color=color)
        ax.axvline(20, color="#444444", ls=":", lw=1.0)
        ax.set_xscale("log")
        ax.set_xlabel("number of network evaluations (steps)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_yscale("log")
        ax.legend(fontsize=6.5, loc="upper right")
    fig.suptitle("Flow matching and DDIM reach the target in tens of steps; DDPM needs about fifty", y=1.02)
    save_figure(fig, "part3-steps.png")

    payload = {
        "seed": SEED,
        "velocity": {
            "s": S_VALUES,
            "relative_rms": velocity_rel,
            "relative_rms_grid": velocity_rel_grid,
            "relative_rms_wide_grid": velocity_rel_wide,
            "n_samples": N_SWEEP,
            "evaluation": "relative RMS over 4000 exact path samples; grid values use a 17x17 grid over the sample support and a 17x17 grid on [-6, 6]",
            "training": {
                "updates": UPDATES,
                "batch_size": BATCH,
                "learning_rate": LR,
                "learning_rate_schedule": "cosine decay to zero",
                "optimizer": "Adam",
                "seed": SEED,
                "final_loss": float(history["loss"][-1]),
                "seconds": history["seconds"],
                "loss_history": history,
            },
        },
        "sweep": results,
    }
    dump_json("part3_metrics.json", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
