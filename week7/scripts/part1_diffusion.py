"""Part 1: DDPM on a four-cluster Gaussian mixture.

Produces the forward-diffusion movie, trains the time-conditioned noise
predictor (25000 updates), saves ten training stages, runs the DDPM sampler and
measures populations, centres and spreads. Also performs the optional denoiser
check against the exact posterior mean.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import _common
from _common import COLORS, ARTIFACTS, EVIDENCE, dump_json, save_figure, save_gif, scatter_cloud, setup_matplotlib, style_axes

sys.path.insert(0, str(_common.ROOT))
from src import diffusion, metrics, mixture, model, schedule, train

SEED = 20260707
UPDATES = 25000
BATCH = 512
LR = 1e-3
STAGES = (0, 100, 300, 1000, 3000, 6000, 10000, 15000, 20000, 25000)
N_FORWARD = 2000
N_GENERATE = 2000
N_DENOISE = 1000


def main() -> int:
    setup_matplotlib()
    _common.ensure_dirs()
    (ARTIFACTS / "part1").mkdir(parents=True, exist_ok=True)
    sched = schedule.linear_schedule()
    key = jax.random.PRNGKey(SEED)
    key_fwd, key_train, key_gen, key_denoise = jax.random.split(key, 4)

    print("=== Part 1: diffusion on a Gaussian mixture ===")
    print(f"T={len(sched)}  beta_1={sched.betas[0]:.5f}  beta_T={sched.betas[-1]:.4f}  abar_T={sched.abars[-1]:.3e}")

    # ---------------------------------------------------------------- forward
    x0 = mixture.sample(key_fwd, N_FORWARD)
    forward_levels = [0, 25, 100, 200, 350, 500, 700, 850, 1000]
    states = diffusion.forward_movie_states(x0, sched, forward_levels, key_fwd)
    x_T = np.asarray(states[-1])
    fwd_mean = x_T.mean(axis=0)
    fwd_var = x_T.var(axis=0)
    print(f"t=1000 mean={fwd_mean} variance={fwd_var}")

    fig, axes = plt.subplots(2, 5, figsize=(15, 6.2), sharex=True, sharey=True)
    for ax, level, state in zip(axes.flat, forward_levels, states):
        scatter_cloud(ax, state, COLORS["data"])
        ax.set_title(f"$t={level}$")
        style_axes(ax)
        if level >= 700:
            ax.text(0.04, 0.04, f"$\\mu=({fwd_mean[0]:.2f},{fwd_mean[1]:.2f})$\n$\\sigma^2=({fwd_var[0]:.2f},{fwd_var[1]:.2f})$",
                    transform=ax.transAxes, fontsize=7, color="#333333")
    axes.flat[-1].axis("off")
    axes.flat[-1].text(0.0, 0.5,
                       "Forward diffusion\n\n"
                       "Same 2000 samples\n"
                       "fixed scale [-8, 8]\n\n"
                       f"t=1000 mean {fwd_mean[0]:.3f}, {fwd_mean[1]:.3f}\n"
                       f"t=1000 var  {fwd_var[0]:.3f}, {fwd_var[1]:.3f}\n\n"
                       "Criterion: |mean|<0.1\nand |var-1|<0.1",
                       fontsize=9, va="center")
    fig.suptitle("Forward diffusion turns four clusters into one Gaussian", y=1.0)
    save_figure(fig, "part1-forward.png")

    frames = []
    for level, state in zip(forward_levels, states):
        fig, ax = plt.subplots(figsize=(4.4, 4.4))
        scatter_cloud(ax, state, COLORS["data"])
        style_axes(ax)
        ax.set_title(f"forward diffusion, $t={level}$")
        frames.append(fig)
    save_gif(frames, "part1-forward.gif", fps=2)

    # ---------------------------------------------------------------- training
    trained_dir = ARTIFACTS / "part1"
    reuse = "--reuse-stages" in sys.argv
    if reuse and (trained_dir / f"stage_{UPDATES:05d}.npz").exists():
        print("reusing the saved training stages")
        stages = {u: model.load_params(trained_dir / f"stage_{u:05d}.npz") for u in STAGES}
        history = json.loads((trained_dir / "history.json").read_text(encoding="utf-8"))
    else:
        print(f"training the noise predictor for {UPDATES} updates (batch {BATCH}) ...")
        stages, history = train.train_noise_predictor(
            key_train, updates=UPDATES, batch=BATCH, lr=LR, stages=STAGES, sched=sched
        )
        for update, params in stages.items():
            model.save_params(trained_dir / f"stage_{update:05d}.npz", params)
        (trained_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    final_params = stages[UPDATES]
    print(f"training finished in {history['seconds']:.1f}s, final loss {history['loss'][-1]:.4f}")

    # training-stage movie: shared starts and shared reverse noise
    n_paths = 10
    train_levels = diffusion.even_levels(1000)
    starts = jax.random.normal(jax.random.fold_in(key_gen, 11), (n_paths, 2))
    noise_key = jax.random.fold_in(key_gen, 12)
    tape = jnp.stack(
        [jax.random.normal(jax.random.fold_in(noise_key, i), (n_paths, 2)) for i in range(len(train_levels) - 1)]
    )
    stage_paths, loss_at = {}, {}
    for update in STAGES:
        eps_fn = model.noise_predictor(stages[update])
        path = diffusion.ddpm_sample_path(eps_fn, starts, sched, train_levels, noise=tape)
        stage_paths[update] = {level: np.asarray(state) for level, state in path.items()}
        idx = min(range(len(history["updates"])), key=lambda i: abs(history["updates"][i] - update))
        loss_at[update] = history["loss"][idx]
        ends = np.asarray(stage_paths[update][0])
        print(f"  stage {update:6d}: loss {loss_at[update]:.4f}  max|endpoint| {np.abs(ends).max():8.2f}")

    gif_stages = STAGES
    frames = []
    for update in gif_stages:
        fig, ax = plt.subplots(figsize=(4.8, 4.8))
        for center in mixture.CENTERS:
            ax.add_patch(plt.Circle(center, 2 * mixture.STD, fill=False, ls=":", lw=0.8, color=COLORS["noise"]))
        for j in range(n_paths):
            path = np.stack([stage_paths[update][lvl][j] for lvl in (1000, 500, 100, 0)])
            ax.plot(path[:, 0], path[:, 1], color=COLORS["learned"], lw=0.6, alpha=0.7)
            ax.scatter(path[0, 0], path[0, 1], s=14, facecolors="none", edgecolors=COLORS["noise"])
            ax.scatter(path[-1, 0], path[-1, 1], s=18, color=COLORS["learned"])
        style_axes(ax, (-8, 8), (-8, 8))
        ax.set_title(f"update {update}  loss {loss_at[update]:.4f}")
        frames.append(fig)
    save_gif(frames, "part1-training.gif", fps=1.4)

    fig = plt.figure(figsize=(16.5, 3.9))
    gs = fig.add_gridspec(1, 5, wspace=0.32, width_ratios=[1, 1, 1.25, 1, 0.9])
    show = [0, 1000, 6000, 25000]
    for i, update in enumerate(show):
        ax = fig.add_subplot(gs[0, i])
        for center in mixture.CENTERS:
            ax.add_patch(plt.Circle(center, 2 * mixture.STD, fill=False, ls=":", lw=0.8, color=COLORS["noise"]))
        for j in range(n_paths):
            path = np.stack([stage_paths[update][lvl][j] for lvl in (1000, 700, 400, 200, 100, 50, 20, 0)])
            ax.plot(path[:, 0], path[:, 1], color=COLORS["learned"], lw=0.7, alpha=0.8)
            ax.scatter(path[0, 0], path[0, 1], s=16, facecolors="none", edgecolors="#555555")
            ax.scatter(path[-1, 0], path[-1, 1], s=20, color=COLORS["learned"])
        ax.set_title(f"update {update}, loss {loss_at[update]:.3f}", fontsize=8)
        style_axes(ax, (-8, 8), (-8, 8))
        if update == 0:
            ax.text(0.03, 0.03, "paths start at $t=1000$\nand leave this square\n(endpoints up to "
                    f"{np.abs(np.asarray(stage_paths[update][0])).max():.0f})",
                    transform=ax.transAxes, fontsize=6.5, va="bottom")
    ax = fig.add_subplot(gs[0, 4])
    ax.plot(history["updates"], history["loss"], color=COLORS["data"], lw=0.9)
    for update in STAGES:
        ax.axvline(update, color=COLORS["learned"], ls=":", lw=0.7)
        ax.scatter([update], [loss_at[update]], color=COLORS["learned"], s=12, zorder=4)
    ax.set_xlabel("update", fontsize=8)
    ax.set_ylabel("noise-prediction loss", fontsize=8)
    ax.set_title("loss history\n(dotted: saved stages)", fontsize=8)
    ax.set_ylim(0.2, 1.3)
    fig.suptitle("Endpoints move onto the four clusters as the loss falls (shared starts and step noise)", y=1.06)
    save_figure(fig, "part1-training-stages.png")

    # ---------------------------------------------------------------- generation
    gen_levels = diffusion.even_levels(1000)
    eps_final = model.noise_predictor(final_params)
    big, key_gen = jax.random.split(key_gen)
    x_T_gen = jax.random.normal(big, (N_GENERATE, 2))
    gen_path = diffusion.ddpm_sample_path(eps_final, x_T_gen, sched, gen_levels, key=key_gen)
    generated = np.asarray(gen_path[0])
    stats = metrics.cluster_stats(generated)
    exact = np.asarray(mixture.sample(jax.random.fold_in(key_gen, 3), N_GENERATE))
    energy = metrics.energy_distance(generated, exact)
    print("generated populations", stats["populations"])
    print("generated centres\n", stats["centres"])
    print("generated spreads\n", stats["spreads"])
    print(f"energy distance to exact samples: {energy:.4e}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    scatter_cloud(axes[0], exact, COLORS["exact"], label="exact target samples", size=3)
    axes[0].set_title(f"exact draws (n={N_GENERATE})")
    style_axes(axes[0])
    scatter_cloud(axes[1], generated, COLORS["learned"], label="DDPM samples", size=3)
    axes[1].set_title(f"DDPM samples, 1000 steps (n={N_GENERATE})")
    style_axes(axes[1])
    for ax, samples, title, color in ((axes[0], exact, "exact", COLORS["exact"]), (axes[1], generated, "generated", COLORS["learned"])):
        for center in mixture.CENTERS:
            ax.add_patch(plt.Circle(center, 2 * mixture.STD, fill=False, ls=":", lw=0.9, color=color))
    ax = axes[2]
    width = 0.36
    idx = np.arange(4)
    ax.bar(idx - width / 2, stats["populations"], width, color=COLORS["learned"], label="generated")
    ax.bar(idx + width / 2, np.full(4, 0.25), width, color=COLORS["exact"], alpha=0.6, label="target 0.25")
    ax.axhspan(0.25 - 0.04, 0.25 + 0.04, color=COLORS["accent"], alpha=0.12, label="allowance 0.04")
    ax.set_xticks(idx)
    ax.set_xticklabels([f"$\\mu_{i + 1}$" for i in idx])
    ax.set_ylabel("population")
    ax.set_title("mode populations")
    ax.legend(fontsize=7)
    fig.suptitle("Generated samples recover the target distribution", y=1.02)
    save_figure(fig, "part1-generation.png")

    # reverse movie
    frames = []
    movie_levels = [1000, 900, 750, 600, 450, 300, 200, 120, 60, 20, 0]
    for level in movie_levels:
        fig, ax = plt.subplots(figsize=(4.4, 4.4))
        scatter_cloud(ax, gen_path[level], COLORS["learned"], size=3)
        for center in mixture.CENTERS:
            ax.add_patch(plt.Circle(center, 2 * mixture.STD, fill=False, ls=":", lw=0.8, color=COLORS["noise"]))
        style_axes(ax)
        ax.set_title(f"reverse process, $t={level}$")
        frames.append(fig)
    save_gif(frames, "part1-reverse.gif", fps=1.6)

    # ---------------------------------------------------------------- denoiser
    denoise_ts = [100, 300, 600, 900]
    denoise_rel, denoise_panels = [], []
    for t in denoise_ts:
        k0, key_denoise = jax.random.split(key_denoise)
        clean = mixture.sample(k0, N_DENOISE)
        eps = jax.random.normal(jax.random.fold_in(key_denoise, t), clean.shape)
        abar = float(sched.abars[t])
        x_t = diffusion.direct_noise(clean, abar, eps)
        denoised = diffusion.tweedie_denoise(eps_final, x_t, sched, t)
        posterior_mean = mixture.exact_posterior_mean(x_t, abar)
        posterior_std = mixture.exact_posterior_std(x_t, abar)
        rms_err = float(jnp.sqrt(jnp.mean(jnp.sum((denoised - posterior_mean) ** 2, axis=-1))))
        scale = float(jnp.sqrt(jnp.mean(posterior_std**2)))
        rel = rms_err / scale
        denoise_rel.append(rel)
        denoise_panels.append((t, np.asarray(x_t), np.asarray(posterior_mean), np.asarray(denoised)))
        print(f"denoiser t={t:3d}: relative RMS error {rel:.4f}")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    for ax, (t, x_t, post, den) in zip(axes, denoise_panels):
        idx = np.arange(0, len(x_t), max(1, len(x_t) // 250))
        for i in idx:
            ax.plot([x_t[i, 0], post[i, 0]], [x_t[i, 1], post[i, 1]], color="#cccccc", lw=0.5, zorder=1)
        ax.scatter(x_t[idx, 0], x_t[idx, 1], s=10, color=COLORS["noise"], label="noisy $x_t$", zorder=2)
        ax.scatter(post[idx, 0], post[idx, 1], s=10, color=COLORS["exact"], label="exact $E[x_0|x_t]$", zorder=3)
        ax.scatter(den[idx, 0], den[idx, 1], s=10, color=COLORS["learned"], label="Tweedie denoiser", zorder=4, alpha=0.8)
        ax.set_title(f"$t={t}$, rel. RMS {denoise_rel[denoise_ts.index(t)]:.3f}")
        ax.set_xlim(-8, 8)
        ax.set_ylim(-8, 8)
        ax.set_aspect("equal")
    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle("The learned denoiser matches the exact posterior mean up to t=600", y=1.02)
    save_figure(fig, "part1-denoiser.png")

    payload = {
        "seed": SEED,
        "schedule": {
            "T": len(sched),
            "beta_start": float(sched.betas[0]),
            "beta_end": float(sched.betas[-1]),
            "abar_T": float(sched.abars[-1]),
        },
        "forward": {
            "n": N_FORWARD,
            "levels": forward_levels,
            "mean": fwd_mean.tolist(),
            "variance": fwd_var.tolist(),
        },
        "training": {
            "updates": UPDATES,
            "batch_size": BATCH,
            "learning_rate": LR,
            "learning_rate_schedule": "cosine decay to zero",
            "optimizer": "Adam",
            "stage_updates": list(STAGES),
            "final_loss": float(history["loss"][-1]),
            "loss_history": history,
            "parameter_count": model.parameter_count(final_params),
            "seconds": history["seconds"],
        },
        "generation": {
            "n": N_GENERATE,
            "steps": 1000,
            "populations": stats["populations"].tolist(),
            "centres": np.nan_to_num(stats["centres"]).tolist(),
            "spreads": np.nan_to_num(stats["spreads"]).tolist(),
            "energy_distance_to_exact": energy,
        },
        "denoiser": {"t": denoise_ts, "relative_rms": denoise_rel},
    }
    dump_json("part1_metrics.json", payload)
    np.savez_compressed(ARTIFACTS / "part1" / "generated.npz", samples=generated, exact=exact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


