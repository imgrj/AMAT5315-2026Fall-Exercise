"""Part 5: RePaint inpainting with the downloaded CIFAR-10 predictor.

Hides rows and columns 6:26 in all three channels of the four prepared test
images, completes each with seeds 7201-7204, and checks that every observed
pixel is preserved while the hidden square varies between completions.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import _common
from _common import ARTIFACTS, dump_json, save_figure, setup_matplotlib

sys.path.insert(0, str(_common.ROOT))
from src import repaint, schedule, unet

SEED_IMAGES = 7201
SEEDS = [7201, 7202, 7203, 7204]
MODEL_DIR = _common.ROOT / "inputs" / "cifar-model"
IMAGES = _common.ROOT / "inputs" / "images.npz"
ATTEMPTS = 2
JUMP = 20


def to_display(x: np.ndarray) -> np.ndarray:
    return np.clip((np.asarray(x) + 1.0) / 2.0, 0.0, 1.0)


def main() -> int:
    setup_matplotlib()
    _common.ensure_dirs()
    (ARTIFACTS / "part5").mkdir(parents=True, exist_ok=True)
    if not IMAGES.exists():
        raise SystemExit(f"missing {IMAGES}; download it with\n"
                         "  curl -fL https://giggleliu.github.io/AMAT5315-2026Fall/downloads/week7-images.npz -o inputs/images.npz")
    print("=== Part 5: filling in missing pixels with RePaint ===")
    data = np.load(IMAGES)
    truth = np.asarray(data["test"], dtype=np.float32).transpose(0, 2, 3, 1)  # (4, 32, 32, 3)
    n_images = truth.shape[0]
    mask = repaint.square_mask(truth.shape[1:])
    observed = truth * mask[None]  # hidden entries are not passed to the sampler
    levels = repaint.repaint_levels(JUMP)
    evaluations = (len(levels) - 2) * ATTEMPTS + 1
    print(f"{n_images} test images, mask rows/cols 6:26, seeds {SEEDS}")
    print(f"{len(levels) - 1} jumps, {ATTEMPTS} attempts except the last -> {evaluations} network evaluations each")

    loaded = unet.load_unet(MODEL_DIR)
    sched = schedule.linear_schedule()

    def eps_fn(x, t):
        index = jnp.full((x.shape[0],), t - 1, dtype=jnp.int32)
        return unet.predict_eps(loaded, x, index)

    # one batch holds every (image, seed) pair: 4 x 4 = 16 completions
    starts, keys, image_ids, seed_ids = [], [], [], []
    key = jax.random.PRNGKey(SEED_IMAGES)
    for i in range(n_images):
        for s in SEEDS:
            key, sub = jax.random.split(key)
            starts.append(jax.random.normal(sub, (32, 32, 3)))
            keys.append(int(s))
            image_ids.append(i)
            seed_ids.append(s)
    x_T = jnp.stack(starts)
    observed_batch = jnp.stack([observed[i] for i in image_ids])
    mask_batch = jnp.broadcast_to(jnp.asarray(mask), x_T.shape)
    started = time.time()
    completions = np.asarray(
        repaint.repaint_complete(eps_fn, sched, observed_batch, mask_batch, x_T, key, levels, ATTEMPTS)
    )
    elapsed = time.time() - started
    print(f"finished {completions.shape[0]} completions in {elapsed:.0f}s")

    hidden = mask == 0.0
    diff = completions - np.stack([truth[i] for i in image_ids])
    hidden_mse = np.zeros((n_images, len(SEEDS)))
    for i in range(n_images):
        for j in range(len(SEEDS)):
            hidden_mse[i, j] = float(np.mean(diff[i * len(SEEDS) + j][hidden] ** 2))
    observed_change = float(np.max(np.abs(diff[:, ~hidden])))
    hidden_values = completions[:, hidden]  # (16, 1200)
    var_per_image = hidden_values.reshape(n_images, len(SEEDS), -1).var(axis=1).mean(axis=1)
    outside = completions[:, ~hidden]
    outside_spread = float(np.max(np.abs(outside.reshape(n_images, len(SEEDS), -1) - outside.reshape(n_images, len(SEEDS), -1)[:, :1, :])))
    print(f"hidden-pixel MSE per image/seed:\n{np.round(hidden_mse, 3)}")
    print(f"largest change of an observed pixel: {observed_change:.2e}")
    print(f"largest disagreement outside the hole: {outside_spread:.2e}")
    print(f"mean hidden-pixel variance across seeds: {np.round(var_per_image, 4)}")

    # ------------------------------------------------------------------ figure
    fig, axes = plt.subplots(n_images, 7, figsize=(16.5, 9.4))
    for i in range(n_images):
        panels = [
            (truth[i], "truth"),
            (None, "observation"),
        ]
        for j in range(len(SEEDS)):
            panels.append((completions[i * len(SEEDS) + j], f"seed {SEEDS[j]}\nMSE {hidden_mse[i, j]:.2f}"))
        variance_map = completions[i * len(SEEDS):(i + 1) * len(SEEDS)].var(axis=0).mean(axis=-1)
        panels.append((variance_map, "pixelwise\nvariance"))
        for col, (image, title) in enumerate(panels):
            ax = axes[i, col]
            if col == 1:
                shown = to_display(observed[i])
                shown = shown.copy()
                shown[6:26, 6:26] = 0.5
                ax.imshow(shown, interpolation="nearest")
            elif col == len(panels) - 1:
                ax.imshow(image, cmap="magma", interpolation="nearest")
            else:
                ax.imshow(to_display(image), interpolation="nearest")
            ax.axis("off")
            if i == 0:
                ax.set_title(title, fontsize=8)
    fig.suptitle("RePaint completions keep every observed pixel and disagree inside the hidden square", y=0.995)
    save_figure(fig, "part5-inpainting.png")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    width = 0.2
    for j, s in enumerate(SEEDS):
        axes[0].bar(np.arange(n_images) + (j - 1.5) * width, hidden_mse[:, j], width, label=f"seed {s}")
    axes[0].set_xticks(np.arange(n_images))
    axes[0].set_xticklabels([f"image {i + 1}" for i in range(n_images)])
    axes[0].set_ylabel("hidden-pixel MSE")
    axes[0].set_title("error against the withheld truth")
    axes[0].legend(fontsize=7)
    axes[1].bar(np.arange(n_images), var_per_image, color="#2ca02c")
    axes[1].set_xticks(np.arange(n_images))
    axes[1].set_xticklabels([f"image {i + 1}" for i in range(n_images)])
    axes[1].set_ylabel("mean variance over 4 completions")
    axes[1].set_title("variation across seeds inside the hole")
    fig.suptitle("Error against the particular truth versus variation between plausible completions", y=1.02)
    save_figure(fig, "part5-variation.png")

    payload = {
        "seeds": SEEDS,
        "jumps": len(levels) - 1,
        "attempts_per_jump": ATTEMPTS,
        "network_evaluations": evaluations,
        "mask": "rows and columns 6:26, all three channels",
        "hidden_pixels": int(hidden.sum()),
        "hidden_mse": hidden_mse.tolist(),
        "observed_max_change": observed_change,
        "outside_hole_max_difference": outside_spread,
        "hidden_variance_mean": var_per_image.tolist(),
        "seconds": elapsed,
        "levels": levels,
    }
    dump_json("part5_metrics.json", payload)
    np.savez_compressed(
        ARTIFACTS / "part5" / "completions.npz",
        completions=completions,
        truth=truth,
        mask=mask,
        image_ids=np.asarray(image_ids),
        seeds=np.asarray(seed_ids),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


