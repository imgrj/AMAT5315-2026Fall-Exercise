"""Part 3: the downloaded CIFAR-10 noise predictor on photographs.

Loads the pinned ``google/ddpm-cifar10-32`` weights as a fixed JAX noise
predictor, verifies the port against the diffusers v0.40.0 reference outputs,
and runs Part 1's 1000-step DDPM sampler unchanged on four images.
"""

from __future__ import annotations

import json
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import _common
from _common import ARTIFACTS, dump_json, save_figure, setup_matplotlib

sys.path.insert(0, str(_common.ROOT))
from src import diffusion, schedule, unet

SEED = 20260709
REVISION = "267b167dc01f0e4e61923ea244e8b988f84deb80"
MODEL_DIR = _common.ROOT / "inputs" / "cifar-model"
N_IMAGES = 4
SHOW_LEVELS = [1000, 800, 500, 200, 100, 50, 20, 0]


def load_real_cifar(count: int = 8) -> np.ndarray:
    """Return ``count`` real CIFAR-10 training images in [-1, 1], NHWC."""
    candidates = [
        _common.ROOT / "inputs" / "cifar10" / "cifar-10-binary.tar.gz",
        _common.ROOT.parent / "week6" / "inputs" / "cifar10" / "cifar-10-binary.tar.gz",
    ]
    archive = next((path for path in candidates if path.exists()), None)
    if archive is None:
        archive = _common.ROOT / "inputs" / "cifar10" / "cifar-10-binary.tar.gz"
        archive.parent.mkdir(parents=True, exist_ok=True)
        url = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
        print(f"downloading CIFAR-10 from {url}")
        urllib.request.urlretrieve(url, archive)
    with tarfile.open(archive, "r:gz") as tar:
        raw = tar.extractfile("cifar-10-batches-bin/data_batch_1.bin").read()
    record = 1 + 3072
    records = np.frombuffer(raw[: record * count], dtype=np.uint8).reshape(count, record)
    images = records[:, 1:].reshape(count, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32)
    return images / 127.5 - 1.0


def to_display(x: np.ndarray) -> np.ndarray:
    """Map clean values from [-1, 1] to [0, 1] for display only."""
    return np.clip((x + 1.0) / 2.0, 0.0, 1.0)


def main() -> int:
    setup_matplotlib()
    _common.ensure_dirs()
    (ARTIFACTS / "part3").mkdir(parents=True, exist_ok=True)
    if not (MODEL_DIR / "diffusion_pytorch_model.safetensors").exists():
        raise SystemExit(
            "missing inputs/cifar-model; download it with\n"
            "  uv run --with huggingface-hub==1.33.0 hf download google/ddpm-cifar10-32 \\\n"
            "    config.json diffusion_pytorch_model.safetensors --revision " + REVISION + " \\\n"
            "    --local-dir inputs/cifar-model"
        )
    print("=== Part 3: the pretrained CIFAR-10 noise predictor ===")
    started = time.time()
    loaded = unet.load_unet(MODEL_DIR)
    params = loaded["params"]
    count = unet.parameter_count(params)
    print(f"loaded {count:,} parameters from the pinned revision {REVISION}")

    # ------------------------------------------------ reference verification
    ref_path = ARTIFACTS / "part3" / "reference_unet.npz"
    reference_diff = None
    if ref_path.exists():
        data = np.load(ref_path)
        x_ref = jnp.asarray(data["x"].transpose(0, 2, 3, 1))
        t_ref = jnp.asarray(data["t"])
        eps_ref = data["eps"].transpose(0, 2, 3, 1)
        eps_jax = np.asarray(unet.predict_eps(loaded, x_ref, t_ref))
        reference_diff = float(np.max(np.abs(eps_jax - eps_ref)))
        print(f"JAX vs diffusers v0.40.0 max |difference|: {reference_diff:.2e}")
    else:
        print(f"no {ref_path.name}; run scripts/export_reference_unet.py with the pinned torch env")

    sched = schedule.linear_schedule()

    def eps_fn(x, t):
        index = jnp.full((x.shape[0],), t - 1, dtype=jnp.int32)
        return unet.predict_eps(loaded, x, index)

    key = jax.random.PRNGKey(SEED)
    x_T = jax.random.normal(key, (N_IMAGES, 32, 32, 3))
    levels = diffusion.even_levels(1000)
    snapshots = {}
    x = x_T
    snapshots[1000] = np.asarray(x)
    step_times = []
    for i, (t, t_next) in enumerate(zip(levels[:-1], levels[1:])):
        t0 = time.time()
        mean = diffusion.ddpm_skipped_mean(eps_fn, x, sched, t, t_next)
        if t_next == 0:
            x = mean
        else:
            a = float(sched.abars[t]) / float(sched.abars[t_next])
            z = jax.random.normal(jax.random.fold_in(key, t), x.shape)
            x = mean + z * np.sqrt(1.0 - a)
        step_times.append(time.time() - t0)
        if t_next in SHOW_LEVELS:
            snapshots[t_next] = np.asarray(x)
        if (i + 1) % 100 == 0:
            print(f"  reverse step {i + 1}/1000")
    samples = snapshots[0]
    elapsed = time.time() - started
    print(f"sampled {N_IMAGES} images in {elapsed:.0f}s ({np.mean(step_times) * 1000:.0f} ms per 4-image evaluation)")

    # ------------------------------------------------------------- figures
    fig, axes = plt.subplots(N_IMAGES, len(SHOW_LEVELS), figsize=(14.5, 7.0))
    for row in range(N_IMAGES):
        for col, level in enumerate(SHOW_LEVELS):
            ax = axes[row, col]
            ax.imshow(to_display(snapshots[level][row]), interpolation="nearest")
            ax.axis("off")
            if row == 0:
                ax.set_title(f"$t={level}$", fontsize=9)
    fig.suptitle("Downloaded weights plus Part 1's DDPM update: structure appears between $t=200$ and $t=50$", y=0.99)
    save_figure(fig, "part3-cifar-generation.png")

    real = load_real_cifar(4)
    fig, axes = plt.subplots(2, 5, figsize=(13.5, 5.6))
    for col in range(4):
        axes[0, col].imshow(to_display(samples[col]), interpolation="nearest")
        axes[1, col].imshow(to_display(real[col]), interpolation="nearest")
        axes[0, col].axis("off")
        axes[1, col].axis("off")
        if col == 0:
            axes[0, col].set_title("generated", fontsize=9)
            axes[1, col].set_title("real CIFAR-10", fontsize=9)
    axes[0, 4].axis("off")
    axes[1, 4].axis("off")
    stats = samples.reshape(N_IMAGES, -1)
    text = "\n".join(
        f"image {i + 1}: mean {stats[i].mean():+.2f}, min {stats[i].min():+.2f}, max {stats[i].max():+.2f}"
        for i in range(N_IMAGES)
    )
    axes[0, 4].text(0.0, 0.5, "Unclipped sampling states\n\n" + text, fontsize=7.5, va="center")
    axes[1, 4].text(0.0, 0.6,
                    "Learned before download:\nnetwork weights (35.7M)\n\n"
                    "Computed now:\nDDPM reverse updates,\n1000 network evaluations",
                    fontsize=7.5, va="center")
    fig.suptitle("Generated photographs from Gaussian noise, compared with real CIFAR-10 images", y=1.0)
    save_figure(fig, "part3-cifar-samples.png")

    # ------------------------------------------------------------- metrics
    pixel_stats = []
    for i in range(N_IMAGES):
        flat = samples[i]
        pixel_stats.append([float(flat.mean()), float(flat.min()), float(flat.max()), float(flat.std())])
    payload = {
        "seed": SEED,
        "revision": REVISION,
        "model_dir": "inputs/cifar-model",
        "download_command": (
            "uv run --with huggingface-hub==1.33.0 hf download google/ddpm-cifar10-32 "
            "config.json diffusion_pytorch_model.safetensors --revision " + REVISION + " --local-dir inputs/cifar-model"
        ),
        "param_count": count,
        "images": N_IMAGES,
        "levels": 1000,
        "model_index_rule": "pass level t as model index t-1",
        "pixel_stats": pixel_stats,
        "reference_max_abs_diff": reference_diff,
        "seconds": elapsed,
        "ms_per_batch_evaluation": float(np.mean(step_times) * 1000),
        "show_levels": SHOW_LEVELS,
    }
    dump_json("part3_metrics.json", payload)
    np.savez_compressed(
        ARTIFACTS / "part3" / "samples.npz",
        samples=samples,
        **{f"level_{level}": snapshots[level] for level in SHOW_LEVELS},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
