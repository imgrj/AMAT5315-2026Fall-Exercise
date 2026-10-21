"""Build the Week 7 interactive lab (docs/week7-diffusion-lab.html).

Embeds the trained noise predictor and small PNG tiles for the photograph
gallery and the inpainting comparison, so the page is a single
self-contained file with no external requests.
"""

from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from PIL import Image

import _common

sys.path.insert(0, str(_common.ROOT))
from src import model

ROOT = _common.ROOT
DOCS = ROOT.parent / "docs"
TEMPLATE = ROOT / "web" / "lab_template.html"
OUT = DOCS / "week7-diffusion-lab.html"
SHOW_LEVELS = [1000, 800, 500, 200, 100, 50, 20, 0]


def float32_b64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array, dtype="<f4").tobytes()).decode("ascii")


def photo_b64(array: np.ndarray, size: int = 256) -> str:
    """Encode a small image as a smooth (Lanczos) JPEG data URL for display.

    CIFAR-10 samples are natively 32 x 32.  Blowing them up with nearest-neighbour
    interpolation is what makes a photograph look like coloured blocks, so the lab
    resamples smoothly once here and lets the browser scale the result.
    """
    array = np.asarray(array)
    if array.dtype != np.uint8:
        array = (np.clip(array, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    image = Image.fromarray(array).resize((size, size), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def heat_b64(values: np.ndarray, size: int = 256) -> str:
    """Encode a scalar map with the magma colour map (smooth JPEG)."""
    import matplotlib.cm as cm

    values = np.clip(np.asarray(values, dtype=np.float64) / max(0.2, float(np.max(values)) + 1e-12), 0.0, 1.0)
    rgb = (cm.magma(values)[..., :3] * 255.0).astype(np.uint8)
    return photo_b64(rgb, size=size)


def to_display(x: np.ndarray) -> np.ndarray:
    return np.clip((np.asarray(x) + 1.0) / 2.0, 0.0, 1.0)


def main() -> int:
    params = model.load_params(ROOT / "artifacts" / "part1" / "stage_25000.npz")
    weights = {
        "shapes": [[np.asarray(w).shape for w in params["weights"]], [np.asarray(b).shape for b in params["biases"]]],
        "w": [float32_b64(np.asarray(w)) for w in params["weights"]],
        "b": [float32_b64(np.asarray(b)) for b in params["biases"]],
    }

    # Part 4: reverse-process stages of the first image, four final images, real CIFAR
    part4 = np.load(ROOT / "artifacts" / "part4" / "samples.npz")
    stage_tiles = [photo_b64(to_display(part4[f"level_{level}"][0]), size=256) for level in SHOW_LEVELS]
    generated = [photo_b64(to_display(part4["samples"][i]), size=256) for i in range(4)]
    from part4_cifar import load_real_cifar

    real = [photo_b64(to_display(image), size=256) for image in load_real_cifar(4)]

    # Part 5: truth, observation, four completions per image, variance map
    part5 = np.load(ROOT / "artifacts" / "part5" / "completions.npz")
    completions = part5["completions"]
    truth = part5["truth"]
    mask = part5["mask"]
    seeds = part5["seeds"]
    hidden = mask == 0
    inpaint_images = []
    for i in range(4):
        obs = to_display(truth[i]).copy()
        obs[6:26, 6:26] = 0.5
        row = {
            "truth": photo_b64(to_display(truth[i]), size=192),
            "observation": photo_b64(obs, size=192),
            "completions": [],
            "variance": heat_b64(completions[i * 4:(i + 1) * 4].var(axis=0).mean(axis=-1), size=192),
        }
        for j in range(4):
            row["completions"].append(photo_b64(to_display(completions[i * 4 + j]), size=192))
        inpaint_images.append(row)
    metrics = json.loads((ROOT / "evidence" / "part5_metrics.json").read_text(encoding="utf-8"))

    part4_metrics = json.loads((ROOT / "evidence" / "part4_metrics.json").read_text(encoding="utf-8"))
    payload = {
        "weights": weights,
        "cifar": {"stages": stage_tiles, "levels": SHOW_LEVELS, "generated": generated, "real": real},
        "inpaint": {
            "images": inpaint_images,
            "seeds": seeds.tolist(),
            "hidden_mse": metrics["hidden_mse"],
            "observed_max_change": metrics["observed_max_change"],
            "jumps": metrics["jumps"],
            "evaluations": metrics["network_evaluations"],
        },
        "part1": {
            "loss": json.loads((ROOT / "evidence" / "part1_metrics.json").read_text())["training"]["final_loss"],
        },
        "part4": {
            "reference_max_abs_diff": part4_metrics["reference_max_abs_diff"],
            "param_count": part4_metrics["param_count"],
        },
    }

    # evidence figures are served with the site: the GitHub Pages root is docs/
    asset_dir = DOCS / "assets" / "week7"
    asset_dir.mkdir(parents=True, exist_ok=True)
    gallery_files = sorted(p.name for p in (ROOT / "evidence").iterdir() if p.suffix in {".png", ".gif"})
    for name in gallery_files:
        data = (ROOT / "evidence" / name).read_bytes()
        target = asset_dir / name
        if not target.exists() or target.read_bytes() != data:
            target.write_bytes(data)
    for stale in asset_dir.iterdir():
        if stale.is_file() and stale.name not in gallery_files:
            stale.unlink()
    print(f"copied {len(gallery_files)} evidence figures to {asset_dir}")

    template = TEMPLATE.read_text(encoding="utf-8")
    html = template.replace("__DATA__", json.dumps(payload, separators=(",", ":")))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT} ({len(html) / 1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
