"""Shared paths, plotting style and JSON helpers for the week 7 scripts."""

from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EVIDENCE = ROOT / "evidence"
ARTIFACTS = ROOT / "artifacts"

COLORS = {
    "data": "#1f77b4",
    "noise": "#7f7f7f",
    "learned": "#d62728",
    "exact": "#111111",
    "accent": "#2ca02c",
    "aux": "#9467bd",
}


def ensure_dirs() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 130,
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    )


def dump_json(name: str, payload) -> Path:
    path = EVIDENCE / name
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")
    return path


def save_figure(fig, name: str) -> Path:
    path = EVIDENCE / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")
    return path


def save_gif(frames, name: str, fps: float = 2.0) -> Path:
    import io

    from PIL import Image

    path = EVIDENCE / name
    images = []
    for fig in frames:
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight")
        buffer.seek(0)
        images.append(Image.open(buffer).convert("RGB"))
        plt.close(fig)
    images[0].save(path, save_all=True, append_images=images[1:], duration=int(1000 / fps), loop=0)
    print(f"wrote {path.relative_to(ROOT)}")
    return path


@contextmanager
def timed(label: str, store: dict | None = None):
    started = time.time()
    yield
    seconds = time.time() - started
    print(f"{label}: {seconds:.1f}s")
    if store is not None:
        store[label] = seconds


def style_axes(ax, xlim=(-8, 8), ylim=(-8, 8)) -> None:
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")


def scatter_cloud(ax, x, color, label=None, alpha=0.45, size=4):
    x = np.asarray(x)
    return ax.scatter(x[:, 0], x[:, 1], s=size, c=color, alpha=alpha, label=label, linewidths=0)
