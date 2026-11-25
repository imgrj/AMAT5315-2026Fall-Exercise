"""Shared helpers for the Week 12 drivers, evidence plots and web build."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
EVIDENCE = ROOT / "evidence"

COLORS = {
    "most-frac": "#7aa2f7",
    "strong": "#e06c75",
    "neural": "#4ade80",
    "root-neural": "#f6a45c",
    "lp": "#38bdf8",
    "exact": "#f6a45c",
    "gap": "#c084fc",
    "excess": "#fbbf24",
    "grid": "#26344c",
    "ink": "#dce6f2",
    "muted": "#9aaec5",
}


def style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 150,
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "axes.facecolor": "#0f172a",
            "figure.facecolor": "#080c16",
            "savefig.facecolor": "#080c16",
            "axes.edgecolor": "#3b4a63",
            "axes.labelcolor": "#dce6f2",
            "text.color": "#dce6f2",
            "xtick.color": "#9aaec5",
            "ytick.color": "#9aaec5",
            "grid.color": "#26344c",
            "legend.facecolor": "#0f172a",
            "legend.edgecolor": "#3b4a63",
        }
    )


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


BINARY: Path | None = None


def binary() -> Path:
    """Path of the built `branchopt` binary, building it once if it is missing."""
    global BINARY
    if BINARY is not None and BINARY.exists():
        return BINARY
    metadata = json.loads(
        subprocess.check_output(
            ["cargo", "metadata", "--format-version", "1", "--no-deps"], cwd=ROOT, text=True
        )
    )
    target = Path(metadata["target_directory"]) / "release"
    exe = target / ("branchopt.exe" if sys.platform == "win32" else "branchopt")
    if not exe.exists():
        # HiGHS is compiled from source here, so the build needs cmake and a C++
        # compiler on PATH; see README.md for the recorded environment.
        subprocess.run(["cargo", "build", "--release"], cwd=ROOT, check=True)
    assert exe.exists(), f"{exe} is missing; run `cargo build --release` in week12/"
    BINARY = exe
    return exe


def run(args: list[str]) -> dict:
    """Run `branchopt` and return its saved run.json."""
    completed = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"{' '.join(args)}\n{completed.stdout}\n{completed.stderr}")
    return read_json(Path(args[args.index("--out") + 1]) / "run.json")


def save(figure, path: Path, tight: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    print(f"wrote {path.relative_to(ROOT)}")


def sorted_nodes(run_record: dict, policy: str) -> list[float]:
    return sorted(
        float(entry["policies"][policy]["nodes"]) for entry in run_record["instances"]
    )


def sorted_nodes_bnb(run_record: dict) -> list[float]:
    return sorted(float(entry["nodes"]) for entry in run_record["instances"])
