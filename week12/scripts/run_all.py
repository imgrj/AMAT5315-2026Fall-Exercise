"""Run the whole Week 12 pipeline in the order of the learning sheet.

From `week12/`:  python scripts/run_all.py

Stages 1-2 are the integer-program benchmark, 3 the exact Branch-and-Bound
comparison, 4-5 the imitation-learning round, 6 the laptop and size-transfer
measurement, and 7 the four evidence figures.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common as common  # noqa: E402


def call(args: list[str], cwd: Path | None = None) -> None:
    print(f"$ {' '.join(str(a) for a in args)}", flush=True)
    subprocess.run(args, cwd=cwd or common.ROOT, check=True)


def main() -> None:
    exe = str(common.binary())
    py = sys.executable

    # Part 1: the MILP formulation, its relaxation and the diet LP.
    call([exe, "--mode", "gap", "--spins", "22", "--edges", "52",
          "--count", "16", "--seed", "2026", "--out", "artifacts/gap"])
    call([py, "scripts/plot_gap.py"])

    # Part 2: exact Branch-and-Bound, cheap rule against Strong Branching.
    call([exe, "--mode", "bnb", "--spins", "22", "--edges", "52", "--count", "48",
          "--policy", "most-frac", "--seed", "2026", "--out", "artifacts/bnb_most"])
    call([exe, "--mode", "bnb", "--spins", "22", "--edges", "52", "--count", "48",
          "--policy", "strong", "--seed", "2026", "--out", "artifacts/bnb_strong"])

    # Part 3: record Strong Branching, train the network, evaluate it. The label
    # run reuses the Part 2 output directory (the sheet's command), so the plain
    # Part 2 run is repeated afterwards to leave the timing record of `bnb.png`
    # free of label-export overhead.
    call([exe, "--mode", "bnb", "--spins", "22", "--edges", "52", "--count", "48",
          "--policy", "strong", "--export-labels", "--seed", "2026",
          "--out", "artifacts/bnb_strong"])
    call([exe, "--mode", "bnb", "--spins", "22", "--edges", "52", "--count", "48",
          "--policy", "strong", "--seed", "2026", "--out", "artifacts/bnb_strong"])
    call([py, "scripts/plot_bnb.py"])
    call(["uv", "run", "--with", "jax==0.12.1", "python", "train_branch.py",
          "--dataset", "artifacts/bnb_strong/dataset.json",
          "--out", "artifacts/model.json"])
    call([exe, "--mode", "eval", "--spins", "22", "--edges", "52", "--count", "16",
          "--seed", "3026", "--model", "artifacts/model.json",
          "--out", "artifacts/eval_test"])

    # Part 4: laptop timing and the 30-spin, 72-bond transfer test.
    call([exe, "--mode", "eval", "--spins", "30", "--edges", "72", "--count", "8",
          "--seed", "4026", "--model", "artifacts/model.json",
          "--out", "artifacts/eval_transfer"])
    call([py, "scripts/plot_neural_bnb.py"])
    call([py, "scripts/plot_eval.py"])


if __name__ == "__main__":
    main()
