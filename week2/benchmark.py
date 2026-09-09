#!/usr/bin/env python3
"""Benchmark md run: --force naive vs --force cells at several N.

Each cell: three runs of
    md run -n N --temp 1.0 --dt 0.005 --steps 500 --equil 100 \
        --force naive|cells --seed 1 --out /tmp/mdbench_...
Reported: median wall seconds with (min-max); speedup = median naive / median cells.
Seconds per step = wall time / (equil + steps) = wall / 600.

Run from week2/ after `cargo build --release`:
    python3 benchmark.py
Writes week2/scaling.png and prints the markdown table.
"""
import statistics
import subprocess
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BIN = "target/release/md"
N_SIZES = [100, 400, 1600]
STEPS = 500
EQUIL = 100
REPS = 3


def run_once(n: int, force: str) -> float:
    cmd = [
        BIN, "run",
        "-n", str(n),
        "--temp", "1.0",
        "--dt", "0.005",
        "--steps", str(STEPS),
        "--equil", str(EQUIL),
        "--force", force,
        "--seed", "1",
        "--out", f"/tmp/mdbench_{n}_{force}.txt",
    ]
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True, capture_output=True)
    return time.perf_counter() - t0


results: dict[int, dict[str, list[float]]] = {}
for n in N_SIZES:
    results[n] = {"naive": [], "cells": []}

# Interleave the strategies so machine drift does not bias one side.
for _ in range(REPS):
    for n in N_SIZES:
        for force in ("naive", "cells"):
            results[n][force].append(run_once(n, force))

print("| N | naive (s) median (min-max) | cells (s) median (min-max) | speedup |")
print("|---|---|---|---|")
table = {}
for n in N_SIZES:
    med = {f: statistics.median(results[n][f]) for f in ("naive", "cells")}
    rng = {f: (min(results[n][f]), max(results[n][f])) for f in ("naive", "cells")}
    speedup = med["naive"] / med["cells"]
    table[n] = (med, rng, speedup)
    print(
        f"| {n} | {med['naive']:.3f} ({rng['naive'][0]:.3f}-{rng['naive'][1]:.3f})"
        f" | {med['cells']:.3f} ({rng['cells'][0]:.3f}-{rng['cells'][1]:.3f})"
        f" | {speedup:.1f}x |"
    )

# scaling.png: seconds per (equil+recorded) step against N, log-log.
xs = N_SIZES
fig, ax = plt.subplots(figsize=(6.4, 4.4))
for force, color, marker in [("naive", "#dc1e3c", "o"), ("cells", "#0d6efd", "s")]:
    ys = [statistics.median(results[n][force]) / (STEPS + EQUIL) for n in xs]
    ax.plot(xs, ys, label=force, color=color, marker=marker, linewidth=1.6)
    for n, y in zip(xs, ys):
        ax.annotate(f"{y:.2e}", (n, y), textcoords="offset points", xytext=(6, 6), fontsize=8)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("N (atoms)")
ax.set_ylabel("seconds per step (equil + recorded)")
ax.set_title("md run: naive vs cell-list force")
ax.legend()
ax.grid(which="both", alpha=0.3)
fig.tight_layout()
fig.savefig("scaling.png", dpi=150)
print("wrote scaling.png")
