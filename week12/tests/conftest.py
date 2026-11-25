"""Fixtures and an independent NumPy re-implementation of the Week 12 model.

Nothing here imports the Rust code: the generator, the energies and the
combinatorial ground state are recomputed from the design file, so a defect in
`branchopt` cannot hide behind a shared implementation.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"

MASK64 = (1 << 64) - 1
GAMMA = 0x9E3779B97F4A7C15
MUL1 = 0xBF58476D1CE4E5B9
MUL2 = 0x94D049BB133111EB
ID_SALT = 0xD1B54A32D192ED03


class SplitMix64:
    """The generator written out in `branchopt.design.toml`."""

    def __init__(self, state: int) -> None:
        self.state = state & MASK64

    def next_u64(self) -> int:
        self.state = (self.state + GAMMA) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * MUL1) & MASK64
        z = ((z ^ (z >> 27)) * MUL2) & MASK64
        return (z ^ (z >> 31)) & MASK64

    def below(self, k: int) -> int:
        return self.next_u64() % k


def build_instance(seed: int, ident: int, n: int, m: int):
    """Edges, couplings and fields of instance `ident`, from the design file."""
    salted = SplitMix64(seed ^ ((ident * ID_SALT) & MASK64))
    rng = SplitMix64(salted.next_u64())

    edges: set[tuple[int, int]] = set()
    degree = [0] * n
    min_degree = min(2, n - 1)
    for i in range(n):
        while degree[i] < min_degree:
            j = rng.below(n)
            if j == i:
                continue
            u, v = min(i, j), max(i, j)
            if (u, v) not in edges:
                edges.add((u, v))
                degree[u] += 1
                degree[v] += 1
    while len(edges) < m:
        i, j = rng.below(n), rng.below(n)
        if i == j:
            continue
        edges.add((min(i, j), max(i, j)))

    ordered = sorted(edges)
    couplings = [1.0 if (rng.next_u64() & 1) == 0 else -1.0 for _ in ordered]
    fields = [float(rng.below(3) - 1) for _ in range(n)]
    return ordered, couplings, fields


def energy(edges, couplings, fields, spins) -> float:
    total = sum(j * spins[u] * spins[v] for (u, v), j in zip(edges, couplings))
    total += sum(h * s for h, s in zip(fields, spins))
    return float(total)


def brute_force_ground_state(edges, couplings, fields) -> tuple[float, np.ndarray]:
    """Lowest energy over all 2^n spin states, by enumeration."""
    n = len(fields)
    states = 1 - 2 * ((np.arange(2**n)[:, None] >> np.arange(n)[None, :]) & 1)
    value = states @ np.asarray(fields, dtype=float)
    for (u, v), j in zip(edges, couplings):
        value = value + j * states[:, u] * states[:, v]
    best = int(np.argmin(value))
    return float(value[best]), states[best]


def milp_energy(inst: dict, x, z) -> float:
    """`H = C + sum_i (-2 h_i) x_i + sum_e (-2 J_e) z_e` of Eq. 6."""
    couplings = np.asarray(inst["couplings"], dtype=float)
    fields = np.asarray(inst["fields"], dtype=float)
    offset = couplings.sum() + fields.sum()
    return float(
        offset + (-2.0 * fields * np.asarray(x, dtype=float)).sum()
        + (-2.0 * couplings * np.asarray(z, dtype=float)).sum()
    )


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def binary() -> Path:
    metadata = json.loads(
        subprocess.check_output(
            ["cargo", "metadata", "--format-version", "1", "--no-deps"], cwd=ROOT, text=True
        )
    )
    exe = Path(metadata["target_directory"]) / "release" / (
        "branchopt.exe" if sys.platform == "win32" else "branchopt"
    )
    return exe


def require(path: Path, command: str) -> dict:
    if not path.exists():
        pytest.skip(f"{path.relative_to(ROOT)} is missing; run `{command}` first")
    return read_json(path)


@pytest.fixture(scope="session")
def gap() -> dict:
    return require(ARTIFACTS / "gap" / "run.json", "python scripts/run_all.py")


@pytest.fixture(scope="session")
def bnb_most() -> dict:
    return require(ARTIFACTS / "bnb_most" / "run.json", "python scripts/run_all.py")


@pytest.fixture(scope="session")
def bnb_strong() -> dict:
    return require(ARTIFACTS / "bnb_strong" / "run.json", "python scripts/run_all.py")


@pytest.fixture(scope="session")
def dataset() -> dict:
    return require(ARTIFACTS / "bnb_strong" / "dataset.json", "python scripts/run_all.py")


@pytest.fixture(scope="session")
def model() -> dict:
    return require(ARTIFACTS / "model.json", "python scripts/run_all.py")


@pytest.fixture(scope="session")
def eval_test() -> dict:
    return require(ARTIFACTS / "eval_test" / "run.json", "python scripts/run_all.py")


@pytest.fixture(scope="session")
def eval_transfer() -> dict:
    return require(ARTIFACTS / "eval_transfer" / "run.json", "python scripts/run_all.py")


@pytest.fixture(scope="session")
def small_run(tmp_path_factory) -> dict:
    """A 12-spin family small enough to enumerate all 2^12 spin states."""
    out = tmp_path_factory.mktemp("small-gap")
    exe = binary()
    if not exe.exists():
        pytest.skip("branchopt is not built; run `cargo build --release` in week12/")
    subprocess.run(
        [str(exe), "--mode", "gap", "--spins", "12", "--edges", "30",
         "--count", "3", "--seed", "2026", "--out", str(out)],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    return read_json(out / "run.json")
