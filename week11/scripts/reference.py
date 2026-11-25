"""Independent references for the Week 11 sheet.

Nothing here imports the Rust tool: the gates are the sheet's own formulas, the
magnet partitions sums are contracted with NumPy's `einsum` in a greedy order of
our own, small magnets are summed over every one of their `2^n` states, and the
queen rules are checked directly on a board. The generator is reimplemented from
the documented splitmix64 description so the lattice bonds can be reproduced.
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# The documented splitmix64 generator (Steele, Lea & Flood 2014)
# ---------------------------------------------------------------------------
MASK64 = (1 << 64) - 1
STREAM_BONDS = 0x243F6A8885A308D3
STREAM_SAMPLE = 0xA4093822299F31D0


class SplitMix64:
    def __init__(self, seed: int, stream: int) -> None:
        self.state = (seed ^ stream) & MASK64
        self.next_u64()  # one value is discarded, exactly as in src/rng.rs

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def next_f64(self) -> float:
        return ((self.next_u64() >> 11) + 0.5) / 9007199254740992.0


def lattice_bonds(l: int, p_af: float, seed: int) -> list[tuple[int, int, float]]:
    """Open `l x l` bonds `(i, j, K)` with `E = sum K s_i s_j`.

    Row-major sites, the horizontal bond first, one uniform draw per bond; a draw
    below `p_af` makes `J = -1`, i.e. `K = +1`.
    """
    rng = SplitMix64(seed, STREAM_BONDS)
    bonds = []
    for r in range(l):
        for c in range(l):
            i = r * l + c
            if c + 1 < l:
                bonds.append((i, r * l + c + 1, 1.0 if rng.next_f64() < p_af else -1.0))
            if r + 1 < l:
                bonds.append((i, (r + 1) * l + c, 1.0 if rng.next_f64() < p_af else -1.0))
    return bonds


# ---------------------------------------------------------------------------
# The sheet's gate energies, written out independently of the Rust source
# ---------------------------------------------------------------------------
def gates() -> dict[str, tuple[list[str], dict]]:
    """Every gadget's bit names and the energy of each of its bit rows."""
    out: dict[str, tuple[list[str], dict]] = {}

    def and_(a: int, b: int, z: int) -> float:
        sa, sb, sz = 2 * a - 1, 2 * b - 1, 2 * z - 1
        return sa * sb - 2 * sa * sz - 2 * sb * sz - sa - sb + 2 * sz

    def or_(a: int, b: int, z: int) -> float:
        sa, sb, sz = 2 * a - 1, 2 * b - 1, 2 * z - 1
        return sa * sb - 2 * sa * sz - 2 * sb * sz + sa + sb - 2 * sz

    def xor_(a: int, b: int, z: int, w: int) -> float:
        sa, sb, sz, sw = 2 * a - 1, 2 * b - 1, 2 * z - 1, 2 * w - 1
        return (sa * sb - sa * sz - 2 * sa * sw - sb * sz - 2 * sb * sw + 2 * sz * sw
                - sa - sb + sz + 2 * sw)

    out["AND"] = (["a", "b", "z"],
                  {(a, b, z): and_(a, b, z) for a in (0, 1) for b in (0, 1) for z in (0, 1)})
    out["OR"] = (["a", "b", "z"],
                 {(a, b, z): or_(a, b, z) for a in (0, 1) for b in (0, 1) for z in (0, 1)})
    out["NOT"] = (["a", "z"], {(a, z): (2 * a - 1) * (2 * z - 1) for a in (0, 1) for z in (0, 1)})
    out["COPY"] = (["a", "z"], {(a, z): -(2 * a - 1) * (2 * z - 1) for a in (0, 1) for z in (0, 1)})
    out["SET0"] = (["z"], {(z,): 2 * z - 1 for z in (0, 1)})
    out["SET1"] = (["z"], {(z,): -(2 * z - 1) for z in (0, 1)})
    out["XOR"] = (["a", "b", "z"],
                  {(a, b, z): min(xor_(a, b, z, w) for w in (0, 1))
                   for a in (0, 1) for b in (0, 1) for z in (0, 1)})
    return out


def gate_allowed(name: str, row: tuple[int, ...]) -> bool:
    table = {
        "AND": lambda a, b, z: z == (a & b),
        "OR": lambda a, b, z: z == (a | b),
        "NOT": lambda a, z: z == 1 - a,
        "COPY": lambda a, z: z == a,
        "SET0": lambda z: z == 0,
        "SET1": lambda z: z == 1,
        "XOR": lambda a, b, z: z == (a ^ b),
    }
    return bool(table[name](*row))


def circuit_energy(a: int, b: int, t: int, o: int) -> float:
    sa, sb, st, so = 2 * a - 1, 2 * b - 1, 2 * t - 1, 2 * o - 1
    h_and = sa * sb - 2 * sa * st - 2 * sb * st - sa - sb + 2 * st
    h_not = st * so
    h_set1 = -so
    return h_and + h_not + h_set1


# ---------------------------------------------------------------------------
# Magnets: exact enumeration and a NumPy contraction of the same network
# ---------------------------------------------------------------------------
def bundle_magnet(path: Path) -> dict:
    """The bundle's `E = sum K_ij s_i s_j + sum h_i s_i` in bit variables."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    data = value["target"]["data"]
    edges = data["graph"]["edges"]
    couplings = data["couplings"]
    fields = [float(h) for h in data["fields"]]
    nspins = int(data["graph"]["num_vertices"])
    bonds = [(int(e[0]), int(e[1]), float(k)) for e, k in zip(edges, couplings) if float(k) != 0.0]
    return {"nspins": nspins, "bonds": bonds, "fields": fields}


def magnet_energies(magnet: dict) -> np.ndarray:
    """The energy of every one of the `2^n` spin states, in bit order."""
    n = magnet["nspins"]
    assert n <= 20, "exact enumeration is only for the small checks"
    states = np.arange(1 << n)[:, None]
    spins = 2.0 * ((states >> np.arange(n)[None, :]) & 1) - 1.0
    energy = np.zeros(1 << n)
    for i, j, k in magnet["bonds"]:
        energy += k * spins[:, i] * spins[:, j]
    for i, h in enumerate(magnet["fields"]):
        if h != 0.0:
            energy += h * spins[:, i]
    return energy


def magnet_terms(magnet: dict, beta: float) -> list[tuple[tuple[int, ...], np.ndarray]]:
    """The Boltzmann network `exp(-beta E)` as `(labels, array)` terms."""
    terms: list[tuple[tuple[int, ...], np.ndarray]] = []
    for i, j, k in magnet["bonds"]:
        weight = np.array([[math.exp(-beta * k * si * sj) for sj in (-1.0, 1.0)]
                           for si in (-1.0, 1.0)])
        terms.append(((i, j), weight))
    for i, h in enumerate(magnet["fields"]):
        if h != 0.0:
            terms.append(((i,), np.array([math.exp(beta * h), math.exp(-beta * h)])))
    return terms


def _pick_pair(sizes: dict[int, int], labels: list[tuple[int, ...]]) -> tuple[int, int]:
    """Greedy: contract the pair whose union of labels is smallest."""
    best = None
    for p in range(len(labels)):
        for q in range(p + 1, len(labels)):
            union = set(labels[p]) | set(labels[q])
            size = 1
            for label in union:
                size *= sizes[label]
            key = (size, p, q)
            if best is None or key < best:
                best = key
    assert best is not None
    return best[1], best[2]


def contract_log(terms: list[tuple[tuple[int, ...], np.ndarray]], sizes: dict[int, int]) -> float:
    """`log` of the full contraction, rescaling every intermediate like the tool.

    The order is a greedy minimum-intermediate-size order of our own, so this
    checks the tool's answer without sharing its `omeco` plan.
    """
    tensors = [(tuple(labels), np.array(array, dtype=float)) for labels, array in terms]
    log_scale = 0.0
    while len(tensors) > 1:
        labels = [labels for labels, _ in tensors]
        p, q = _pick_pair(sizes, labels)
        (la, a), (lb, b) = tensors[p], tensors[q]
        needed: set[int] = set()
        for index, (other, _) in enumerate(tensors):
            if index not in (p, q):
                needed |= set(other)
        combined = list(dict.fromkeys(list(la) + list(lb)))
        out = [label for label in combined if label in needed]
        subscript = {label: n for n, label in enumerate(combined)}
        result = np.einsum(
            a, [subscript[label] for label in la],
            b, [subscript[label] for label in lb],
            [subscript[label] for label in out],
        )
        result = np.asarray(result, dtype=float)
        scale = float(np.max(np.abs(result))) if result.size else 0.0
        if scale > 0.0 and scale != 1.0:
            result = result / scale
            log_scale += math.log(scale)
        tensors = [t for index, t in enumerate(tensors) if index not in (p, q)]
        tensors.append((tuple(out), result))
    labels, array = tensors[0]
    assert not labels
    value = float(array)
    return log_scale + math.log(value)


def log_partition(magnet: dict, beta: float) -> float:
    terms = magnet_terms(magnet, beta)
    sizes = {label: 2 for label in range(magnet["nspins"])}
    return contract_log(terms, sizes)


def ground_state(magnet: dict, beta: float = 20.0) -> tuple[float, int]:
    """`(E_min, g_0)` from `log Z` at two low temperatures, by the sheet's (13).

    With `log Z(beta) = log g_0 - beta E_min + log(1 + ...)`, the slope over two
    large betas gives `E_min` and the intercept gives `g_0`, which must be an
    integer.
    """
    beta2 = beta + 4.0
    first = log_partition(magnet, beta)
    second = log_partition(magnet, beta2)
    e_min = (second - first) / (beta - beta2)
    log_g = first + beta * e_min
    g0 = round(math.exp(log_g))
    return e_min, int(g0)


# ---------------------------------------------------------------------------
# Boltzmann sampling theory
# ---------------------------------------------------------------------------
def p_ground(g0: int, e_min: float, beta: float, log_z: float) -> float:
    return math.exp(math.log(g0) - beta * e_min - log_z)


def binomial_spread(samples: int, p: float) -> float:
    return math.sqrt(samples * p * (1.0 - p))


# ---------------------------------------------------------------------------
# N-queens
# ---------------------------------------------------------------------------
#: OEIS A000170, the number of ways to place N non-attacking queens.
KNOWN_QUEENS = {1: 1, 2: 0, 3: 0, 4: 2, 5: 10, 6: 4, 7: 40, 8: 92, 9: 352,
                10: 724, 11: 2680, 12: 14200}


def valid_board(board: list[int]) -> bool:
    """The queen rules on a board given as the queen's column in every row."""
    n = len(board)
    if any(not 0 <= column < n for column in board):
        return False
    if len(set(board)) != n:
        return False
    if len({row - column for row, column in enumerate(board)}) != n:
        return False
    if len({row + column for row, column in enumerate(board)}) != n:
        return False
    return True


def enumerate_boards(n: int) -> list[list[int]]:
    """Every valid `n`-queens board, by trying every column permutation."""
    good = []
    for permutation in itertools.permutations(range(n)):
        board = list(permutation)
        if valid_board(board):
            good.append(board)
    return good


def uniform_bookkeeping(count: int, samples: int) -> float:
    """The expected number of distinct boards seen in `samples` uniform draws."""
    return count * (1.0 - ((count - 1.0) / count) ** samples)
