"""Independent Week 9 reference implementations.

Nothing here is imported by the Rust implementation: the parsers, the ASIA
enumeration, the optimal-order dynamic program and the small tensor examples
are written from the learning sheet and SPEC.md so they can judge the tool.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


# --- UAI (MARKOV) model file -------------------------------------------------

@dataclass(frozen=True)
class Factor:
    scope: tuple[int, ...]
    dims: tuple[int, ...]
    entries: tuple[float, ...]  # row-major over scope, last scope variable fastest

    def table(self) -> np.ndarray:
        return np.asarray(self.entries, dtype=np.float64).reshape(self.dims)


@dataclass(frozen=True)
class Model:
    path: Path
    variables: int
    dims: tuple[int, ...]
    factors: tuple[Factor, ...]


def read_uai(path: Path) -> Model:
    tokens = path.read_text().split()
    assert tokens[0] == 'MARKOV', tokens[0]
    at = 1
    variables = int(tokens[at]); at += 1
    dims = tuple(int(t) for t in tokens[at:at + variables]); at += variables
    count = int(tokens[at]); at += 1
    scopes = []
    for _ in range(count):
        rank = int(tokens[at]); at += 1
        scopes.append(tuple(int(t) for t in tokens[at:at + rank])); at += rank
    factors = []
    for scope in scopes:
        size = int(tokens[at]); at += 1
        expected = math.prod(dims[v] for v in scope)
        assert size == expected, (scope, size, expected)
        values = tuple(float(t) for t in tokens[at:at + size]); at += size
        factors.append(Factor(scope, tuple(dims[v] for v in scope), values))
    assert at == len(tokens), (at, len(tokens))
    return Model(path, variables, dims, tuple(factors))


def read_evid(path: Path) -> dict[int, int]:
    tokens = path.read_text().split()
    count = int(tokens[0])
    assert len(tokens) == 1 + 2 * count, tokens[:4]
    return {int(tokens[1 + 2 * i]): int(tokens[2 + 2 * i]) for i in range(count)}


def read_mar(path: Path) -> list[np.ndarray]:
    """State-ordered marginals, one array per variable, in variable order."""
    tokens = path.read_text().split()
    assert tokens[0] == 'MAR', tokens[0]
    count = int(tokens[1])
    at = 2
    marginals = []
    for _ in range(count):
        size = int(tokens[at]); at += 1
        marginals.append(np.asarray([float(t) for t in tokens[at:at + size]]))
        at += size
    assert at == len(tokens), (at, len(tokens))
    return marginals


def read_pr(path: Path) -> float:
    tokens = path.read_text().split()
    assert tokens[0] == 'PR', tokens[0]
    return float(tokens[1])


# --- ASIA enumeration --------------------------------------------------------

def row_probability(model: Model, assignment: dict[int, int]) -> float:
    product = 1.0
    for factor in model.factors:
        flat = 0
        for axis, variable in enumerate(factor.scope):
            flat = flat * factor.dims[axis] + assignment[variable]
        product *= factor.entries[flat]
    return product


@dataclass(frozen=True)
class AsisResult:
    mass: float
    marginals: dict[int, np.ndarray]
    joints: dict[tuple[int, ...], np.ndarray]


def enumerate_query(model: Model, observations: dict[int, int],
                    joints: list[tuple[int, ...]] = (), limit: int = 12) -> AsisResult:
    """Sum every joint row consistent with the observations, keeping the requested joints."""
    if model.variables > limit:
        raise ValueError('enumeration is only used for the small ASIA model')
    mass = 0.0
    marginals = {v: np.zeros(size) for v, size in enumerate(model.dims)}
    joint_arrays = {key: np.zeros(tuple(model.dims[v] for v in key)) for key in joints}
    for states in itertools.product(*(range(size) for size in model.dims)):
        assignment = dict(enumerate(states))
        if any(assignment[v] != state for v, state in observations.items()):
            continue
        p = row_probability(model, assignment)
        mass += p
        for v in marginals:
            marginals[v][states[v]] += p
        for key, array in joint_arrays.items():
            array[tuple(states[v] for v in key)] += p
    normalized = {v: array / mass for v, array in marginals.items()}
    return AsisResult(mass, normalized, {k: a / mass for k, a in joint_arrays.items()})


# --- Contraction-order cost model -------------------------------------------

def pair_work(left: frozenset[int], right: frozenset[int], dims) -> float:
    return float(np.prod([dims[l] for l in sorted(left | right)]))


def optimal_work(scopes, dims, query: frozenset[int] = frozenset()) -> float:
    """Exhaustive dynamic program over factor subsets (the sheet's pair-order search).

    A leaf carries every axis of its table; an intermediate carries the labels that
    another tensor or the output still needs.
    """
    n = len(scopes)
    labels = [frozenset(s) for s in scopes]
    carried: dict[int, frozenset[int]] = {}
    for mask in range(1, 1 << n):
        inside = frozenset().union(*(labels[i] for i in range(n) if mask >> i & 1))
        if mask & (mask - 1) == 0:
            carried[mask] = inside
        else:
            rest = frozenset().union(*(labels[i] for i in range(n) if not mask >> i & 1), query)
            carried[mask] = inside & rest
    cost = {1 << i: 0.0 for i in range(n)}
    for mask in range(1, 1 << n):
        if mask & (mask - 1) == 0:
            continue
        low = mask & -mask
        sub = (mask - 1) & mask
        best = math.inf
        while sub:
            if sub & low:
                other = mask ^ sub
                if other:
                    best = min(best, cost[sub] + cost[other]
                               + pair_work(carried[sub], carried[other], dims))
            sub = (sub - 1) & mask
        cost[mask] = best
    return cost[(1 << n) - 1]


# --- Small tensor identities -------------------------------------------------

A_MATRIX = np.array([[1.0, 2.0], [3.0, 4.0]])
B_MATRIX = np.array([[2.0, 0.0], [1.0, 2.0]])


def matrix_product() -> np.ndarray:
    return A_MATRIX @ B_MATRIX


def matrix_trace() -> float:
    return float(np.trace(A_MATRIX @ B_MATRIX))


def matrix_total() -> float:
    return float(np.sum(A_MATRIX @ B_MATRIX))


def lung_bronchitis_example() -> np.ndarray:
    """p(l,b) for the sheet's three-factor example: sum over smoking only."""
    p_s = np.array([0.5, 0.5])
    p_ls = np.array([[0.1, 0.01], [0.9, 0.99]])   # rows L, columns S
    p_bs = np.array([[0.6, 0.3], [0.4, 0.7]])     # rows B, columns S
    return np.einsum('s,ls,bs->lb', p_s, p_ls, p_bs)


def ising_ring(beta: float = 0.5, coupling: float = 1.0) -> float:
    total = 0.0
    for s1, s2, s3 in itertools.product((-1, 1), repeat=3):
        total += math.exp(beta * coupling * (s1 * s2 + s2 * s3 + s3 * s1))
    return total


# --- ASIA cap network --------------------------------------------------------

ASIA_SCOPES = ((0,), (1, 0), (2,), (3, 2), (4, 2), (5, 3, 1), (6, 5), (7, 5, 4))


def cap_scopes(model: Model, observations: dict[int, int]):
    """Observed variables enter as (1,0) cap tensors, as drawn in the sheet."""
    scopes = [factor.scope for factor in model.factors]
    scopes += [(v,) for v in sorted(observations)]
    return scopes
