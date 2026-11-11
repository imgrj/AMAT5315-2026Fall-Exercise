"""Independent NumPy reference for the zero-field square-lattice Ising model.

Everything here is written separately from the Rust crate, so agreement between
the two is evidence about the algorithms rather than about shared code:

* `enumerate_square` sums every spin configuration for small squares (the ground
  truth used by the acceptance checks),
* `boundary_mps` is the sheet's Algorithm 4 with the exact SVD canonicalisation
  of Algorithm 3 (the Rust tool uses an equivalent LQ gauge fix),
* `numpy_trg` is the sheet's Algorithm 5 with `numpy.linalg.svd`.

Conventions: `s_i = +1` for bit 0 of the configuration code, `H = -sum s_i s_j`,
weights `exp(K s_i s_j)`, `g = log Z / L^2`, `c_V = K^2 Var(H) / N`.
"""
from __future__ import annotations

import math

import numpy as np


def square_bonds(length: int, boundary: str) -> list[tuple[int, int]]:
    """Bonds of the square lattice: one per site in each positive direction."""
    bonds: list[tuple[int, int]] = []
    for y in range(length):
        for x in range(length):
            site = y * length + x
            if boundary == 'open':
                if x + 1 < length:
                    bonds.append((site, site + 1))
                if y + 1 < length:
                    bonds.append((site, site + length))
            elif boundary == 'periodic':
                bonds.append((site, y * length + (x + 1) % length))
                bonds.append((site, ((y + 1) % length) * length + x))
            else:
                raise ValueError(boundary)
    return bonds


def energies(length: int, boundary: str) -> np.ndarray:
    """Energy of every one of the `2^(L^2)` configurations."""
    sites = length * length
    if sites > 20:
        raise ValueError(f'{sites} sites is too many to enumerate')
    codes = np.arange(1 << sites, dtype=np.int64)[:, None]
    bits = np.arange(sites, dtype=np.int64)[None, :]
    spins = 1.0 - 2.0 * ((codes >> bits) & 1)
    total = np.zeros(1 << sites)
    for i, j in square_bonds(length, boundary):
        total -= spins[:, i] * spins[:, j]
    return total


def enumerate_square(length: int, k: float, boundary: str) -> dict:
    """`log Z`, `g`, `Var(H)` and `c_V` from a full sum over configurations."""
    if k < 0.0:
        raise ValueError('K must be non-negative')
    heat = energies(length, boundary)
    weights = np.exp(k * heat)
    partition = weights.sum()
    mean = float((weights * heat).sum() / partition)
    variance = float((weights * (heat - mean) ** 2).sum() / partition)
    sites = length * length
    log_z = math.log(float(partition))
    return {
        'boundary': boundary,
        'l': length,
        'k': k,
        'configurations': 1 << sites,
        'bonds': len(square_bonds(length, boundary)),
        'partition': float(partition),
        'log_z': log_z,
        'g': log_z / sites,
        'mean_energy': mean,
        'variance_energy': variance,
        'specific_heat': k * k * variance / sites,
    }


def split_factors(array: np.ndarray, dims: tuple[int, int, int, int],
                  group_a: tuple[int, ...], group_b: tuple[int, ...],
                  chi: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sheet's Algorithm 2: split a tensor along two ordered leg groups."""
    order = group_a + group_b
    rows = int(np.prod([dims[axis] for axis in group_a]))
    cols = int(np.prod([dims[axis] for axis in group_b]))
    assert rows == cols, 'the two groups must have the same size'
    matrix = np.transpose(array, order).reshape(rows, cols)
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    keep = min(chi, len(s))
    root = np.sqrt(s[:keep])
    first = (u[:, :keep] * root).reshape(tuple(dims[axis] for axis in group_a) + (keep,))
    second = (root[:, None] * vt[:keep, :]).reshape((keep,) + tuple(dims[axis] for axis in group_b))
    return first, second, s


def _site_tensor(k: float) -> tuple[np.ndarray, np.ndarray]:
    w = np.array([[math.sqrt(math.cosh(k)), math.sqrt(math.sinh(k))],
                  [math.sqrt(math.cosh(k)), -math.sqrt(math.sinh(k))]])
    site = np.einsum('sr,su,sl,sd->ruld', w, w, w, w)
    close = np.array([1.0 / math.sqrt(math.cosh(k)), 0.0])
    return site, close


def boundary_mps(length: int, k: float, chi: int) -> dict:
    """Sheet's Algorithm 4 on the open square, with the exact SVD canonicalisation."""
    site, close = _site_tensor(k)
    cores = [close.reshape(1, 2, 1).copy() for _ in range(length)]
    ell = 0.0
    for y in range(length):
        absorbed = []
        for i in range(length):
            row = np.transpose(site, (2, 1, 3, 0))  # (l, u, d, r)
            if i == 0:
                row = np.tensordot(close, row, axes=([0], [0]))[None, ...]
            if i == length - 1:
                row = np.tensordot(row, close, axes=([-1], [0]))[..., None]
            contracted = np.einsum('aub,ludr->aldbr', cores[i], row)
            left, physical, right = contracted.shape[0] * contracted.shape[1], contracted.shape[2], contracted.shape[3] * contracted.shape[4]
            absorbed.append(contracted.reshape(left, physical, right))
        cores = absorbed
        # Right-to-left canonicalisation with exact SVDs (Algorithm 3, lines 1-6).
        for i in range(length - 1, 0, -1):
            matrix = cores[i].reshape(cores[i].shape[0], -1)
            u, s, vt = np.linalg.svd(matrix, full_matrices=False)
            cores[i] = vt.reshape(vt.shape[0], 2, cores[i].shape[2])
            cores[i - 1] = np.tensordot(cores[i - 1], u * s, axes=([2], [0]))
        # Left-to-right truncation (Algorithm 3, lines 7-12).
        for i in range(length - 1):
            matrix = cores[i].reshape(-1, cores[i].shape[2])
            u, s, vt = np.linalg.svd(matrix, full_matrices=False)
            keep = min(chi, len(s))
            cores[i] = u[:, :keep].reshape(cores[i].shape[0], 2, keep)
            cores[i + 1] = np.tensordot(s[:keep, None] * vt[:keep, :], cores[i + 1],
                                        axes=([1], [0]))
        scale = float(np.linalg.norm(cores[length - 1]))
        cores[length - 1] = cores[length - 1] / scale
        ell += math.log(scale)
    vector = np.array([1.0])
    for core in cores:
        vector = vector @ np.tensordot(core, close, axes=([1], [0]))
    bottom = float(vector[0])
    log_z = ell + math.log(bottom)
    return {
        'boundary': 'open',
        'l': length,
        'k': k,
        'chi': chi,
        'log_z': log_z,
        'g': log_z / (length * length),
        'normalisation': ell,
        'bottom': bottom,
        'bond_dims': [int(core.shape[2]) for core in cores],
    }


def numpy_trg(length: int, k: float, chi: int) -> dict:
    """Sheet's Algorithm 5 on the periodic square."""
    if length < 2 or (length & (length - 1)) != 0:
        raise ValueError('TRG needs a power-of-two side length')
    site, _ = _site_tensor(k)
    array = site.copy()
    dims = (2, 2, 2, 2)
    steps = 2 * int(math.log2(length))
    tensors = length * length
    ell = 0.0
    spectra = []
    for _ in range(steps):
        scale = float(np.abs(array).max())
        array = array / scale
        ell += tensors * math.log(scale)
        s1, s3, first_spectrum = split_factors(array, dims, (2, 1), (0, 3), chi)
        s2, s4, second_spectrum = split_factors(array, dims, (3, 2), (1, 0), chi)
        array = np.einsum('war,abu,lbg,dgw->ruld', s1, s2, s3, s4)
        dims = array.shape
        spectra.append({'scale': scale, 'dims': list(dims),
                        'split1': first_spectrum.tolist(), 'split2': second_spectrum.tolist()})
        tensors //= 2
    closure = float(np.einsum('ijij->', array))
    log_z = ell + math.log(abs(closure))
    return {
        'boundary': 'periodic',
        'l': length,
        'k': k,
        'chi': chi,
        'log_z': log_z,
        'g': log_z / (length * length),
        'normalisation': ell,
        'closure': closure,
        'steps': spectra,
    }
