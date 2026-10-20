"""Distribution measurements for the mixture: cluster stats, energy distance, KS."""

from __future__ import annotations

import jax
import numpy as np
from scipy.spatial.distance import cdist

from . import mixture


def cluster_stats(samples) -> dict:
    """Population, centre and per-coordinate spread of each nearest-centre group."""
    x = np.asarray(samples, dtype=np.float64)
    labels = mixture.nearest_component(x)
    populations = np.zeros(mixture.COMPONENTS)
    centres = np.full((mixture.COMPONENTS, 2), np.nan)
    spreads = np.full((mixture.COMPONENTS, 2), np.nan)
    for k in range(mixture.COMPONENTS):
        members = x[labels == k]
        populations[k] = members.shape[0] / x.shape[0]
        if members.shape[0] > 1:
            centres[k] = members.mean(axis=0)
            spreads[k] = members.std(axis=0, ddof=0)
    return {
        "populations": populations,
        "centres": centres,
        "spreads": spreads,
        "labels": labels,
    }


def max_population_error(samples) -> float:
    return float(np.max(np.abs(cluster_stats(samples)["populations"] - 0.25)))


def max_centre_error(samples) -> float:
    return float(np.max(np.abs(cluster_stats(samples)["centres"] - mixture.CENTERS)))


def max_width_error(samples) -> float:
    return float(np.max(np.abs(cluster_stats(samples)["spreads"] - mixture.STD)))


def _mean_pairwise(x, y) -> float:
    """Mean Euclidean distance over distinct pairs (``x`` may be ``y``)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    same = x is y
    total = 0.0
    chunk = 512
    for start in range(0, x.shape[0], chunk):
        block = cdist(x[start : start + chunk], y)
        if same:
            rows = np.arange(block.shape[0])
            block[rows, start + rows] = 0.0
            total += block.sum()
        else:
            total += block.sum()
    count = x.shape[0] * (x.shape[0] - 1) if same else x.shape[0] * y.shape[0]
    return total / count


def energy_distance(x, y) -> float:
    """Equation (18): ``2 E||X-Y|| - E||X-X'|| - E||Y-Y'||``."""
    x = np.asarray(x)
    y = np.asarray(y)
    return float(2.0 * _mean_pairwise(x, y) - _mean_pairwise(x, x) - _mean_pairwise(y, y))


def null_scale(key, size: int = 4000, pairs: int = 20) -> dict:
    """95% quantile of the energy distance between two exact sample sets."""
    values = []
    for i in range(pairs):
        k1, k2, key = jax.random.split(key, 3)
        a = np.asarray(mixture.sample(k1, size))
        b = np.asarray(mixture.sample(k2, size))
        values.append(energy_distance(a, b))
    values = np.asarray(values)
    return {
        "values": values,
        "q95": float(np.quantile(values, 0.95)),
        "size": size,
        "pairs": pairs,
    }


def ks_distance(a, b) -> float:
    """Largest gap between the two empirical cumulative distribution functions."""
    a = np.sort(np.asarray(a, dtype=np.float64).ravel())
    b = np.sort(np.asarray(b, dtype=np.float64).ravel())
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / a.size
    cdf_b = np.searchsorted(b, grid, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))
