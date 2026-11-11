"""Specific-heat helpers: finite differences, peak location and size extrapolation.

Every function here is a pure calculation on saved numbers, so the tests can
check it against analytic curves and the scan drivers can share it.
"""
from __future__ import annotations

import math

ONSAGER_T = 2.0 / math.log(1.0 + math.sqrt(2.0))
ONSAGER_K = 1.0 / ONSAGER_T


def specific_heat(ks: list[float], gs: list[float], h: float) -> list[tuple[float, float]]:
    """`c_V(K; h) = K^2 [g(K+h) - 2 g(K) + g(K-h)] / h^2` on the sampled grid.

    Only interior points whose two neighbours exist at distance `h` are returned.
    """
    assert len(ks) == len(gs)
    table = {round(k, 9): g for k, g in zip(ks, gs)}
    points = []
    for k, g in zip(ks, gs):
        low = table.get(round(k - h, 9))
        high = table.get(round(k + h, 9))
        if low is None or high is None:
            continue
        second = (high - 2.0 * g + low) / (h * h)
        points.append((k, k * k * second))
    return points


def parabola_vertex(ks: list[float], values: list[float]) -> tuple[float, float]:
    """Vertex of the parabola through three (K, c_V) points, plus its curvature."""
    assert len(ks) == 3 and len(values) == 3
    (x0, y0), (x1, y1), (x2, y2) = zip(ks, values)
    denominator = (x0 - x1) * (x0 - x2) * (x1 - x2)
    assert denominator != 0.0, 'the three points must be distinct'
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denominator
    b = (
        x2 * x2 * (y0 - y1)
        + x1 * x1 * (y2 - y0)
        + x0 * x0 * (y1 - y2)
    ) / denominator
    assert a < 0.0, f'the refinement points must bracket a maximum, curvature {a}'
    return -b / (2.0 * a), a


def peak_from_points(points: list[tuple[float, float]]) -> dict:
    """Locate the maximum and turn it into `T_peak` with a three-point parabola."""
    assert len(points) >= 3, 'need at least three c_V values around the peak'
    best = max(range(len(points)), key=lambda index: points[index][1])
    if best == 0 or best == len(points) - 1:
        raise ValueError('the peak sits on the edge of the sampled window')
    window = points[best - 1:best + 2]
    k_peak, curvature = parabola_vertex([p[0] for p in window], [p[1] for p in window])
    return {
        'grid_k': points[best][0],
        'grid_cv': points[best][1],
        'k_peak': k_peak,
        't_peak': 1.0 / k_peak,
        'curvature': curvature,
        'window': [[k, c] for k, c in window],
    }


def linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Least-squares `y = intercept + slope x`."""
    assert len(xs) == len(ys) and len(xs) >= 2
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    variance = sum((x - mean_x) ** 2 for x in xs)
    slope = covariance / variance
    return mean_y - slope * mean_x, slope


def tpeak_fit(sizes: list[int], tpeaks: list[float]) -> dict:
    """Fit `T_peak(L) = T_c + a/L`; the intercept estimates `T_c`."""
    xs = [1.0 / size for size in sizes]
    intercept, slope = linear_fit(xs, tpeaks)
    return {
        'sizes': sizes,
        't_peaks': tpeaks,
        't_c': intercept,
        'slope': slope,
        'residuals': [y - (intercept + slope * x) for x, y in zip(xs, tpeaks)],
    }


def relative_difference(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b))


def storage_entries(side: int, rank: int) -> tuple[int, int]:
    """Stored entries of the two thin factors and of the original `side x side` matrix."""
    return side * rank + rank * side, side * side
