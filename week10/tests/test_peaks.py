"""The peak analysis itself: finite differences, parabola vertices and fits.

These checks use analytic curves, so they test the arithmetic the peak study
depends on instead of re-reading the data it produced.
"""
from __future__ import annotations

import math

import pytest

import peaks


def test_parabola_vertex_is_exact_for_a_quadratic():
    # y = -3 (x - 0.47)^2 + 2
    ks = [0.46, 0.47, 0.48]
    values = [-3.0 * (x - 0.47) ** 2 + 2.0 for x in ks]
    vertex, curvature = peaks.parabola_vertex(ks, values)
    assert abs(vertex - 0.47) < 1e-12
    assert abs(curvature + 3.0) < 1e-10


def test_parabola_vertex_rejects_a_minimum():
    with pytest.raises(AssertionError):
        peaks.parabola_vertex([0.4, 0.45, 0.5], [0.0, 1.0, 2.0])


def test_specific_heat_reproduces_an_analytic_curvature():
    # g(K) = a + b K^2  =>  c_V = K^2 * 2b exactly.
    b = 1.7
    ks = [0.40 + 0.0025 * index for index in range(9)]
    gs = [0.5 + b * k * k for k in ks]
    points = peaks.specific_heat(ks, gs, 0.0025)
    assert len(points) == 7
    for k, cv in points:
        assert abs(cv - k * k * 2.0 * b) < 1e-10


def test_specific_heat_needs_both_neighbours():
    ks = [0.4, 0.4025, 0.405, 0.4075]
    gs = [k * k for k in ks]
    points = peaks.specific_heat(ks, gs, 0.0025)
    assert [k for k, _ in points] == [0.4025, 0.405]


def test_peak_from_points_uses_the_three_point_window():
    ks = [0.45, 0.46, 0.47, 0.48, 0.49]
    values = [1.0, 1.5, 1.8, 1.6, 1.1]
    peak = peaks.peak_from_points(list(zip(ks, values)))
    assert peak['grid_k'] == 0.47
    assert peak['grid_cv'] == 1.8
    assert [entry[0] for entry in peak['window']] == [0.46, 0.47, 0.48]
    expected, _ = peaks.parabola_vertex([0.46, 0.47, 0.48], [1.5, 1.8, 1.6])
    assert abs(peak['k_peak'] - expected) < 1e-12
    assert abs(peak['t_peak'] - 1.0 / expected) < 1e-12


def test_peak_on_the_edge_is_rejected():
    with pytest.raises(ValueError):
        peaks.peak_from_points([(0.4, 1.0), (0.4025, 2.0), (0.405, 3.0)])


def test_tpeak_fit_recovers_a_linear_intercept():
    sizes = [16, 32, 64]
    t_c, slope = 2.2692, 0.31
    t_peaks = [t_c + slope / size for size in sizes]
    fit = peaks.tpeak_fit(sizes, t_peaks)
    assert abs(fit['t_c'] - t_c) < 1e-12
    assert abs(fit['slope'] - slope) < 1e-12
    assert max(abs(value) for value in fit['residuals']) < 1e-12


def test_onsager_constants_are_consistent():
    assert abs(peaks.ONSAGER_T - 2.269185314213022) < 1e-12
    assert abs(peaks.ONSAGER_K - 0.44068679350977147) < 1e-12
    assert abs(1.0 / peaks.ONSAGER_K - peaks.ONSAGER_T) < 1e-12


def test_storage_entries_counts_both_factors():
    stored, full = peaks.storage_entries(512, 256)
    assert stored == full == 262144
    stored, full = peaks.storage_entries(512, 16)
    assert stored == 2 * 512 * 16
    assert stored / full == 0.0625


def test_relative_difference_is_symmetric_in_name_only():
    assert peaks.relative_difference(2.0, 2.02) == pytest.approx(0.00990099, rel=1e-6)
    assert peaks.relative_difference(2.02, 2.0) == pytest.approx(0.00990099, rel=1e-6)


def test_specific_heat_matches_a_numerical_second_derivative():
    # g(K) = log(2 cosh K) is smooth; c_V = K^2 g''.
    def g(k: float) -> float:
        return math.log(2.0 * math.cosh(k))

    ks = [0.4 + 0.00125 * index for index in range(9)]
    gs = [g(k) for k in ks]
    points = peaks.specific_heat(ks, gs, 0.0025)
    for k, cv in points:
        analytic = k * k * (1.0 - math.tanh(k) ** 2)
        assert abs(cv - analytic) < 1e-6
