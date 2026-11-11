"""Specific-heat peak figures for Part 2 (open) and Part 3 (periodic).

Run from week10/:
    python scripts/plot_critical.py --method boundary   ->   artifacts/boundary-critical.png
    python scripts/plot_critical.py --method trg        ->   artifacts/trg-critical.png
"""
from __future__ import annotations

import argparse

import numpy as np

import _common as common
import peaks

import matplotlib.pyplot as plt

SIZES = {'boundary': [16, 32, 64], 'trg': [64, 256, 1024]}
LABEL = {'boundary': 'open', 'trg': 'periodic'}
TITLES = {
    'boundary': ('(a) Open-square specific-heat peaks', '(b) Extrapolate the peak to $L \\to \\infty$'),
    'trg': ('(a) Periodic specific-heat peaks', '(b) Periodic peak extrapolation'),
}
SMALL_CHECK = {
    'boundary': '4x4 open, $\\chi$ = 8 vs enumeration: |dg| < 1e-12, $c_V$ within 1e-3',
    'trg': '2x2 periodic, $\\chi$ = 16 exact; 4x4 at $\\chi$ = 8 within 2e-3',
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['boundary', 'trg'], required=True)
    arguments = parser.parse_args()
    method = arguments.method
    common.style()
    scan = common.read_json(common.TABLES / f'{method}-scan.json')
    refine = common.read_json(common.TABLES / f'{method}-refine.json')
    primary = refine['variants']['chi32_h0.00125']
    sizes = SIZES[method]

    figure = plt.figure(figsize=(15.0, 4.7))
    grid = figure.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 0.95], wspace=0.28)
    heat = figure.add_subplot(grid[0, 0])
    drift = figure.add_subplot(grid[0, 1])
    table = figure.add_subplot(grid[0, 2])
    table.axis('off')

    for length in sizes:
        colour = common.COLORS[f'L{length}']
        points = np.array(scan['specific_heat'][str(length)])
        heat.plot(1.0 / points[:, 0], points[:, 1], color=colour, linewidth=1.5,
                  label=f'L = {length}, $\\chi$ = 16, h = 0.0025')
        refined = np.array(primary['sizes'][str(length)]['specific_heat'])
        heat.plot(1.0 / refined[:, 0], refined[:, 1], '.', color=colour, markersize=3.5,
                  alpha=0.8)
        marker = primary['sizes'][str(length)]['peak']
        heat.plot([marker['t_peak']], [marker['grid_cv']], '*', color=colour, markersize=14,
                  markeredgecolor='#0b1120', zorder=5)
    heat.axvline(peaks.ONSAGER_T, color=common.COLORS['onsager'], linestyle='--', linewidth=1.1,
                 label='Onsager $T_c$ = 2.269185')
    heat.set_xlabel('temperature $T$')
    heat.set_ylabel('specific heat per spin $c_V$')
    heat.set_title(TITLES[method][0])
    heat.grid(alpha=0.35)
    heat.legend(loc='upper left', fontsize=7.8, framealpha=0.9)

    t_peaks = np.array([primary['sizes'][str(length)]['peak']['t_peak'] for length in sizes])
    inverses = np.array([1.0 / length for length in sizes])
    smallest = inverses.min()
    for length, t_peak in zip(sizes, t_peaks):
        drift.plot([1.0 / length], [t_peak], 'o', color=common.COLORS[f'L{length}'], markersize=8)
        near_edge = 1.0 / length <= 1.4 * smallest
        drift.annotate(f'L = {length}', xy=(1.0 / length, t_peak),
                       xytext=(-8 if near_edge else 6, -10), textcoords='offset points',
                       ha='right' if near_edge else 'left', fontsize=8.5,
                       color=common.COLORS[f'L{length}'])
    for key, label, style in (('all_sizes', 'all sizes', '-'),
                              ('largest_two', 'largest two', '--')):
        fit = primary['fit'][key]
        drift.plot(inverses, fit['t_c'] + fit['slope'] * inverses, style, color='#dce6f2',
                   linewidth=1.1, label=f"{label}: $T_c$ = {fit['t_c']:.4f}")
    drift.axhline(peaks.ONSAGER_T, color=common.COLORS['onsager'], linestyle=':', linewidth=1.2,
                  label='Onsager 2.2692')
    drift.set_xlabel('inverse side length $1/L$')
    drift.set_ylabel('peak temperature $T_{\\mathrm{peak}}$')
    drift.set_title(TITLES[method][1])
    drift.grid(alpha=0.35)
    drift.legend(loc='lower right', fontsize=8.0, framealpha=0.9)
    span = inverses.max()
    drift.set_xlim(-0.06 * span, 1.12 * span)

    lines = [
        f'{LABEL[method]} squares, refined peaks',
        '$\\chi$ = 32, h = 0.00125',
        '',
        ' L      K_peak    T_peak    c_V peak',
    ]
    for length in sizes:
        marker = primary['sizes'][str(length)]['peak']
        lines.append(f' {length:<5d}  {marker["k_peak"]:.5f}  {marker["t_peak"]:.5f}  '
                     f'{marker["grid_cv"]:.4f}')
    lines += [
        '',
        f"fit all        T_c = {primary['fit']['all_sizes']['t_c']:.5f}",
        f"fit largest 2  T_c = {primary['fit']['largest_two']['t_c']:.5f}",
        f"Onsager        T_c = {peaks.ONSAGER_T:.5f}",
        f"agreement          {100 * primary['fit']['onsager_relative_error_all']:.3f}% "
        f"(largest two {100 * primary['fit']['onsager_relative_error_largest_two']:.3f}%)",
        '',
        'rank and step stability',
        f"  $\\chi$ 16 -> 32:  {100 * refine['comparison']['t_c_rank_shift']:.3f}%",
        f"  h 0.0025 -> 0.00125: {100 * refine['comparison']['t_c_step_shift']:.3f}%",
        '',
        SMALL_CHECK[method],
    ]
    table.text(0.0, 1.0, '\n'.join(lines), transform=table.transAxes, va='top', ha='left',
               fontsize=7.8, family='monospace', color='#dce6f2')

    common.save(figure, common.ARTIFACTS / f'{method}-critical.png')
    print('  refined peaks:',
          {length: round(primary['sizes'][str(length)]['peak']['t_peak'], 6) for length in sizes})


if __name__ == '__main__':
    main()
