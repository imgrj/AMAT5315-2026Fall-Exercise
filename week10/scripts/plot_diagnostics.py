"""Diagnostic figures: one stated case per method, and both geometries together.

Run from week10/: python scripts/plot_diagnostics.py
Writes artifacts/boundary-rows.png, artifacts/trg-steps.png,
artifacts/g-curves.png and artifacts/phase-compare.png.
"""
from __future__ import annotations

import numpy as np

import _common as common
import peaks

import matplotlib.pyplot as plt


def boundary_rows() -> None:
    record = common.read_json(common.RUNS / 'boundary64' / 'run.json')
    rows = record['rows']
    index = np.arange(1, len(rows) + 1)
    scales = np.array([row['scale'] for row in rows])
    ell = np.array([row['ell'] for row in rows])
    discarded = np.array([row['relative_discarded_weight'] for row in rows])
    bonds_before = np.array([max(row['bond_dims_before']) for row in rows])
    bonds_after = np.array([max(row['bond_dims_after']) for row in rows])

    figure, axes = plt.subplots(1, 3, figsize=(14.2, 3.9))
    axes[0].semilogy(index, scales, color=common.COLORS['measured'], linewidth=1.1,
                     label='row scale $c_y$')
    axes[0].semilogy(index, ell, color=common.COLORS['prediction'], linewidth=1.1,
                     label='accumulated $\\ell$')
    axes[0].set_xlabel('absorbed row $y$')
    axes[0].set_ylabel('scale')
    axes[0].set_title('(a) One scale per row, restored at the end')
    axes[0].legend(fontsize=8.5)
    axes[0].grid(alpha=0.35, which='both')

    axes[1].semilogy(index, np.maximum(discarded, 1e-18), color=common.COLORS['onsager'],
                     linewidth=1.1)
    axes[1].set_xlabel('absorbed row $y$')
    axes[1].set_ylabel('relative discarded weight')
    axes[1].set_title('(b) Truncation loss per row (at most $\\chi$ directions)')
    axes[1].grid(alpha=0.35, which='both')

    axes[2].step(index, bonds_before, where='mid', color=common.COLORS['threshold'],
                 linewidth=1.1, label='after absorption (up to $2\\chi$)')
    axes[2].step(index, bonds_after, where='mid', color=common.COLORS['measured'],
                 linewidth=1.1, label='after compression (at most $\\chi$)')
    axes[2].axhline(record['settings']['chi'], color='#9aaec5', linestyle=':', linewidth=1.0)
    axes[2].set_xlabel('absorbed row $y$')
    axes[2].set_ylabel('widest horizontal bond')
    axes[2].set_title('(c) Bonds grow to $2\\chi$ and come back')
    axes[2].legend(fontsize=8.5)
    axes[2].grid(alpha=0.35)
    common.save(figure, common.ARTIFACTS / 'boundary-rows.png')


def trg_steps() -> None:
    record = common.read_json(common.RUNS / 'trg1024' / 'run.json')
    steps = record['steps']
    index = np.arange(1, len(steps) + 1)
    scales = np.array([step['max_abs'] for step in steps])
    ell = np.array([step['ell'] for step in steps])
    bond = np.array([max(step['bond_dims']) for step in steps])

    figure, axes = plt.subplots(1, 3, figsize=(14.2, 3.9))
    axes[0].semilogy(index, scales, 'o-', color=common.COLORS['measured'], markersize=3.5,
                     linewidth=1.0, label='max $|A|$ of the round')
    axes[0].plot(index, ell, color=common.COLORS['prediction'], linewidth=1.2,
                 label='accumulated $\\ell$')
    axes[0].set_xlabel('TRG round')
    axes[0].set_ylabel('scale')
    axes[0].set_title('(a) 20 rounds of max-absolute normalisation')
    axes[0].legend(fontsize=8.5)
    axes[0].grid(alpha=0.35, which='both')

    for step, colour in zip(steps, plt.cm.viridis(np.linspace(0.15, 0.9, len(steps)))):
        spectrum = np.array(step['split1_singular_values'])
        axes[1].semilogy(np.arange(1, len(spectrum) + 1), spectrum, 'o-', markersize=2.2,
                         linewidth=0.8, color=colour, alpha=0.85)
    axes[1].set_xlabel('retained direction')
    axes[1].set_ylabel('split singular value')
    axes[1].set_title('(b) First split spectrum, one line per round')
    axes[1].grid(alpha=0.35, which='both')

    axes[2].plot(index, bond, 'o-', color=common.COLORS['L1024'], markersize=4, linewidth=1.1)
    axes[2].axhline(record['settings']['chi'], color='#9aaec5', linestyle=':', linewidth=1.0,
                    label='$\\chi$ = 16')
    axes[2].set_xlabel('TRG round')
    axes[2].set_ylabel('bond dimension')
    axes[2].set_title('(c) Coarse legs stop growing at $\\chi$')
    axes[2].legend(fontsize=8.5)
    axes[2].grid(alpha=0.35)
    common.save(figure, common.ARTIFACTS / 'trg-steps.png')


def g_curves() -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.4, 4.0), sharey=True)
    for axis, method, title in ((axes[0], 'boundary', 'open squares'),
                                (axes[1], 'trg', 'periodic squares')):
        scan = common.read_json(common.TABLES / f'{method}-scan.json')
        for length in scan['sizes']:
            ks = np.array(scan['ks'])
            values = np.array(scan['g'][str(length)])
            axis.plot(1.0 / ks, values, color=common.COLORS[f'L{length}'], linewidth=1.4,
                      label=f'L = {length}')
        axis.axvline(peaks.ONSAGER_T, color=common.COLORS['onsager'], linestyle='--',
                     linewidth=1.0)
        axis.set_xlabel('temperature $T$')
        axis.set_ylabel('$g = \\log Z / L^2$')
        axis.set_title(f'$g$ for {title}, $\\chi$ = 16')
        axis.grid(alpha=0.35)
        axis.legend(fontsize=8.5)
    axes[0].set_xlim(1.95, 2.55)
    common.save(figure, common.ARTIFACTS / 'g-curves.png')


def phase_compare() -> None:
    figure, axis = plt.subplots(figsize=(7.4, 4.2))
    for method, sizes, style in (('boundary', [16, 32, 64], '-'),
                                 ('trg', [64, 256, 1024], '--')):
        scan = common.read_json(common.TABLES / f'{method}-scan.json')
        for length in sizes:
            points = np.array(scan['specific_heat'][str(length)])
            axis.plot(1.0 / points[:, 0], points[:, 1], style, color=common.COLORS[f'L{length}'],
                      linewidth=1.4, label=f'{scan["boundary"]} L = {length}')
    axis.axvline(peaks.ONSAGER_T, color=common.COLORS['onsager'], linestyle=':', linewidth=1.2,
                 label='Onsager $T_c$ = 2.269185')
    axis.set_xlabel('temperature $T$')
    axis.set_ylabel('specific heat per spin $c_V$')
    axis.set_title('Finite-size peaks approach $T_c$ from either side')
    axis.set_xlim(2.04, 2.40)
    axis.set_ylim(0.0, 3.2)
    axis.grid(alpha=0.35)
    axis.legend(fontsize=8.0, ncol=2)
    common.save(figure, common.ARTIFACTS / 'phase-compare.png')


def main() -> None:
    common.style()
    boundary_rows()
    trg_steps()
    g_curves()
    phase_compare()


if __name__ == '__main__':
    main()
