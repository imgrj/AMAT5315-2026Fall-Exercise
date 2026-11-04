"""evidence/promedus-orders.png and evidence/promedus-bp.png.

Run from week9/: python scripts/plot_promedus.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _common import COLORS, read_run, save, style

QUERIES = ('59', '102', '139')
REFERENCE = {'greedy': (667_111_481, 4_194_304), 'treesa': (10_635_220, 131_072)}


def main() -> None:
    style()
    greedy = read_run('promedus/greedy')
    treesa = read_run('promedus/treesa')
    bp = read_run('promedus/bp')

    figure, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
    methods = ('greedy', 'treesa')
    work = [greedy['ordering']['work'], treesa['ordering']['work']]
    entries = [greedy['ordering']['largest_entries'], treesa['ordering']['largest_entries']]
    positions = np.arange(len(methods))
    axes[0].bar(positions, work, color=[COLORS['greedy'], COLORS['treesa']], width=0.55)
    axes[0].set_yscale('log')
    axes[0].set_xticks(positions, ['omeco greedy', 'TreeSA (time only)'])
    axes[0].set_ylabel('work: terms in the pair sums')
    axes[0].set_title('Exact contraction work', loc='left')
    for index, value in enumerate(work):
        axes[0].text(index, value * 1.15, f'{value:,.0f}', ha='center', fontsize=8.5)
    axes[1].bar(positions, entries, color=[COLORS['greedy'], COLORS['treesa']], width=0.55)
    axes[1].set_yscale('log')
    axes[1].set_xticks(positions, ['omeco greedy', 'TreeSA (time only)'])
    axes[1].set_ylabel('largest tensor: entries')
    axes[1].set_title('Peak intermediate', loc='left')
    for index, value in enumerate(entries):
        axes[1].text(index, value * 1.15, f'{value:,}', ha='center', fontsize=8.5)
    ratio = work[0] / work[1]
    figure.suptitle(f'Same evidence, two exact plans: TreeSA uses {ratio:.0f}x less work '
                    f'({work[0]:,} → {work[1]:,} terms)', color='#dce6f2', fontsize=10)
    figure.text(0.01, 0.005,
                f'log10 Z = {treesa["contraction"]["log10_mass"]:.15f} for both plans; '
                f'published PR {treesa["reference"]["pr"]["published"]} '
                f'(|diff| {treesa["reference"]["pr"]["difference"]:.1e}). '
                f'The sheet reports greedy 667,111,481 / 4,194,304 and TreeSA 10,635,220 / 131,072 '
                f'with its own searcher; our recount is what is plotted.',
                fontsize=7.6, color='#9aaec5')
    save(figure, 'promedus-orders.png')

    figure, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
    sweeps = [record['sweep'] for record in bp['sweeps']]
    residuals = [record['residual'] for record in bp['sweeps']]
    axes[0].semilogy(sweeps, residuals, color='#f7768e', linewidth=1.3)
    axes[0].axhline(bp['settings']['tolerance'], color='#9aaec5', linestyle='--', linewidth=0.9)
    axes[0].text(4, bp['settings']['tolerance'] * 1.4, 'stop tolerance 1e-10',
                 color='#9aaec5', fontsize=8)
    axes[0].set_xlabel('sweeps')
    axes[0].set_ylabel('undamped residual')
    axes[0].set_title(f'BP does not converge in {bp["final"]["sweeps"]} sweeps', loc='left')
    axes[0].grid(alpha=0.3)
    colours = {'59': '#38bdf8', '102': '#f6a45c', '139': '#4ade80'}
    for variable in QUERIES:
        errors = [record['max_node_error'] for record in bp['sweeps']]
        if len(QUERIES) == 1:
            values = errors
        else:
            values = [abs(record['beliefs'][variable][1] - bp['exact']['node_marginals'][variable][1])
                      for record in bp['sweeps']]
        axes[1].semilogy(sweeps, np.maximum(values, 1e-16), color=colours[variable],
                         linewidth=1.3, label=f'ID {variable}: {bp["bp"]["node_errors"][variable]:.4f}')
    axes[1].set_xlabel('sweeps')
    axes[1].set_ylabel('|state-1 belief - exact marginal|')
    axes[1].set_title('Selected query error stays large', loc='left')
    axes[1].grid(alpha=0.3, which='both')
    axes[1].legend(fontsize=8)
    figure.suptitle(f'Promedus_14 with the ASIA BP settings: residual '
                    f'{bp["final"]["residual"]:.2e} at {bp["final"]["sweeps"]} sweeps, '
                    f'unconverged and inaccurate', color='#dce6f2', fontsize=9.5)
    save(figure, 'promedus-bp.png')


if __name__ == '__main__':
    main()
