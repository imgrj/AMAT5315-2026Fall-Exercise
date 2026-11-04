"""evidence/posteriors.png and evidence/joint.png.

Run from week9/: python scripts/plot_posteriors.py
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _common import CASES, CASE_LABELS, COLORS, NAMES, VARIABLES, read_run, save, style

DISEASES = (1, 3, 4)      # T, L, B
TESTS = (5, 6, 7)         # E, X, D


def posteriors() -> None:
    runs = {case: read_run(case) for case in CASES}
    figure, axes = plt.subplots(1, 2, figsize=(11.6, 4.4), gridspec_kw={'width_ratios': [1.25, 1]})
    for axis, variables, title in (
        (axes[0], DISEASES, 'Diseases'),
        (axes[1], TESTS, 'Shared effect and test variables'),
    ):
        positions = np.arange(len(variables))
        width = 0.26
        for offset, case in enumerate(CASES):
            values = [runs[case]['contraction']['node_marginals'][str(v)][0] for v in variables]
            enumeration = [runs[case]['enumeration']['node_marginals'][str(v)][0] for v in variables]
            axis.bar(positions + (offset - 1) * width, values, width * 0.92,
                     color=COLORS[case], label=CASE_LABELS[case], alpha=0.9,
                     edgecolor='#0b1120')
            axis.scatter(positions + (offset - 1) * width, enumeration, marker='_', s=110,
                         color='#edf3fa', zorder=3, linewidths=1.6)
        axis.set_xticks(positions)
        axis.set_xticklabels([f'{VARIABLES[v]}\n{NAMES[VARIABLES[v]]}' for v in variables], fontsize=8.5)
        axis.set_ylabel('p(state 0 = yes / present / positive)')
        axis.set_title(title, loc='left')
        axis.grid(axis='y', alpha=0.35)
        axis.set_axisbelow(True)
        axis.legend(fontsize=8, loc='upper left')
    worst = max(run['contraction']['max_marginal_error'] for run in runs.values())
    figure.suptitle(f'Bars: tensor contraction. White ticks: independent enumeration. '
                    f'Largest difference {worst:.1e}', color='#dce6f2', fontsize=10)
    save(figure, 'posteriors.png')


def joint() -> None:
    runs = {case: read_run(case) for case in CASES}
    figure, axes = plt.subplots(3, 3, figsize=(10.4, 9.4))
    for column, case in enumerate(CASES):
        run = runs[case]
        values = np.array(run['contraction']['joints']['1,3'])
        rows = np.array(run['contraction']['node_marginals']['1'])
        columns = np.array(run['contraction']['node_marginals']['3'])
        product = np.outer(rows, columns)
        dependence = values - product
        for row, (matrix, title) in enumerate((
            (values, 'joint p(T,L)'),
            (product, 'product of marginals'),
            (dependence, 'dependence = joint - product'),
        )):
            axis = axes[row][column]
            colour_map = 'magma' if row < 2 else 'RdBu_r'
            limit = np.max(np.abs(dependence)) if row == 2 else np.max(matrix)
            image = axis.imshow(matrix, cmap=colour_map, vmin=-limit if row == 2 else 0,
                                vmax=limit)
            for t in range(2):
                for l in range(2):
                    axis.text(l, t, f'{matrix[t, l]:.4f}', ha='center', va='center',
                              fontsize=8.5,
                              color='#080c16' if row == 2 and abs(matrix[t, l]) > 0.6 * limit else '#dce6f2')
            axis.set_xticks([0, 1], ['L=0', 'L=1'])
            axis.set_yticks([0, 1], ['T=0', 'T=1'])
            if row == 0:
                axis.set_title(CASE_LABELS[case], loc='left', fontsize=9.5)
            if column == 0:
                axis.set_ylabel(title, fontsize=9)
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
    # Numerical labels make the dependence explicit in the caption text.
    lines = []
    for case in CASES:
        run = runs[case]
        dependence = np.array(run['contraction']['dependence']['1,3'])
        lines.append(f"{case}: p(T=0,L=0) - p(T=0)p(L=0) = {dependence[0, 0]:+.5f}")
    figure.suptitle('The joint keeps the dependence the product drops\n' + '   |   '.join(lines),
                    color='#dce6f2', fontsize=9.5)
    save(figure, 'joint.png')


if __name__ == '__main__':
    style()
    posteriors()
    joint()
