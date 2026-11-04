"""evidence/graphs.svg: the AD tuberculosis query as a picture of the factors.

Run from week9/: python scripts/plot_graphs.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

from _common import ROOT, VARIABLES, save, style

FACTORS = [
    ('p(a)', ('A',)),
    ('p(t|a)', ('T', 'A')),
    ('p(s)', ('S',)),
    ('p(l|s)', ('L', 'S')),
    ('p(b|s)', ('B', 'S')),
    ('p(e|t,l)', ('E', 'T', 'L')),
    ('p(x|e)', ('X', 'E')),
    ('p(d|e,b)', ('D', 'E', 'B')),
]
OBSERVED = {'A', 'D'}
QUERY = 'T'


def main() -> None:
    style()
    figure, axis = plt.subplots(figsize=(9.2, 5.4))
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 6.2)
    axis.axis('off')

    variable_x = {name: 0.9 + index * 1.14 for index, name in enumerate(VARIABLES)}
    variable_y = 4.9
    for name, x in variable_x.items():
        observed = name in OBSERVED
        colour = '#f6a45c' if observed else ('#38bdf8' if name == QUERY else '#9aaec5')
        axis.add_patch(Circle((x, variable_y), 0.24, facecolor='#16243b',
                              edgecolor=colour, linewidth=2.4 if name == QUERY else 1.3))
        axis.text(x, variable_y, name, ha='center', va='center', fontsize=11,
                  color=colour, fontweight='bold')
        if observed:
            axis.add_patch(FancyBboxPatch((x - 0.33, variable_y - 1.05), 0.66, 0.42,
                                          boxstyle='round,pad=0.03',
                                          facecolor='#2a2118', edgecolor='#f6a45c', linewidth=1.2))
            axis.text(x, variable_y - 0.84, '(1,0)', ha='center', va='center',
                      fontsize=8.5, color='#f6a45c')
            axis.plot([x, x], [variable_y - 0.24, variable_y - 0.63], color='#f6a45c', linewidth=1.4)
    axis.annotate('open output: p(T)', xy=(variable_x[QUERY], variable_y + 0.26),
                  xytext=(variable_x[QUERY] - 0.4, 5.95), fontsize=9, color='#38bdf8',
                  arrowprops={'arrowstyle': '-|>', 'color': '#38bdf8', 'lw': 1.4})

    factor_y = 2.5
    factor_x = {label: 0.95 + index * 1.16 for index, (label, _) in enumerate(FACTORS)}
    for label, scope in FACTORS:
        x = factor_x[label]
        axis.add_patch(FancyBboxPatch((x - 0.5, factor_y - 0.31), 1.0, 0.62,
                                      boxstyle='round,pad=0.04',
                                      facecolor='#0f172a', edgecolor='#4b6390', linewidth=1.2))
        axis.text(x, factor_y + 0.09, label, ha='center', va='center', fontsize=9.5, color='#dce6f2')
        axis.text(x, factor_y - 0.17, 'axes ' + ','.join(scope), ha='center', va='center',
                  fontsize=7.2, color='#7f93ad')
        for name in scope:
            colour = '#f6a45c' if name in OBSERVED else ('#38bdf8' if name == QUERY else '#4b6390')
            axis.plot([x, variable_x[name]], [factor_y + 0.31, variable_y - 0.24],
                      color=colour, linewidth=1.5 if colour != '#4b6390' else 1.0,
                      zorder=1)
    axis.text(0.2, 1.55, 'Shared labels join: S appears in p(s), p(l|s) and p(b|s); '
                         'T joins p(t|a) and p(e|t,l).', fontsize=9, color='#9aaec5')
    axis.text(0.2, 1.2, 'Observed A and D carry (1,0) caps; summing the other labels '
                        'leaves the tuberculosis marginal p(T).', fontsize=9, color='#9aaec5')
    axis.text(0.2, 0.85, 'Only T stays open, so the same picture answers the closed '
                         'evidence query by leaving no label open.', fontsize=9, color='#9aaec5')
    axis.set_title('ASIA AD query: T remains open while every other label is summed',
                   loc='left', color='#dce6f2')
    figure.tight_layout()
    path = ROOT / 'evidence' / 'graphs.svg'
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, format='svg', transparent=False)
    plt.close(figure)
    print('wrote evidence/graphs.svg')


if __name__ == '__main__':
    main()
