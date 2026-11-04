"""evidence/bp-convergence.png and evidence/bp-accuracy.png.

Run from week9/: python scripts/plot_bp.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt

from _common import CASES, CASE_LABELS, COLORS, read_run, save, style


def main() -> None:
    style()
    runs = {case: read_run(f'bp-{case}') for case in CASES}
    tolerance = runs['prior']['settings']['tolerance']

    figure, axis = plt.subplots(figsize=(7.6, 4.6))
    for case, run in runs.items():
        sweeps = [record['sweep'] for record in run['sweeps']]
        residuals = [record['residual'] for record in run['sweeps']]
        axis.semilogy(sweeps, residuals, color=COLORS[case], linewidth=1.7,
                      label=f'{CASE_LABELS[case]} · {run["final"]["sweeps"]} sweeps')
        axis.scatter([sweeps[-1]], [residuals[-1]], color=COLORS[case], s=26, zorder=3)
    axis.axhline(tolerance, color='#9aaec5', linestyle='--', linewidth=1.0)
    axis.text(2, tolerance * 1.25, 'stop tolerance 1e-10', color='#9aaec5', fontsize=8)
    axis.set_xlabel('synchronous damped sweeps')
    axis.set_ylabel('max |F(m) - m|  (undamped residual)')
    axis.set_title('The messages stop changing', loc='left')
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    save(figure, 'bp-convergence.png')

    figure, axis = plt.subplots(figsize=(7.6, 4.6))
    for case, run in runs.items():
        sweeps = [record['sweep'] for record in run['sweeps']]
        errors = [max(record['max_node_error'], 1e-16) for record in run['sweeps']]
        axis.loglog(sweeps, errors, color=COLORS[case], linewidth=1.7,
                    label=f'{case} · final {run["bp"]["max_node_error"]:.5f}')
    axis.set_xlabel('sweeps')
    axis.set_ylabel('max |belief - exact marginal|')
    axis.set_title('Convergence is not accuracy: the loopy beliefs settle away from exact',
                   loc='left')
    axis.grid(alpha=0.3, which='both')
    axis.legend(fontsize=8)
    figure.text(0.01, 0.005,
                f'ADX stops at residual {runs["ADX"]["final"]["residual"]:.2e} after '
                f'{runs["ADX"]["final"]["sweeps"]} sweeps but stays '
                f'{runs["ADX"]["bp"]["max_node_error"]:.5f} away from the exact marginal.',
                fontsize=8, color='#9aaec5')
    save(figure, 'bp-accuracy.png')


if __name__ == '__main__':
    main()
