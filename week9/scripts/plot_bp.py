"""evidence/bp-convergence.png, bp-accuracy.png and bp-comparison.png.

Run from week9/: python scripts/plot_bp.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _common import CASES, CASE_LABELS, COLORS, read_run, save, style

VARIABLES = [('Tuberculosis', 1), ('Lung cancer', 3), ('Bronchitis', 4), ('Smoking', 2)]


def comparison(runs: dict) -> None:
    """The lecture slide's view: final probabilities beside convergence."""
    exact = read_run('ADX')['contraction']['node_marginals']
    loopy = runs['ADX']['final']['beliefs']
    figure, (left, right) = plt.subplots(1, 2, figsize=(11.2, 4.6),
                                         gridspec_kw={'width_ratios': [1.15, 1]})

    positions = np.arange(len(VARIABLES))
    height = 0.36
    exact_values = [exact[str(variable)][0] for _, variable in VARIABLES]
    loopy_values = [loopy[str(variable)][0] for _, variable in VARIABLES]
    left.barh(positions - height / 2, exact_values, height, color='#38bdf8', label='Exact')
    left.barh(positions + height / 2, loopy_values, height, color='#f7768e', label='Loopy BP')
    for index, (value, other) in enumerate(zip(exact_values, loopy_values)):
        left.text(value + 0.012, index - height / 2, f'{value:.3f}', va='center', fontsize=8,
                  color='#dce6f2')
        left.text(other + 0.012, index + height / 2, f'{other:.3f}', va='center', fontsize=8,
                  color='#dce6f2')
    left.set_yticks(positions, [name for name, _ in VARIABLES])
    left.invert_yaxis()
    left.set_xlim(0, 1.0)
    left.set_xlabel('probability of being present')
    left.set_title('(a) Final probabilities', loc='left', fontsize=10)
    left.legend(fontsize=8, loc='lower right')
    left.grid(axis='x', alpha=0.3)
    left.set_axisbelow(True)

    run = runs['ADX']
    sweeps = [record['sweep'] for record in run['sweeps']]
    residuals = [record['residual'] for record in run['sweeps']]
    errors = [record['max_node_error'] for record in run['sweeps']]
    right.plot(sweeps, residuals, color='#38bdf8', linewidth=1.8, label='message residual')
    right.plot(sweeps, errors, color='#f7768e', linewidth=1.8, linestyle='--',
               label='largest probability error')
    right.set_xlabel('sweeps')
    right.set_ylabel('absolute difference')
    right.set_ylim(0, max(errors) * 1.35)
    right.set_title('(b) Convergence', loc='left', fontsize=10)
    right.grid(alpha=0.3)
    right.legend(fontsize=8, loc='upper right')
    right.annotate(f'message residual → {run["final"]["residual"]:.2e}',
                   xy=(sweeps[-1] - 1, residuals[-1]), xytext=(sweeps[-1] * 0.40, 0.115),
                   fontsize=8, color='#38bdf8',
                   arrowprops={'arrowstyle': '-|>', 'color': '#38bdf8', 'lw': 1.0})
    right.annotate(f'largest probability error stays at {run["bp"]["max_node_error"]:.3f}',
                   xy=(sweeps[-22], run['bp']['max_node_error']),
                   xytext=(sweeps[-1] * 0.34, run['bp']['max_node_error'] * 3.4),
                   fontsize=8, color='#f7768e',
                   arrowprops={'arrowstyle': '-|>', 'color': '#f7768e', 'lw': 1.0})

    figure.suptitle('ASIA, visit + dyspnoea + positive X-ray: loopy BP is close, not exact\n'
                    f'residual below 1e-10 after {run["final"]["sweeps"]} sweeps; '
                    f'the largest marginal error is {run["bp"]["max_node_error"]:.5f} and does not fall with it',
                    color='#dce6f2', fontsize=9.5)
    save(figure, 'bp-comparison.png')


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

    comparison(runs)


if __name__ == '__main__':
    main()
