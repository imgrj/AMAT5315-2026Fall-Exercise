"""Draw `artifacts/factoring.png` from the planning and sampling runs.

Panel (a): the planned space complexity `sc` against the factor bit length `k` for
`greedy` and `treesa`, with the 8 GB line and the ~2k+1 seam. Panel (b): the
sampled energies of `factor_15` at beta = 2 beside the Boltzmann prediction
`2000 g_0 exp(-beta E_min) / Z` for the ground state.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / 'artifacts'
BITS = [2, 3, 4, 5, 6, 7, 8]
BETA = 2.0


def read(*parts: str) -> dict:
    return json.loads((ROOT / 'runs' / Path(*parts) / 'run.json').read_text(encoding='utf-8'))


def main() -> None:
    plan = {order: [read(f'plan_k{k}_{order}')['results'] for k in BITS]
            for order in ('greedy', 'treesa')}
    sample = read('sample_f15')
    results = sample['results']

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.7), gridspec_kw={'width_ratios': [1, 1.1]})
    axis = axes[0]
    for order, marker, color in (('greedy', 'o', '#d97706'), ('treesa', 's', '#1d4ed8')):
        values = [row['sc'] for row in plan[order]]
        axis.plot(BITS, values, marker=marker, color=color, label=order, lw=1.6)
        for bit, value in zip(BITS, values):
            if bit in (2, 3, 8):
                axis.annotate(f'{value:g}', (bit, value), textcoords='offset points',
                              xytext=(4, 5), fontsize=7.5, color=color)
    axis.plot(BITS, [2 * k + 1 for k in BITS], ls=':', color='#1d4ed8', lw=1.0,
              label=r'$sc = 2k+1$ (the seam)')
    axis.axhline(30.0, color='#b91c1c', ls='--', lw=1.0)
    axis.text(2.05, 30.6, 'sc = 30 ($2^{30}$ entries, about 8 GB)', color='#b91c1c', fontsize=7.8)
    axis.set_xlabel('bits per factor $k$')
    axis.set_ylabel(r'space complexity $sc$ ($\log_2$ entries)')
    axis.set_title('(a) Contraction memory cost of the multiplier')
    axis.grid(color='#e2e8f0')
    axis.legend(frameon=False, fontsize=8, loc='upper left')
    axis.set_xticks(BITS)
    axis.set_ylim(0, 60)

    axis = axes[1]
    histogram = sorted(results['histogram'], key=lambda row: row['energy'])
    energies = [float(row['energy']) for row in histogram]
    counts = [int(row['count']) for row in histogram]
    e_min = float(results['e_min'])
    predicted = float(results['predicted_ground_draws'])
    axis.bar(energies, counts, width=1.4, color='#64748b', label='2000 sampled states')
    axis.bar([e_min], [predicted], width=1.4, facecolor='none', edgecolor='#1d4ed8',
             lw=1.8, label='predicted $2000 g_0 e^{-\\beta E_{min}}/Z$')
    axis.annotate(f'observed {int(results["ground_draws"])}',
                  (e_min, counts[0]), textcoords='offset points', xytext=(10, 2),
                  ha='left', fontsize=8.5, color='#0f172a')
    axis.annotate(f'predicted {predicted:.0f}', (e_min, predicted),
                  textcoords='offset points', xytext=(10, 8), ha='left', fontsize=8,
                  color='#1d4ed8')
    axis.text(0.035, 0.52, 'ground state\n(decodes to 3 x 5)', transform=axis.transAxes,
              fontsize=8, color='#0f172a', va='center')
    for energy, count in list(zip(energies, counts))[1:6]:
        axis.annotate(str(count), (energy, count), textcoords='offset points',
                      xytext=(0, 4), ha='center', fontsize=7, color='#334155')
    axis.set_xlabel('spin-glass energy $E$ at $\\beta = 2$')
    axis.set_ylabel('samples (out of 2000)')
    axis.set_title(f'(b) $N = 15$: $E_{{min}} = {e_min:g}$, $g_0 = {results["g0"]}$, '
                   f'$\\log Z = {float(results["log_z"]):.4f}$')
    axis.grid(axis='y', color='#e2e8f0')
    axis.set_ylim(0, max(counts) * 1.28)
    axis.legend(frameon=False, fontsize=8, loc='upper center')

    fig.suptitle('Week 11 - factoring 15 through a 104-spin magnet: memory grows, the ground state dominates',
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out = ARTIFACTS / 'factoring.png'
    fig.savefig(out, dpi=170)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
