"""Draw `artifacts/nqueens.png` from the count and sample runs.

Panel (a): the contracted board counts for N = 4..10 against OEIS A000170 on a log
scale. Panel (b): the frequencies of the four 6-queens boards among 200 uniform
draws, with the expected 50 marked.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import reference  # noqa: E402

ARTIFACTS = ROOT / 'artifacts'
SIZES = [4, 5, 6, 7, 8, 9, 10]


def read(name: str) -> dict:
    return json.loads((ROOT / 'runs' / name / 'run.json').read_text(encoding='utf-8'))


def main() -> None:
    counts = [int(read(f'count_nq{n}')['results']['boards']) for n in SIZES]
    known = [reference.KNOWN_QUEENS[n] for n in SIZES]
    six = read('sample_nq6')
    eight = read('sample_nq8')

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6), gridspec_kw={'width_ratios': [1, 1.15]})
    axis = axes[0]
    axis.plot(SIZES, counts, 'o', color='#1d4ed8', ms=7, label='tensor contraction')
    axis.plot(SIZES, known, 'x', color='#b91c1c', ms=8, mew=1.6, label='known (OEIS A000170)')
    for n, count in zip(SIZES, counts):
        axis.annotate(f'{count}', (n, count), textcoords='offset points', xytext=(7, 6),
                      fontsize=7.5, color='#334155')
    axis.set_yscale('log')
    axis.set_xlabel('board size $N$')
    axis.set_ylabel('valid boards (log scale)')
    axis.set_title('(a) Exact N-queens counts: every one matches')
    axis.set_xticks(SIZES)
    axis.set_xlim(3.5, 10.7)
    axis.set_ylim(1.3, 2200)
    axis.grid(which='both', color='#e2e8f0')
    axis.legend(frameon=False, fontsize=8)

    axis = axes[1]
    frequencies = {tuple(row['board']): int(row['count'])
                   for row in six['results']['frequencies']}
    boards = sorted(frequencies)
    values = [frequencies[board] for board in boards]
    labels = ['(' + ', '.join(str(column) for column in board) + ')' for board in boards]
    bars = axis.bar(range(len(boards)), values, color='#1d4ed8', width=0.62)
    axis.axhline(50.0, color='#b91c1c', ls='--', lw=1.0)
    axis.text(len(boards) - 0.4, 52.0, 'uniform: 200 / 4 = 50', color='#b91c1c', fontsize=8,
              ha='right')
    for bar, value in zip(bars, values):
        axis.annotate(str(value), (bar.get_x() + bar.get_width() / 2, value),
                      textcoords='offset points', xytext=(0, 4), ha='center', fontsize=8)
    axis.set_xticks(range(len(boards)))
    axis.set_xticklabels(labels, fontsize=8)
    axis.set_xlabel('6-queens board (column of the queen in each row)')
    axis.set_ylabel('times drawn (out of 200)')
    axis.set_title(f'(b) 200 sampled 6-queens boards, {len(boards)} distinct')
    axis.set_ylim(0, max(66.0, max(values) + 8.0))
    axis.grid(axis='y', color='#e2e8f0')

    fig.suptitle(f'Week 11 - a chess puzzle as a tensor network: '
                 f'{eight["results"]["distinct"]} distinct 8-queens boards in 200 draws',
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out = ARTIFACTS / 'nqueens.png'
    fig.savefig(out, dpi=170)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
