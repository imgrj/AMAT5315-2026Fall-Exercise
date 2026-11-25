"""Draw `artifacts/gates.png` from `runs/gadgets/run.json`.

Panel (a): the energy of every truth-table row of every gadget, with the rows the
gate allows marked and one line per gate floor. Panel (b): all sixteen states of
the joined AND-NOT-SET1 circuit, with the three states on its floor marked.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs' / 'gadgets' / 'run.json'
OUT = ROOT / 'artifacts' / 'gates.png'
NAMES = ['AND', 'OR', 'NOT', 'COPY', 'SET0', 'SET1', 'XOR']


def main() -> None:
    record = json.loads(RUN.read_text(encoding='utf-8'))
    gates = record['gates']

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0), gridspec_kw={'width_ratios': [1.3, 1]})
    axis = axes[0]
    for position, name in enumerate(NAMES):
        gate = gates[name]
        energies = [float(row['energy']) for row in gate['rows']]
        offsets = np.linspace(-0.2, 0.2, len(energies))
        for offset, energy, row in zip(offsets, energies, gate['rows']):
            ok = bool(row['allowed'])
            axis.scatter(position + offset, energy, s=38,
                         color='#1d4ed8' if ok else '#d14343',
                         marker='o' if ok else 'x', zorder=3,
                         linewidths=1.1)
        floor = float(gate['floor'])
        axis.hlines(floor, position - 0.36, position + 0.36, color='#0f172a', lw=1.2,
                    zorder=2)
        if name == 'XOR':
            allowed_rows = [''.join(str(bit) for bit in row['bits'])
                            for row in gate['rows'] if row['allowed']]
            axis.text(position + 0.05, -4.95, 'rows ' + ', '.join(allowed_rows),
                      ha='left', va='top', fontsize=6.8, color='#1d4ed8')

    axis.set_xticks(range(len(NAMES)))
    axis.set_xticklabels(NAMES)
    axis.set_ylabel('energy of a truth-table row')
    axis.set_title('(a) Gate energies: allowed rows sit on the floor, every forbidden row is above')
    axis.axhline(0.0, color='#cbd5e1', lw=0.8, zorder=1)
    axis.grid(axis='y', color='#e2e8f0', zorder=0)
    axis.set_ylim(-5.4, 11.5)
    axis.set_xlim(-0.55, 7.75)
    axis.legend(handles=[
        Line2D([], [], color='#1d4ed8', marker='o', ls='', label='allowed row'),
        Line2D([], [], color='#d14343', marker='x', ls='', label='forbidden row'),
        Line2D([], [], color='#0f172a', lw=1.2, label='gate floor'),
    ], frameon=False, fontsize=8, loc='upper left', ncol=3)
    for position, name in enumerate(NAMES):
        axis.text(position, -5.0, f'({len(gates[name]["rows"])} rows)', ha='center',
                  fontsize=7, color='#475569')
    

    circuit = record['circuit']
    rows = circuit['rows']
    labels = [''.join(str(bit) for bit in row['bits']) for row in rows]
    energies = [float(row['energy']) for row in rows]
    floor = float(circuit['floor'])
    axis = axes[1]
    colors = ['#1d4ed8' if energy == floor else '#94a3b8' for energy in energies]
    axis.bar(range(len(rows)), energies, color=colors, width=0.72)
    axis.axhline(floor, color='#0f172a', lw=1.2)
    axis.set_xticks(range(len(rows)))
    axis.set_xticklabels(labels, rotation=90, fontsize=7)
    axis.set_xlabel('state $a\\,b\\,t\\,out$ of AND(a,b,t) + NOT(t,out) + SET1(out)')
    axis.set_ylabel('energy of a state')
    axis.set_title(f'(b) Three gates joined: {circuit["states_at_floor"]} of 16 states at '
                   f'$E_0 = {floor:g}$')
    axis.grid(axis='y', color='#e2e8f0', zorder=0)
    axis.set_ylim(min(energies) - 1.6, max(energies) + 1.4)
    floor_states = [labels[index] for index, energy in enumerate(energies) if energy == floor]
    axis.annotate('floor states ' + ', '.join(floor_states) + '\n'
                  '= inputs (a,b) 00, 01, 10 with NOT(AND(a,b)) = 1',
                  (0.02, 0.98), xycoords='axes fraction', fontsize=7.5, color='#0f172a',
                  va='top')

    fig.suptitle('Week 11 - one magnet per logic rule: every truth table on its gate floor',
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170)
    print(f'wrote {OUT} from {RUN}')


if __name__ == '__main__':
    main()
