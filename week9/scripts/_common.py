"""Shared helpers for the Week 9 evidence plots."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'evidence'
ARTIFACTS = ROOT / 'artifacts'

VARIABLES = 'ATSLBEXD'
NAMES = {
    'A': 'visit Asia',
    'T': 'tuberculosis',
    'S': 'smoking',
    'L': 'lung cancer',
    'B': 'bronchitis',
    'E': 'TB or cancer',
    'X': 'positive X-ray',
    'D': 'dyspnoea',
}
CASES = ('prior', 'AD', 'ADX')
CASE_LABELS = {
    'prior': 'prior (no observations)',
    'AD': 'AD: visit + dyspnoea',
    'ADX': 'ADX: plus positive X-ray',
}
COLORS = {
    'prior': '#7aa2f7',
    'AD': '#f6a45c',
    'ADX': '#e06c75',
    'greedy': '#4ade80',
    'random': '#f6a45c',
    'exhaustive': '#38bdf8',
    'treesa': '#4ade80',
    'exact': '#38bdf8',
    'bp': '#f7768e',
}
FACTOR_LABEL = {
    (0,): 'p(a)',
    (1, 0): 'p(t|a)',
    (2,): 'p(s)',
    (3, 2): 'p(l|s)',
    (4, 2): 'p(b|s)',
    (5, 3, 1): 'p(e|t,l)',
    (6, 5): 'p(x|e)',
    (7, 5, 4): 'p(d|e,b)',
}


def style() -> None:
    plt.rcParams.update({
        'figure.dpi': 130,
        'savefig.dpi': 150,
        'font.size': 9.5,
        'axes.titlesize': 10.5,
        'axes.labelsize': 9.5,
        'axes.facecolor': '#0f172a',
        'figure.facecolor': '#080c16',
        'savefig.facecolor': '#080c16',
        'axes.edgecolor': '#3b4a63',
        'axes.labelcolor': '#dce6f2',
        'text.color': '#dce6f2',
        'xtick.color': '#9aaec5',
        'ytick.color': '#9aaec5',
        'grid.color': '#26344c',
        'legend.facecolor': '#0f172a',
        'legend.edgecolor': '#3b4a63',
    })


def read_run(name: str) -> dict:
    return json.loads((ARTIFACTS / name / 'run.json').read_text())


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def save(figure, name: str, tight: bool = True) -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    if tight:
        figure.tight_layout()
    figure.savefig(EVIDENCE / name)
    plt.close(figure)
    print(f'wrote evidence/{name}')


def marginal(run: dict, variable: int) -> list[float]:
    return run['contraction']['node_marginals'][str(variable)]


def factor_names(order: dict) -> list[str]:
    """Human labels for a plan's leaves, from the saved factors and observations."""
    factors = [FACTOR_LABEL[tuple(f['scope'])] for f in order['model']['factors']]
    observed = sorted(int(variable) for variable in order['evidence']['observations'])
    caps = [f'{VARIABLES[v]} cap' for v in observed]
    return factors + caps


def plan_leaves(run: dict) -> list[str]:
    """Leaves of the saved scalar plan, in leaf-index order."""
    factors = [FACTOR_LABEL[tuple(f['scope'])] for f in run['model']['factors']]
    names = list(factors)
    for variable in run['evidence']['observations']:
        names.append(f'{VARIABLES[int(variable)]} cap')
    return names
