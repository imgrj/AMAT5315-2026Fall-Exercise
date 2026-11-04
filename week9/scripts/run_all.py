"""Run every Week 9 inference command in order, filling artifacts/ for the plots.

Run from week9/: python scripts/run_all.py [--skip-promedus]
Uses the binary built by `cargo +1.96.0 build --release`.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def binary() -> Path:
    subprocess.run(['cargo', '+1.96.0', 'build', '--release'], cwd=ROOT, check=True)
    metadata = subprocess.run(['cargo', '+1.96.0', 'metadata', '--format-version', '1', '--no-deps'],
                              cwd=ROOT, check=True, capture_output=True, text=True)
    target = Path(json.loads(metadata.stdout)['target_directory'])
    return target / 'release' / ('infer.exe' if sys.platform == 'win32' else 'infer')


ASIA = [
    ('--model', 'asia.uai', '--evidence', 'prior', '--out', 'artifacts/prior'),
    ('--model', 'asia.uai', '--evidence', 'AD', '--out', 'artifacts/AD'),
    ('--model', 'asia.uai', '--evidence', 'ADX', '--out', 'artifacts/ADX'),
]
ORDERS = [
    ('--model', 'asia.uai', '--evidence', 'AD', '--order', 'random', '--seed', '42', '--out', 'artifacts/runs/random'),
    ('--model', 'asia.uai', '--evidence', 'AD', '--order', 'greedy', '--out', 'artifacts/runs/greedy'),
    ('--model', 'asia.uai', '--evidence', 'AD', '--analyze', 'artifacts/runs/random/order.json', '--out', 'artifacts/runs/analysis'),
    ('--model', 'asia.uai', '--evidence', 'AD', '--plan', 'artifacts/runs/greedy/order.json', '--out', 'artifacts/runs/replay'),
    ('--model', 'asia.uai', '--evidence', 'AD', '--order', 'exhaustive', '--out', 'artifacts/runs/exhaustive'),
]
OPTIMA = [
    ('--model', 'asia.uai', '--evidence', 'prior', '--order', 'exhaustive', '--out', 'artifacts/runs/exhaustive-prior'),
    ('--model', 'asia.uai', '--evidence', 'AD', '--order', 'exhaustive', '--out', 'artifacts/runs/exhaustive-AD'),
    ('--model', 'asia.uai', '--evidence', 'ADX', '--order', 'exhaustive', '--out', 'artifacts/runs/exhaustive-ADX'),
]
BP = [
    ('--model', 'asia.uai', '--evidence', case, '--method', 'bp', '--damping', '0.5',
     '--tolerance', '1e-10', '--max-sweeps', '1000', '--out', f'artifacts/bp-{case}')
    for case in ('prior', 'AD', 'ADX')
]
TREE = [
    ('--model', 'tests/data/tree.uai', '--out', 'artifacts/tree/exact'),
    ('--model', 'tests/data/tree.uai', '--method', 'bp', '--damping', '0.5',
     '--tolerance', '1e-14', '--max-sweeps', '500', '--out', 'artifacts/tree/bp'),
]
PROMEDUS = [
    ('--model', 'promedus/Promedus_14.uai', '--query', '59,102,139', '--order', 'greedy',
     '--out', 'artifacts/promedus/greedy'),
    ('--model', 'promedus/Promedus_14.uai', '--query', '59,102,139', '--order', 'treesa',
     '--out', 'artifacts/promedus/treesa'),
    ('--model', 'promedus/Promedus_14.uai', '--query', '59,102,139', '--order', 'treesa',
     '--method', 'bp', '--damping', '0.5', '--tolerance', '1e-10', '--max-sweeps', '1000',
     '--out', 'artifacts/promedus/bp'),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-promedus', action='store_true', help='only run the ASIA cases')
    arguments = parser.parse_args()
    exe = binary()
    commands = ASIA + ORDERS + OPTIMA + BP + TREE + ([] if arguments.skip_promedus else PROMEDUS)
    for index, command in enumerate(commands, 1):
        print(f'[{index}/{len(commands)}] infer ' + ' '.join(command))
        subprocess.run([str(exe), *command], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)


if __name__ == '__main__':
    main()
