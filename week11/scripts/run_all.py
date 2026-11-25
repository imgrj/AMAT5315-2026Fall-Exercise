"""Run every command the Week 11 learning sheet prints, from `week11/`.

The order follows the sheet: the `pred` round trip of Part 1, the gadgets of Part 2,
the counting, sampling and planning of Part 3, and the N-queens counts and draws of
Part 4. Then the three figures are drawn from the saved runs and the acceptance
summary is written. `runs/manifest.json` records every command with its exit code
and wall time.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'runs'
PRED = RUNS / 'pred'
PYTHON = sys.executable
BITS = [2, 3, 4, 5, 6, 7, 8]
SIZES = [4, 5, 6, 7, 8, 9, 10]
manifest: list[dict] = []


def run(command: list[str]) -> dict:
    started = time.time()
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                               encoding='utf-8', errors='replace')
    entry = {
        'command': ' '.join(str(part) for part in command),
        'returncode': completed.returncode,
        'seconds': round(time.time() - started, 3),
        'stdout': completed.stdout,
        'stderr': completed.stderr,
    }
    manifest.append(entry)
    if completed.returncode != 0:
        raise SystemExit(f"failed: {entry['command']}\n{completed.stdout}\n{completed.stderr}")
    return entry


def tnsample(*args: str) -> str:
    return run(['tnsample', *args])['stdout']


def main() -> None:
    if shutil.which('tnsample') is None:
        raise SystemExit('install the tool first: cd tnsample && cargo install --path . --force')
    if shutil.which('pred') is None:
        raise SystemExit('install pred: cargo install problemreductions-cli@0.6.0 --locked')
    PRED.mkdir(parents=True, exist_ok=True)

    # Part 1: the factoring round trip, and the bundles Parts 3 and 4 read.
    run([PYTHON, 'scripts/run_pred.py'])

    # Part 2: every gate truth table and the joined circuit.
    tnsample('--mode', 'gadgets', '--seed', '2026', '--out', 'runs/gadgets')

    # Part 3: the stated counts and the 2000 draws of the 104-spin magnet.
    tnsample('--mode', 'count', '--l', '10', '--p-af', '0', '--beta', '2.0',
             '--seed', '2026', '--out', 'runs/count_p0')
    tnsample('--mode', 'count', '--input', str(PRED / 'factor_15.bundle.json'), '--beta', '2.0',
             '--seed', '2026', '--out', 'runs/count_f15')
    tnsample('--mode', 'sample', '--input', str(PRED / 'factor_15.bundle.json'), '--beta', '2.0',
             '--samples', '2000', '--seed', '2026', '--out', 'runs/sample_f15')
    for k in BITS:
        target = (2 ** k - 1) ** 2
        source = PRED / f'scale_{k}.json'
        bundle = PRED / f'scale_{k}.bundle.json'
        run(['pred', 'create', 'Factoring', '--target', str(target), '--m', str(k),
             '--n', str(k), '-o', str(source)])
        run(['pred', 'reduce', str(source), '--to', 'SpinGlass', '-o', str(bundle)])
        for order in ('greedy', 'treesa'):
            tnsample('--mode', 'plan', '--order', order, '--input', str(bundle),
                     '--seed', '2026', '--out', f'runs/plan_k{k}_{order}')

    # Part 3 closes with pred extract on the lowest-energy sample.
    sample = json.loads((RUNS / 'sample_f15' / 'run.json').read_text(encoding='utf-8'))
    bits = sample['lowest_energy_bits']
    extract = run(['pred', 'extract', str(PRED / 'factor_15.bundle.json'),
                   '--config', ','.join(bits), '--json'])
    text = extract['stdout']
    answer = json.loads(text[text.find('{'):])
    (PRED / 'extract_f15.json').write_text(
        json.dumps({'config': bits, 'spin_glass_energy': sample['results']['e_min'],
                    'answer': answer}, indent=2), encoding='utf-8')
    print(f"pred extract of the lowest sample ({sample['results']['lowest_sample']:g}): "
          f"{answer['solution']} -> {answer['evaluation']}")

    # Part 4: every board count and 200 uniform draws for N = 6 and N = 8.
    for n in SIZES:
        tnsample('--mode', 'count', '--n', str(n), '--seed', '2026', '--out', f'runs/count_nq{n}')
    for n in (6, 8):
        tnsample('--mode', 'sample', '--n', str(n), '--samples', '200', '--seed', '2026',
                 '--out', f'runs/sample_nq{n}')

    # Figures from the saved runs, then the acceptance summary.
    for script in ('plot_gates.py', 'plot_factoring.py', 'plot_nqueens.py', 'check_sheet.py'):
        run([PYTHON, 'scripts/' + script])

    (RUNS / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    total = sum(entry['seconds'] for entry in manifest)
    print(f'{len(manifest)} commands, {total:.1f} s total; runs/manifest.json written')


if __name__ == '__main__':
    main()
