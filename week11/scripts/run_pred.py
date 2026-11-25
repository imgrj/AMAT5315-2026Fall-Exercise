"""Part 1 of the Week 11 sheet: send factoring around the reduction with `pred`.

Runs the sheet's commands from `week11/`, keeps every transcript in
`runs/pred/transcript.json`, and distils the three answers into
`evidence/reduction.json`. `pred` is the `problemreductions-cli` crate at the
version the sheet pins (0.6.0).
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'runs' / 'pred'
EVIDENCE = ROOT / 'evidence'


def run(command: list[str]) -> dict:
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                               encoding='utf-8', errors='replace')
    return {
        'command': ' '.join(command),
        'returncode': completed.returncode,
        'stdout': completed.stdout,
        'stderr': completed.stderr,
    }


def solve(command: list[str]) -> dict:
    record = run(command)
    if record['returncode'] != 0:
        raise SystemExit(f"failed: {record['command']}\n{record['stderr']}")
    # pred 0.6.0 prints three lines about a panicked thread before its JSON answer.
    text = record['stdout']
    start = text.find('{')
    return json.loads(text[start:])


def product(bits: list[int]) -> tuple[int, int]:
    """The decoded pair `p, q`, lowest bit first, as the sheet reads it."""
    half = len(bits) // 2
    first = sum(bit << position for position, bit in enumerate(bits[:half]))
    second = sum(bit << position for position, bit in enumerate(bits[half:]))
    return first, second


def main() -> None:
    if shutil.which('pred') is None:
        raise SystemExit('install pred with: cargo install problemreductions-cli@0.6.0 --locked')
    RUNS.mkdir(parents=True, exist_ok=True)
    version = run(['pred', '--version'])
    commands = [version]
    path = run(['pred', 'path', 'Factoring', 'SpinGlass'])
    commands.append(path)

    answers: dict[str, dict] = {}
    for target, m, n in ((15, 2, 3), (6, 2, 2), (7, 2, 2)):
        source = RUNS / f'factor_{target}.json'
        bundle = RUNS / f'factor_{target}.bundle.json'
        commands.append(run(['pred', 'create', 'Factoring', '--target', str(target),
                             '--m', str(m), '--n', str(n), '-o', str(source)]))
        commands.append(run(['pred', 'reduce', str(source), '--to', 'SpinGlass',
                             '-o', str(bundle)]))
        solved = solve(['pred', 'solve', str(bundle)])
        commands.append(run(['pred', 'solve', str(bundle)]))
        bits = solved['solution']
        first, second = product(bits)
        answers[str(target)] = {
            'bits': bits,
            'factors': [first, second],
            'product': first * second,
            'error': abs(first * second - target),
            'evaluation': solved['evaluation'],
            'spin_glass_evaluation': solved['intermediate']['evaluation'],
            'spin_glass_spins': len(solved['intermediate']['solution']),
            'solver': solved['solver'],
        }

    # The sheet's own check on the original problem, for the 15 case.
    evaluate = solve(['pred', 'evaluate', str(RUNS / 'factor_15.json'),
                      '--config', '1,1,1,0,1'])
    commands.append(run(['pred', 'evaluate', str(RUNS / 'factor_15.json'),
                         '--config', '1,1,1,0,1']))

    route_line = next((line for line in path['stdout'].splitlines()
                       if line.startswith('Path')), '')
    route = []
    if ':' in route_line:
        route = [step.split('/')[0].strip() for step in route_line.split(':', 1)[1].split('→')]
    transcript = {
        'pred_version': version['stdout'].strip() or version['stderr'].strip(),
        'run_from': str(ROOT),
        'commands': commands,
        'path_line': route_line,
        'path': route,
        'answers': answers,
        'evaluate_15': evaluate,
    }
    (RUNS / 'transcript.json').write_text(json.dumps(transcript, indent=2), encoding='utf-8')

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    distilled = {
        'pred_version': transcript['pred_version'],
        'path_line': route_line,
        'path': route,
        'factoring': answers,
        'evaluate_15': {'config': evaluate['config'], 'result': evaluate['result']},
    }
    (EVIDENCE / 'reduction.json').write_text(json.dumps(distilled, indent=2), encoding='utf-8')

    for target, answer in answers.items():
        print(f"N = {target}: bits {answer['bits']} -> {answer['factors'][0]} x "
              f"{answer['factors'][1]} = {answer['product']}, "
              f"{answer['evaluation']}, magnet {answer['spin_glass_evaluation']} "
              f"on {answer['spin_glass_spins']} spins")
    print(f"pred: {transcript['pred_version']}")
    print(f"route: {transcript['path']}")


if __name__ == '__main__':
    main()
