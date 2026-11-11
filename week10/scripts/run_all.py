"""Run every Week 10 computation, figure and check in the documented order.

Run from week10/: python scripts/run_all.py [--skip-scan]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

import _common as common

STEPS = [
    ('run_stated.py', 'the sheet\'s stated cases: runs/boundary64 and runs/trg1024'),
    ('run_checks.py', 'the small squares checked against spin enumeration'),
    ('run_scan.py --method boundary', 'open-square K sweep and peak refinement'),
    ('run_scan.py --method trg', 'periodic K sweep and peak refinement'),
    ('plot_svd.py', 'artifacts/svd.png'),
    ('plot_critical.py --method boundary', 'artifacts/boundary-critical.png'),
    ('plot_critical.py --method trg', 'artifacts/trg-critical.png'),
    ('verify_results.py', 'evidence/acceptance.json'),
    ('build_web_lab.py', 'docs/week10-tensor-lab.html'),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-scan', action='store_true',
                        help='reuse the saved sweep tables instead of rerunning the scans')
    arguments = parser.parse_args()
    for script, description in STEPS:
        if arguments.skip_scan and script.startswith('run_scan.py'):
            print(f'skip  {script} ({description})')
            continue
        print(f'run   {script} ({description})')
        started = time.perf_counter()
        completed = subprocess.run([sys.executable, 'scripts/' + script.split()[0]]
                                   + script.split()[1:], cwd=common.ROOT)
        if completed.returncode != 0:
            raise SystemExit(f'{script} failed with {completed.returncode}')
        print(f'      done in {time.perf_counter() - started:.1f} s')


if __name__ == '__main__':
    main()
