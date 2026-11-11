"""Record the negative controls: run the sabotaged build and show which checks fail.

This harness is meant to be run while a deliberate defect is in the crate (see
the git history: the "Sabotage (negative control)" commit). It measures the
same three acceptance numbers the tests measure and writes
`evidence/negative-controls.json`, so the record survives after the fix.

Run from week10/: python scripts/check_negative_controls.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np

import _common as common
import ising_reference as reference

COUPLING = 0.44


def boundary_control() -> dict:
    out = common.RUNS / 'negative-controls' / 'boundary'
    record = common.run(common.command('boundary', out, l=4, k=COUPLING, chi=8))
    truth = reference.enumerate_square(4, COUPLING, 'open')
    difference = abs(record['g'] - truth['g'])
    return {
        'defect': 'the accumulated normalisation scales are never advanced, while the last '
                  'core is still divided by its Frobenius norm',
        'case': 'thermo --method boundary --l 4 --k 0.44 --chi 8',
        'quantity': '|g - g_enumeration|',
        'measured': difference,
        'requirement': 1e-8,
        'detected': difference > 1e-8,
        'values': {'tool_g': record['g'], 'enumeration_g': truth['g']},
    }


def trg_control() -> dict:
    out = common.RUNS / 'negative-controls' / 'trg'
    record = common.run(common.command('trg', out, l=2, k=COUPLING, chi=16))
    truth = reference.enumerate_square(2, COUPLING, 'periodic')
    difference = abs(record['g'] - truth['g'])
    return {
        'defect': 'the closing contraction uses sum_i A[i,i,i,i] instead of '
                  'sum_{i,j} A[i,j,i,j], setting all four indices equal',
        'case': 'thermo --method trg --l 2 --k 0.44 --chi 16',
        'quantity': '|g - g_enumeration|',
        'measured': difference,
        'requirement': 1e-8,
        'detected': difference > 1e-8,
        'values': {'tool_g': record['g'], 'enumeration_g': truth['g']},
    }


def svd_control() -> dict:
    out = common.RUNS / 'negative-controls' / 'svd'
    record = common.run(common.command('svd', out, l=64, seed=2026, ranks='8,32'))
    singular = np.load(out / 'singular-values.npy')
    total = float(np.sum(singular ** 2))
    worst = 0.0
    details = []
    for entry in record['reconstructions']:
        truth = math.sqrt(float(np.sum(singular[entry['rank']:] ** 2)) / total)
        difference = abs(entry['discarded_spectrum_prediction'] - truth)
        worst = max(worst, difference)
        details.append({'rank': entry['rank'],
                        'reported_prediction': entry['discarded_spectrum_prediction'],
                        'discarded_spectrum_prediction': truth,
                        'difference': difference})
    return {
        'defect': 'the reported prediction uses the retained spectrum weight instead of the '
                  'discarded singular values',
        'case': 'thermo --method svd --l 64 --seed 2026 --ranks 8,32',
        'quantity': 'max |reported prediction - prediction from the saved singular values|',
        'measured': worst,
        'requirement': 1e-8,
        'detected': worst > 1e-8,
        'values': details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', action='store_true',
                        help='write evidence/negative-controls.json (use with the sabotaged build)')
    arguments = parser.parse_args()
    controls = {
        'purpose': 'each deliberate defect must make the acceptance check fail',
        'reference': 'scripts/verify_results.py and the pytest suite measure the same numbers',
        'controls': [boundary_control(), trg_control(), svd_control()],
    }
    if arguments.record:
        common.write_json(common.EVIDENCE / 'negative-controls.json', controls)
    failed_to_detect = [entry['case'] for entry in controls['controls'] if not entry['detected']]
    print(json.dumps({entry['case']: {'measured': entry['measured'],
                                      'detected': entry['detected']}
                      for entry in controls['controls']}, indent=2))
    if failed_to_detect:
        print(f'NOT DETECTED (a correct build should show this): {failed_to_detect}', file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
