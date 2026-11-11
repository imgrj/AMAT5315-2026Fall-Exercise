"""Recompute every Week 10 acceptance number from the saved runs.

Run from week10/: python scripts/verify_results.py

Nothing here trusts the Rust tool: the Gaussian errors are recomputed from the
saved matrices with NumPy, the boundary contraction is recomputed from the saved
cores, and the small squares are compared with a full sum over spin
configurations. Writes `evidence/acceptance.json`.
"""
from __future__ import annotations

import datetime
import json
import math
from pathlib import Path

import numpy as np

import _common as common
import ising_reference as reference
import peaks

TOLERANCES = {
    'svd_error_vs_prediction': 1e-8,
    'svd_independent_error': 1e-8,
    'svd_full_rank': 1e-8,
    'boundary_enumeration_g': 1e-8,
    'boundary_specific_heat': 1e-3,
    'boundary_zero_coupling': 1e-12,
    'boundary_cores_recompute': 1e-12,
    'boundary_numpy_reference': 1e-10,
    'trg_enumeration_exact': 1e-8,
    'trg_enumeration_truncated': 2e-3,
    'trg_specific_heat': 1e-3,
    'trg_zero_coupling': 1e-12,
    'trg_numpy_reference': 1e-10,
    'onsager': 0.01,
    'peak_stability': 0.01,
}


def relative_difference(reference_value: float, measured: float) -> float:
    scale = max(abs(reference_value), abs(measured), 1e-300)
    return abs(measured - reference_value) / scale


def check(record: dict, name: str, requirement: str, measured, passed: bool, detail=None) -> None:
    record.setdefault('checks', []).append({
        'name': name,
        'requirement': requirement,
        'measured': measured,
        'passed': bool(passed),
        **({'detail': detail} if detail is not None else {}),
    })


def svd_section(record: dict) -> dict:
    run = common.RUNS / 'svd'
    data = common.read_json(run / 'run.json')
    matrix = np.load(run / 'matrix.npy')
    singular = np.load(run / 'singular-values.npy')
    total = float(np.sum(singular ** 2))
    norm = math.sqrt(total)
    entries = []
    independent_errors = {}
    for entry in data['reconstructions']:
        rebuilt = np.load(run / entry['file'])
        measured = float(np.linalg.norm(matrix - rebuilt) / norm)
        prediction = float(math.sqrt(float(np.sum(singular[entry['rank']:] ** 2)) / total))
        independent_errors[entry['rank']] = measured
        entries.append({
            'rank': entry['rank'],
            'tool_error': entry['relative_frobenius_error'],
            'independent_error': measured,
            'prediction': prediction,
            'tool_minus_independent': abs(entry['relative_frobenius_error'] - measured),
            'tool_minus_prediction': abs(entry['relative_frobenius_error'] - prediction),
            'storage_entries': entry['saved_entries'],
            'storage_ratio': entry['storage_ratio'],
        })
    worst_prediction = max(entry['tool_minus_prediction'] for entry in entries)
    worst_independent = max(entry['tool_minus_independent'] for entry in entries)
    storage_rank = (matrix.shape[0] * matrix.shape[1]) // (matrix.shape[0] + matrix.shape[1])
    crossing = None
    for rank in range(1, matrix.shape[0] + 1):
        if math.sqrt(float(np.sum(singular[rank:] ** 2)) / total) <= 0.1:
            crossing = rank
            break
    crossing_ratio = (matrix.shape[0] + matrix.shape[1]) * crossing / matrix.size
    section = {
        'generator': data['generator'],
        'sample_mean': data['sample_mean'],
        'sample_standard_deviation': data['sample_standard_deviation'],
        'ranks': entries,
        'worst_tool_minus_prediction': worst_prediction,
        'worst_tool_minus_independent': worst_independent,
        'full_rank_error': independent_errors.get(matrix.shape[0]),
        'storage_crossover_rank': storage_rank,
        'ten_percent_crossing_rank': crossing,
        'ten_percent_storage_ratio': crossing_ratio,
        'tool_ten_percent_crossing_rank': data['ten_percent_crossing']['rank'],
        'solver_time_s': data['solver_time_s'],
        'answer': (
            f"The 10% error is first reached at chi = {crossing}, which needs "
            f"{(matrix.shape[0] + matrix.shape[1]) * crossing / matrix.size:.2f} times the "
            f"original storage; the two factors only start saving storage at chi = {storage_rank}. "
            "So this matrix does not admit a storage-saving approximation at 10% error."
        ),
    }
    for entry in entries:
        check(record, f"svd chi={entry['rank']} measured vs prediction",
              f"<= {TOLERANCES['svd_error_vs_prediction']:.0e}",
              f"{entry['tool_minus_prediction']:.2e}",
              entry['tool_minus_prediction'] <= TOLERANCES['svd_error_vs_prediction'])
        check(record, f"svd chi={entry['rank']} independent recomputation",
              f"<= {TOLERANCES['svd_independent_error']:.0e}",
              f"{entry['tool_minus_independent']:.2e}",
              entry['tool_minus_independent'] <= TOLERANCES['svd_independent_error'])
    full_rank = independent_errors[matrix.shape[0]]
    check(record, 'svd full-rank reconstruction',
          f"e_{matrix.shape[0]} < {TOLERANCES['svd_full_rank']:.0e}", f'{full_rank:.2e}',
          full_rank < TOLERANCES['svd_full_rank'])
    check(record, 'svd storage crossover', f"rank = mn/(m+n) = {storage_rank}", storage_rank,
          data['storage_crossover']['rank'] == storage_rank)
    check(record, 'svd 10% crossing from the saved spectrum', 'same rank as the tool reports',
          f'{crossing} (tool {data["ten_percent_crossing"]["rank"]})',
          crossing == data['ten_percent_crossing']['rank'])
    return section


def recompute_boundary_from_cores(out: Path) -> dict:
    """Recompute `g` from the saved cores alone: g = (ell + log v_1)/L^2."""
    run = common.read_json(out / 'run.json')
    cores = common.read_json(out / 'cores.json')
    length = run['geometry']['l']
    close = np.array(run['model']['b_vector'])
    vector = np.array([1.0])
    for (left, physical, right), offset in zip(cores['dims'], np.cumsum(
            [0] + [l * p * r for l, p, r in cores['dims'][:-1]])):
        block = np.array(cores['data'][offset:offset + left * physical * right])
        core = block.reshape(left, physical, right)
        vector = vector @ np.tensordot(core, close, axes=([1], [0]))
    ell = sum(math.log(scale) for scale in cores['scales'])
    log_z = ell + math.log(float(vector[0]))
    return {
        'g': log_z / (length * length),
        'log_z': log_z,
        'ell': ell,
        'bottom': float(vector[0]),
        'tool_g': run['g'],
        'tool_log_z': run['log_z'],
    }


def boundary_section(record: dict) -> dict:
    checks = common.read_json(common.TABLES / 'small-checks.json')
    scan = common.read_json(common.TABLES / 'boundary-scan.json')
    refine = common.read_json(common.TABLES / 'boundary-refine.json')
    primary = refine['variants']['chi32_h0.00125']
    section = {'boundary_condition': 'open', 'scan': scan['grid_peaks'],
               'refined': {str(length): primary['sizes'][str(length)]['peak']
                           for length in scan['sizes']},
               'fit': primary['fit'], 'comparison': refine['comparison']}

    exact = reference.enumerate_square(4, 0.44, 'open')
    small = {entry['k']: entry for entry in checks['boundary']['open_4x4_chi8']}
    comparisons = []
    for k in (0.0, 0.44, 0.7):
        truth = reference.enumerate_square(4, k, 'open')
        measured = small[k]['g']
        difference = abs(measured - truth['g'])
        comparisons.append({'k': k, 'tool_g': measured, 'enumeration_g': truth['g'],
                            'difference': difference})
        check(record, f'boundary 4x4 open K={k} vs enumeration',
              f"|dg| <= {TOLERANCES['boundary_enumeration_g']:.0e}", f'{difference:.2e}',
              difference <= TOLERANCES['boundary_enumeration_g'])
    h = checks['stencil_h']
    finite_difference = (small[0.44 + h]['g'] - 2.0 * small[0.44]['g'] + small[0.44 - h]['g']) / h ** 2
    cv_tool = 0.44 ** 2 * finite_difference
    cv_truth = exact['specific_heat']
    difference = abs(cv_tool - cv_truth)
    section['small_square'] = {
        'k': 0.44, 'g_tool': small[0.44]['g'], 'g_enumeration': exact['g'],
        'cv_tool': cv_tool, 'cv_variance': cv_truth, 'cv_difference': difference,
        'bonds': exact['bonds'], 'configurations': exact['configurations'],
        'comparisons': comparisons,
    }
    check(record, 'boundary 4x4 c_V vs K^2 Var(H)/16',
          f"<= {TOLERANCES['boundary_specific_heat']:.0e}", f'{difference:.2e}',
          difference <= TOLERANCES['boundary_specific_heat'])

    zeros = checks['boundary']['zero_coupling']
    worst = max(abs(entry['g'] - math.log(2.0)) for entry in zeros)
    section['zero_coupling'] = {'entries': zeros, 'worst_difference': worst}
    check(record, 'boundary g(K=0) = log 2', f"<= {TOLERANCES['boundary_zero_coupling']:.0e}",
          f'{worst:.2e}', worst <= TOLERANCES['boundary_zero_coupling'])

    stated = recompute_boundary_from_cores(common.RUNS / 'boundary64')
    section['stated_case'] = stated
    difference = abs(stated['g'] - stated['tool_g'])
    check(record, 'boundary stated case recomputed from cores.json',
          f"<= {TOLERANCES['boundary_cores_recompute']:.0e}", f'{difference:.2e}',
          difference <= TOLERANCES['boundary_cores_recompute'])

    reference_runs = []
    for length, chi in ((6, 8), (8, 8), (8, 16)):
        out = common.run_path('reference', 'boundary', length, chi, 0.44)
        record_run = common.run(common.command('boundary', out, l=length, k=0.44, chi=chi))
        independent = reference.boundary_mps(length, 0.44, chi)
        difference = abs(record_run['g'] - independent['g'])
        reference_runs.append({'l': length, 'chi': chi, 'tool_g': record_run['g'],
                               'numpy_g': independent['g'], 'difference': difference})
        check(record, f'boundary L={length} chi={chi} vs NumPy reference',
              f"<= {TOLERANCES['boundary_numpy_reference']:.0e}", f'{difference:.2e}',
              difference <= TOLERANCES['boundary_numpy_reference'])
    section['numpy_reference'] = reference_runs

    for key, label in (('all_sizes', 'all three sizes'), ('largest_two', 'largest two')):
        fit = primary['fit'][key]
        error = peaks.relative_difference(fit['t_c'], peaks.ONSAGER_T)
        check(record, f'open T_c from the {label} fit',
              f"within {100 * TOLERANCES['onsager']:.0f}% of Onsager 2.269185",
              f"{fit['t_c']:.5f} ({100 * error:.3f}%)", error <= TOLERANCES['onsager'])
    for label, value in refine['comparison']['rank_shift'].items():
        check(record, f'open peak L={label} chi 16 vs 32',
              f"within {100 * TOLERANCES['peak_stability']:.0f}%", f'{100 * value:.3f}%',
              value <= TOLERANCES['peak_stability'])
    for label, value in refine['comparison']['step_shift'].items():
        check(record, f'open peak L={label} h 0.0025 vs 0.00125',
              f"within {100 * TOLERANCES['peak_stability']:.0f}%", f'{100 * value:.3f}%',
              value <= TOLERANCES['peak_stability'])
    return section


def trg_section(record: dict) -> dict:
    checks = common.read_json(common.TABLES / 'small-checks.json')
    scan = common.read_json(common.TABLES / 'trg-scan.json')
    refine = common.read_json(common.TABLES / 'trg-refine.json')
    primary = refine['variants']['chi32_h0.00125']
    section = {'boundary_condition': 'periodic', 'scan': scan['grid_peaks'],
               'refined': {str(length): primary['sizes'][str(length)]['peak']
                           for length in scan['sizes']},
               'fit': primary['fit'], 'comparison': refine['comparison']}

    exact_two = reference.enumerate_square(2, 0.44, 'periodic')
    two = {entry['k']: entry for entry in checks['trg']['periodic_2x2_chi16']}
    difference = abs(two[0.44]['g'] - exact_two['g'])
    h = checks['stencil_h']
    finite_difference = (two[0.44 + h]['g'] - 2.0 * two[0.44]['g'] + two[0.44 - h]['g']) / h ** 2
    cv_tool = 0.44 ** 2 * finite_difference
    section['two_by_two'] = {
        'g_tool': two[0.44]['g'], 'g_enumeration': exact_two['g'], 'difference': difference,
        'cv_tool': cv_tool, 'cv_variance': exact_two['specific_heat'],
        'cv_difference': abs(cv_tool - exact_two['specific_heat']),
        'bonds': exact_two['bonds'], 'configurations': exact_two['configurations'],
    }
    check(record, 'trg 2x2 periodic chi=16 vs enumeration',
          f"|dg| <= {TOLERANCES['trg_enumeration_exact']:.0e}", f'{difference:.2e}',
          difference <= TOLERANCES['trg_enumeration_exact'])
    check(record, 'trg 2x2 c_V vs K^2 Var(H)/4',
          f"<= {TOLERANCES['trg_specific_heat']:.0e}", f'{section["two_by_two"]["cv_difference"]:.2e}',
          section['two_by_two']['cv_difference'] <= TOLERANCES['trg_specific_heat'])

    exact_four = reference.enumerate_square(4, 0.44, 'periodic')
    four = {entry['k']: entry for entry in checks['trg']['periodic_4x4_chi8']}
    difference = abs(four[0.44]['g'] - exact_four['g'])
    section['four_by_four'] = {'g_tool': four[0.44]['g'], 'g_enumeration': exact_four['g'],
                               'difference': difference, 'bonds': exact_four['bonds']}
    check(record, 'trg 4x4 periodic chi=8 vs enumeration',
          f"|dg| <= {TOLERANCES['trg_enumeration_truncated']:.0e}", f'{difference:.2e}',
          difference <= TOLERANCES['trg_enumeration_truncated'])

    zeros = checks['trg']['zero_coupling']
    worst = max(abs(entry['g'] - math.log(2.0)) for entry in zeros)
    section['zero_coupling'] = {'entries': zeros, 'worst_difference': worst}
    check(record, 'trg g(K=0) = log 2', f"<= {TOLERANCES['trg_zero_coupling']:.0e}",
          f'{worst:.2e}', worst <= TOLERANCES['trg_zero_coupling'])

    reference_runs = []
    for length, chi in ((4, 4), (4, 8), (8, 8)):
        out = common.run_path('reference', 'trg', length, chi, 0.44)
        record_run = common.run(common.command('trg', out, l=length, k=0.44, chi=chi))
        independent = reference.numpy_trg(length, 0.44, chi)
        difference = abs(record_run['g'] - independent['g'])
        reference_runs.append({'l': length, 'chi': chi, 'tool_g': record_run['g'],
                               'numpy_g': independent['g'], 'difference': difference})
        check(record, f'trg L={length} chi={chi} vs NumPy reference',
              f"<= {TOLERANCES['trg_numpy_reference']:.0e}", f'{difference:.2e}',
              difference <= TOLERANCES['trg_numpy_reference'])
    section['numpy_reference'] = reference_runs

    stated = common.read_json(common.RUNS / 'trg1024' / 'run.json')
    section['stated_case'] = {
        'l': stated['geometry']['l'], 'k': stated['settings']['k'],
        'chi': stated['settings']['chi'], 'g': stated['g'], 'log_z': stated['log_z'],
        'steps': len(stated['steps']), 'solver_time_s': stated['solver_time_s'],
        'tensors_per_step': [step['tensors'] for step in stated['steps']],
        'bond_dims': [step['bond_dims'] for step in stated['steps']],
        'final_shape': stated['final_tensor_shape'],
    }
    check(record, 'trg stated case halves the tensor count',
          'L=1024 needs 20 steps from 1048576 to 2 tensors',
          f"{len(stated['steps'])} steps, first {stated['steps'][0]['tensors']}, "
          f"last {stated['steps'][-1]['tensors']}",
          len(stated['steps']) == 20 and stated['steps'][-1]['tensors'] == 2)

    for key, label in (('all_sizes', 'all three sizes'), ('largest_two', 'largest two')):
        fit = primary['fit'][key]
        error = peaks.relative_difference(fit['t_c'], peaks.ONSAGER_T)
        check(record, f'periodic T_c from the {label} fit',
              f"within {100 * TOLERANCES['onsager']:.0f}% of Onsager 2.269185",
              f"{fit['t_c']:.5f} ({100 * error:.3f}%)", error <= TOLERANCES['onsager'])
    for label, value in refine['comparison']['rank_shift'].items():
        check(record, f'periodic peak L={label} chi 16 vs 32',
              f"within {100 * TOLERANCES['peak_stability']:.0f}%", f'{100 * value:.3f}%',
              value <= TOLERANCES['peak_stability'])
    for label, value in refine['comparison']['step_shift'].items():
        check(record, f'periodic peak L={label} h 0.0025 vs 0.00125',
              f"within {100 * TOLERANCES['peak_stability']:.0f}%", f'{100 * value:.3f}%',
              value <= TOLERANCES['peak_stability'])
    return section


def main() -> None:
    record: dict = {
        'generated': datetime.datetime.now().isoformat(timespec='seconds'),
        'crate': 'thermo 0.1.0',
        'settings': {
            'grid_k': '0.40 .. 0.50 step 0.0025',
            'refinement': '9 points step 0.00125 around the nearest grid point',
            'stencil_h': {'coarse': 0.0025, 'refined_chi16': 0.0025, 'refined_chi32': 0.00125},
            'chi': [1, 8, 16, 32],
            'open_sizes': [16, 32, 64],
            'periodic_sizes': [64, 256, 1024],
        },
        'tolerances': TOLERANCES,
    }
    record['svd'] = svd_section(record)
    record['boundary'] = boundary_section(record)
    record['trg'] = trg_section(record)
    controls = common.EVIDENCE / 'negative-controls.json'
    if controls.exists():
        record['negative_controls'] = common.read_json(controls)
    passed = sum(1 for entry in record['checks'] if entry['passed'])
    record['summary'] = {
        'checks': len(record['checks']),
        'passed': passed,
        'failed': [entry['name'] for entry in record['checks'] if not entry['passed']],
    }
    common.write_json(common.EVIDENCE / 'acceptance.json', record)
    print(f"{passed}/{len(record['checks'])} acceptance checks passed")
    for entry in record['checks']:
        if not entry['passed']:
            print(f"  FAILED {entry['name']}: {entry['measured']} (needs {entry['requirement']})")


if __name__ == '__main__':
    main()
