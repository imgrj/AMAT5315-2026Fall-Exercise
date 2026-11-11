"""Build docs/week10-tensor-lab.html from the recorded runs.

Run from week10/: python scripts/build_web_lab.py
Copies the evidence figures into docs/assets/week10/.
"""
from __future__ import annotations

import json
import math
import shutil

import numpy as np

import _common as common
import peaks

GALLERY = [
    ('svd.png', '1 / What a truncated SVD keeps',
     'plot_svd.py',
     'The spectrum and the error of the seeded 512x512 Gaussian matrix. Keeping chi directions discards '
     'sqrt(sum_{alpha>chi} s_alpha^2) of the Frobenius weight, and every measured point sits on that '
     'prediction: e_16 = 0.9423, e_64 = 0.7910, e_128 = 0.6143, e_256 = 0.3261, and the full-rank '
     'reconstruction is exact to 1e-15. The two factors only start saving storage below chi = 256, but '
     '10% error is not reached until chi = 395, which costs 1.54 times the original storage. This matrix '
     'therefore does not admit a storage-saving approximation at 10% error.'),
    ('boundary-critical.png', '2 / Open squares: where the peak sits',
     'plot_critical.py --method boundary',
     "Specific heat from the boundary-MPS contraction of open squares. The peaks move from T = 2.0931 "
     "(L = 16) through 2.1797 (L = 32) to 2.2242 (L = 64) and grow taller, so larger regions fluctuate "
     "together. Fitting T_peak = T_c + a/L gives T_c = 2.2675 from all three sizes and 2.2688 from the "
     "largest two, against Onsager's 2.26919 (0.07% and 0.02%). Doubling chi and halving h moves T_c by "
     "0.001% and 0.001%."),
    ('trg-critical.png', '3 / Periodic squares: the same transition',
     'plot_critical.py --method trg',
     "The same study with TRG on periodic squares, up to 1024x1024 (over a million spins, 20 coarse-graining "
     "rounds). Periodic boundaries remove the open surface, so the peaks approach the transition from above: "
     "2.2816, 2.2723, 2.2700 for L = 64, 256, 1024. The fits give T_c = 2.26920 (all sizes) and 2.26926 "
     "(largest two), 0.001% and 0.003% from Onsager, and the peak shifts under chi and h stay below 0.02%."),
    ('g-curves.png', '4 / The partition function behind the peaks',
     'plot_diagnostics.py',
     'g = log Z / L^2 for every size and both geometries. At K = 0 all configurations have equal weight and '
     'every curve meets log 2 = 0.6931; the curves then rise with K and bend over near the transition, which '
     'is where the curvature c_V peaks. The open and periodic squares differ by surface terms, which is why '
     'their finite-size peaks sit on opposite sides of T_c.'),
    ('phase-compare.png', '5 / Both geometries on one axis',
     'plot_diagnostics.py',
     'The two peak families together. Open squares (solid) approach T_c from below, periodic squares (dashed) '
     'from above, and both are heading to the same vertical line at 2.269185. The peak heights differ too: '
     'periodic boundaries let the whole finite system fluctuate together, so its c_V peak is about twice as '
     'tall at the same chi and the same finite-difference step.'),
    ('boundary-rows.png', '2 / One absorbed row at a time',
     'plot_diagnostics.py',
     'Diagnostics of the stated case (L = 64, K = 0.45, chi = 16). Each row multiplies the boundary by about '
     'e^60, so the row scale c_y is around 1e27 while the accumulated ell stays near 1.5e3; restoring the '
     'scales at the end gives log Z = 3837.48. The relative truncation loss per row grows from 1e-8 at the '
     'top edge to 3.4e-6 in the middle, and every horizontal bond is compressed back to chi after the row '
     'grows it towards 2 chi.'),
    ('trg-steps.png', '3 / One TRG round at a time',
     'plot_diagnostics.py',
     'Diagnostics of the stated TRG case (L = 1024, K = 0.44, chi = 16). Each round divides the representative '
     'tensor by its largest entry and adds N log c to ell, where N halves every round: 1,048,576 tensors at '
     'the start, 2 before the last round, then one closure z = 1.978 (< 1e-3 of the total). The bond '
     'dimensions follow the rank of the split, and the split spectra both show which directions the cap '
     'discards.'),
]


def spectrum_section() -> dict:
    run = common.RUNS / 'svd'
    data = common.read_json(run / 'run.json')
    singular = np.load(run / 'singular-values.npy')
    total = float(np.sum(singular ** 2))
    cumulative = np.cumsum(singular[::-1] ** 2)[::-1]
    # curve[r-1] is the relative error of the rank-r truncation: the weight of the
    # singular values alpha > r, which is cumulative[r] in zero-based indexing.
    curve = [math.sqrt(max(float(value), 0.0) / total) for value in cumulative[1:]] + [0.0]
    return {
        'side': int(data['geometry']['m']),
        'seed': int(data['generator']['seed']),
        'mean': data['sample_mean'],
        'standardDeviation': data['sample_standard_deviation'],
        'generator': data['generator'],
        'spectrum': [float(value) for value in singular],
        'predictionCurve': curve,
        'measured': [{'rank': entry['rank'],
                      'error': entry['relative_frobenius_error'],
                      'savedEntries': entry['saved_entries'],
                      'fullEntries': entry['full_entries']} for entry in data['reconstructions']],
        'storageCrossover': data['storage_crossover']['rank'],
        'crossing': data['ten_percent_crossing']['rank'],
        'crossingRatio': data['ten_percent_crossing']['storage_ratio'],
        'solverTime': data['solver_time_s'],
    }


def boundary_section() -> dict:
    run = common.read_json(common.RUNS / 'boundary64' / 'run.json')
    scan = common.read_json(common.TABLES / 'boundary-scan.json')
    refine = common.read_json(common.TABLES / 'boundary-refine.json')
    primary = refine['variants']['chi32_h0.00125']
    sizes = scan['sizes']
    rows = []
    for row in run['rows']:
        rows.append({
            'y': row['row'],
            'scale': row['scale'],
            'ell': row['ell'],
            'before': row['bond_dims_before'],
            'after': row['bond_dims_after'],
            'relativeDiscarded': row['relative_discarded_weight'],
            'maxDiscarded': row['max_discarded_weight'],
            'entries': row['contraction_entries'],
        })
    return {
        'l': run['geometry']['l'],
        'k': run['settings']['k'],
        'chi': run['settings']['chi'],
        'g': run['g'],
        'logZ': run['log_z'],
        'solverTime': run['solver_time_s'],
        'ells': run['normalisation']['ell'],
        'bottom': run['normalisation']['bottom_value'],
        'rows': rows,
        'peaks': [{'l': length,
                   'tPeak': primary['sizes'][str(length)]['peak']['t_peak'],
                   'kPeak': primary['sizes'][str(length)]['peak']['k_peak'],
                   'cvPeak': primary['sizes'][str(length)]['peak']['grid_cv']} for length in sizes],
        'fitAll': primary['fit']['all_sizes']['t_c'],
        'fitLargestTwo': primary['fit']['largest_two']['t_c'],
        'onsagerRelativeError': primary['fit']['onsager_relative_error_all'],
        'comparison': refine['comparison'],
    }


def trg_section() -> dict:
    run = common.read_json(common.RUNS / 'trg1024' / 'run.json')
    scan = common.read_json(common.TABLES / 'trg-scan.json')
    refine = common.read_json(common.TABLES / 'trg-refine.json')
    primary = refine['variants']['chi32_h0.00125']
    sizes = scan['sizes']
    steps = []
    for step in run['steps']:
        steps.append({
            'step': step['step'],
            'tensors': step['tensors'],
            'dims': step['bond_dims'],
            'maxAbs': step['max_abs'],
            'ell': step['ell'],
            'logScale': step['log_scale'],
            'spectrum': step['split1_singular_values'][:8],
            'spectrum2': step['split2_singular_values'][:8],
            'regroupEntries': step['regroup_entries'],
        })
    return {
        'l': run['geometry']['l'],
        'k': run['settings']['k'],
        'chi': run['settings']['chi'],
        'g': run['g'],
        'logZ': run['log_z'],
        'solverTime': run['solver_time_s'],
        'ells': run['normalisation']['ell'],
        'closure': run['normalisation']['z'],
        'steps': steps,
        'peaks': [{'l': length,
                   'tPeak': primary['sizes'][str(length)]['peak']['t_peak'],
                   'kPeak': primary['sizes'][str(length)]['peak']['k_peak'],
                   'cvPeak': primary['sizes'][str(length)]['peak']['grid_cv']} for length in sizes],
        'fitAll': primary['fit']['all_sizes']['t_c'],
        'fitLargestTwo': primary['fit']['largest_two']['t_c'],
        'onsagerRelativeError': primary['fit']['onsager_relative_error_all'],
        'comparison': refine['comparison'],
    }


def main() -> None:
    docs = common.ROOT.parent / 'docs'
    assets = docs / 'assets' / 'week10'
    assets.mkdir(parents=True, exist_ok=True)
    acceptance = common.read_json(common.EVIDENCE / 'acceptance.json')
    sheet = common.read_json(common.EVIDENCE / 'sheet.json')
    checks = common.read_json(common.TABLES / 'small-checks.json')
    controls = common.read_json(common.EVIDENCE / 'negative-controls.json')
    data = {
        'sheet': sheet,
        'summary': acceptance['summary'],
        'tolerances': acceptance['tolerances'],
        'svd': spectrum_section(),
        'boundary': boundary_section(),
        'trg': trg_section(),
        'small': {
            'open4': checks['boundary']['open_4x4_chi8'],
            'periodic2': checks['trg']['periodic_2x2_chi16'],
            'periodic4': checks['trg']['periodic_4x4_chi8'],
            'stencil': checks['stencil_h'],
            'cvOpen4': acceptance['boundary']['small_square'],
            'cvOpen4Truth': acceptance['boundary']['small_square']['cv_variance'],
            'cvPeriodic2': acceptance['trg']['two_by_two'],
            'openEnum': acceptance['boundary']['small_square']['comparisons'],
            'periodic2G': acceptance['trg']['two_by_two']['g_tool'],
            'periodic2Enum': acceptance['trg']['two_by_two']['g_enumeration'],
            'periodic2Difference': acceptance['trg']['two_by_two']['difference'],
            'periodic4G': acceptance['trg']['four_by_four']['g_tool'],
            'periodic4Enum': acceptance['trg']['four_by_four']['g_enumeration'],
            'periodic4Difference': acceptance['trg']['four_by_four']['difference'],
        },
        'reference': {
            'svd': {'e16': 0.943, 'e256': 0.326, 'crossing': 395, 'ratio': 1.54},
            'openPeaks': [2.093, 2.180, 2.224],
            'openTc': [2.268, 2.269],
            'openCv4': 0.439397,
            'periodicPeaks': [2.282, 2.272, 2.270],
            'periodicTc': [2.2692, 2.2693],
            'periodicCv2': 0.404278,
            'onsager': peaks.ONSAGER_T,
        },
        'controls': controls['controls'],
        'gallery': GALLERY,
    }
    data['svd']['worstPrediction'] = acceptance['svd']['worst_tool_minus_prediction']
    data['svd']['worstIndependent'] = acceptance['svd']['worst_tool_minus_independent']
    data['svd']['fullRankError'] = acceptance['svd']['full_rank_error']
    for name, _, _, _ in GALLERY:
        shutil.copyfile(common.ARTIFACTS / name, assets / name)
    template = (common.ROOT / 'web' / 'lab_template.html').read_text(encoding='utf-8')
    rendered = template.replace('__DATA__', json.dumps(data, ensure_ascii=True).replace('</', '<\\/'))
    (docs / 'week10-tensor-lab.html').write_text(rendered, encoding='utf-8')
    print(f'built docs/week10-tensor-lab.html with {len(GALLERY)} evidence figures')


if __name__ == '__main__':
    main()
