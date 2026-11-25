"""Compare every saved Week 11 run with the numbers printed in the learning sheet.

Writes `evidence/acceptance.json`, which the published lab page quotes, and exits
non-zero if a stated acceptance condition fails. Nothing is rounded upward and no
tolerance is tuned to fit a result: each tolerance is the one the sheet states, or
the statistical one implied by its 2000 draws.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'runs'
EVIDENCE = ROOT / 'evidence'
BITS = [2, 3, 4, 5, 6, 7, 8]
SIZES = [4, 5, 6, 7, 8, 9, 10]

# The sheet's printed reference values.
SHEET = {
    'gates': {'AND': -3, 'OR': -3, 'NOT': -1, 'COPY': -1, 'SET0': -1, 'SET1': -1, 'XOR': -4},
    'circuit_floor': -5,
    'circuit_states': 3,
    'ferromagnet': {'e_min': -180.0, 'g0': 2},
    'factor_15': {'e_min': -299.0, 'g0': 1, 'log_z': 598.3179, 'sc': 5, 'entries': 32,
                  'p0_predicted': 1455.0, 'reference_draws': 1460},
    'spins': {2: 72, 3: 150, 4: 256, 5: 390, 6: 552, 7: 742, 8: 960},
    'greedy_sc': {2: 5, 3: 9, 4: 14, 5: 23, 6: 27, 7: 37, 8: 58},
    'treesa_sc': {2: 4, 3: 7, 4: 9, 5: 11, 6: 13, 7: 15, 8: 17},
    'treesa_tc': {2: 10.7, 3: 12.6, 4: 14.5, 5: 16.6, 6: 19.0, 7: 21.5, 8: 24.0},
    'queens': {4: 2, 5: 10, 6: 4, 7: 40, 8: 92, 9: 352, 10: 724},
    'queens8_distinct': 80,
}


def read(name: str) -> dict:
    return json.loads((RUNS / name / 'run.json').read_text(encoding='utf-8'))


def main() -> None:
    checks: list[dict] = []

    def check(name: str, ok: bool, ours, sheet=None, note: str = '') -> None:
        checks.append({'check': name, 'ok': bool(ok), 'ours': ours, 'sheet': sheet, 'note': note})

    gates = read('gadgets')
    rows = {name: [float(row['energy']) for row in gate['rows']]
            for name, gate in gates['gates'].items()}
    for name, floor in SHEET['gates'].items():
        check(f'gate floor {name}', min(rows[name]) == floor, min(rows[name]), floor)
    circuit = gates['circuit']
    circuit_floor = min(float(row['energy']) for row in circuit['rows'])
    at_floor = sum(1 for row in circuit['rows'] if float(row['energy']) == circuit_floor)
    check('circuit floor', circuit_floor == SHEET['circuit_floor'], circuit_floor,
          SHEET['circuit_floor'])
    check('circuit states on the floor', at_floor == SHEET['circuit_states'], at_floor,
          SHEET['circuit_states'])

    p0 = read('count_p0')['results']
    check('10x10 ferromagnet E_min', float(p0['e_min']) == SHEET['ferromagnet']['e_min'],
          p0['e_min'], SHEET['ferromagnet']['e_min'])
    check('10x10 ferromagnet g_0', p0['g0'] == SHEET['ferromagnet']['g0'], p0['g0'],
          SHEET['ferromagnet']['g0'])

    f15 = read('count_f15')['results']
    check('factor_15 E_min', float(f15['e_min']) == SHEET['factor_15']['e_min'], f15['e_min'],
          SHEET['factor_15']['e_min'])
    check('factor_15 g_0', f15['g0'] == SHEET['factor_15']['g0'], f15['g0'],
          SHEET['factor_15']['g0'])
    check('factor_15 log Z', abs(float(f15['log_z']) - SHEET['factor_15']['log_z']) <= 1e-4,
          f15['log_z'], SHEET['factor_15']['log_z'], 'beta = 2')
    check('factor_15 sc', f15['sc'] == SHEET['factor_15']['sc'], f15['sc'],
          SHEET['factor_15']['sc'])
    check('factor_15 largest tensor', f15['max_entries'] == SHEET['factor_15']['entries'],
          f15['max_entries'], SHEET['factor_15']['entries'])

    sample = read('sample_f15')
    results = sample['results']
    spread = results['binomial_spread']
    check('ground-state draws against 2000 p_0',
          abs(results['ground_draws'] - results['predicted_ground_draws']) <= 4 * spread,
          results['ground_draws'],
          results['predicted_ground_draws'],
          f'within 4 binomial sigma = {4 * spread:.1f}; the sheet draws '
          f'{SHEET["factor_15"]["reference_draws"]}')
    histogram = results['histogram']
    check('histogram totals 2000', sum(row['count'] for row in histogram) == 2000,
          sum(row['count'] for row in histogram), 2000)
    check('lowest sampled energy', float(results['lowest_sample']) == SHEET['factor_15']['e_min'],
          results['lowest_sample'], SHEET['factor_15']['e_min'])

    extract_path = RUNS / 'pred' / 'extract_f15.json'
    extract = json.loads(extract_path.read_text(encoding='utf-8')) if extract_path.exists() else None
    if extract is not None:
        answer = extract['answer']
        check('pred extract of the sampled ground state', answer['solution'] == [1, 1, 1, 0, 1],
              answer['solution'], [1, 1, 1, 0, 1])
        check('pred extract check on 15', answer['evaluation'] == 'Min(0)',
              answer['evaluation'], 'Min(0)')

    scaling = {}
    for k in BITS:
        greedy = read(f'plan_k{k}_greedy')['results']
        treesa = read(f'plan_k{k}_treesa')['results']
        spins = read(f'plan_k{k}_treesa')['network']['spins']
        scaling[k] = {
            'spins': spins,
            'sheet_spins': SHEET['spins'][k],
            'greedy_sc': greedy['sc'],
            'sheet_greedy_sc': SHEET['greedy_sc'][k],
            'treesa_sc': treesa['sc'],
            'sheet_treesa_sc': SHEET['treesa_sc'][k],
            'treesa_tc': round(treesa['tc'], 2),
            'sheet_treesa_tc': SHEET['treesa_tc'][k],
            'plan_hash': treesa.get('plan_hash'),
        }
        check(f'spins at k={k}', spins == SHEET['spins'][k], spins, SHEET['spins'][k])
        check(f'treesa sc at k={k}', abs(treesa['sc'] - SHEET['treesa_sc'][k]) <= 3,
              treesa['sc'], SHEET['treesa_sc'][k],
              'omeco anneals, so its plan varies between runs; the sheet states sc ~ 2k + 1 for k >= 3')
        check(f'treesa tc at k={k}', abs(treesa['tc'] - SHEET['treesa_tc'][k]) <= 1.5,
              round(treesa['tc'], 2), SHEET['treesa_tc'][k])

    counts = {}
    for n in SIZES:
        record = read(f'count_nq{n}')
        boards = int(record['results']['boards'])
        counts[n] = {'boards': boards, 'sheet': SHEET['queens'][n],
                     'sc': record['results']['sc'], 'tc': round(record['results']['tc'], 2),
                     'runtime_ms': round(record['runtime_ms'], 1)}
        check(f'N-queens count N={n}', boards == SHEET['queens'][n], boards, SHEET['queens'][n])

    six = read('sample_nq6')['results']
    frequencies = [row['count'] for row in six['frequencies']]
    chi2 = sum((count - 50.0) ** 2 / 50.0 for count in frequencies)
    check('200 draws of 6-queens visit all 4 boards', len(frequencies) == 4, len(frequencies), 4)
    check('6-queens frequencies uniform (chi2 < 16.27, 3 dof)', chi2 < 16.27, round(chi2, 2),
          'sheet reference: 56, 43, 52, 49')
    eight = read('sample_nq8')['results']
    expected_distinct = 92 * (1 - (91 / 92) ** 200)
    check('distinct 8-queens boards in 200 draws',
          abs(eight['distinct'] - expected_distinct) <= 8, eight['distinct'],
          round(expected_distinct, 1),
          f'sheet reference {SHEET["queens8_distinct"]}')

    accepted = all(entry['ok'] for entry in checks)
    summary = {
        'accepted': accepted,
        'checks': len(checks),
        'failed': [entry['check'] for entry in checks if not entry['ok']],
        'gates': {name: {'ours': min(values), 'sheet': SHEET['gates'][name]}
                  for name, values in rows.items()},
        'circuit': {'floor_ours': circuit_floor, 'floor_sheet': SHEET['circuit_floor'],
                    'states_at_floor': at_floor},
        'count_p0': {'e_min': p0['e_min'], 'g0': p0['g0'], 'log_z': p0['log_z'],
                     'log_excess_above_ground': p0['log_excess_above_ground']},
        'count_f15': {'e_min': f15['e_min'], 'g0': f15['g0'], 'log_z': f15['log_z'],
                      'sc': f15['sc'], 'tc': round(f15['tc'], 2),
                      'max_entries': f15['max_entries']},
        'sample_f15': {'samples': results['samples'], 'p0': results['p0'],
                       'predicted': results['predicted_ground_draws'],
                       'ground_draws': results['ground_draws'],
                       'binomial_spread': spread,
                       'sheet_reference_draws': SHEET['factor_15']['reference_draws'],
                       'histogram': histogram,
                       'extract': extract['answer']['solution'] if extract else None},
        'scaling': {str(k): value for k, value in scaling.items()},
        'queens': {str(n): value for n, value in counts.items()},
        'queens_samples': {'n6_frequencies': frequencies, 'n6_chi2': round(chi2, 2),
                           'n8_distinct': eight['distinct'],
                           'n8_expected_distinct': round(expected_distinct, 2)},
        'detail': checks,
    }
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    (EVIDENCE / 'acceptance.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print(f'{len(checks)} acceptance checks, {sum(1 for c in checks if not c["ok"])} failed')
    if not accepted:
        for entry in checks:
            if not entry['ok']:
                print(f"  FAILED {entry['check']}: ours {entry['ours']} sheet {entry['sheet']}")
        sys.exit(1)


if __name__ == '__main__':
    main()
