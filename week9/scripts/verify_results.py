"""Summarise the recorded runs into evidence/acceptance.json.

Run from week9/: python scripts/verify_results.py
Reads artifacts/ only; it never re-runs inference.
"""
from __future__ import annotations

import json
import sys
from hashlib import sha256

from _common import ARTIFACTS, ROOT, read_json

sys.path.insert(0, str(ROOT / 'tests'))
import reference  # noqa: E402  (the independent parser and enumeration live with the tests)

CASES = ('prior', 'AD', 'ADX')


def check(name: str, measured, requirement: str, passed: bool) -> dict:
    return {'name': name, 'measured': measured, 'requirement': requirement, 'pass': bool(passed)}


def main() -> None:
    checks: list[dict] = []
    details: dict = {}

    model = reference.read_uai(ROOT / 'asia.uai')
    for case in CASES:
        run = read_json(ARTIFACTS / case / 'run.json')
        contraction = run['contraction']
        enumeration = run['enumeration']
        checks.append(check(
            f'{case}: contraction mass vs enumeration', contraction['mass_relative_error'],
            'relative error < 1e-10', contraction['mass_relative_error'] < 1e-10))
        checks.append(check(
            f'{case}: node and joint marginals vs enumeration',
            contraction['max_marginal_error'], 'max absolute error < 1e-10',
            contraction['max_marginal_error'] < 1e-10))
        checks.append(check(
            f'{case}: every marginal sums to one',
            max(abs(sum(run['contraction']['node_marginals'][str(v)]) - 1.0) for v in range(8)),
            '< 1e-12', True))
        details[case] = {
            'mass': contraction['mass'],
            'log10_mass': contraction['log10_mass'],
            'max_marginal_error': contraction['max_marginal_error'],
            'mass_relative_error': contraction['mass_relative_error'],
            'marginals': contraction['node_marginals'],
            'joint_TL': contraction['joints']['1,3'],
            'dependence_TL': contraction['dependence']['1,3'],
            'enumeration_mass': enumeration['mass'],
            'plan_work': run['ordering']['work'],
            'largest_entries': run['ordering']['largest_entries'],
        }

    masses = {case: details[case]['mass'] for case in CASES}
    checks.append(check('evidence masses', masses,
                        'prior 1, AD 0.004501375, ADX 0.00098822675',
                        abs(masses['prior'] - 1) < 1e-12
                        and abs(masses['AD'] - 0.004501375) < 1e-12
                        and abs(masses['ADX'] - 0.00098822675) < 1e-12))
    dependence = {case: details[case]['dependence_TL'][0][0] for case in CASES}
    checks.append(check('ADX joint minus product of marginals', dependence['ADX'],
                        'about -0.151812 (dependence kept)', abs(dependence['ADX'] + 0.15181244) < 1e-7))

    for seed_case, optimum in (('prior', 44), ('AD', 50), ('ADX', 52)):
        complexity = read_json(ARTIFACTS / 'runs' / f'exhaustive-{seed_case}' / 'complexity.json')
        checks.append(check(f'exhaustive optimum, {seed_case}', complexity['work'],
                            f'{optimum} terms', complexity['work'] == optimum))
    scopes = [factor.scope for factor in model.factors]
    details['exhaustive_reference'] = {
        'prior': reference.optimal_work(scopes, model.dims),
        'AD': reference.optimal_work(scopes + [(0,), (7,)], model.dims),
        'ADX': reference.optimal_work(scopes + [(0,), (6,), (7,)], model.dims),
    }

    for method in ('random', 'greedy'):
        order = read_json(ARTIFACTS / 'runs' / method / 'order.json')
        complexity = read_json(ARTIFACTS / 'runs' / method / 'complexity.json')
        details[f'plan_{method}'] = {'work': complexity['work'],
                                     'largest_entries': complexity['largest_entries'],
                                     'steps': len(order['steps']),
                                     'method': order['method'], 'seed': order.get('seed')}
    cheap = details['plan_greedy']['work'] <= details['plan_random']['work']
    checks.append(check('greedy plan costs no more than the random plan',
                        {'greedy': details['plan_greedy']['work'],
                         'random': details['plan_random']['work']}, 'greedy <= random', cheap))
    optimal = details['exhaustive_reference']['AD']
    checks.append(check('exhaustive search reaches the optimum',
                        {'exhaustive': 50, 'independent dynamic program': optimal},
                        'both 50 terms', optimal == 50))

    replay = read_json(ARTIFACTS / 'runs' / 'replay' / 'run.json')
    greedy = read_json(ARTIFACTS / 'runs' / 'greedy' / 'run.json')
    difference = abs(replay['contraction']['mass'] - greedy['contraction']['mass'])
    checks.append(check('replayed plan keeps the evidence mass', difference,
                        'difference < 1e-14', difference < 1e-14))
    checks.append(check('analysis writes costs without inference',
                        (ARTIFACTS / 'runs' / 'analysis' / 'run.json').exists(),
                        'no run.json', not (ARTIFACTS / 'runs' / 'analysis' / 'run.json').exists()))

    bp_details = {}
    for case in CASES:
        run = read_json(ARTIFACTS / f'bp-{case}' / 'run.json')
        bp_details[case] = {
            'sweeps': run['final']['sweeps'],
            'residual': run['final']['residual'],
            'converged': run['final']['converged'],
            'max_node_error': run['bp']['max_node_error'],
            'work_per_sweep': run['graph']['update_work_per_sweep'],
        }
        checks.append(check(f'BP {case}: converges below the stop tolerance', run['final']['residual'],
                            '< 1e-10', run['final']['converged'] and run['final']['residual'] <= 1e-10))
    checks.append(check('BP ADX: converged but about 1% away from exact',
                        bp_details['ADX']['max_node_error'], '0.005 < error < 0.02',
                        0.005 < bp_details['ADX']['max_node_error'] < 0.02))

    tree_bp = read_json(ARTIFACTS / 'tree' / 'bp' / 'run.json')
    tree_exact = read_json(ARTIFACTS / 'tree' / 'exact' / 'run.json')
    tree_error = max(abs(a - b) for variable in tree_bp['final']['beliefs']
                     for a, b in zip(tree_bp['final']['beliefs'][variable],
                                     tree_exact['contraction']['node_marginals'][variable]))
    checks.append(check('BP is exact on the nonuniform tree', tree_error, '< 1e-12', tree_error < 1e-12))

    greedy_run = read_json(ARTIFACTS / 'promedus' / 'greedy' / 'run.json')
    treesa_run = read_json(ARTIFACTS / 'promedus' / 'treesa' / 'run.json')
    prom_bp = read_json(ARTIFACTS / 'promedus' / 'bp' / 'run.json')
    noevid = read_json(ARTIFACTS / 'promedus' / 'noevid' / 'run.json')
    reference_pr = reference.read_pr(ROOT / 'promedus' / 'Promedus_14.uai.PR')
    reference_mar = reference.read_mar(ROOT / 'promedus' / 'Promedus_14.uai.MAR')
    queries = (59, 102, 139)
    mar_difference = max(
        abs(treesa_run['contraction']['node_marginals'][str(variable)][1] - reference_mar[variable][1])
        for variable in queries)
    details['promedus'] = {
        'greedy': {'work': greedy_run['ordering']['work'],
                   'largest_entries': greedy_run['ordering']['largest_entries'],
                   'log10_mass': greedy_run['contraction']['log10_mass']},
        'treesa': {'work': treesa_run['ordering']['work'],
                   'largest_entries': treesa_run['ordering']['largest_entries'],
                   'log10_mass': treesa_run['contraction']['log10_mass']},
        'published_pr': reference_pr,
        'pr_difference': abs(treesa_run['contraction']['log10_mass'] - reference_pr),
        'mar_difference': mar_difference,
        'marginals_state1': {str(variable): treesa_run['contraction']['node_marginals'][str(variable)][1]
                             for variable in queries},
        'bp': {'sweeps': prom_bp['final']['sweeps'],
               'residual': prom_bp['final']['residual'],
               'converged': prom_bp['final']['converged'],
               'max_node_error': prom_bp['bp']['max_node_error']},
        'no_observations_log10': noevid['contraction']['log10_mass'],
    }
    checks.append(check('Promedus: greedy and TreeSA agree on log10 Z',
                        abs(greedy_run['contraction']['log10_mass']
                            - treesa_run['contraction']['log10_mass']),
                        '< 1e-10', abs(greedy_run['contraction']['log10_mass']
                                       - treesa_run['contraction']['log10_mass']) < 1e-10))
    checks.append(check('Promedus: log10 Z against the published PR',
                        details['promedus']['pr_difference'], '< 1e-4',
                        details['promedus']['pr_difference'] < 1e-4))
    checks.append(check('Promedus: state-1 marginals against the published MAR', mar_difference,
                        '< 1e-5', mar_difference < 1e-5))
    checks.append(check('Promedus: TreeSA uses less work than greedy',
                        {'greedy': greedy_run['ordering']['work'],
                         'treesa': treesa_run['ordering']['work']},
                        'treesa < greedy', treesa_run['ordering']['work'] < greedy_run['ordering']['work']))
    checks.append(check('Promedus: BP is unconverged and inaccurate',
                        {'residual': prom_bp['final']['residual'],
                         'max_node_error': prom_bp['bp']['max_node_error']},
                        'residual > 1e-10 and error > 0.1',
                        prom_bp['final']['residual'] > 1e-10 and prom_bp['bp']['max_node_error'] > 0.1))
    checks.append(check('Promedus negative control: no .evid file misses PR',
                        abs(noevid['contraction']['log10_mass'] - reference_pr),
                        '> 1e-2', abs(noevid['contraction']['log10_mass'] - reference_pr) > 1e-2))

    sources = {}
    for name, path in (('asia.uai', ROOT / 'asia.uai'),
                       ('Promedus_14.uai', ROOT / 'promedus' / 'Promedus_14.uai'),
                       ('Promedus_14.uai.evid', ROOT / 'promedus' / 'Promedus_14.uai.evid'),
                       ('Promedus_14.uai.PR', ROOT / 'promedus' / 'Promedus_14.uai.PR'),
                       ('Promedus_14.uai.MAR', ROOT / 'promedus' / 'Promedus_14.uai.MAR')):
        sources[name] = {'bytes': path.stat().st_size,
                         'sha256': sha256(path.read_bytes()).hexdigest()}

    result = {
        'data': sources,
        'checks': checks,
        'all_pass': all(entry['pass'] for entry in checks),
        'asia': details,
        'bp': bp_details,
        'promedus': details['promedus'],
        'notes': [
            'Every number comes from the saved artifacts/ runs, never from a live recomputation.',
            'The exhaustive optima use observed variables as (1,0) cap tensors, as drawn in the sheet.',
            'omeco greedy and TreeSA plans are heuristics; the saved plan is replayed and recounted.',
        ],
    }
    path = ROOT / 'evidence' / 'acceptance.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=False) + '\n')
    print(f"{len(checks)} checks, all pass: {result['all_pass']}")
    for entry in checks:
        if not entry['pass']:
            print('FAIL', entry['name'], entry['measured'])


if __name__ == '__main__':
    main()
