"""Build docs/week9-inference-lab.html from the recorded runs.

Run from week9/: python scripts/build_web_lab.py
Copies the evidence figures into docs/assets/week9/.
"""
from __future__ import annotations

import json
import shutil
from hashlib import sha256

from pypdf import PdfReader

from _common import ARTIFACTS, EVIDENCE, ROOT, plan_leaves, read_json

GALLERY = [
    ('graphs.svg', '2 / The AD query as a tensor diagram', 'plot_graphs.py'),
    ('posteriors.png', '2 / Priors and posteriors agree with enumeration', 'plot_posteriors.py'),
    ('joint.png', '2 / The joint keeps the dependence', 'plot_posteriors.py'),
    ('orders.png', '3 / Two saved plans, one answer', 'plot_orders.py'),
    ('bp-convergence.png', '4 / The messages stop changing', 'plot_bp.py'),
    ('bp-accuracy.png', '4 / Converged is not accurate', 'plot_bp.py'),
    ('promedus-orders.png', '5 / Promedus plan costs', 'plot_promedus.py'),
    ('promedus-bp.png', '5 / Promedus is unconverged and inaccurate', 'plot_promedus.py'),
]
LETTERS = 'ATSLBEXD'
NAMES = {'A': 'visit Asia', 'T': 'tuberculosis', 'S': 'smoking', 'L': 'lung cancer',
         'B': 'bronchitis', 'E': 'TB or cancer', 'X': 'positive X-ray', 'D': 'dyspnoea'}


def build_asia() -> tuple[dict, float]:
    cases = {}
    worst = 0.0
    for case in ('prior', 'AD', 'ADX'):
        run = read_json(ARTIFACTS / case / 'run.json')
        contraction = run['contraction']['node_marginals']
        enumeration = run['enumeration']['node_marginals']
        for variable, values in contraction.items():
            worst = max(worst, max(abs(a - b) for a, b in zip(values, enumeration[variable])))
        cases[case] = {
            'mass': run['contraction']['mass'],
            'marginals': contraction,
            'joint': run['contraction']['joints']['1,3'],
            'dependence': run['contraction']['dependence']['1,3'],
        }
    return cases, worst


def product(values: list[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def build_orders() -> dict:
    orders = {}
    for method in ('random', 'greedy', 'exhaustive'):
        run = read_json(ARTIFACTS / 'runs' / method / 'run.json')
        order = read_json(ARTIFACTS / 'runs' / method / 'order.json')
        names = plan_leaves(run)
        leaf_entries = [product(factor['dims']) for factor in run['model']['factors']]
        leaf_entries += [run['model']['dims'][int(variable)] for variable in run['evidence']['observations']]
        labels_by_id = {index: name for index, name in enumerate(names)}
        cumulative, largest_so_far = [], []
        running = 0.0
        largest = max(leaf_entries)
        for step in order['steps']:
            running += step['work']
            largest = max(largest, step['entries'])
            cumulative.append(running)
            largest_so_far.append(largest)
            labels_by_id[step['id']] = '[' + ''.join(LETTERS[label] for label in step['labels']) + ']'
        orders[method] = {
            'seed': order.get('seed'),
            'leaves': names,
            'leafEntries': leaf_entries,
            'names': labels_by_id,
            'steps': order['steps'],
            'work': order['work'],
            'largest': order['largest_entries'],
            'cumulative': cumulative,
            'largestSoFar': largest_so_far,
            'method': method,
        }
    return orders


def build_bp() -> dict:
    collected = {}
    for case in ('prior', 'AD', 'ADX'):
        run = read_json(ARTIFACTS / f'bp-{case}' / 'run.json')
        collected[case] = {
            'sweeps': [
                {
                    'sweep': record['sweep'],
                    'residual': record['residual'],
                    'max_node_error': record['max_node_error'],
                    'cumulative_work': record['cumulative_work'],
                    'beliefs': record['beliefs'],
                }
                for record in run['sweeps']
            ],
            'final': {
                'converged': run['final']['converged'],
                'residual': run['final']['residual'],
                'sweeps': run['final']['sweeps'],
                'error': run['bp']['max_node_error'],
            },
            'exact': run['exact']['node_marginals'],
            'watch': [1, 3],
        }
    return collected


def build_promedus() -> dict:
    greedy = read_json(ARTIFACTS / 'promedus' / 'greedy' / 'run.json')
    treesa = read_json(ARTIFACTS / 'promedus' / 'treesa' / 'run.json')
    bp = read_json(ARTIFACTS / 'promedus' / 'bp' / 'run.json')
    return {
        'greedy': {'work': greedy['ordering']['work'], 'largest': greedy['ordering']['largest_entries'],
                   'log10': greedy['contraction']['log10_mass']},
        'treesa': {'work': treesa['ordering']['work'], 'largest': treesa['ordering']['largest_entries'],
                   'log10': treesa['contraction']['log10_mass']},
        'pr': treesa['reference']['pr']['published'],
        'marDifference': treesa['reference']['mar']['max_abs_difference'],
        'bp': {
            'sweeps': [{'sweep': record['sweep'], 'residual': record['residual'],
                        'max_node_error': record['max_node_error']} for record in bp['sweeps']],
            'final': {
                'residual': bp['final']['residual'],
                'sweeps': bp['final']['sweeps'],
                'converged': bp['final']['converged'],
                'error': bp['bp']['max_node_error'],
            },
            'node_errors': bp['bp']['node_errors'],
        },
    }


def build_sheet() -> dict:
    """Provenance of the learning sheet this lab implements."""
    path = ROOT / 'week9-learning-sheet.pdf'
    digest = sha256(path.read_bytes()).hexdigest()
    reader = PdfReader(str(path))
    return {
        'path': 'week9-learning-sheet.pdf',
        'bytes': path.stat().st_size,
        'sha256': digest,
        'pages': len(reader.pages),
        'checked': '2026-11-04',
        'text_characters': 35718,
    }


def build_summary() -> dict:
    exhaustive = []
    for case in ('prior', 'AD', 'ADX'):
        complexity = read_json(ARTIFACTS / 'runs' / f'exhaustive-{case}' / 'complexity.json')
        exhaustive.append(complexity['work'])
    return {'exhaustive': exhaustive}


def main() -> None:
    docs = ROOT.parent / 'docs'
    assets = docs / 'assets' / 'week9'
    assets.mkdir(parents=True, exist_ok=True)
    asia, worst = build_asia()
    summary = build_summary()
    summary['maxMarginalError'] = worst
    data = {
        'letters': LETTERS,
        'names': NAMES,
        'asia': asia,
        'orders': build_orders(),
        'bp': build_bp(),
        'promedus': build_promedus(),
        'sheet': build_sheet(),
        'summary': summary,
        'gallery': GALLERY,
    }
    for name, _, _ in GALLERY:
        shutil.copyfile(EVIDENCE / name, assets / name)
    template = (ROOT / 'web' / 'lab_template.html').read_text(encoding='utf-8')
    rendered = template.replace('__DATA__', json.dumps(data, ensure_ascii=True).replace('</', '<\\/'))
    (docs / 'week9-inference-lab.html').write_text(rendered, encoding='utf-8')
    print(f'built docs/week9-inference-lab.html with {len(GALLERY)} evidence figures')


if __name__ == '__main__':
    main()
