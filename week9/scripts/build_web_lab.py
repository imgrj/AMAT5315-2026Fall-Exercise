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
    ('graphs.svg', '2 / The AD query as a tensor diagram', 'plot_graphs.py',
     'The AD tuberculosis query drawn as a tensor network. Each factor is a box whose legs are the axes of its '
     'UAI table; a label is one shared index, joined in every table that uses it. The visit to Asia and the '
     'dyspnoea observation carry (1,0) caps, smoking joins p(s), p(l|s) and p(b|s), and only T stays open, so '
     'the same picture answers p(T | A=0, D=0). Replacing both caps by (1,1) closes the network and returns the '
     'evidence mass 0.004501375.'),
    ('posteriors.png', '2 / Priors and posteriors agree with enumeration', 'plot_posteriors.py',
     'Exact node marginals for the three patient cases: the bars are one tensor contraction and the white ticks '
     'are the independent 256-row enumeration. A positive X-ray raises tuberculosis from 0.0878 to 0.3917 and '
     'lung cancer from 0.0995 to 0.4443 while bronchitis falls from 0.8114 to 0.6288. The largest disagreement '
     'between the two methods is 2.2e-16, and every marginal sums to one.'),
    ('joint.png', '2 / The joint keeps the dependence', 'plot_posteriors.py',
     'p(T,L) beside the product of its marginals, and their difference. In ADX the joint gives '
     'p(T=0,L=0) = 0.022214 while the product gives 0.174026, a dependence of -0.151812 that no product of '
     'marginals can represent; the prior, where tuberculosis and lung cancer are independent, shows zero to '
     'machine precision. This is why a joint marginal keeps a shared cause or a shared effect.'),
    ('orders.png', '3 / Two saved plans, one answer', 'plot_orders.py',
     'The random (seed 42) and greedy contraction trees for the same AD evidence query. An internal node keeps '
     'the labels that another tensor or the output still needs and sums the rest; the number above a node is '
     "that step's work, the number under a leaf is its entry count, and the highlighted node is the largest "
     'intermediate. Random costs 204 terms with a 32-entry tensor, greedy 52 with 8, and both return '
     'p(A=0,D=0) = 0.004501375.'),
    ('bp-convergence.png', '4 / The messages stop changing', 'plot_bp.py',
     'Maximum undamped message change max|F(m) - m| per synchronous damped sweep (damping 0.5). All three cases '
     'reach the 1e-10 stop tolerance: prior in 54 sweeps, AD in 93 and ADX in 94, where the residual is '
     '8.23e-11. The stop test uses this undamped residual, not the damped step size.'),
    ('bp-accuracy.png', '4 / Converged is not accurate', 'plot_bp.py',
     'Maximum belief-minus-exact marginal error over the same sweeps. On ASIA loops the error settles above '
     'zero: 0.00334 for prior, 0.01141 for AD and 0.01061 for ADX, so a small residual does not certify '
     'accurate marginals. On the committed nonuniform tree the same code is exact to 8.3e-16, which separates '
     'a correct implementation from the approximation error of correct loopy BP.'),
    ('promedus-orders.png', '5 / Promedus plan costs', 'plot_promedus.py',
     'Exact contraction work and peak intermediate for the two saved plans on the 414-variable benchmark. '
     'Time-only TreeSA uses 11,415,062 terms with a 131,072-entry largest tensor, against greedy 383,630,739 '
     'and 1,048,576. Both plans return log10 Z_evidence = -7.80674630458244, 3.7e-6 from the published PR, '
     'and the same three state-ordered marginals.'),
    ('promedus-bp.png', '5 / Promedus is unconverged and inaccurate', 'plot_promedus.py',
     'Loopy belief propagation with the ASIA settings on Promedus_14. After the 1000-sweep limit the residual '
     'is still 2.17e-10, above the 1e-10 tolerance, and the state-1 beliefs of IDs 59, 102 and 139 are off by '
     '0.1975, 0.1495 and 0.3824. It is reported as unconverged and inaccurate for these queries rather than '
     'tuned to look converged.'),
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
    for name, _, _, _ in GALLERY:
        shutil.copyfile(EVIDENCE / name, assets / name)
    template = (ROOT / 'web' / 'lab_template.html').read_text(encoding='utf-8')
    rendered = template.replace('__DATA__', json.dumps(data, ensure_ascii=True).replace('</', '<\\/'))
    (docs / 'week9-inference-lab.html').write_text(rendered, encoding='utf-8')
    print(f'built docs/week9-inference-lab.html with {len(GALLERY)} evidence figures')


if __name__ == '__main__':
    main()
