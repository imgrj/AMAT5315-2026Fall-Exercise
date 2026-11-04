"""evidence/orders.png: the two saved contraction trees and their costs.

Run from week9/: python scripts/plot_orders.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from _common import ARTIFACTS, COLORS, plan_leaves, read_json, save, style

CASES = ('random', 'greedy')


class Node:
    def __init__(self, identifier, labels, work=0.0, entries=1, leaf=None, left=None, right=None):
        self.id = identifier
        self.labels = labels
        self.work = work
        self.entries = entries
        self.leaf = leaf
        self.left = left
        self.right = right


def load(method: str):
    run = read_json(ARTIFACTS / 'runs' / method / 'run.json')
    order = read_json(ARTIFACTS / 'runs' / method / 'order.json')
    names = plan_leaves(run)
    factors = run['model']['factors']
    observed = sorted(int(variable) for variable in run['evidence']['observations'])
    nodes = {}
    for index, factor in enumerate(factors):
        nodes[index] = Node(index, factor['scope'], leaf=names[index])
    for offset, variable in enumerate(observed):
        index = len(factors) + offset
        nodes[index] = Node(index, [variable], leaf=names[index])
    for step in order['steps']:
        nodes[step['id']] = Node(step['id'], step['labels'], step['work'], step['entries'],
                                 left=nodes[step['left']], right=nodes[step['right']])
    root = nodes[max(nodes)]
    return run, order, root, names


def layout(node, depth, counter, positions, depths):
    if node.leaf is not None:
        positions[node.id] = counter[0]
        depths[node.id] = depth
        counter[0] += 1
        return
    layout(node.left, depth + 1, counter, positions, depths)
    layout(node.right, depth + 1, counter, positions, depths)
    positions[node.id] = (positions[node.left.id] + positions[node.right.id]) / 2
    depths[node.id] = depth


def draw(axis, root, names, colour, title, note):
    positions, depths = {}, {}
    layout(root, 0, [0], positions, depths)
    largest = max(((node.entries, node.id) for node in walk(root)))
    for node in walk(root):
        x, y = positions[node.id], -depths[node.id]
        for child in (node.left, node.right):
            if child is not None:
                axis.plot([x, positions[child.id]], [y - 0.18, -depths[child.id] + 0.18],
                          color='#3b4a63', linewidth=1.1, zorder=1)
    for node in walk(root):
        x, y = positions[node.id], -depths[node.id]
        if node.leaf is not None:
            axis.add_patch(FancyBboxPatch((x - 0.42, y - 0.16), 0.84, 0.32,
                                          boxstyle='round,pad=0.02', facecolor='#16243b',
                                          edgecolor='#4b6390', linewidth=1.1, zorder=2))
            axis.text(x, y, node.leaf, ha='center', va='center', fontsize=7.6, color='#dce6f2',
                      zorder=3)
            axis.text(x, y - 0.27, f'{node.entries}', ha='center', va='center', fontsize=6.2,
                      color='#7f93ad', zorder=3)
        else:
            biggest = (node.entries, node.id) == largest
            axis.add_patch(FancyBboxPatch((x - 0.38, y - 0.15), 0.76, 0.3,
                                          boxstyle='round,pad=0.02',
                                          facecolor='#7a2c33' if biggest else '#0f172a',
                                          edgecolor='#f7768e' if biggest else colour,
                                          linewidth=1.6 if biggest else 1.0, zorder=2))
            axis.text(x, y, ','.join(names_of(node.labels)), ha='center', va='center',
                      fontsize=6.8, color='#dce6f2', zorder=3)
            axis.text(x, y + 0.23, f'{node.work:,.0f}', ha='center', va='bottom', fontsize=6.2,
                      color='#9aaec5', zorder=3)
    axis.set_title(title, loc='left', color='#dce6f2', fontsize=10)
    axis.text(0.0, -max(depths.values()) - 0.85, note, fontsize=8.2, color='#9aaec5')
    axis.set_ylim(-max(depths.values()) - 1.1, 1.0)
    axis.set_xlim(-0.6, max(positions.values()) + 0.6)
    axis.axis('off')


def names_of(labels) -> list[str]:
    letters = 'ATSLBEXD'
    return [letters[label] for label in labels]


def walk(node):
    yield node
    for child in (node.left, node.right):
        if child is not None:
            yield from walk(child)


def main() -> None:
    style()
    figure, axes = plt.subplots(2, 1, figsize=(11.2, 8.4))
    summary = []
    for axis, method in zip(axes, CASES):
        run, order, root, names = load(method)
        complexity = read_json(ARTIFACTS / 'runs' / method / 'complexity.json')
        largest = max((node.entries, node.id) for node in walk(root))[0]
        note = (f'work {order["work"]:,.0f} terms  ·  largest tensor {order["largest_entries"]:,} entries  ·  '
                f'mass {run["contraction"]["mass"]:.15f}  ·  {len(order["steps"])} pair steps')
        draw(axis, root, names, COLORS[method],
             f'{method} order (seed {order.get("seed")})' if order.get('seed') else f'{method} order',
             note)
        summary.append((method, order['work'], order['largest_entries']))
    figure.suptitle('Contraction trees for the same AD evidence query: the answer is identical, '
                    'the cost is not\n'
                    'internal nodes keep the labels still needed; the highlighted node is the '
                    'largest intermediate (entry counts under the leaves)',
                    color='#dce6f2', fontsize=9.5)
    save(figure, 'orders.png')
    for method, work, entries in summary:
        print(f'{method}: work {work:,.0f}, largest {entries:,}')


if __name__ == '__main__':
    main()
