"""Run the small ground checks whose exact answers are known by enumeration.

Run from week10/:
    python scripts/run_checks.py

Writes `artifacts/tables/small-checks.json` with every g value that
`scripts/verify_results.py` and the pytest suite compare against an independent
sum over spin configurations.
"""
from __future__ import annotations

import _common as common

STENCIL = 0.0025
COUPLING = 0.44


def main() -> None:
    table: dict = {
        'stencil_h': STENCIL,
        'coupling': COUPLING,
        'boundary': {'open_4x4_chi8': [], 'zero_coupling': []},
        'trg': {'periodic_2x2_chi16': [], 'periodic_4x4_chi8': [], 'zero_coupling': []},
    }
    commands = []
    order = []

    # Open 4x4 at chi = 8: the sheet's three couplings plus the c_V stencil.
    for k in (0.0, COUPLING - STENCIL, COUPLING, COUPLING + STENCIL, 0.7):
        order.append(('boundary', 'open_4x4_chi8', 4, 8, k))
    # Open squares at K = 0: every configuration has equal weight, so g = log 2.
    for length in (8, 16):
        order.append(('boundary', 'zero_coupling', length, 8, 0.0))
    # Periodic 2x2 at chi = 16 and 4x4 at chi = 8.
    for k in (0.0, COUPLING - STENCIL, COUPLING, COUPLING + STENCIL):
        order.append(('trg', 'periodic_2x2_chi16', 2, 16, k))
    for k in (0.0, COUPLING):
        order.append(('trg', 'periodic_4x4_chi8', 4, 8, k))
    for length in (8, 16):
        order.append(('trg', 'zero_coupling', length, 8, 0.0))

    for method, _, length, chi, k in order:
        commands.append(common.command(
            method, common.run_path('checks', method, length, chi, k),
            l=length, k=f'{k:.6f}', chi=chi))
    runs = common.run_many(commands)
    for (method, group, length, chi, k), record in zip(order, runs):
        table[method][group].append({
            'l': length,
            'chi': chi,
            'k': k,
            'g': record['g'],
            'log_z': record['log_z'],
            'solver_seconds': record['solver_time_s'],
            'boundary_condition': record['geometry']['boundary'],
            'out': str(common.run_path('checks', method, length, chi, k).relative_to(common.ROOT)),
        })
        print(f"{method} {group} L={length} chi={chi} K={k}: g={record['g']:.15f}")
    common.write_json(common.TABLES / 'small-checks.json', table)


if __name__ == '__main__':
    main()
