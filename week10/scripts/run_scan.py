"""Sweep `K`, refine every specific-heat peak and save the Week 10 tables.

Run from week10/:
    python scripts/run_scan.py --method boundary
    python scripts/run_scan.py --method trg

The sheet fixes the coarse grid (`K = 0.40 .. 0.50`, step 0.0025) and the
refinement (nine points spaced 0.00125 around the nearest grid point of each
peak, at chi = 16 and 32). The c_V stencil is h = 0.0025 on the coarse grid and
on the chi = 16 refinement, and h = 0.00125 on the chi = 32 refinement; every
other refined sample also gives the wider-step estimate at chi = 32.
"""
from __future__ import annotations

import argparse
import time

import _common as common
import peaks

GRID_STEP = 0.0025
COARSE_KS = [round(0.40 + GRID_STEP * index, 6) for index in range(41)]
REFINE_STEP = 0.00125
REFINE_OFFSETS = [round(REFINE_STEP * (index - 4), 6) for index in range(9)]
SIZES = {'boundary': [16, 32, 64], 'trg': [64, 256, 1024]}
BOUNDARY = {'boundary': 'open', 'trg': 'periodic'}


def sweep(method: str, chi: int, ks: dict[int, list[float]], group: str) -> dict:
    commands = []
    order = []
    for length in SIZES[method]:
        for k in ks[length]:
            order.append(length)
            commands.append(common.command(
                method, common.run_path(group, method, length, chi, k),
                l=length, k=f'{k:.6f}', chi=chi))
    started = time.perf_counter()
    runs = common.run_many(commands)
    elapsed = time.perf_counter() - started
    print(f'{method} chi={chi} {group}: {len(commands)} runs in {elapsed:.1f} s')
    collected: dict[str, dict] = {}
    for length in SIZES[method]:
        records = [record for record, owner in zip(runs, order) if owner == length]
        collected[str(length)] = {
            'g': [record['g'] for record in records],
            'solver_seconds': sum(record['solver_time_s'] for record in records),
        }
    return {'wall_seconds': elapsed, 'sizes': collected}


def peak_table(ks: list[float], values: list[float], h: float) -> dict:
    points = peaks.specific_heat(ks, values, h)
    return {
        'stencil_h': h,
        'points': [[k, value] for k, value in points],
        'peak': peaks.peak_from_points(points),
    }


def fit_summary(method: str, chi: int, variant: str, tpeaks: dict[int, float]) -> dict:
    sizes = SIZES[method]
    all_sizes = peaks.tpeak_fit(sizes, [tpeaks[length] for length in sizes])
    largest_two = peaks.tpeak_fit(sizes[1:], [tpeaks[length] for length in sizes[1:]])
    return {
        'method': method,
        'chi': chi,
        'variant': variant,
        'geometry': BOUNDARY[method],
        't_peaks': {str(length): tpeaks[length] for length in sizes},
        'all_sizes': all_sizes,
        'largest_two': largest_two,
        'onsager_t': peaks.ONSAGER_T,
        'onsager_relative_error_all': peaks.relative_difference(all_sizes['t_c'], peaks.ONSAGER_T),
        'onsager_relative_error_largest_two': peaks.relative_difference(
            largest_two['t_c'], peaks.ONSAGER_T),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['boundary', 'trg'], required=True)
    arguments = parser.parse_args()
    method = arguments.method
    geometry = BOUNDARY[method]
    sizes = SIZES[method]

    # --- coarse sweep at chi = 16 over the sheet's grid ---------------------
    coarse = sweep(method, 16, {length: COARSE_KS for length in sizes}, 'scan')
    scan = {
        'method': method,
        'boundary': geometry,
        'chi': 16,
        'stencil_h': GRID_STEP,
        'ks': COARSE_KS,
        'sizes': sizes,
        'g': {key: value['g'] for key, value in coarse['sizes'].items()},
        'specific_heat': {},
        'grid_peaks': {},
        'solver_seconds': {key: value['solver_seconds'] for key, value in coarse['sizes'].items()},
        'wall_seconds': coarse['wall_seconds'],
    }
    for length in sizes:
        table = peak_table(COARSE_KS, scan['g'][str(length)], GRID_STEP)
        scan['specific_heat'][str(length)] = table['points']
        scan['grid_peaks'][str(length)] = table['peak']
        print(f"  L={length:5d} grid peak K={table['peak']['grid_k']:.4f} "
              f"T={table['peak']['t_peak']:.5f}")
    common.write_json(common.TABLES / f'{method}-scan.json', scan)

    # --- refinement around each peak, at chi = 16 and 32 -------------------
    centres = {length: scan['grid_peaks'][str(length)]['grid_k'] for length in sizes}
    refinement = {
        'method': method,
        'boundary': geometry,
        'centres': {str(length): centres[length] for length in sizes},
        'refine_offsets': REFINE_OFFSETS,
        'variants': {},
        'runs': {},
    }
    refine_ks = {length: [round(centres[length] + offset, 6) for offset in REFINE_OFFSETS]
                 for length in sizes}
    for chi in (16, 32):
        stage = sweep(method, chi, refine_ks, 'refine')
        refinement['runs'][f'chi{chi}'] = {
            'wall_seconds': stage['wall_seconds'],
            'solver_seconds': {key: value['solver_seconds'] for key, value in stage['sizes'].items()},
            'g': {key: value['g'] for key, value in stage['sizes'].items()},
        }
        for h in ((0.0025,) if chi == 16 else (0.00125, 0.0025)):
            label = f'chi{chi}_h{h}'
            detail = {}
            tpeaks = {}
            for length in sizes:
                key = str(length)
                values = refinement['runs'][f'chi{chi}']['g'][key]
                table = peak_table(refine_ks[length], values, h)
                detail[key] = {
                    'ks': refine_ks[length],
                    'g': values,
                    'specific_heat': table['points'],
                    'peak': table['peak'],
                }
                tpeaks[length] = table['peak']['t_peak']
            refinement['variants'][label] = {
                'chi': chi,
                'stencil_h': h,
                'sizes': detail,
                'fit': fit_summary(method, chi, label, tpeaks),
            }

    primary = refinement['variants']['chi32_h0.00125']['fit']
    wide = refinement['variants']['chi32_h0.0025']['fit']
    low_rank = refinement['variants']['chi16_h0.0025']['fit']
    comparison = {
        'rank_shift': {
            str(length): peaks.relative_difference(
                refinement['variants']['chi32_h0.00125']['sizes'][str(length)]['peak']['t_peak'],
                refinement['variants']['chi16_h0.0025']['sizes'][str(length)]['peak']['t_peak'])
            for length in sizes
        },
        'step_shift': {
            str(length): peaks.relative_difference(
                refinement['variants']['chi32_h0.00125']['sizes'][str(length)]['peak']['t_peak'],
                refinement['variants']['chi32_h0.0025']['sizes'][str(length)]['peak']['t_peak'])
            for length in sizes
        },
        't_c_rank_shift': peaks.relative_difference(primary['all_sizes']['t_c'],
                                                    low_rank['all_sizes']['t_c']),
        't_c_step_shift': peaks.relative_difference(primary['all_sizes']['t_c'],
                                                    wide['all_sizes']['t_c']),
        'onsager_relative_error': primary['onsager_relative_error_all'],
        'onsager_relative_error_largest_two': primary['onsager_relative_error_largest_two'],
    }
    refinement['comparison'] = comparison
    common.write_json(common.TABLES / f'{method}-refine.json', refinement)

    print(f"{method}: T_c all sizes {primary['all_sizes']['t_c']:.6f}, "
          f"largest two {primary['largest_two']['t_c']:.6f}, Onsager {peaks.ONSAGER_T:.6f}")
    print(f"  rank shift {100 * comparison['t_c_rank_shift']:.3f}%, "
          f"step shift {100 * comparison['t_c_step_shift']:.3f}%")
    for length in sizes:
        peak = refinement['variants']['chi32_h0.00125']['sizes'][str(length)]['peak']
        print(f"  L={length:5d} refined T_peak={peak['t_peak']:.5f} at K={peak['grid_k']:.5f}")


if __name__ == '__main__':
    main()
