"""Part 1 figure: singular spectrum and relative error against retained rank.

Run from week10/: python scripts/plot_svd.py   ->   artifacts/svd.png
"""
from __future__ import annotations

import numpy as np

import _common as common

import matplotlib.pyplot as plt


def main() -> None:
    common.style()
    run = common.RUNS / 'svd'
    record = common.read_json(run / 'run.json')
    singular = np.load(run / 'singular-values.npy')
    total = float(np.sum(singular ** 2))
    ranks = np.arange(1, len(singular) + 1)
    discarded = np.sqrt(np.cumsum(singular[::-1] ** 2)[::-1])
    prediction = np.zeros(len(singular) + 1)
    prediction[1:] = np.concatenate([[np.sqrt(total)], discarded[:-1]]) / np.sqrt(total)
    prediction = prediction[1:]  # prediction[chi-1] = error of the rank-chi truncation

    figure, (spectrum, errors) = plt.subplots(1, 2, figsize=(11.2, 4.3))

    spectrum.plot(ranks, singular, color=common.COLORS['measured'], linewidth=1.1)
    spectrum.set_xlabel('singular-value index')
    spectrum.set_ylabel('singular value')
    spectrum.set_title('(a) Singular spectrum of the seeded Gaussian matrix')
    spectrum.grid(alpha=0.35)
    spectrum.set_xlim(0, len(singular))
    spectrum.text(0.03, 0.06, f'{record["geometry"]["m"]}x{record["geometry"]["n"]}, '
                   f'seed {record["generator"]["seed"]}, '
                   f'mean {record["sample_mean"]:+.4f}, sd {record["sample_standard_deviation"]:.4f}',
                   transform=spectrum.transAxes, color='#9aaec5', fontsize=8.5)

    measured = [max(entry['relative_frobenius_error'], 1e-16)
                for entry in record['reconstructions']]
    saved_ranks = [entry['rank'] for entry in record['reconstructions']]
    errors.semilogy(ranks, np.maximum(prediction, 1e-16), color=common.COLORS['prediction'],
                    linewidth=1.1, label='discarded-spectrum prediction')
    errors.semilogy(saved_ranks, measured, 'o', color=common.COLORS['measured'],
                    markersize=6, label='measured from the saved matrices')
    crossover = record['storage_crossover']['rank']
    crossing = record['ten_percent_crossing']
    errors.axvline(crossover, color=common.COLORS['threshold'], linestyle=':', linewidth=1.2)
    errors.axhline(0.1, color=common.COLORS['threshold'], linestyle='--', linewidth=1.0)
    errors.axvline(crossing['rank'], color=common.COLORS['onsager'], linestyle='-.', linewidth=1.0)
    errors.annotate('no storage saving\nfor $\\chi > 256$', xy=(crossover, 0.03),
                    xytext=(0.55, 0.62), textcoords='axes fraction',
                    color=common.COLORS['threshold'], fontsize=8.5,
                    arrowprops={'arrowstyle': '->', 'color': common.COLORS['threshold'], 'lw': 0.9})
    errors.annotate(f"10% threshold: $\\chi = {crossing['rank']}$\n"
                    f"({crossing['storage_ratio']:.2f}$\\times$ the original storage)",
                    xy=(crossing['rank'], 0.1), xytext=(0.30, 0.06), textcoords='axes fraction',
                    color='#dce6f2', fontsize=8.5,
                    arrowprops={'arrowstyle': '->', 'color': '#9aaec5', 'lw': 0.9})
    errors.set_xlabel('retained rank $\\chi$')
    errors.set_ylabel('relative Frobenius error $e_\\chi$')
    errors.set_title('(b) Error against retained rank')
    errors.set_xlim(0, len(singular))
    errors.set_ylim(1e-16, 2.0)
    errors.grid(alpha=0.35, which='both')
    errors.legend(loc='center left', fontsize=8.5, framealpha=0.9)

    e16 = record['reconstructions'][0]
    e256 = record['reconstructions'][3]
    errors.text(0.03, 0.93,
                f"$e_{{16}}$ = {e16['relative_frobenius_error']:.4f}   "
                f"$e_{{256}}$ = {e256['relative_frobenius_error']:.4f}   "
                f"$e_{{512}}$ < 1e-15",
                transform=errors.transAxes, color='#dce6f2', fontsize=8.0)

    common.save(figure, common.ARTIFACTS / 'svd.png')
    print(f"  measurement-minus-prediction: "
          f"{max(abs(entry['difference']) for entry in record['reconstructions']):.2e}")


if __name__ == '__main__':
    main()
