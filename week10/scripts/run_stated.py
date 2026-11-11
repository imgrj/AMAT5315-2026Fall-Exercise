"""Run the sheet's two stated cases and the Part 1 Gaussian-matrix evidence.

Run from week10/:
    python scripts/run_stated.py      # runs/boundary64 and runs/trg1024
    python scripts/run_svd.py         # runs/svd
"""
from __future__ import annotations

import _common as common

SVD_RANKS = '16,64,128,256,512'


def stated() -> None:
    cases = [
        ('boundary', 64, 0.45, 16),
        ('trg', 1024, 0.44, 16),
    ]
    for method, length, k, chi in cases:
        out = common.RUNS / ('boundary64' if method == 'boundary' else 'trg1024')
        record = common.run(common.command(method, out, l=length, k=k, chi=chi))
        print(f"{method} L={length} K={k} chi={chi}: g={record['g']:.12f} "
              f"logZ={record['log_z']:.6f} t={record['solver_time_s']:.3f}s -> {out}")


def svd() -> None:
    out = common.RUNS / 'svd'
    record = common.run(common.command('svd', out, l=512, seed=2026, ranks=SVD_RANKS))
    print(f"svd 512x512 seed 2026: t={record['solver_time_s']:.3f}s, "
          f"10% crossing chi={record['ten_percent_crossing']['rank']}")
    for entry in record['reconstructions']:
        print(f"  chi={entry['rank']:4d} measured={entry['relative_frobenius_error']:.12f} "
              f"prediction={entry['discarded_spectrum_prediction']:.12f} "
              f"storage={entry['storage_ratio']:.4f}x")


def main() -> None:
    stated()
    svd()


if __name__ == '__main__':
    main()
