import json
import os
import numpy as np
import matplotlib.pyplot as plt

def load_energies(jsonl_path):
    energies = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            energies.append(data['E'] * 4096.0)
    return np.array(energies)

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    t30_path = os.path.join(base_dir, "runs", "T3.0", "series.jsonl")
    t31_path = os.path.join(base_dir, "runs", "T3.1", "series.jsonl")
    evidence_dir = os.path.join(base_dir, "evidence")
    os.makedirs(evidence_dir, exist_ok=True)
    out_png = os.path.join(evidence_dir, "boltzmann.png")

    e30 = load_energies(t30_path)
    e31 = load_energies(t31_path)

    bin_width = 40.0
    min_e = min(e30.min(), e31.min())
    max_e = max(e30.max(), e31.max())

    bin_start = np.floor(min_e / bin_width) * bin_width
    bin_end = np.ceil(max_e / bin_width) * bin_width
    bins = np.arange(bin_start, bin_end + bin_width, bin_width)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0

    counts30, _ = np.histogram(e30, bins=bins)
    counts31, _ = np.histogram(e31, bins=bins)

    valid = (counts30 >= 5) & (counts31 >= 5)
    valid_centers = bin_centers[valid]
    log_ratios = np.log(counts31[valid].astype(float) / counts30[valid].astype(float))

    slope = 1.0 / 3.0 - 1.0 / 3.1 # 0.010752688...

    # Compute intercept that best aligns with the data
    # ln(P3.1/P3.0) = slope * E + const
    const = np.mean(log_ratios - slope * valid_centers)
    theory_line = slope * valid_centers + const

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=200)

    # Left: Histograms
    # Step plot for histogram outlines
    ax1.step(bin_centers, counts30, where='mid', label=r'$T = 3.0$', color='#1f77b4', lw=1.5)
    ax1.step(bin_centers, counts31, where='mid', label=r'$T = 3.1$', color='#d62728', lw=1.5)
    ax1.set_xlabel('total energy $E$', fontsize=11)
    ax1.set_ylabel('sweeps in bin', fontsize=11)
    ax1.legend(frameon=True, fontsize=10)
    ax1.grid(True, alpha=0.3, ls=':')

    # Right: Log ratio
    ax2.plot(valid_centers, log_ratios, 'ko', ms=4, label='measured ratio')
    ax2.plot(valid_centers, theory_line, 'k--', lw=1.5, label=f'slope = {slope:.5f}')
    ax2.set_xlabel('total energy $E$', fontsize=11)
    ax2.set_ylabel(r'$\ln(P_{3.1} / P_{3.0})$', fontsize=11)
    ax2.legend(frameon=True, fontsize=10)
    ax2.grid(True, alpha=0.3, ls=':')

    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"Saved {out_png} successfully.")

if __name__ == '__main__':
    main()
