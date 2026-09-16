import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def onsager_magnetization(T):
    Tc = 2.0 / np.log(1.0 + np.sqrt(2.0))
    res = np.zeros_like(T)
    mask = T < Tc
    sinh_val = np.sinh(2.0 / T[mask])
    res[mask] = (1.0 - sinh_val**(-4))**(1.0 / 8.0)
    return res, Tc

def load_metropolis_m(base_dir, size):
    # Merge coarse and window runs
    coarse_path = os.path.join(base_dir, "artifacts", f"coarse-l{size}", "series.jsonl")
    window_path = os.path.join(base_dir, "artifacts", f"window-l{size}", "series.jsonl")

    t_data = {}

    for path in [coarse_path, window_path]:
        if not os.path.exists(path): continue
        with open(path, 'r') as f:
            for line in f:
                if not line.strip(): continue
                d = json.loads(line)
                t = round(float(d['T']), 4)
                m = abs(float(d['M']))
                if t not in t_data:
                    t_data[t] = []
                t_data[t].append(m)

    temps = sorted(t_data.keys())
    means = [np.mean(t_data[t]) for t in temps]
    return np.array(temps), np.array(means)

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    evidence_dir = os.path.join(base_dir, "evidence")
    os.makedirs(evidence_dir, exist_ok=True)
    out_png = os.path.join(evidence_dir, "magnetization.png")

    t_64, m_64 = load_metropolis_m(base_dir, 64)

    t_fine = np.linspace(1.5, 3.5, 500)
    m_exact, Tc = onsager_magnetization(t_fine)

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=200)
    ax.plot(t_fine, m_exact, 'k-', lw=1.5, label='Onsager, infinite lattice')
    ax.plot(t_64, m_64, 's-', color='#1f77b4', ms=4, lw=1.2, label='measured, $L = 64$')
    ax.axvline(Tc, color='k', ls='--', lw=1.2, label=f'$T_c = {Tc:.4f}$')

    ax.set_xlabel('temperature $T$', fontsize=11)
    ax.set_ylabel(r'$\langle |m| \rangle$', fontsize=11)
    ax.set_xlim(1.4, 3.6)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3, ls=':')

    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"Saved {out_png}")

if __name__ == '__main__':
    main()
