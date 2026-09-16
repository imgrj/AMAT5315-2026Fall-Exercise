import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def load_data(base_dir, size):
    coarse_path = os.path.join(base_dir, "artifacts", f"coarse-l{size}", "series.jsonl")
    window_path = os.path.join(base_dir, "artifacts", f"window-l{size}", "series.jsonl")

    # In case of overlap, window run (longer statistics) overrides coarse run
    t_dict = {}
    if os.path.exists(coarse_path):
        with open(coarse_path, 'r') as f:
            for line in f:
                if not line.strip(): continue
                d = json.loads(line)
                t = round(float(d['T']), 4)
                if t not in t_dict: t_dict[t] = []
                t_dict[t].append(float(d['M']))

    if os.path.exists(window_path):
        # replace window temperatures with higher-stat runs
        w_dict = {}
        with open(window_path, 'r') as f:
            for line in f:
                if not line.strip(): continue
                d = json.loads(line)
                t = round(float(d['T']), 4)
                if t not in w_dict: w_dict[t] = []
                w_dict[t].append(float(d['M']))
        for t, m_list in w_dict.items():
            t_dict[t] = m_list

    temps = sorted(t_dict.keys())
    chi_list = []
    mean_abs_m_list = []

    for t in temps:
        m_arr = np.array(t_dict[t])
        mean_m2 = np.mean(m_arr**2)
        mean_abs_m = np.mean(np.abs(m_arr))
        chi = (size**2) * (mean_m2 - mean_abs_m**2) / t
        chi_list.append(chi)
        mean_abs_m_list.append(mean_abs_m)

    return np.array(temps), np.array(chi_list), np.array(mean_abs_m_list)

def fit_peak(temps, chis):
    idx_max = np.argmax(chis)
    start = max(0, idx_max - 2)
    end = min(len(temps), start + 5)
    if end - start < 5:
        start = max(0, end - 5)
    t_5 = temps[start:end]
    chi_5 = chis[start:end]

    # Quadratic fit: chi = a*T^2 + b*T + c
    poly = np.polyfit(t_5, chi_5, 2)
    a, b, c = poly
    t_peak = -b / (2.0 * a)
    chi_peak = np.polyval(poly, t_peak)
    return t_peak, chi_peak, poly, t_5, chi_5

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    evidence_dir = os.path.join(base_dir, "evidence")
    os.makedirs(evidence_dir, exist_ok=True)

    t32, chi32, m32 = load_data(base_dir, 32)
    t64, chi64, m64 = load_data(base_dir, 64)

    t_peak_32, chi_peak_32, poly32, t5_32, chi5_32 = fit_peak(t32, chi32)
    t_peak_64, chi_peak_64, poly64, t5_64, chi5_64 = fit_peak(t64, chi64)

    # Extrapolate Tc
    tc_extrap = 2.0 * t_peak_64 - t_peak_32
    tc_exact = 2.0 / np.log(1.0 + np.sqrt(2.0))
    deviation_pct = (tc_extrap - tc_exact) / tc_exact * 100.0

    cold_m32 = m32[0] # at T=1.5
    cold_m64 = m64[0]

    report = f"""L = 32:
  cold mean |M| = {cold_m32:.4f}
  fitted peak T_peak = {t_peak_32:.4f}
L = 64:
  cold mean |M| = {cold_m64:.4f}
  fitted peak T_peak = {t_peak_64:.4f}

Extrapolated T_c = {tc_extrap:.4f} (Onsager exact = {tc_exact:.5f}, deviation = {deviation_pct:+.2f}%)
Pass condition (< 2%): {abs(deviation_pct) < 2.0}
"""
    print(report)

    peaks_txt_path = os.path.join(evidence_dir, "peaks.txt")
    with open(peaks_txt_path, 'w') as f:
        f.write(report)
    print(f"Saved {peaks_txt_path}")

    # Plot susceptibility
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=200)
    ax.plot(t32, chi32, 'o-', color='#1f77b4', ms=4, lw=1.2, label=f'$L = 32$, peak {t_peak_32:.4f}')
    ax.plot(t64, chi64, 's-', color='#ff7f0e', ms=4, lw=1.2, label=f'$L = 64$, peak {t_peak_64:.4f}')

    # Fit curves
    t_fit32 = np.linspace(t5_32[0] - 0.02, t5_32[-1] + 0.02, 100)
    t_fit64 = np.linspace(t5_64[0] - 0.02, t5_64[-1] + 0.02, 100)
    ax.plot(t_fit32, np.polyval(poly32, t_fit32), ':', color='#1f77b4', lw=1.5)
    ax.plot(t_fit64, np.polyval(poly64, t_fit64), ':', color='#ff7f0e', lw=1.5)

    ax.axvline(tc_exact, color='k', ls='--', lw=1.2, label=f'$T_c = {tc_exact:.4f}$')
    ax.axvline(t_peak_32, color='#1f77b4', ls=':', lw=1.0)
    ax.axvline(t_peak_64, color='#ff7f0e', ls=':', lw=1.0)

    ax.set_xlabel('temperature $T$', fontsize=11)
    ax.set_ylabel(r'$\chi(T)$', fontsize=11)
    ax.set_xlim(2.0, 2.75)
    ax.legend(frameon=True, fontsize=9.5)
    ax.grid(True, alpha=0.3, ls=':')

    plt.tight_layout()
    susc_png_path = os.path.join(evidence_dir, "susceptibility.png")
    plt.savefig(susc_png_path)
    plt.close()
    print(f"Saved {susc_png_path}")

if __name__ == '__main__':
    main()
