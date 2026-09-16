import json
import os
import numpy as np
import matplotlib.pyplot as plt

def load_data(series_path):
    t_dict = {}
    c_dict = {}
    with open(series_path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            d = json.loads(line)
            t = round(float(d['T']), 4)
            if t not in t_dict:
                t_dict[t] = []
                c_dict[t] = []
            t_dict[t].append(float(d['M']))
            if 'cluster_size' in d:
                c_dict[t].append(float(d['cluster_size']))
    return t_dict, c_dict

def compute_autocorr(x, max_lag=10000):
    n = len(x)
    mu = np.mean(x)
    var = np.var(x)
    if var < 1e-15:
        return 0.5
    max_lag = min(max_lag, n - 1)
    x_zero = x - mu
    n_fft = 1 << (2 * n - 1).bit_length()
    fx = np.fft.rfft(x_zero, n=n_fft)
    r = np.fft.irfft(fx * np.conj(fx), n=n_fft)[:n]
    norm = np.arange(n, 0, -1)
    acf = (r / norm) / var
    acf = acf[:max_lag]

    tau_int = 0.5
    for t in range(1, len(acf)):
        tau_int += acf[t]
        if t > 6.0 * tau_int:
            break
    return max(0.5, tau_int)

def block_bootstrap_error(x, block_len=2000, reps=500, rng=None):
    if rng is None:
        rng = np.random.default_rng(2026)
    n = len(x)
    nb = n // block_len
    blocks = x[:nb * block_len].reshape((nb, block_len))
    boot_means = []
    for _ in range(reps):
        sample_idx = rng.integers(0, nb, size=nb)
        boot_means.append(np.mean(blocks[sample_idx]))
    return np.std(boot_means)

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    evidence_dir = os.path.join(base_dir, "evidence")
    os.makedirs(evidence_dir, exist_ok=True)

    tc_exact = 2.0 / np.log(1.0 + np.sqrt(2.0))

    # Load runs
    m64_path = os.path.join(base_dir, "artifacts", "window-l64", "series.jsonl")
    w64_path = os.path.join(base_dir, "artifacts", "wolff-l64", "series.jsonl")
    w32_path = os.path.join(base_dir, "artifacts", "wolff-l32", "series.jsonl")

    m64_dict, _ = load_data(m64_path)
    w64_dict, w64_c = load_data(w64_path)
    w32_dict, w32_c = load_data(w32_path)

    temps = sorted(w64_dict.keys())
    rng = np.random.default_rng(42)

    # 1. Magnetization comparison and Wolff critical temperature
    m_metro_means = []
    m_metro_errs = []
    m_wolff_means = []
    m_wolff_errs = []

    chi_wolff_32 = []
    chi_wolff_64 = []

    for t in temps:
        # Metropolis L=64
        m_arr_m = np.abs(m64_dict[t])
        m_metro_means.append(np.mean(m_arr_m))
        m_metro_errs.append(block_bootstrap_error(m_arr_m, block_len=2000, reps=300, rng=rng))

        # Wolff L=64
        m_arr_w = np.abs(w64_dict[t])
        m_wolff_means.append(np.mean(m_arr_w))
        m_wolff_errs.append(block_bootstrap_error(m_arr_w, block_len=2000, reps=300, rng=rng))

        # Wolff Chi
        w64_raw = np.array(w64_dict[t])
        chi_wolff_64.append(64**2 * (np.mean(w64_raw**2) - np.mean(np.abs(w64_raw))**2) / t)

        w32_raw = np.array(w32_dict[t])
        chi_wolff_32.append(32**2 * (np.mean(w32_raw**2) - np.mean(np.abs(w32_raw))**2) / t)

    m_metro_means = np.array(m_metro_means)
    m_metro_errs = np.array(m_metro_errs)
    m_wolff_means = np.array(m_wolff_means)
    m_wolff_errs = np.array(m_wolff_errs)
    chi_wolff_32 = np.array(chi_wolff_32)
    chi_wolff_64 = np.array(chi_wolff_64)

    # Check agreement at T=2.3
    idx_23 = temps.index(2.3)
    val_m = m_metro_means[idx_23]
    err_m = m_metro_errs[idx_23]
    val_w = m_wolff_means[idx_23]
    err_w = m_wolff_errs[idx_23]
    d_stat = abs(val_w - val_m) / np.sqrt(err_m**2 + err_w**2)

    print(f"\nSampler agreement at L=64, T=2.3:")
    print(f"  Wolff <|m|>:      {val_w:.4f} +/- {err_w:.4f}")
    print(f"  Metropolis <|m|>: {val_m:.4f} +/- {err_m:.4f}")
    print(f"  d statistic:      {d_stat:.2f} (Pass condition d <= 3: {d_stat <= 3.0})")

    # Fit Wolff peaks
    i32 = np.argmax(chi_wolff_32)
    s32 = max(0, i32 - 2); e32 = min(len(temps), s32 + 5)
    if e32 - s32 < 5: s32 = max(0, e32 - 5)
    p32 = np.polyfit(temps[s32:e32], chi_wolff_32[s32:e32], 2)
    tp_wolff_32 = -p32[1] / (2.0 * p32[0])

    i64 = np.argmax(chi_wolff_64)
    s64 = max(0, i64 - 2); e64 = min(len(temps), s64 + 5)
    if e64 - s64 < 5: s64 = max(0, e64 - 5)
    p64 = np.polyfit(temps[s64:e64], chi_wolff_64[s64:e64], 2)
    tp_wolff_64 = -p64[1] / (2.0 * p64[0])

    tc_wolff_extrap = 2.0 * tp_wolff_64 - tp_wolff_32
    wolff_dev_pct = (tc_wolff_extrap - tc_exact) / tc_exact * 100.0
    print(f"\nWolff critical temperature:")
    print(f"  L=32 peak: {tp_wolff_32:.4f}")
    print(f"  L=64 peak: {tp_wolff_64:.4f}")
    print(f"  Extrapolated Tc: {tc_wolff_extrap:.4f} (dev: {wolff_dev_pct:+.2f}%)")

    # Plot magnetization-compare.png
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), dpi=200)

    # Left panel: Magnetization comparison
    ax1.errorbar(temps, m_metro_means, yerr=m_metro_errs, fmt='s-', color='#1f77b4', ms=4, capsize=3, lw=1.2, label='Metropolis ($L=64$)')
    ax1.errorbar(temps, m_wolff_means, yerr=m_wolff_errs, fmt='o-', color='#2ca02c', ms=4, capsize=3, lw=1.2, label='Wolff ($L=64$)')
    ax1.axvline(tc_exact, color='k', ls='--', lw=1.0, label='2.26919')
    ax1.set_xlabel('temperature $T$', fontsize=11)
    ax1.set_ylabel(r'$\langle |m| \rangle$', fontsize=11)
    ax1.legend(frameon=True, fontsize=10)
    ax1.grid(True, alpha=0.3, ls=':')

    # Right panel: Wolff Susceptibility
    t_fine = np.linspace(2.0, 2.6, 200)
    t_fit32 = np.linspace(temps[s32], temps[e32-1], 100)
    t_fit64 = np.linspace(temps[s64], temps[e64-1], 100)

    ax2.plot(temps, chi_wolff_32, 'o-', color='#1f77b4', ms=4, lw=1.2, label=f'$L = 32$, peak {tp_wolff_32:.4f}')
    ax2.plot(temps, chi_wolff_64, 's-', color='#ff7f0e', ms=4, lw=1.2, label=f'$L = 64$, peak {tp_wolff_64:.4f}')
    ax2.plot(t_fit32, np.polyval(p32, t_fit32), ':', color='#1f77b4', lw=1.5)
    ax2.plot(t_fit64, np.polyval(p64, t_fit64), ':', color='#ff7f0e', lw=1.5)
    ax2.axvline(tc_exact, color='k', ls='--', lw=1.0, label=f'$T_c = {tc_exact:.4f}$')
    ax2.set_xlabel('temperature $T$', fontsize=11)
    ax2.set_ylabel(r'$\chi(T)$ (Wolff)', fontsize=11)
    ax2.legend(frameon=True, fontsize=9.5)
    ax2.grid(True, alpha=0.3, ls=':')

    plt.tight_layout()
    mag_cmp_path = os.path.join(evidence_dir, "magnetization-compare.png")
    plt.savefig(mag_cmp_path)
    plt.close()
    print(f"Saved {mag_cmp_path}")

    # 2. TAU-COMPARE.PNG
    print("\nComputing work-normalized autocorrelation times...")
    tau_metro = []
    tau_wolff_work = []
    tau_wolff_raw = []
    mean_clusters = []

    for t in temps:
        # Metropolis
        tau_m = compute_autocorr(np.abs(m64_dict[t]), max_lag=15000)
        tau_metro.append(tau_m)

        # Wolff
        c_mean = np.mean(w64_c[t])
        tau_w = compute_autocorr(np.abs(w64_dict[t]), max_lag=500)
        tau_w_work = tau_w * (c_mean / (64.0**2))

        tau_wolff_raw.append(tau_w)
        mean_clusters.append(c_mean)
        tau_wolff_work.append(tau_w_work)

    # Print summary at T=2.3
    idx_23 = temps.index(2.3)
    tau_m_23 = tau_metro[idx_23]
    tau_w_raw_23 = tau_wolff_raw[idx_23]
    c_23 = mean_clusters[idx_23]
    tau_w_work_23 = tau_wolff_work[idx_23]
    speedup = tau_m_23 / tau_w_work_23

    print(f"\nAutocorrelation and Work at T=2.3 (L=64):")
    print(f"  Metropolis tau:     {tau_m_23:.2f} sweeps")
    print(f"  Wolff raw tau:      {tau_w_raw_23:.3f} moves")
    print(f"  Mean cluster size:  {c_23:.1f}")
    print(f"  Wolff tau_work:     {tau_w_work_23:.3f} spin-update work units")
    print(f"  Efficiency Speedup: {speedup:.1f}x less work per independent sample!")

    # Plot tau-compare.png
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=200)
    ax.plot(temps, tau_metro, 's-', color='#d62728', ms=4, lw=1.2, label='Metropolis')
    ax.plot(temps, tau_wolff_work, 'o-', color='#2ca02c', ms=4, lw=1.2, label='Wolff, one move per row')
    ax.axvline(tc_exact, color='k', ls='--', lw=1.0, label='2.26919')

    ax.set_yscale('log')
    ax.set_xlabel('temperature $T$', fontsize=11)
    ax.set_ylabel(r'$\tau_{\mathrm{work}}$, $L = 64$', fontsize=11)
    ax.set_xlim(2.0, 2.6)
    ax.set_ylim(0.05, 3000)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3, ls=':')

    plt.tight_layout()
    tau_cmp_path = os.path.join(evidence_dir, "tau-compare.png")
    plt.savefig(tau_cmp_path)
    plt.close()
    print(f"Saved {tau_cmp_path}")

if __name__ == '__main__':
    main()
