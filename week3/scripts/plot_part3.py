import json
import os
import numpy as np
import matplotlib.pyplot as plt

def load_series(path):
    t_dict = {}
    with open(path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            d = json.loads(line)
            t = round(float(d['T']), 4)
            if t not in t_dict: t_dict[t] = []
            t_dict[t].append(float(d['M']))
    return t_dict

def compute_acf(x, max_lag=3000):
    n = len(x)
    mu = np.mean(x)
    var = np.var(x)
    if var < 1e-15:
        return np.ones(max_lag)
    x_zero = x - mu
    n_fft = 1 << (2 * n - 1).bit_length()
    fx = np.fft.rfft(x_zero, n=n_fft)
    r = np.fft.irfft(fx * np.conj(fx), n=n_fft)[:n]
    norm = np.arange(n, 0, -1)
    acf = (r / norm) / var
    return acf[:min(max_lag, n)]

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    evidence_dir = os.path.join(base_dir, "evidence")
    os.makedirs(evidence_dir, exist_ok=True)

    tc_exact = 2.0 / np.log(1.0 + np.sqrt(2.0))

    # 1. TRACE.PNG
    print("Generating trace.png...")
    w64_path = os.path.join(base_dir, "artifacts", "window-l64", "series.jsonl")
    c64_path = os.path.join(base_dir, "artifacts", "coarse-l64", "series.jsonl")

    w64_dict = load_series(w64_path)
    c64_dict = load_series(c64_path)

    m_23 = np.abs(w64_dict[2.3][:2000])
    m_30 = np.abs(c64_dict[3.0][:2000])

    fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
    ax.plot(m_23, color='#1f77b4', lw=0.9, label=r'$T = 2.3$')
    ax.plot(m_30, color='#d62728', lw=0.7, alpha=0.85, label=r'$T = 3.0$')
    ax.set_xlabel('measurement sweep', fontsize=11)
    ax.set_ylabel(r'$|m|$', fontsize=11)
    ax.set_xlim(0, 2000)
    ax.set_ylim(-0.05, 0.85)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3, ls=':')
    plt.tight_layout()
    plt.savefig(os.path.join(evidence_dir, "trace.png"))
    plt.close()
    print("Saved trace.png")

    # 2. ACF-BINNING.PNG
    print("Generating acf-binning.png...")
    full_23 = np.abs(w64_dict[2.3])
    acf_23 = compute_acf(full_23, max_lag=3000)

    # Binning analysis
    block_lengths = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    errors_binning = []
    n_tot = len(full_23)
    for bl in block_lengths:
        nb = n_tot // bl
        blocks = full_23[:nb * bl].reshape((nb, bl))
        b_means = np.mean(blocks, axis=1)
        se = np.std(b_means, ddof=1) / np.sqrt(nb)
        errors_binning.append(se)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=200)
    ax1.plot(acf_23, 'k-', lw=1.2)
    ax1.set_xlabel('lag $t$ (sweeps)', fontsize=11)
    ax1.set_ylabel(r'$\rho(t)$ of $|m|$ at $T = 2.3$', fontsize=11)
    ax1.set_xlim(0, 3000)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3, ls=':')

    ax2.plot(block_lengths, errors_binning, 'ko-', ms=4, lw=1.2)
    ax2.set_xscale('log')
    ax2.set_xlabel('block length (sweeps)', fontsize=11)
    ax2.set_ylabel(r'error bar on $\langle |m| \rangle$', fontsize=11)
    ax2.grid(True, alpha=0.3, ls=':')
    plt.tight_layout()
    plt.savefig(os.path.join(evidence_dir, "acf-binning.png"))
    plt.close()
    print("Saved acf-binning.png")

    # 3. TAU.PNG
    print("Generating tau.png...")
    # Read errors.txt to get tau_int
    errors_txt = os.path.join(evidence_dir, "errors.txt")
    tau_32 = []
    t_32 = []
    tau_64 = []
    t_64 = []
    with open(errors_txt, 'r') as f:
        lines = f.readlines()[1:] # skip header
        for line in lines:
            if not line.strip(): continue
            parts = line.split('\t')
            sz = int(parts[0])
            t = float(parts[1])
            tau = float(parts[6])
            if sz == 32:
                t_32.append(t); tau_32.append(tau)
            else:
                t_64.append(t); tau_64.append(tau)

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=200)
    ax.plot(t_32, tau_32, 'o-', color='#1f77b4', ms=3.5, lw=1.2, label='$L = 32$')
    ax.plot(t_64, tau_64, 's-', color='#ff7f0e', ms=3.5, lw=1.2, label='$L = 64$')
    ax.axvline(tc_exact, color='k', ls='--', lw=1.2, label=f'$T_c = {tc_exact:.4f}$')
    ax.set_yscale('log')
    ax.set_xlabel('temperature $T$', fontsize=11)
    ax.set_ylabel(r'$\tau_{\mathrm{int}}$ (sweeps)', fontsize=11)
    ax.set_xlim(1.4, 3.6)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3, ls=':')
    plt.tight_layout()
    plt.savefig(os.path.join(evidence_dir, "tau.png"))
    plt.close()
    print("Saved tau.png")

    # 4. CHI-BOOTSTRAP.PNG
    print("Generating chi-bootstrap.png...")
    w32_path = os.path.join(base_dir, "artifacts", "window-l32", "series.jsonl")
    w32_dict = load_series(w32_path)

    window_temps = sorted(w32_dict.keys())

    # Pre-parse arrays
    data32 = {t: np.array(w32_dict[t]) for t in window_temps}
    data64 = {t: np.array(w64_dict[t]) for t in window_temps}

    n_sweeps = len(data32[window_temps[0]])
    reps = 500

    # Store curves for block lengths 2000, 4000, 8000
    t_fine = np.linspace(2.0, 2.6, 200)
    
    # We will compute bootstrap for block length 2000 for envelope shading
    bl = 2000
    nb = n_sweeps // bl

    rng = np.random.default_rng(2026)

    tc_samples_by_bl = {}

    for current_bl in [2000, 4000, 8000]:
        c_nb = n_sweeps // current_bl
        tc_reps = []
        for _ in range(reps):
            # Resample L=32
            chis32 = []
            for t in window_temps:
                m_arr = data32[t]
                blocks = m_arr[:c_nb * current_bl].reshape((c_nb, current_bl))
                sampled_blocks = blocks[rng.integers(0, c_nb, size=c_nb)].ravel()
                m2 = np.mean(sampled_blocks**2)
                absm = np.mean(np.abs(sampled_blocks))
                chis32.append(32**2 * (m2 - absm**2) / t)
            chis32 = np.array(chis32)

            # Resample L=64
            chis64 = []
            for t in window_temps:
                m_arr = data64[t]
                blocks = m_arr[:c_nb * current_bl].reshape((c_nb, current_bl))
                sampled_blocks = blocks[rng.integers(0, c_nb, size=c_nb)].ravel()
                m2 = np.mean(sampled_blocks**2)
                absm = np.mean(np.abs(sampled_blocks))
                chis64.append(64**2 * (m2 - absm**2) / t)
            chis64 = np.array(chis64)

            # Fit peak 32
            i32 = np.argmax(chis32)
            s32 = max(0, i32 - 2); e32 = min(len(window_temps), s32 + 5)
            if e32 - s32 < 5: s32 = max(0, e32 - 5)
            p32 = np.polyfit(window_temps[s32:e32], chis32[s32:e32], 2)

            # Fit peak 64
            i64 = np.argmax(chis64)
            s64 = max(0, i64 - 2); e64 = min(len(window_temps), s64 + 5)
            if e64 - s64 < 5: s64 = max(0, e64 - 5)
            p64 = np.polyfit(window_temps[s64:e64], chis64[s64:e64], 2)

            tp32 = -p32[1] / (2.0 * p32[0])
            tp64 = -p64[1] / (2.0 * p64[0])
            tc_reps.append(2.0 * tp64 - tp32)

        tc_samples_by_bl[current_bl] = np.array(tc_reps)

    print("Bootstrap Tc standard errors:")
    for cur_bl in [2000, 4000, 8000]:
        err = np.std(tc_samples_by_bl[cur_bl])
        print(f"Block length {cur_bl}: std = {err:.5f}")

    # Plot chi bootstrap envelope
    # For L=32 and L=64, compute curves over 500 replicates for bl=2000
    curves32 = []
    curves64 = []
    c_nb = n_sweeps // 2000
    for rep in range(150):
        # 32
        chis32 = []
        for t in window_temps:
            m_arr = data32[t]
            blocks = m_arr[:c_nb * 2000].reshape((c_nb, 2000))
            sb = blocks[rng.integers(0, c_nb, size=c_nb)].ravel()
            chis32.append(32**2 * (np.mean(sb**2) - np.mean(np.abs(sb))**2) / t)
        chis32 = np.array(chis32)
        i32 = np.argmax(chis32)
        s32 = max(0, i32 - 2); e32 = min(len(window_temps), s32 + 5)
        if e32 - s32 < 5: s32 = max(0, e32 - 5)
        p32 = np.polyfit(window_temps[s32:e32], chis32[s32:e32], 2)
        curves32.append(np.polyval(p32, t_fine))

        # 64
        chis64 = []
        for t in window_temps:
            m_arr = data64[t]
            blocks = m_arr[:c_nb * 2000].reshape((c_nb, 2000))
            sb = blocks[rng.integers(0, c_nb, size=c_nb)].ravel()
            chis64.append(64**2 * (np.mean(sb**2) - np.mean(np.abs(sb))**2) / t)
        chis64 = np.array(chis64)
        i64 = np.argmax(chis64)
        s64 = max(0, i64 - 2); e64 = min(len(window_temps), s64 + 5)
        if e64 - s64 < 5: s64 = max(0, e64 - 5)
        p64 = np.polyfit(window_temps[s64:e64], chis64[s64:e64], 2)
        curves64.append(np.polyval(p64, t_fine))

    curves32 = np.array(curves32)
    curves64 = np.array(curves64)

    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=200)
    # Shaded envelopes (10th to 90th percentile)
    ax.fill_between(t_fine, np.percentile(curves32, 5, axis=0), np.percentile(curves32, 95, axis=0), color='#1f77b4', alpha=0.25, label='$L=32$ fit envelope')
    ax.fill_between(t_fine, np.percentile(curves64, 5, axis=0), np.percentile(curves64, 95, axis=0), color='#ff7f0e', alpha=0.25, label='$L=64$ fit envelope')

    # Mean lines
    ax.plot(t_fine, np.median(curves32, axis=0), color='#1f77b4', lw=1.5, label='$L=32$ median')
    ax.plot(t_fine, np.median(curves64, axis=0), color='#ff7f0e', lw=1.5, label='$L=64$ median')

    ax.axvline(tc_exact, color='k', ls='--', lw=1.2, label='2.26919')

    ax.set_xlabel('temperature $T$', fontsize=11)
    ax.set_ylabel(r'$\chi(T)$', fontsize=11)
    ax.set_xlim(2.2, 2.45)
    ax.set_ylim(0, 100)
    ax.legend(frameon=True, fontsize=9.5)
    ax.grid(True, alpha=0.3, ls=':')
    plt.tight_layout()
    plt.savefig(os.path.join(evidence_dir, "chi-bootstrap.png"))
    plt.close()
    print("Saved chi-bootstrap.png")

if __name__ == '__main__':
    main()
