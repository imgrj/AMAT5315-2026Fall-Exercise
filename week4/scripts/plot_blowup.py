import numpy as np
import matplotlib.pyplot as plt

def parse_tsv(path):
    times = []
    energies = []
    with open(path) as f:
        lines = f.readlines()
        for line in lines[1:]: # skip header
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                try:
                    t = float(parts[0])
                    e = float(parts[1])
                    times.append(t)
                    energies.append(e)
                except ValueError:
                    pass
    return np.array(times), np.array(energies)

# Load Taylor-Green runs
t_tg32, e_tg32 = parse_tsv("D:/Desktop/AMAT5315-2026Fall-Exercise/week4/artifacts/scan/tg_rk4_dt0.032.tsv")
t_tg33, e_tg33 = parse_tsv("D:/Desktop/AMAT5315-2026Fall-Exercise/week4/artifacts/scan/tg_rk4_dt0.033.tsv")

# Load Random Flow runs
t_rf_stable, e_rf_stable = parse_tsv("D:/Desktop/AMAT5315-2026Fall-Exercise/week4/artifacts/scan/rf_rk4_dt0.035.tsv")
t_rf_unstable, e_rf_unstable = parse_tsv("D:/Desktop/AMAT5315-2026Fall-Exercise/week4/artifacts/scan/rf_rk4_dt0.038.tsv")
t_rf_euler, e_rf_euler = parse_tsv("D:/Desktop/AMAT5315-2026Fall-Exercise/week4/artifacts/scan/rf_euler_dt0.01.tsv")

fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=200)

# Left Panel: Taylor-Green
ax1 = axes[0]
t_exact = np.linspace(0, 8, 200)
e_exact = 0.25 * np.exp(-0.4 * t_exact)
ax1.plot(t_exact, e_exact, 'k--', label=r'exact $\frac{1}{4}e^{-0.4t}$', linewidth=1.5)

ax1.plot(t_tg32, e_tg32, 'tab:blue', label=r'RK4, $\Delta t = 0.032$', linewidth=1.5)
ax1.plot(t_tg33, e_tg33, 'crimson', label=r'RK4, $\Delta t = 0.033$', linewidth=1.5)

# Mark stopping time of unstable TG run
t_stop_tg = t_tg33[-1]
ax1.annotate(f'non-finite\nat $t = {t_stop_tg:.2f}$',
             xy=(t_stop_tg, min(max(e_tg33[-1], 1e-3), 1e2)),
             xytext=(t_stop_tg - 2.5, 1.0),
             arrowprops=dict(arrowstyle='->', color='crimson', lw=1.2),
             color='crimson', fontsize=9, fontweight='bold')

ax1.set_yscale('log')
ax1.set_xlim(0, 8.5)
ax1.set_ylim(1e-3, 1e2)
ax1.set_xlabel('time t', fontsize=10)
ax1.set_ylabel('energy E(t)', fontsize=10)
ax1.set_title(r'Taylor–Green, $N = 64, \nu = 0.1$', fontsize=11)
ax1.legend(loc='lower left', fontsize=9, framealpha=0.9)
ax1.grid(True, which='both', linestyle=':', alpha=0.5)

text_tg = (r"Predicted $\Delta t_{\rm crit} = 0.0316$ from $\nu |k|^2_{\rm max}\Delta t = 2.785$." "\n"
           r"$\Delta t = 0.032$ stays on the exact $\frac{1}{4}e^{-0.4t}$ (dashed);" "\n"
           r"$0.033$ departs and blows up.")
ax1.text(0.03, 0.35, text_tg, transform=ax1.transAxes, fontsize=8,
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

# Right Panel: Random Flow
ax2 = axes[1]
ax2.plot(t_rf_stable, e_rf_stable, 'tab:blue', label=r'RK4, $\Delta t = 0.035$', linewidth=1.5)
ax2.plot(t_rf_unstable, e_rf_unstable, 'crimson', label=r'RK4, $\Delta t = 0.038$', linewidth=1.5)
ax2.plot(t_rf_euler, e_rf_euler, 'k:', label=r'Euler $0.01$ (dotted)', linewidth=1.5)

# Mark stopping times of unstable RF runs
t_stop_rf = t_rf_unstable[-1]
ax2.annotate(f'RK4 0.038:\nnon-finite at $t = {t_stop_rf:.2f}$',
             xy=(t_stop_rf, min(max(e_rf_unstable[-1], 1e-1), 1e2)),
             xytext=(t_stop_rf + 0.8, 10.0),
             arrowprops=dict(arrowstyle='->', color='crimson', lw=1.2),
             color='crimson', fontsize=9, fontweight='bold')

t_stop_eu = t_rf_euler[-1]
ax2.annotate(f'Euler 0.01:\nnon-finite at $t = {t_stop_eu:.2f}$',
             xy=(t_stop_eu, min(max(e_rf_euler[-1], 1e-1), 1e2)),
             xytext=(t_stop_eu + 0.8, 1.5),
             arrowprops=dict(arrowstyle='->', color='black', lw=1.2),
             color='black', fontsize=9, fontweight='bold')

ax2.set_yscale('log')
ax2.set_xlim(0, 10.5)
ax2.set_ylim(1e-1, 1e2)
ax2.set_xlabel('time t', fontsize=10)
ax2.set_ylabel('energy E(t)', fontsize=10)
ax2.set_title(r'random case, $N = 128, \nu = 0.004$', fontsize=11)
ax2.legend(loc='center right', fontsize=9, framealpha=0.9)
ax2.grid(True, which='both', linestyle=':', alpha=0.5)

text_rf = (r"Predicted $\Delta t_{\rm crit} = 0.0194$ from $U_{\rm max}|k|_{\rm max}\Delta t = 2.83$." "\n"
           r"Measured $0.035$ to $0.038$, $\sim 1.8\times$ bound." "\n"
           r"Euler 0.01 fails early: no imaginary interval.")
ax2.text(0.03, 0.05, text_rf, transform=ax2.transAxes, fontsize=8,
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

plt.tight_layout()
out_png = 'D:/Desktop/AMAT5315-2026Fall-Exercise/week4/evidence/blowup.png'
plt.savefig(out_png, bbox_inches='tight')
print('Saved', out_png)
