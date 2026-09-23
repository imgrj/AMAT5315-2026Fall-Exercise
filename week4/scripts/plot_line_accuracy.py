import json
import numpy as np
import matplotlib.pyplot as plt

# 1. Load pulse profiles
with open('D:/Desktop/AMAT5315-2026Fall-Exercise/week4/artifacts/pulse_profiles.json') as f:
    dp = json.load(f)

# 2. Load line convergence
with open('D:/Desktop/AMAT5315-2026Fall-Exercise/week4/artifacts/line_convergence.json') as f:
    dc = json.load(f)

fig = plt.figure(figsize=(13, 5), dpi=200)

# Left Panel: Pulse Profiles
ax1 = plt.subplot(1, 2, 1)
x = np.array(dp['x'])
ax1.plot(x, dp['exact'], 'k-', linewidth=1.8, label='exact, after one lap')
ax1.plot(x, dp['rk4_fourier'], 'b--', linewidth=1.5, label='Fourier derivative, RK4, $h = 0.02$')
ax1.plot(x, dp['rk4_fd'], color='darkorange', linestyle='-', linewidth=1.2, label='centred differences, RK4, $h = 0.02$')
ax1.plot(x, dp['euler_fourier'], color='forestgreen', linestyle='-', linewidth=1.2, label='Fourier derivative, forward Euler, $h = 0.005$')
ax1.plot(x, dp['start'], color='gray', linestyle=':', linewidth=1.2, label='start')

ax1.set_xlim(0, 2*np.pi)
ax1.set_ylim(-0.75, 1.35)
ax1.set_xlabel('x', fontsize=10)
ax1.set_ylabel('u', fontsize=10)
ax1.legend(fontsize=8, loc='upper left', framealpha=0.9)
ax1.grid(True, linestyle=':', alpha=0.5)
ax1.text(0.95, 0.05, r'$N = 64, \nu = 0.002, \sigma = 0.25$', transform=ax1.transAxes,
         fontsize=8, horizontalalignment='right', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Right Panel: Error Convergence
ax2 = plt.subplot(1, 2, 2)
dts = np.array(dc['dts'])
e_err = np.array(dc['euler_errs'])
m_err = np.array(dc['midpoint_errs'])
r4_err = np.array(dc['rk4_errs'])
r4eq_err = np.array(dc['rk4_equal_errs'])

def fit_line(x, y):
    p = np.polyfit(np.log(x), np.log(y), 1)
    slope = p[0]
    fit_y = np.exp(p[1]) * (x**slope)
    return slope, fit_y

s_e, fit_e = fit_line(dts, e_err)
s_m, fit_m = fit_line(dts, m_err)
s_r4, fit_r4 = fit_line(dts, r4_err)
s_r4eq, fit_r4eq = fit_line(dts, r4eq_err)

ax2.loglog(dts, e_err, 's', color='crimson', markersize=5)
ax2.loglog(dts, fit_e, '-', color='crimson', label=f'forward Euler, slope {s_e:.2f}')

ax2.loglog(dts, m_err, 'o', color='darkorange', markersize=5)
ax2.loglog(dts, fit_m, '-', color='darkorange', label=f'midpoint, slope {s_m:.2f}')

ax2.loglog(dts, r4_err, '^', color='tab:blue', markersize=5)
ax2.loglog(dts, fit_r4, '-', color='tab:blue', label=f'RK4, slope {s_r4:.2f}')

ax2.loglog(dts, r4eq_err, 'x', color='tab:gray', markersize=5)
ax2.loglog(dts, fit_r4eq, '--', color='tab:gray', label=f'RK4, weights 1,1,1,1/4, slope {s_r4eq:.2f}')

ax2.set_xlabel('h', fontsize=10)
ax2.set_ylabel(r'$\max |u(T) - u_{\rm exact}(T)|$', fontsize=10)
ax2.legend(fontsize=8, loc='lower right', framealpha=0.9)
ax2.grid(True, which='both', linestyle=':', alpha=0.5)
ax2.text(0.05, 0.95, r'$T = 1, \nu = 0.05$', transform=ax2.transAxes,
         fontsize=8, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

plt.tight_layout()
out_path = 'D:/Desktop/AMAT5315-2026Fall-Exercise/week4/evidence/line-accuracy.png'
plt.savefig(out_path, bbox_inches='tight')
print('Saved', out_path)
print(f'Euler slope: {s_e:.2f}, Midpoint slope: {s_m:.2f}, RK4 slope: {s_r4:.2f}, Equal weights slope: {s_r4eq:.2f}')
