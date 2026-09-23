import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# 1. Compute stability regions in complex plane
# z = x + 1j * y
x = np.linspace(-4.5, 0.8, 800)
y = np.linspace(-3.5, 3.5, 800)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y

# RK4 growth factor
R_rk4 = 1.0 + Z + (Z**2)/2.0 + (Z**3)/6.0 + (Z**4)/24.0
growth_rk4 = np.abs(R_rk4)

# Euler & Midpoint
R_euler = 1.0 + Z
R_midpoint = 1.0 + Z + (Z**2)/2.0

# Modes of the line: nu = 0.05, n = 64, c = 1
nu = 0.05
n = 64
c = 1.0
ks = np.arange(-n//2, n//2)
lam = np.zeros(len(ks), dtype=complex)
for idx, k in enumerate(ks):
    if k == -n//2:
        lam[idx] = -nu * (k**2)
    else:
        lam[idx] = -nu * (k**2) - 1j * c * k

# Load simulation data
with open('D:/Desktop/AMAT5315-2026Fall-Exercise/week4/artifacts/stability_0.045.json') as f:
    d45 = json.load(f)
with open('D:/Desktop/AMAT5315-2026Fall-Exercise/week4/artifacts/stability_0.056.json') as f:
    d56 = json.load(f)

u45 = np.array(d45['u_xt'])
t45 = np.array(d45['times'])
u56 = np.array(d56['u_xt'])
t56 = np.array(d56['times'])
x_grid = np.linspace(0, 2*np.pi, n, endpoint=False)

fig = plt.figure(figsize=(14, 4.5), dpi=200)

# Panel 1: Stability region
ax1 = plt.subplot(1, 3, 1)
# Plot growth factor in region where growth <= 3.5
norm = LogNorm(vmin=0.3, vmax=3.0)
cf = ax1.pcolormesh(X, Y, growth_rk4, norm=norm, cmap='coolwarm', shading='auto', rasterized=True)
cbar = plt.colorbar(cf, ax=ax1, fraction=0.046, pad=0.04)
cbar.set_label('growth factor per step, RK4', fontsize=9)

# Contours |R| = 1
ax1.contour(X, Y, np.abs(R_euler), levels=[1.0], colors='gray', linestyles='--', linewidths=1.2)
ax1.contour(X, Y, np.abs(R_midpoint), levels=[1.0], colors='darkgray', linestyles='--', linewidths=1.2)
ax1.contour(X, Y, growth_rk4, levels=[1.0], colors='black', linewidths=1.8)

# Modes dots
z45 = lam * 0.045
z56 = lam * 0.056
ax1.plot(z45.real, z45.imag, '.', color='magenta', markersize=4, label='h = 0.0450')
ax1.plot(z56.real, z56.imag, '.', color='red', markersize=4, label='h = 0.0560')

ax1.set_xlim(-4.5, 0.8)
ax1.set_ylim(-3.5, 3.5)
ax1.set_xlabel(r'Re $\lambda h$', fontsize=10)
ax1.set_ylabel(r'Im $\lambda h$', fontsize=10)
ax1.legend(title=r'modes $\lambda_k h$', fontsize=8, loc='upper left', framealpha=0.9)
ax1.grid(True, linestyle=':', alpha=0.5)

# Panel 2: u(x, t) for h = 0.045
ax2 = plt.subplot(1, 3, 2)
im2 = ax2.imshow(u45, extent=[0, 2*np.pi, t45[-1], 0], aspect='auto', cmap='plasma', vmin=-0.2, vmax=1.0)
ax2.set_xlabel('x', fontsize=10)
ax2.set_ylabel('t', fontsize=10)
ax2.set_title('h = 0.045', fontsize=11)

# Panel 3: u(x, t) for h = 0.056
ax3 = plt.subplot(1, 3, 3)
im3 = ax3.imshow(u56, extent=[0, 2*np.pi, t56[-1], 0], aspect='auto', cmap='plasma', vmin=-0.2, vmax=1.0)
ax3.set_xlabel('x', fontsize=10)
ax3.set_ylabel('t', fontsize=10)
ax3.set_title('h = 0.056', fontsize=11)

plt.tight_layout()
out_path = 'D:/Desktop/AMAT5315-2026Fall-Exercise/week4/evidence/line-stability.png'
plt.savefig(out_path, bbox_inches='tight')
print('Saved', out_path)
