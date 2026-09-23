import json
import numpy as np
import matplotlib.pyplot as plt

# 1. Load exact t=1
with open('D:/Desktop/AMAT5315-2026Fall-Exercise/week4/artifacts/taylor-green/exact-t1.json') as f:
    exact = json.load(f)

# 2. Load fields.jsonl
with open('D:/Desktop/AMAT5315-2026Fall-Exercise/week4/artifacts/taylor-green/fields.jsonl') as f:
    frames = [json.loads(line) for line in f]

f0 = frames[0]
f1 = frames[-1]

n = f0['n'] if 'n' in f0 else 64
u_exact = np.array(exact['u'])
v_exact = np.array(exact['v'])
u_comp = np.array(f1['u'])
v_comp = np.array(f1['v'])

# Compute relative error
diff_sq = np.sum((u_comp - u_exact)**2 + (v_comp - v_exact)**2)
exact_sq = np.sum(u_exact**2 + v_exact**2)
rel_err = np.sqrt(diff_sq / exact_sq)
print(f"Relative error of velocity: {rel_err:.2e}")

# Prepare grids for plotting
dx = 2.0 * np.pi / n
x = np.arange(n) * dx
y = np.arange(n) * dx
X, Y = np.meshgrid(x, y, indexing='xy')

omega0 = np.array(f0['omega']).reshape((n, n))
u0 = np.array(f0['u']).reshape((n, n))
v0 = np.array(f0['v']).reshape((n, n))

omega1 = np.array(f1['omega']).reshape((n, n))
u1 = np.array(f1['u']).reshape((n, n))
v1 = np.array(f1['v']).reshape((n, n))

max_w0 = np.max(np.abs(omega0))
max_w1 = np.max(np.abs(omega1))

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=200)

# Subsampling for velocity arrows
skip = 4
Xs = X[::skip, ::skip]
Ys = Y[::skip, ::skip]

# Panel 1: t = 0
ax = axes[0]
im0 = ax.imshow(omega0, extent=[0, 2*np.pi, 0, 2*np.pi], origin='lower',
                cmap='RdBu_r', vmin=-2.0, vmax=2.0)
ax.quiver(Xs, Ys, u0[::skip, ::skip], v0[::skip, ::skip], color='black', alpha=0.7, scale=12.0)
ax.set_title(f"$t = 0,\\ \\max|\\omega| = {max_w0:.1f}$", fontsize=11)
ax.set_xlabel("x", fontsize=10)
ax.set_ylabel("y", fontsize=10)
ax.set_xticks([0, np.pi, 2*np.pi], ["0", r"$\pi$", r"$2\pi$"])
ax.set_yticks([0, np.pi, 2*np.pi], ["0", r"$\pi$", r"$2\pi$"])

# Panel 2: t = 1
ax = axes[1]
im1 = ax.imshow(omega1, extent=[0, 2*np.pi, 0, 2*np.pi], origin='lower',
                cmap='RdBu_r', vmin=-2.0, vmax=2.0)
ax.quiver(Xs, Ys, u1[::skip, ::skip], v1[::skip, ::skip], color='black', alpha=0.7, scale=12.0)
ax.set_title(f"$t = 1,\\ \\max|\\omega| = {max_w1:.3f}$", fontsize=11)
ax.set_xlabel("x", fontsize=10)
ax.set_ylabel("y", fontsize=10)
ax.set_xticks([0, np.pi, 2*np.pi], ["0", r"$\pi$", r"$2\pi$"])
ax.set_yticks([0, np.pi, 2*np.pi], ["0", r"$\pi$", r"$2\pi$"])

plt.tight_layout()
out_png = 'D:/Desktop/AMAT5315-2026Fall-Exercise/week4/evidence/taylor-green.png'
plt.savefig(out_png, bbox_inches='tight')
print('Saved', out_png)
