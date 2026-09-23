import json
import numpy as np
import matplotlib.pyplot as plt

# Load fields.jsonl from artifacts/random/
frames = {}
with open('D:/Desktop/AMAT5315-2026Fall-Exercise/week4/artifacts/random/fields.jsonl') as f:
    for line in f:
        data = json.loads(line)
        t = round(data['t'], 1)
        if t in [0.0, 2.0, 5.0, 10.0] and t not in frames:
            frames[t] = data

target_times = [0.0, 2.0, 5.0, 10.0]
fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), dpi=200)

# All frames use the same colour scale: blue -10.97, white 0, red 10.97
vmin = -10.97
vmax = 10.97

for idx, t in enumerate(target_times):
    ax = axes[idx]
    frame = frames[t]
    omega = np.array(frame['omega']).reshape((128, 128))
    u = np.array(frame['u'])
    v = np.array(frame['v'])
    
    e = 0.5 * np.mean(u**2 + v**2)
    z = 0.5 * np.mean(omega**2)
    
    im = ax.imshow(omega, extent=[0, 2*np.pi, 0, 2*np.pi], origin='lower',
                   cmap='RdBu_r', vmin=vmin, vmax=vmax)
    t_str = f"t = {int(t)}" if t == int(t) else f"t = {t}"
    ax.set_title(f"{t_str}\n$E = {e:.3f},\\ Z = {z:.3f}$", fontsize=11)
    ax.set_xlabel("x", fontsize=10)
    ax.set_ylabel("y", fontsize=10)
    ax.set_xticks([0, np.pi, 2*np.pi], ["0", r"$\pi$", r"$2\pi$"])
    ax.set_yticks([0, np.pi, 2*np.pi], ["0", r"$\pi$", r"$2\pi$"])

plt.tight_layout()
out_png = 'D:/Desktop/AMAT5315-2026Fall-Exercise/week4/evidence/random.png'
plt.savefig(out_png, bbox_inches='tight')
print('Saved', out_png)
