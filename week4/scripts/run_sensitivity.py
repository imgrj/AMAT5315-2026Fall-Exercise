import subprocess
import os
import json
import numpy as np
import matplotlib.pyplot as plt

os.makedirs('D:/Desktop/AMAT5315-2026Fall-Exercise/week4/artifacts/sensitivity', exist_ok=True)

def add_perturbation(field_json_str):
    data = json.loads(field_json_str)
    n = data['n']
    u = np.array(data['u'], dtype=float)
    v = np.array(data['v'], dtype=float)
    
    # M is largest absolute value among all initial u and v
    M = max(np.max(np.abs(u)), np.max(np.abs(v)))
    
    dx = 2.0 * np.pi / n
    y = np.arange(n) * dx
    x = np.arange(n) * dx
    X, Y = np.meshgrid(x, y, indexing='xy')
    
    # delta_omega = -7e-5 * M * cos(3x) * cos(4y)
    # Corresponding velocity perturbation:
    # u_delta = (28e-5 * M / 25) * cos(3x) * sin(4y)
    # v_delta = (-21e-5 * M / 25) * sin(3x) * cos(4y)
    coeff = -7e-5 * M
    u_delta = (-4.0 / 25.0) * coeff * np.cos(3.0 * X) * np.sin(4.0 * Y)
    v_delta = (-3.0 / 25.0) * coeff * np.sin(3.0 * X) * np.cos(4.0 * Y)
    
    u_pert = u + u_delta.flatten()
    v_pert = v + v_delta.flatten()
    
    data['u'] = u_pert.tolist()
    data['v'] = v_pert.tolist()
    return json.dumps(data)

# 1. Taylor-Green
print("Running Taylor-Green sensitivity...")
cmd_tg = "field taylor-green --n 64"
res_tg = subprocess.run(cmd_tg, shell=True, capture_output=True, text=True, check=True)
tg_unpert = res_tg.stdout
tg_pert = add_perturbation(tg_unpert)

dir_tg_unpert = "D:/Desktop/AMAT5315-2026Fall-Exercise/week4/artifacts/sensitivity/tg_unpert"
dir_tg_pert = "D:/Desktop/AMAT5315-2026Fall-Exercise/week4/artifacts/sensitivity/tg_pert"

cmd_fluid_tg = "fluid --method rk4 --nu 0.1 --dt 0.01 --t-end 20 --every 0.5 --out "
subprocess.run(cmd_fluid_tg + dir_tg_unpert, shell=True, input=tg_unpert, text=True, capture_output=True, check=True)
subprocess.run(cmd_fluid_tg + dir_tg_pert, shell=True, input=tg_pert, text=True, capture_output=True, check=True)

# 2. Random Flow
print("Running Random Flow sensitivity...")
cmd_rf = "field random --n 128 --seed 2026 --k-min 2 --k-max 6"
res_rf = subprocess.run(cmd_rf, shell=True, capture_output=True, text=True, check=True)
rf_unpert = res_rf.stdout
rf_pert = add_perturbation(rf_unpert)

dir_rf_unpert = "D:/Desktop/AMAT5315-2026Fall-Exercise/week4/artifacts/sensitivity/rf_unpert"
dir_rf_pert = "D:/Desktop/AMAT5315-2026Fall-Exercise/week4/artifacts/sensitivity/rf_pert"

cmd_fluid_rf = "fluid --method rk4 --nu 0.004 --dt 0.01 --t-end 20 --every 0.5 --out "
subprocess.run(cmd_fluid_rf + dir_rf_unpert, shell=True, input=rf_unpert, text=True, capture_output=True, check=True)
subprocess.run(cmd_fluid_rf + dir_rf_pert, shell=True, input=rf_pert, text=True, capture_output=True, check=True)

def load_frames(jsonl_path):
    frames = {}
    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            t = round(d['t'], 2)
            frames[t] = np.array(d['omega'], dtype=float)
    return frames

frames_tg_u = load_frames(f"{dir_tg_unpert}/fields.jsonl")
frames_tg_p = load_frames(f"{dir_tg_pert}/fields.jsonl")

frames_rf_u = load_frames(f"{dir_rf_unpert}/fields.jsonl")
frames_rf_p = load_frames(f"{dir_rf_pert}/fields.jsonl")

# Compute relative distances: ||w1 - w2|| / ||w1||
times_tg = sorted(frames_tg_u.keys())
dist_tg = []
for t in times_tg:
    w1 = frames_tg_u[t]
    w2 = frames_tg_p[t]
    d = np.linalg.norm(w1 - w2) / max(np.linalg.norm(w1), 1e-12)
    dist_tg.append(d)

times_rf = sorted(frames_rf_u.keys())
dist_rf = []
for t in times_rf:
    w1 = frames_rf_u[t]
    w2 = frames_rf_p[t]
    d = np.linalg.norm(w1 - w2) / max(np.linalg.norm(w1), 1e-12)
    dist_rf.append(d)

# Plot sensitivity
plt.figure(figsize=(6.5, 5), dpi=200)
plt.plot(times_rf, dist_rf, 'k-', linewidth=1.5, label='random')
plt.plot(times_tg, dist_tg, color='tab:gray', linestyle='-', linewidth=1.2, label='Taylor–Green')

plt.yscale('log')
plt.xlim(0, 20.5)
plt.ylim(1e-7, 1e-2)
plt.xlabel('time t', fontsize=10)
plt.ylabel(r'$\|\omega_1 - \omega_2\| / \|\omega_1\|$', fontsize=10)
plt.title(r'same step $\Delta t = 0.01$, perturbed start', fontsize=11)
plt.legend(loc='center right', fontsize=9, framealpha=0.9)
plt.grid(True, which='both', linestyle=':', alpha=0.5)

plt.annotate('rounding floor', xy=(7, 1.5e-6), xytext=(7, 4e-6),
             arrowprops=dict(arrowstyle='->', color='gray', lw=1),
             color='gray', fontsize=8)

text_box = ("Perturbation $10^{-5}$ in one mode;\n"
            "random grows $\\sim 40\\times$,\n"
            "Taylor–Green decays to floor.")
plt.text(0.04, 0.05, text_box, transform=plt.gca().transAxes, fontsize=8,
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

plt.tight_layout()
out_png = 'D:/Desktop/AMAT5315-2026Fall-Exercise/week4/evidence/sensitivity.png'
plt.savefig(out_png, bbox_inches='tight')
print('Saved', out_png)
print(f"Random initial dist: {dist_rf[0]:.2e}, t=20 dist: {dist_rf[-1]:.2e}, growth factor: {dist_rf[-1]/dist_rf[0]:.1f}x")
print(f"Taylor-Green initial dist: {dist_tg[0]:.2e}, t=20 dist: {dist_tg[-1]:.2e}")
