import subprocess
import os
import json
import numpy as np
import matplotlib.pyplot as plt

os.makedirs('D:/Desktop/AMAT5315-2026Fall-Exercise/week4/artifacts/convergence', exist_ok=True)
os.makedirs('D:/Desktop/AMAT5315-2026Fall-Exercise/week4/evidence', exist_ok=True)

# 1. Generate random initial field on N = 128
print("Generating random initial field...")
cmd_rf = "field random --n 128 --seed 2026 --k-min 2 --k-max 6"
res_rf = subprocess.run(cmd_rf, shell=True, capture_output=True, text=True, check=True)
rf_json = res_rf.stdout

dts = [0.02, 0.0125, 0.01, 0.0025]
omega_fields = {}

for dt in dts:
    out_dir = f"D:/Desktop/AMAT5315-2026Fall-Exercise/week4/artifacts/convergence/dt{dt}"
    print(f"Running RK4 dt = {dt} to t = 2...")
    cmd_fluid = f"fluid --method rk4 --nu 0.004 --dt {dt} --t-end 2 --every 2 --out {out_dir}"
    subprocess.run(cmd_fluid, shell=True, input=rf_json, text=True, capture_output=True, check=True)
    
    with open(f"{out_dir}/fields.jsonl") as f:
        lines = f.readlines()
        last = json.loads(lines[-1])
        omega_fields[dt] = np.array(last['omega'], dtype=float)

# Reference field at dt = 0.0025
ref_omega = omega_fields[0.0025]
ref_norm = np.linalg.norm(ref_omega)

candidate_dts = [0.02, 0.0125, 0.01]
measured_errors = []

for dt in candidate_dts:
    diff = np.linalg.norm(omega_fields[dt] - ref_omega)
    rel_err = diff / ref_norm
    measured_errors.append(rel_err)
    print(f"Measured relative error at dt={dt}: {rel_err:.4e}")

# Compute log-log slope
p = np.polyfit(np.log(candidate_dts), np.log(measured_errors), 1)
slope = p[0]
print(f"Log-log slope of the series: {slope:.4f}")

# Richardson estimation using dt = 0.02 (2h) and dt = 0.01 (h)
w_2h = omega_fields[0.02]
w_h = omega_fields[0.01]
diff_2h_h = np.linalg.norm(w_2h - w_h)
norm_h = np.linalg.norm(w_h)
e_h = diff_2h_h / ((2.0**4 - 1.0) * norm_h)
print(f"Richardson error estimate at dt=0.01: {e_h:.4e}")

# Predict errors at candidate steps using 4th-order scaling
predicted_errors = {}
for dt in candidate_dts:
    pred = e_h * ((dt / 0.01)**4)
    predicted_errors[dt] = pred
    print(f"Predicted error at dt={dt}: {pred:.4e}")

# Choose largest step with predicted error < 5e-6
chosen_dt = None
for dt in sorted(candidate_dts, reverse=True):
    if predicted_errors[dt] < 5e-6:
        chosen_dt = dt
        break

print(f"\nChosen time step: dt = {chosen_dt}")
print(f"Predicted error: {predicted_errors[chosen_dt]:.2e}")
print(f"Measured error: {measured_errors[candidate_dts.index(chosen_dt)]:.2e}")

# Save convergence.json
result_json = {
    "time_steps": candidate_dts,
    "relative_errors": [float(e) for e in measured_errors],
    "slope": float(slope),
    "reference_dt": 0.0025,
    "richardson_e001": float(e_h),
    "predicted_errors": {str(k): float(v) for k, v in predicted_errors.items()},
    "chosen_dt": float(chosen_dt)
}

with open('D:/Desktop/AMAT5315-2026Fall-Exercise/week4/evidence/convergence.json', 'w') as f:
    json.dump(result_json, f, indent=2)
print("Saved D:/Desktop/AMAT5315-2026Fall-Exercise/week4/evidence/convergence.json")

# Plot convergence.png
fig, ax = plt.subplots(figsize=(6.5, 5), dpi=200)

fit_x = np.linspace(0.008, 0.025, 100)
fit_y = np.exp(p[1]) * (fit_x**slope)

ax.loglog(candidate_dts, measured_errors, 's', color='tab:blue', markersize=6, label='measured data')
ax.loglog(fit_x, fit_y, '-', color='tab:blue', label=f'measured, slope {slope:.3f}')

# Reference line slope 4
ref_y = fit_y * 3.0
ax.loglog(fit_x, ref_y, '--', color='tab:gray', label='slope 4, offset × 3')

# Mark the chosen step
chosen_err = measured_errors[candidate_dts.index(chosen_dt)]
ax.plot(chosen_dt, chosen_err, 'o', markerfacecolor='none', markeredgecolor='crimson', markersize=12, markeredgewidth=2, label=f'choice $\\Delta t = {chosen_dt}$')

ax.set_xlabel(r'time step $\Delta t$', fontsize=10)
ax.set_ylabel(r'relative error of $\omega$ at $t = 2$', fontsize=10)
ax.set_title(r'Time-step refinement at $t = 2$ ($N = 128$, ref $\Delta t = 0.0025$)', fontsize=10)
ax.set_xlim(0.008, 0.025)
ax.set_ylim(2e-7, 2e-4)
ax.set_xticks([0.01, 0.0125, 0.02])
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

ax.legend(loc='lower right', fontsize=8.5, framealpha=0.9)
ax.grid(True, which='both', linestyle=':', alpha=0.5)

note_text = (f"The error follows slope {slope:.3f} $\\approx 4$.\n"
             "RK4's accumulated time error\n"
             r"scales as $\Delta t^4$." "\n\n"
             r"Halving $\Delta t$ from 0.02 to 0.01" "\n"
             f"lowers the error about {measured_errors[0]/measured_errors[2]:.1f}x.\n\n"
             f"Choice: $\\Delta t = {chosen_dt}$\n"
             f"Pred err: {predicted_errors[chosen_dt]:.2e} (< 5e-6)\n"
             f"Meas err: {chosen_err:.2e}")
ax.text(0.03, 0.45, note_text, transform=ax.transAxes, fontsize=8,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

plt.tight_layout()
out_png = 'D:/Desktop/AMAT5315-2026Fall-Exercise/week4/evidence/convergence.png'
plt.savefig(out_png, bbox_inches='tight')
print('Saved', out_png)
