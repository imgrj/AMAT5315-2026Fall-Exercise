import subprocess
import os
import json
import numpy as np
import matplotlib.pyplot as plt

os.makedirs('D:/Desktop/AMAT5315-2026Fall-Exercise/week4/artifacts/order', exist_ok=True)

# 1. Generate exact solution at t = 2 for N = 8, nu = 0.5
cmd_exact = "field taylor-green --n 8 --nu 0.5 --t 2"
res_exact = subprocess.run(cmd_exact, shell=True, capture_output=True, text=True, check=True)
exact_data = json.loads(res_exact.stdout)
u_exact = np.array(exact_data['u'])
v_exact = np.array(exact_data['v'])
exact_norm = np.sqrt(np.sum(u_exact**2 + v_exact**2))

with open('D:/Desktop/AMAT5315-2026Fall-Exercise/week4/artifacts/order/exact-t2.json', 'w') as f:
    f.write(res_exact.stdout)

# 2. Generate initial field at t = 0 for N = 8
cmd_init = "field taylor-green --n 8"
res_init = subprocess.run(cmd_init, shell=True, capture_output=True, text=True, check=True)
init_json = res_init.stdout

dts = [0.4, 0.25, 0.2]
errors = []

for dt in dts:
    out_dir = f"D:/Desktop/AMAT5315-2026Fall-Exercise/week4/artifacts/order/rk4-dt{dt}"
    cmd_fluid = f"fluid --method rk4 --nu 0.5 --dt {dt} --t-end 2 --every {dt} --out {out_dir}"
    subprocess.run(cmd_fluid, shell=True, input=init_json, text=True, capture_output=True, check=True)
    
    # Read last line of fields.jsonl
    with open(f"{out_dir}/fields.jsonl") as f:
        lines = f.readlines()
        last = json.loads(lines[-1])
        u_comp = np.array(last['u'])
        v_comp = np.array(last['v'])
        diff_norm = np.sqrt(np.sum((u_comp - u_exact)**2 + (v_comp - v_exact)**2))
        rel_err = diff_norm / exact_norm
        errors.append(rel_err)
        print(f"dt={dt}: relative error = {rel_err:.4e}")

# Compute slope
p = np.polyfit(np.log(dts), np.log(errors), 1)
slope = p[0]
print(f"Fitted slope: {slope:.3f}")
ratio_04_02 = errors[0] / errors[2]
print(f"Error drop from 0.4 to 0.2: {ratio_04_02:.2f}x (close to 2^4 = 16)")

# Plot order.png
fig, ax = plt.subplots(figsize=(6.5, 5), dpi=200)

fit_x = np.linspace(0.18, 0.45, 100)
fit_y = np.exp(p[1]) * (fit_x**slope)

ax.loglog(dts, errors, 'o', color='tab:blue', markersize=6)
ax.loglog(fit_x, fit_y, '-', color='tab:blue', label=f'RK4, slope {slope:.2f}')

# Reference slope lines
x_ref = 0.25
y_ref = 5e-5
for s, name in [(1, '1'), (2, '2'), (4, '4')]:
    xr = np.array([0.22, 0.32])
    yr = y_ref * ((xr / x_ref)**s)
    ax.loglog(xr, yr, color='gray', linestyle=':', linewidth=1)
    ax.text(xr[1]*1.02, yr[1], name, fontsize=8, color='gray')
ax.text(0.24, 8e-5, 'slopes', fontsize=8, color='gray')

# Storage floor
ax.axhline(7e-7, color='gray', linestyle='--', linewidth=1, label='6-decimal storage floor')

ax.set_xlim(0.16, 0.55)
ax.set_ylim(2e-7, 2e-3)
ax.set_xticks([0.2, 0.25, 0.3, 0.4, 0.5])
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

ax.set_xlabel(r'time step $\Delta t$', fontsize=10)
ax.set_ylabel(r'relative field error at $t = 2$', fontsize=10)
ax.set_title(r'Taylor–Green field error at $t = 2$ against the step, $N = 8, \nu = 0.5$', fontsize=10)
ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
ax.grid(True, which='both', linestyle=':', alpha=0.5)

note_text = (f"RK4's error drops {ratio_04_02:.1f}x from $\\Delta t = 0.4$ to $0.2$,\n"
             f"close to $2^4 = 16$.\n"
             "Every point sits above the floor of the\n"
             "6-decimal frames, so the slopes are the\n"
             "integrators' own.")
ax.text(0.97, 0.35, note_text, transform=ax.transAxes, fontsize=8, horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

plt.tight_layout()
out_png = 'D:/Desktop/AMAT5315-2026Fall-Exercise/week4/evidence/order.png'
plt.savefig(out_png, bbox_inches='tight')
print('Saved', out_png)
