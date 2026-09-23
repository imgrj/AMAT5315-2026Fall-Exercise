import subprocess
import os
import json
import numpy as np
import matplotlib.pyplot as plt

os.makedirs('D:/Desktop/AMAT5315-2026Fall-Exercise/week4/artifacts/scan', exist_ok=True)

# 1. Run Taylor-Green cases
print("Running Taylor-Green scan...")
cmd_tg = "field taylor-green --n 64"
res_tg = subprocess.run(cmd_tg, shell=True, capture_output=True, text=True, check=True)
tg_json = res_tg.stdout

for dt in [0.032, 0.033]:
    out_dir = f"D:/Desktop/AMAT5315-2026Fall-Exercise/week4/artifacts/scan/tg_rk4_dt{dt}"
    cmd_fluid = f"fluid --method rk4 --nu 0.1 --dt {dt} --t-end 8 --every 0.5 --out {out_dir}"
    tsv_path = f"{out_dir}.tsv"
    p = subprocess.run(cmd_fluid, shell=True, input=tg_json, text=True, capture_output=True)
    with open(tsv_path, "w") as f:
        f.write(p.stdout)
    print(f"TG RK4 dt={dt}: exit code {p.returncode}")

# 2. Run Random Flow cases
print("Running Random Flow scan...")
cmd_rf = "field random --n 128 --seed 2026 --k-min 2 --k-max 6"
res_rf = subprocess.run(cmd_rf, shell=True, capture_output=True, text=True, check=True)
rf_json = res_rf.stdout

rf_data = json.loads(rf_json)
u = np.array(rf_data['u'])
v = np.array(rf_data['v'])
speeds = np.sqrt(u**2 + v**2)
u_max = np.max(speeds)
print(f"Largest speed of the random initial field: U_max = {u_max:.4f}")

# Euler dt = 0.01
out_dir_euler = "D:/Desktop/AMAT5315-2026Fall-Exercise/week4/artifacts/scan/rf_euler_dt0.01"
cmd_fluid = f"fluid --method euler --nu 0.004 --dt 0.01 --t-end 10 --every 0.5 --out {out_dir_euler}"
p_e = subprocess.run(cmd_fluid, shell=True, input=rf_json, text=True, capture_output=True)
with open(f"{out_dir_euler}.tsv", "w") as f:
    f.write(p_e.stdout)
print(f"RF Euler dt=0.01: exit code {p_e.returncode}")

# RK4 scans: start with 0.038 and 0.040
dt_stable = None
dt_unstable = None

for dt in [0.038, 0.040, 0.035, 0.030, 0.042, 0.045]:
    out_dir = f"D:/Desktop/AMAT5315-2026Fall-Exercise/week4/artifacts/scan/rf_rk4_dt{dt}"
    cmd_fluid = f"fluid --method rk4 --nu 0.004 --dt {dt} --t-end 10 --every 0.5 --out {out_dir}"
    p = subprocess.run(cmd_fluid, shell=True, input=rf_json, text=True, capture_output=True)
    with open(f"{out_dir}.tsv", "w") as f:
        f.write(p.stdout)
    print(f"RF RK4 dt={dt}: exit code {p.returncode}")
    if p.returncode == 0 and dt_stable is None:
        dt_stable = dt
    elif p.returncode != 0 and dt_unstable is None:
        dt_unstable = dt

print(f"Selected for RF plot: stable dt={dt_stable}, unstable dt={dt_unstable}")
