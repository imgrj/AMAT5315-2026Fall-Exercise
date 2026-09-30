import unicodedata
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def main():
    exp_path = "inputs/reflector.json"
    with open(exp_path) as f:
        exp = json.load(f)
        
    traces = np.load("artifacts/forward/traces.npy") # [shots, steps, receivers]
    
    # 1. Print verification metrics
    total_l2 = np.linalg.norm(traces)
    print(f"Total traces L2 norm: {total_l2:.6f} (reference: 11.574770)")
    rel_err_l2 = abs(total_l2 - 11.574770) / 11.574770
    print(f"L2 norm relative error: {rel_err_l2:.2e}")
    
    shots = exp["shots"]
    receivers = np.array(exp["receivers"])
    dt = exp["dt"]
    length_unit_m = exp["length_unit_m"]
    time_unit_s = exp["time_unit_s"]
    dx = exp["dx"]
    
    for s_idx in range(len(shots)):
        shot_tr = traces[s_idx]
        max_flat_idx = np.argmax(np.abs(shot_tr))
        step_idx, recv_idx = np.unravel_index(max_flat_idx, shot_tr.shape)
        val = shot_tr[step_idx, recv_idx]
        rx = receivers[recv_idx, 0] * dx * (length_unit_m / 1000.0)
        rz = receivers[recv_idx, 1] * dx * (length_unit_m / 1000.0)
        t_sec = (step_idx + 1) * dt * time_unit_s
        print(f"Shot {s_idx}: max pressure = {val:.8f} at trace index {step_idx} (step {step_idx+1}, t={t_sec*10:.1f} red units = {t_sec:.2f} s), receiver {recv_idx} at ({rx:.1f}, {rz:.1f}) km")

    # 2. Plot shot gathers: artifacts/forward/gathers.png
    recv_x_km = receivers[:, 0] * dx * (length_unit_m / 1000.0)
    time_sec = (np.arange(exp["steps"]) + 1) * dt * time_unit_s
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)
    vmax = 0.6
    vmin = -0.6
    
    for s_idx in range(3):
        ax = axes[s_idx]
        source_x_km = shots[s_idx][0] * dx * (length_unit_m / 1000.0)
        # traces[s_idx] shape is [steps, receivers]
        im = ax.imshow(traces[s_idx], extent=[recv_x_km[0], recv_x_km[-1], time_sec[-1], time_sec[0]],
                       aspect="auto", cmap="coolwarm", vmin=vmin, vmax=vmax)
        ax.set_title(f"Shot {s_idx}; source x = {source_x_km:.1f} km")
        ax.set_xlabel("Receiver position (km)")
        if s_idx == 0:
            ax.set_ylabel("Time (s)")
        if s_idx == 1:
            ax.text(1.5, 0.8, "Strong early event", fontsize=9, color="black")
            ax.text(1.5, 3.8, "Later negative lobe", fontsize=9, color="black")
            
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Pressure (common scale; arbitrary units)")
    
    gathers_path = Path("artifacts/forward/gathers.png")
    plt.savefig(gathers_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {gathers_path}")

    # 3. Plot step 150 forward wavefield and echo
    with open("artifacts/forward/run.json") as f:
        run_meta = json.load(f)
    rec_steps = run_meta["recording"]["steps"]
    idx_150 = rec_steps.index(150)
    
    wf = np.load("artifacts/forward/wavefield.npy") # [n_frames, nz, nx]
    echo = np.load("artifacts/forward/echo.npy")
    
    wf_150 = wf[idx_150]
    echo_150 = echo[idx_150]
    
    nx, nz = exp["nx"], exp["nz"]
    x_km = np.arange(nx) * dx * (length_unit_m / 1000.0)
    z_km = np.arange(nz) * dx * (length_unit_m / 1000.0)
    
    # Wavefield step 150 plot
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(wf_150, extent=[x_km[0], x_km[-1], z_km[-1], z_km[0]],
                   cmap="coolwarm", vmin=-0.25, vmax=0.25, aspect="equal")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pressure (arbitrary units)")
    
    # Overlays
    shot0_x = shots[0][0] * dx * (length_unit_m / 1000.0)
    shot0_z = shots[0][1] * dx * (length_unit_m / 1000.0)
    ax.scatter(recv_x_km, receivers[:, 1] * dx * (length_unit_m / 1000.0),
               marker="v", edgecolors="black", facecolors="white", s=30, label="Receivers")
    ax.scatter([shot0_x], [shot0_z], marker="*", color="red", s=80, label="Active source")
    
    ax.set_title("Forward wavefield; shot 0, t = 3.00 s")
    ax.set_xlabel("Horizontal position (km)")
    ax.set_ylabel("Depth (km)")
    ax.set_xlim(x_km[0], x_km[-1])
    ax.set_ylim(z_km[-1], z_km[0])
    ax.legend(loc="upper right", fontsize=8)
    
    wf_path = Path("artifacts/forward/wavefield.png")
    plt.tight_layout()
    plt.savefig(wf_path, dpi=200)
    plt.close()
    print(f"Saved {wf_path}")
    
    # Echo step 150 plot
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(echo_150, extent=[x_km[0], x_km[-1], z_km[-1], z_km[0]],
                   cmap="coolwarm", vmin=-0.007, vmax=0.007, aspect="equal")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pressure difference (arbitrary units)")
    
    # Mark thin reflector at z=2.1 km
    ax.plot([x_km[0], x_km[-1]], [2.1, 2.1], "--", color="cyan", linewidth=1.5, label="Reflector")
    ax.scatter(recv_x_km, receivers[:, 1] * dx * (length_unit_m / 1000.0),
               marker="v", edgecolors="black", facecolors="white", s=30, label="Receivers")
    ax.scatter([shot0_x], [shot0_z], marker="*", color="red", s=80, label="Active source")
    
    ax.set_title("Reflector echo; shot 0, t = 3.00 s")
    ax.set_xlabel("Horizontal position (km)")
    ax.set_ylabel("Depth (km)")
    ax.set_xlim(x_km[0], x_km[-1])
    ax.set_ylim(z_km[-1], z_km[0])
    ax.legend(loc="upper right", fontsize=8)
    
    echo_path = Path("artifacts/forward/echo.png")
    plt.tight_layout()
    plt.savefig(echo_path, dpi=200)
    plt.close()
    print(f"Saved {echo_path}")
    print(f"Max echo pressure at step 150: {np.max(np.abs(echo_150)):.6f} vs max wavefield pressure: {np.max(np.abs(wf_150)):.6f}")

if __name__ == "__main__":
    main()
