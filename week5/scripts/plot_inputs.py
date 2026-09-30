import unicodedata
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def main():
    with open("inputs/reflector.json") as f:
        exp = json.load(f)
        
    nx = exp["nx"]
    nz = exp["nz"]
    dx = exp["dx"]
    dt = exp["dt"]
    steps = exp["steps"]
    f0 = exp["source_frequency"]
    t0 = exp["source_peak_time"]
    amp = exp["source_amplitude"]
    w_sponge = exp["sponge_width"]
    length_unit_m = exp["length_unit_m"]
    time_unit_s = exp["time_unit_s"]
    
    # Coordinates in km
    x_km = np.arange(nx) * dx * (length_unit_m / 1000.0)
    z_km = np.arange(nz) * dx * (length_unit_m / 1000.0)
    
    bg = np.array(exp["background"])
    pert = np.array(exp["perturbation"])
    total_speed = bg + pert
    
    shots = np.array(exp["shots"])
    receivers = np.array(exp["receivers"])
    
    shots_x_km = shots[:, 0] * dx * (length_unit_m / 1000.0)
    shots_z_km = shots[:, 1] * dx * (length_unit_m / 1000.0)
    
    recv_x_km = receivers[:, 0] * dx * (length_unit_m / 1000.0)
    recv_z_km = receivers[:, 1] * dx * (length_unit_m / 1000.0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
    
    # 1. Geometry panel
    im = ax1.imshow(total_speed, extent=[x_km[0], x_km[-1], z_km[-1], z_km[0]],
                    cmap="viridis", vmin=1.5, vmax=2.1, aspect="equal")
    cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("Speed (km/s)")
    
    # Dashed line for inner edge of sponge
    x_sponge_min = w_sponge * dx * (length_unit_m / 1000.0)
    x_sponge_max = (nx - 1 - w_sponge) * dx * (length_unit_m / 1000.0)
    z_sponge_min = w_sponge * dx * (length_unit_m / 1000.0)
    z_sponge_max = (nz - 1 - w_sponge) * dx * (length_unit_m / 1000.0)
    
    ax1.plot([x_sponge_min, x_sponge_max, x_sponge_max, x_sponge_min, x_sponge_min],
             [z_sponge_min, z_sponge_min, z_sponge_max, z_sponge_max, z_sponge_min],
             '--', color="cyan", linewidth=1.2, label="Dashed line: inner\nedge of the sponge")
             
    # Receivers and Sources
    ax1.scatter(recv_x_km, recv_z_km, marker="v", edgecolors="white", facecolors="none", s=40, label="Receivers")
    ax1.scatter(shots_x_km, shots_z_km, marker="*", color="red", s=70, label="Sources")
    
    # Mark thin reflector
    # Find z of reflector
    refl_z_idx = np.where(pert > 0.05)[0]
    if len(refl_z_idx) > 0:
        z_refl = refl_z_idx[0] * dx * (length_unit_m / 1000.0)
        ax1.plot([x_sponge_min, x_sponge_max], [z_refl, z_refl], color="red", linewidth=2.0, label="Thin reflector")
        
    ax1.set_xlabel("Horizontal position (km)")
    ax1.set_ylabel("Depth (km)")
    ax1.set_title("Seismic acquisition")
    ax1.set_xlim(x_km[0], x_km[-1])
    ax1.set_ylim(z_km[-1], z_km[0])
    ax1.legend(loc="upper right", fontsize=8, framealpha=0.8)
    
    # 2. Source pulse panel
    # Time in seconds
    t_reduced = np.arange(steps) * dt
    t_sec = t_reduced * time_unit_s
    theta = np.pi * f0 * (t_reduced - t0)
    g_t = amp * (1.0 - 2.0 * theta**2) * np.exp(-theta**2)
    
    f0_hz = f0 / time_unit_s
    t0_sec = t0 * time_unit_s
    
    ax2.plot(t_sec, g_t, color="C0", linewidth=1.8)
    ax2.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Source pulse $g(t)$")
    ax2.set_title(f"Ricker pulse; peak frequency {f0_hz:.1f} Hz, peak time {t0_sec:.1f} s")
    ax2.set_xlim(0, 5)
    ax2.set_ylim(-0.5, 1.1)
    ax2.grid(True, linestyle=":", alpha=0.5)
    
    plt.tight_layout()
    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "inputs.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
