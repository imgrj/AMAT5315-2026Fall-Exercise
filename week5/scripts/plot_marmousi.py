import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def main():
    exp_path = Path("inputs/marmousi.json")
    with open(exp_path) as f:
        exp = json.load(f)

    dx = exp["dx"] # 0.25
    dt = exp["dt"] # 0.03
    steps = exp["steps"] # 1200
    length_unit_m = exp.get("length_unit_m", 100.0)
    time_unit_s = exp.get("time_unit_s", 0.1)

    km_per_cell = dx * length_unit_m / 1000.0 # 0.025 km
    nx = exp["nx"] # 805
    nz = exp["nz"] # 269

    x_km = np.arange(nx) * km_per_cell
    z_km = np.arange(nz) * km_per_cell

    bg = np.array(exp["background"]) # already in km/s
    pert = np.array(exp["perturbation"])

    born_data = np.load("artifacts/marmousi-born/born_data.npy") # [9, 1200, 91]
    image = np.load("artifacts/marmousi-image/image.npy") # [269, 805]

    # Shot gather at x = 10 km (shot index 4)
    # receivers: x in [40, ..., 760] -> [1.0 km, 19.0 km]
    receivers = np.array(exp["receivers"])
    recv_x_km = receivers[:, 0] * km_per_cell
    times_sec = np.arange(steps) * dt * time_unit_s # [0, 3.6] s

    shot4_gather = born_data[4] # [1200, 91]

    fig, axes = plt.subplots(4, 1, figsize=(10, 13), constrained_layout=True)

    # 1. Background velocity
    im0 = axes[0].imshow(bg, extent=[x_km[0], x_km[-1], z_km[-1], z_km[0]],
                         cmap="viridis", aspect="auto")
    axes[0].set_title("Smoothed Marmousi Background Velocity $c_0(x, z)$ [km/s]", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Depth z [km]")
    fig.colorbar(im0, ax=axes[0], label="Velocity [km/s]", shrink=0.8, pad=0.02)

    # 2. Perturbation
    vmax_p = np.percentile(np.abs(pert), 99)
    im1 = axes[1].imshow(pert, extent=[x_km[0], x_km[-1], z_km[-1], z_km[0]],
                         cmap="coolwarm", vmin=-vmax_p, vmax=vmax_p, aspect="auto")
    axes[1].set_title("Marmousi Velocity Perturbation $m(x, z)$ [km/s]", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Depth z [km]")
    fig.colorbar(im1, ax=axes[1], label="Perturbation [km/s]", shrink=0.8, pad=0.02)

    # 3. Born synthetic shot gather at x=10 km
    vmax_g = np.percentile(np.abs(shot4_gather), 98)
    im2 = axes[2].imshow(shot4_gather, extent=[recv_x_km[0], recv_x_km[-1], times_sec[-1], times_sec[0]],
                         cmap="gray", vmin=-vmax_g, vmax=vmax_g, aspect="auto")
    axes[2].set_title("Born Synthetic Shot Gather at Shot $x = 10.0$ km", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("Two-Way Time [s]")
    fig.colorbar(im2, ax=axes[2], label="Pressure", shrink=0.8, pad=0.02)

    # 4. Checkpointed migrated image without artificial gain
    vmax_img = np.percentile(np.abs(image), 99.5)
    im3 = axes[3].imshow(image, extent=[x_km[0], x_km[-1], z_km[-1], z_km[0]],
                         cmap="seismic", vmin=-vmax_img, vmax=vmax_img, aspect="auto")
    axes[3].set_title("Checkpointed Reverse-Time Migration Image $J^T d_{\\mathrm{born}}$ (Raw, No Gain)", fontsize=11, fontweight="bold")
    axes[3].set_xlabel("Offset x [km]")
    axes[3].set_ylabel("Depth z [km]")
    fig.colorbar(im3, ax=axes[3], label="Image Amplitude", shrink=0.8, pad=0.02)

    out_file = Path("artifacts/marmousi.png")
    plt.savefig(out_file, dpi=180)
    plt.close()
    print(f"Saved {out_file}")

if __name__ == "__main__":
    main()
