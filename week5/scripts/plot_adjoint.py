import unicodedata
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def main():
    with open("inputs/reflector.json") as f:
        exp = json.load(f)
        
    born_data = np.load("artifacts/born/born_data.npy")
    image = np.load("artifacts/adjoint/image.npy")
    pert = np.array(exp["perturbation"])
    
    # 1. Dot-product identity
    left = np.sum(born_data**2)
    right = np.sum(pert * image)
    rel_diff = abs(left - right) / max(abs(left), abs(right))
    
    print(f"Dot-product identity evaluation:")
    print(f"  left  (||d||^2)       = {left:.8f}")
    print(f"  right (<m, J^T d>)    = {right:.8f}")
    print(f"  relative difference   = {rel_diff:.2e}")
    
    # 2. Draw artifacts/adjoint/image.png
    # Window: x=7..33, z=10..33 inclusive
    dx = exp["dx"]
    length_unit_m = exp["length_unit_m"]
    km_per_cell = dx * length_unit_m / 1000.0
    
    x_min, x_max = 7, 33
    z_min, z_max = 10, 33
    
    sub_pert = pert[z_min:z_max+1, x_min:x_max+1]
    sub_image = image[z_min:z_max+1, x_min:x_max+1]
    
    x_km = np.arange(x_min, x_max + 1) * km_per_cell
    z_km = np.arange(z_min, z_max + 1) * km_per_cell
    
    # Row L2 norm of the windowed image
    row_l2 = np.sqrt(np.sum(sub_image**2, axis=1))
    
    peak_idx = np.argmax(row_l2)
    peak_z_km = z_km[peak_idx]
    true_z_km = 2.1
    depth_diff = abs(peak_z_km - true_z_km)
    
    # Check side lobes
    # Find secondary peak in shallower region z < 2.0 km
    shallow_mask = z_km < 1.9
    shallow_max = np.max(row_l2[shallow_mask]) if np.any(shallow_mask) else 0.0
    side_lobe_ratio = (shallow_max / row_l2[peak_idx]) * 100.0
    
    print(f"Profile peak depth: {peak_z_km:.1f} km, true depth: {true_z_km:.1f} km, difference: {depth_diff:.1f} km")
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4.8), gridspec_kw={'width_ratios': [1, 1, 0.8]})
    
    # Panel 1: Perturbation
    im1 = ax1.imshow(sub_pert, extent=[x_km[0], x_km[-1], z_km[-1], z_km[0]],
                     cmap="coolwarm", vmin=-0.15, vmax=0.15, aspect="auto")
    ax1.set_title("1. Known reflector")
    ax1.set_xlabel("Horizontal position (km)")
    ax1.set_ylabel("Depth (km)")
    ax1.axhline(true_z_km, color="black", linestyle="--", linewidth=1.0)
    ax1.text(x_km[0] + 0.1, true_z_km + 0.25, f"Known depth: {true_z_km:.1f} km", fontsize=8)
    cbar1 = plt.colorbar(im1, ax=ax1, orientation="horizontal", pad=0.2, shrink=0.8)
    cbar1.set_label("Velocity change (km/s)")
    
    # Panel 2: Raw signed RTM image
    vlim = max(abs(np.min(sub_image)), abs(np.max(sub_image)))
    im2 = ax2.imshow(sub_image, extent=[x_km[0], x_km[-1], z_km[-1], z_km[0]],
                     cmap="coolwarm", vmin=-vlim, vmax=vlim, aspect="auto")
    ax2.set_title("2. Raw signed RTM image")
    ax2.set_xlabel("Horizontal position (km)")
    ax2.text(x_km[0] + 0.5, true_z_km - 0.45, "Negative side lobe", fontsize=8)
    ax2.text(x_km[0] + 0.4, true_z_km + 0.05, "Broad reflector response", fontsize=8)
    ax2.text(x_km[0] + 0.5, true_z_km + 0.65, "Negative side lobe", fontsize=8)
    cbar2 = plt.colorbar(im2, ax=ax2, orientation="horizontal", pad=0.2, shrink=0.8)
    cbar2.set_label("Image (arbitrary units)")
    
    # Panel 3: Image depth profile
    ax3.plot(row_l2, z_km, color="C0", linewidth=1.5)
    ax3.axhline(true_z_km, color="red", linestyle="--", linewidth=1.2)
    ax3.scatter([row_l2[peak_idx]], [peak_z_km], color="C0", s=30)
    ax3.text(row_l2[peak_idx] * 0.5, peak_z_km - 0.1, f"Peak: {peak_z_km:.1f} km", fontsize=8)
    ax3.set_title("3. Depth profile")
    ax3.set_xlabel("Row L2 norm\n(arbitrary units)")
    ax3.set_ylabel("Depth (km)")
    ax3.set_ylim(z_km[-1], z_km[0])
    ax3.grid(True, linestyle=":", alpha=0.5)
    
    fig.suptitle(f"RTM locates the reflector; image amplitudes are not velocity\nShallow side-lobe norm: {side_lobe_ratio:.0f}% of peak; depth error: {depth_diff:.1f} km", fontsize=11)
    
    plt.tight_layout()
    img_out = Path("artifacts/adjoint/image.png")
    plt.savefig(img_out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {img_out}")
    
    # 3. Save artifacts/adjoint/wavefield.png at step 132 (2.64 s)
    with open("artifacts/adjoint/run.json") as f:
        run_meta = json.load(f)
    steps_list = run_meta["recording"]["steps"]
    idx_132 = steps_list.index(132)
    
    adj_wf = np.load("artifacts/adjoint/wavefield.npy")
    adj_frame_132 = adj_wf[idx_132]
    
    all_nx = exp["nx"]
    all_nz = exp["nz"]
    full_x_km = np.arange(all_nx) * km_per_cell
    full_z_km = np.arange(all_nz) * km_per_cell
    
    fig, ax = plt.subplots(figsize=(6, 5))
    vlim_wf = 0.08
    im = ax.imshow(adj_frame_132, extent=[full_x_km[0], full_x_km[-1], full_z_km[-1], full_z_km[0]],
                   cmap="coolwarm", vmin=-vlim_wf, vmax=vlim_wf, aspect="equal")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pressure adjoint (arbitrary units)")
    
    ax.plot([full_x_km[0], full_x_km[-1]], [2.1, 2.1], "--", color="cyan", linewidth=1.2, label="Reflector")
    ax.set_title("Adjoint field at step 132; 2.64 s")
    ax.set_xlabel("Horizontal position (km)")
    ax.set_ylabel("Depth (km)")
    ax.set_xlim(full_x_km[0], full_x_km[-1])
    ax.set_ylim(full_z_km[-1], full_z_km[0])
    ax.legend(loc="lower left", fontsize=8)
    
    plt.tight_layout()
    wf_out = Path("artifacts/adjoint/wavefield.png")
    plt.savefig(wf_out, dpi=200)
    plt.close()
    print(f"Saved {wf_out}")

if __name__ == "__main__":
    main()
