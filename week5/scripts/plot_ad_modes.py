import unicodedata
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
import jax
from jax import config
config.update("jax_enable_x64", True)

def analytic_derivative(r):
    return 24.0 * (r**(-7) - 2.0 * r**(-13))

def lj_energy(r):
    a = r**(-6)
    b = a**2
    c = b - a
    return 4.0 * c

def forward_mode(r):
    # Vectorized or loop over r
    # Node by node
    dot_r = 1.0
    a = r**(-6)
    dot_a = -6.0 * (r**(-7)) * dot_r
    b = a**2
    dot_b = 2.0 * a * dot_a
    c = b - a
    dot_c = dot_b - dot_a
    U = 4.0 * c
    dot_U = 4.0 * dot_c
    return dot_U

def reverse_mode(r):
    a = r**(-6)
    b = a**2
    c = b - a
    U = 4.0 * c
    
    bar_U = 1.0
    bar_c = 4.0 * bar_U
    bar_b = bar_c
    bar_a = -bar_c + 2.0 * a * bar_b
    bar_r = -6.0 * (r**(-7)) * bar_a
    return bar_r

def finite_difference(r, h=1e-6):
    return (lj_energy(r + h) - lj_energy(r - h)) / (2.0 * h)

def main():
    r_vals = np.linspace(0.95, 2.5, 601)
    
    exact = analytic_derivative(r_vals)
    fwd = forward_mode(r_vals)
    rev = reverse_mode(r_vals)
    fd = finite_difference(r_vals, h=1e-6)
    
    err_fwd = np.abs(fwd - exact)
    err_rev = np.abs(rev - exact)
    err_fd = np.abs(fd - exact)
    
    max_err_fwd = np.max(err_fwd)
    max_err_rev = np.max(err_rev)
    max_err_fd = np.max(err_fd)
    
    print(f"Maximum forward mode error: {max_err_fwd:.4e}")
    print(f"Maximum reverse mode error: {max_err_rev:.4e}")
    print(f"Maximum finite difference error: {max_err_fd:.4e}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    
    # Left panel: dU/dr
    ax1.plot(r_vals, exact, label="analytic", color="gray", linewidth=2.5)
    ax1.plot(r_vals, fwd, label="forward mode", color="C0", linestyle="-")
    ax1.plot(r_vals, rev, label="reverse mode", color="C1", linestyle="--")
    ax1.plot(r_vals, fd, label="finite difference", color="C2", linestyle=":")
    ax1.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax1.set_xlabel("Separation $r$")
    ax1.set_ylabel("$dU/dr$")
    ax1.set_title("Derivative; zero at $r = 2^{1/6}$")
    ax1.set_xlim(0.9, 2.6)
    ax1.set_ylim(-3.5, 3.2)
    ax1.legend(loc="upper right", framealpha=0.8)
    ax1.grid(True, linestyle=":", alpha=0.5)
    
    # Right panel: absolute error
    # To avoid log(0), clip at 1e-17
    clip_floor = 1e-17
    ax2.plot(r_vals, np.maximum(err_fwd, clip_floor), label="forward mode", color="C0", linestyle="-")
    ax2.plot(r_vals, np.maximum(err_rev, clip_floor), label="reverse mode", color="C1", linestyle="--")
    ax2.plot(r_vals, np.maximum(err_fd, clip_floor), label="finite difference, $h = 10^{-6}$", color="gray", linestyle=":")
    ax2.set_yscale("log")
    ax2.set_xlabel("Separation $r$")
    ax2.set_ylabel("Absolute error")
    ax2.set_title("Error against the analytic derivative")
    ax2.set_xlim(0.9, 2.6)
    ax2.set_ylim(1e-17, 1e-4)
    ax2.legend(loc="upper right", framealpha=0.8)
    ax2.grid(True, which="both", linestyle=":", alpha=0.5)
    
    plt.tight_layout()
    out_dir = Path("artifacts/ad")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "modes.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
