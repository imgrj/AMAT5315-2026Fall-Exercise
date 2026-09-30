import unicodedata
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp

def make_cluster(N, seed=42):
    rng = np.random.default_rng(seed)
    side = int(np.ceil(N**(1/3))) + 2
    pts = []
    for x in range(side):
        for y in range(side):
            for z in range(side):
                pts.append((x, y, z))
    pts = np.array(pts[:N], dtype=np.float64) * (2.0**(1/6))
    pts += rng.normal(0.0, 0.05, size=pts.shape)
    return pts

def compute_analytic_grad(coords, iu):
    grad = np.zeros_like(coords)
    diff = coords[iu[0]] - coords[iu[1]]
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    dU = 24.0 * (dist**(-7) - 2.0 * dist**(-13))
    forces = (dU / dist)[:, None] * diff
    np.add.at(grad, iu[0], forces)
    np.add.at(grad, iu[1], -forces)
    return grad.flatten()

def run_scaling():
    sizes = [64, 128, 256, 512, 1024]
    
    P_vals = []
    ratio_fwd_list = []
    ratio_rev_list = []
    max_err_fwd = 0.0
    max_err_rev = 0.0
    
    print("Running Lennard-Jones cluster gradient scaling benchmark...")
    
    for N in sizes:
        P = 3 * N
        P_vals.append(P)
        coords = make_cluster(N)
        iu = np.triu_indices(N, k=1)
        r_flat = jnp.array(coords.flatten())
        
        # Analytic gradient
        g_analytic = compute_analytic_grad(coords, iu)
        max_force = np.max(np.abs(g_analytic))
        
        # JAX Energy
        def energy(rf):
            r = rf.reshape((N, 3))
            diff = r[iu[0]] - r[iu[1]]
            dist = jnp.sqrt(jnp.sum(diff**2, axis=-1))
            inv6 = dist**(-6)
            return jnp.sum(4.0 * (inv6**2 - inv6))
            
        energy_fn = jax.jit(energy)
        grad_rev_fn = jax.jit(jax.grad(energy))
        
        # Single JVP function
        jvp_single = jax.jit(lambda v: jax.jvp(energy, (r_flat,), (v,))[1])
        
        # Warmup
        _ = energy_fn(r_flat).block_until_ready()
        g_rev = grad_rev_fn(r_flat).block_until_ready()
        v0 = jnp.zeros(P).at[0].set(1.0)
        _ = jvp_single(v0).block_until_ready()
        
        # Reverse mode error
        g_rev_np = np.array(g_rev)
        rel_err_rev = np.max(np.abs(g_rev_np - g_analytic)) / max_force
        max_err_rev = max(max_err_rev, rel_err_rev)
        
        # Compute forward mode gradient (for small P compute all, for large P compute sample to check error)
        # To verify forward mode error:
        eye = jnp.eye(P)
        if P <= 768:
            fwd_all = jax.jit(jax.vmap(lambda v: jax.jvp(energy, (r_flat,), (v,))[1]))
            g_fwd_np = np.array(fwd_all(eye))
            rel_err_fwd = np.max(np.abs(g_fwd_np - g_analytic)) / max_force
        else:
            # Check 100 sample directions
            sample_idx = np.random.choice(P, size=100, replace=False)
            fwd_sample = jax.jit(jax.vmap(lambda v: jax.jvp(energy, (r_flat,), (v,))[1]))
            g_fwd_sample = np.array(fwd_sample(eye[sample_idx]))
            rel_err_fwd = np.max(np.abs(g_fwd_sample - g_analytic[sample_idx])) / max_force
        max_err_fwd = max(max_err_fwd, rel_err_fwd)
        
        # Timings
        # Time energy
        n_repeats = 20 if N <= 256 else 10
        t0 = time.perf_counter()
        for _ in range(n_repeats):
            _ = energy_fn(r_flat).block_until_ready()
        t_energy = (time.perf_counter() - t0) / n_repeats
        
        # Time reverse mode
        t0 = time.perf_counter()
        for _ in range(n_repeats):
            _ = grad_rev_fn(r_flat).block_until_ready()
        t_reverse = (time.perf_counter() - t0) / n_repeats
        
        # Time forward mode: one input direction at a time
        # Time single JVP to get true cost of one input direction
        n_jvp = 20 if N <= 256 else 10
        t0 = time.perf_counter()
        for _ in range(n_jvp):
            _ = jvp_single(v0).block_until_ready()
        t_one_jvp = (time.perf_counter() - t0) / n_jvp
        t_forward = P * t_one_jvp
        
        ratio_rev = t_reverse / t_energy
        ratio_fwd = t_forward / t_energy
        
        ratio_rev_list.append(ratio_rev)
        ratio_fwd_list.append(ratio_fwd)
        
        print(f"N = {N:4d}, P = {P:4d} | t_energy = {t_energy*1e3:.3f} ms | "
              f"ratio_rev = {ratio_rev:.3f} | ratio_fwd = {ratio_fwd:.1f} | "
              f"err_rev = {rel_err_rev:.2e} | err_fwd = {rel_err_fwd:.2e}")
              
    print("\nSummary:")
    print(f"Largest reverse-mode relative error: {max_err_rev:.4e}")
    print(f"Largest forward-mode relative error: {max_err_fwd:.4e}")
    print(f"At P = 3072: ratio_fwd = {ratio_fwd_list[-1]:.1f}, ratio_rev = {ratio_rev_list[-1]:.2f}, factor = {ratio_fwd_list[-1]/ratio_rev_list[-1]:.1f}")
    
    # Plot log-log scaling figure
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.loglog(P_vals, ratio_fwd_list, 'o-', color="C0", label="forward mode, one JVP per input", linewidth=1.5, markersize=6)
    ax.loglog(P_vals, ratio_rev_list, 's-', color="C1", label="reverse mode, one VJP", linewidth=1.5, markersize=6)
    
    # Proportional to P line through forward mode
    c_prop = ratio_fwd_list[0] / P_vals[0]
    p_line = np.array(P_vals)
    ax.loglog(p_line, c_prop * p_line, ':', color="gray", label="proportional to $P$", linewidth=1.2)
    
    ax.set_xticks(P_vals)
    ax.set_xticklabels([str(p) for p in P_vals])
    ax.set_xlabel("Inputs $P = 3N$")
    ax.set_ylabel("Gradient time / energy time")
    ax.set_title("Gradient of a Lennard-Jones cluster energy")
    ax.set_ylim(0.5, max(ratio_fwd_list) * 2.0)
    ax.legend(loc="upper left", framealpha=0.8)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    
    plt.tight_layout()
    out_dir = Path("artifacts/ad")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "scaling.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")

if __name__ == "__main__":
    run_scaling()
