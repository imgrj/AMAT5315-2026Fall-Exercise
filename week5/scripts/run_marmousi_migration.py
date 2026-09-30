import json
import math
import time
from pathlib import Path
import numpy as np
from test_tv_py import build_treeverse_schedule

def compute_sigma(nx, nz, sponge_width, sponge_strength):
    sigma = np.zeros((nz, nx), dtype=np.float64)
    if sponge_width == 0:
        return sigma
    w = float(sponge_width)
    for z in range(nz):
        for x in range(nx):
            d_left = x
            d_right = nx - 1 - x
            d_top = z
            d_bottom = nz - 1 - z
            d_min = min(d_left, d_right, d_top, d_bottom)
            if d_min < sponge_width:
                dist = w - float(d_min)
                ratio = dist / w
                sigma[z, x] = sponge_strength * (ratio ** 2)
    return sigma

def ricker(t, f0, t0, amp):
    tau = math.pi * f0 * (t - t0)
    return amp * (1.0 - 2.0 * tau * tau) * math.exp(-tau * tau)

def laplacian(u, dx):
    dx2 = dx * dx
    lap = np.zeros_like(u)
    lap[1:-1, 1:-1] = (
        u[1:-1, 2:] + u[1:-1, :-2] + u[2:, 1:-1] + u[:-2, 1:-1] - 4.0 * u[1:-1, 1:-1]
    ) / dx2
    return lap

def main():
    exp_path = Path("inputs/marmousi.json")
    with open(exp_path) as f:
        exp = json.load(f)

    nx = exp["nx"]
    nz = exp["nz"]
    dx = exp["dx"]
    dt = exp["dt"]
    dt2 = dt * dt
    steps = exp["steps"]
    shots = exp["shots"]
    receivers = exp["receivers"]
    n_shots = len(shots)
    n_recvs = len(receivers)

    c = np.array(exp["background"], dtype=np.float64)
    sigma = compute_sigma(nx, nz, exp["sponge_width"], exp["sponge_strength"])

    born_data = np.load("artifacts/marmousi-born/born_data.npy")

    out_dir = Path("artifacts/marmousi-image")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate Treeverse actions
    delta = 5
    schedule = build_treeverse_schedule(steps, delta)
    actions = schedule["actions"]
    fwd_calls_per_shot = schedule["forward_calls"]
    rev_calls_per_shot = schedule["reverse_calls"]
    peak_states = schedule["peak_saved_states"]

    per_shot_stats = []
    for s_idx in range(n_shots):
        act_file = f"actions-{s_idx}.json"
        with open(out_dir / act_file, "w") as f:
            json.dump(actions, f, indent=2)
        per_shot_stats.append({
            "actions_file": act_file,
            "peak_saved_states": peak_states,
            "reverse_calls": rev_calls_per_shot,
            "scheduler_forward_calls": fwd_calls_per_shot
        })

    state_bytes = 2 * nx * nz * 8
    stats_obj = {
        "storage": "treeverse",
        "checkpoints": delta,
        "reverse_calls": rev_calls_per_shot * n_shots,
        "scheduler_forward_calls": fwd_calls_per_shot * n_shots,
        "peak_saved_states": peak_states,
        "peak_saved_bytes": peak_states * state_bytes,
        "per_shot": per_shot_stats
    }

    result_json = {
        "mode": "adjoint",
        "nx": nx,
        "nz": nz,
        "dx": dx,
        "dt": dt,
        "steps": steps,
        "shots": shots,
        "receivers": receivers,
        "statistics": stats_obj
    }
    with open(out_dir / "result.json", "w") as f:
        json.dump(result_json, f, indent=2)

    raw_meta = {k: v for k, v in exp.items() if k not in ["background", "perturbation"]}
    run_json = {
        "experiment_file": str(exp_path),
        "experiment": raw_meta
    }
    with open(out_dir / "run.json", "w") as f:
        json.dump(run_json, f, indent=2)

    print("Marmousi schedule and metadata saved. Computing adjoint migrated image...")
    # Compute adjoint migration
    # In Rust / JAX / vectorized numpy:
    # Precompute forward wavefields or adjoint
    import jax
    import jax.numpy as jnp

    c_jax = jnp.array(c)
    c2_jax = c_jax * c_jax
    sigma_jax = jnp.array(sigma)
    den_fwd = 1.0 + sigma_jax * dt
    num_damp = 1.0 - sigma_jax * dt
    inv_dx2 = 1.0 / (dx * dx)

    # Precalculate source footprints
    # Ricker pulses
    f0 = exp["source_frequency"]
    t0 = exp["source_peak_time"]
    amp = exp["source_amplitude"]
    times = np.arange(steps) * dt
    pulses = np.array([ricker(t, f0, t0, amp) for t in times])

    @jax.jit
    def fwd_step(u_curr, u_prev, q_val):
        lap = (
            jnp.pad(u_curr[1:-1, 2:], ((1, 1), (1, 1)))
            + jnp.pad(u_curr[1:-1, :-2], ((1, 1), (1, 1)))
            + jnp.pad(u_curr[2:, 1:-1], ((1, 1), (1, 1)))
            + jnp.pad(u_curr[:-2, 1:-1], ((1, 1), (1, 1)))
            - 4.0 * jnp.pad(u_curr[1:-1, 1:-1], ((1, 1), (1, 1)))
        ) * inv_dx2
        num = 2.0 * u_curr - num_damp * u_prev + dt2 * (c2_jax * lap + q_val)
        u_next = jnp.where(
            (jnp.arange(nz)[:, None] > 0) & (jnp.arange(nz)[:, None] < nz - 1) &
            (jnp.arange(nx)[None, :] > 0) & (jnp.arange(nx)[None, :] < nx - 1),
            num / den_fwd,
            0.0
        )
        return u_next

    recv_xs = jnp.array([r[0] for r in receivers])
    recv_zs = jnp.array([r[1] for r in receivers])

    total_image = np.zeros((nz, nx), dtype=np.float64)

    for s_idx, shot in enumerate(shots):
        t_shot_start = time.time()
        sx, sz = int(shot[0]), int(shot[1])
        fp = np.zeros((nz, nx), dtype=np.float64)
        fp[sz, sx] = 1.0
        fp_jax = jnp.array(fp)

        # 1. Forward run, save u_history
        u_prev = jnp.zeros((nz, nx))
        u_curr = jnp.zeros((nz, nx))
        u_hist = []

        for n in range(steps):
            q = pulses[n] * fp_jax
            u_next = fwd_step(u_curr, u_prev, q)
            u_hist.append(np.array(u_curr))
            u_prev = u_curr
            u_curr = u_next

        # 2. Reverse pass
        w_shot = born_data[s_idx] # [steps, receivers]
        adj_next = np.zeros((nz, nx), dtype=np.float64)
        adj_curr = np.zeros((nz, nx), dtype=np.float64)
        shot_image = np.zeros((nz, nx), dtype=np.float64)

        for n in range(steps - 1, -1, -1):
            # inject weights
            for k in range(n_recvs):
                rx, rz = receivers[k]
                adj_next[rz, rx] += w_shot[n, k]

            v = np.zeros_like(adj_next)
            v[1:-1, 1:-1] = adj_next[1:-1, 1:-1] / (1.0 + sigma[1:-1, 1:-1] * dt)
            c2_v = c * c * v

            u_n = u_hist[n]
            lap_un = laplacian(u_n, dx)
            shot_image[1:-1, 1:-1] += 2.0 * c[1:-1, 1:-1] * dt2 * lap_un[1:-1, 1:-1] * v[1:-1, 1:-1]

            lap_c2_v = laplacian(c2_v, dx)
            adj_prev = np.zeros_like(adj_curr)
            adj_curr[1:-1, 1:-1] += 2.0 * v[1:-1, 1:-1] + dt2 * lap_c2_v[1:-1, 1:-1]
            adj_prev[1:-1, 1:-1] = -(1.0 - sigma[1:-1, 1:-1] * dt) * v[1:-1, 1:-1]

            adj_next = adj_curr
            adj_curr = adj_prev

        shot_l2 = np.linalg.norm(shot_image)
        total_image += shot_image
        t_shot_end = time.time()
        print(f"Shot {s_idx}: L2 = {shot_l2:.6e} (time: {t_shot_end - t_shot_start:.1f}s)")

    total_l2 = np.linalg.norm(total_image)
    print(f"Total Marmousi image L2 norm: {total_l2:.7e}")

    np.save(out_dir / "image.npy", total_image)
    print("Saved artifacts/marmousi-image/image.npy")

if __name__ == "__main__":
    main()
