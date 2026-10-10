import json
import time
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.hopfield import (
    create_hebbian_couplings,
    calculate_unstable_bits,
    crosstalk_prediction_large_n,
    crosstalk_prediction_finite_n,
    asynchronous_recall,
    calculate_overlap,
)

def main():
    root = Path(__file__).resolve().parent.parent
    evidence_dir = root / "evidence"
    artifacts_dir = root / "artifacts" / "hopfield"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("=== Part 3: Store random patterns in a Hopfield network ===")

    N_values = [100, 200, 400, 800]
    alpha_values = np.round(np.arange(0.05, 0.2501, 0.01), 2)
    num_trials = 20
    base_seed = 42

    if (artifacts_dir / "hopfield_capacity_results.json").exists():
        print("Loading existing simulation results from artifacts...")
        with open(artifacts_dir / "hopfield_capacity_results.json", "r") as f:
            results = json.load(f)
    else:
        results = {
        "N_values": N_values,
        "alpha_values": alpha_values.tolist(),
        "unstable_means": {},
        "unstable_sems": {},
        "overlap_means": {},
        "overlap_sems": {},
        "success_fraction": {},
        "runtimes": {},
    }

    t0_all = time.time()
    for N in N_values:
        t0_N = time.time()
        print(f"\nSimulating Hopfield capacity for N = {N} spins...")
        unstable_m = []
        unstable_s = []
        overlap_m = []
        overlap_s = []
        success_frac = []

        for alpha in alpha_values:
            P = int(round(alpha * N))
            trial_unstable = []
            trial_overlaps = []
            trial_success = []

            for trial in range(num_trials):
                seed = base_seed + N * 10000 + int(alpha * 100) * 100 + trial
                rng = np.random.default_rng(seed)

                # Random patterns: +/- 1 with equal probability
                patterns = rng.choice([-1, 1], size=(P, N)).astype(np.int32)
                J = create_hebbian_couplings(patterns)

                # 1. Unstable bits fraction
                frac_u = calculate_unstable_bits(J, patterns)
                trial_unstable.append(frac_u)

                # 2. Recall test: start at first 10 patterns (or all if P < 10)
                num_test = min(10, P)
                for mu in range(num_test):
                    target = patterns[mu]
                    s_recalled = asynchronous_recall(J, target, max_sweeps=100, rng=rng)
                    m = calculate_overlap(s_recalled, target)
                    trial_overlaps.append(m)
                    trial_success.append(1.0 if m >= 0.9 else 0.0)

            unstable_m.append(float(np.mean(trial_unstable)))
            unstable_s.append(float(np.std(trial_unstable, ddof=1) / np.sqrt(num_trials)))
            overlap_m.append(float(np.mean(trial_overlaps)))
            overlap_s.append(float(np.std(trial_overlaps, ddof=1) / np.sqrt(len(trial_overlaps))))
            success_frac.append(float(np.mean(trial_success)))

        elapsed_N = time.time() - t0_N
        print(f"N = {N} completed in {elapsed_N:.1f}s")

        str_N = str(N)
        results["unstable_means"][str_N] = unstable_m
        results["unstable_sems"][str_N] = unstable_s
        results["overlap_means"][str_N] = overlap_m
        results["overlap_sems"][str_N] = overlap_s
        results["success_fraction"][str_N] = success_frac
        results["runtimes"][str_N] = elapsed_N

        total_time = time.time() - t0_all
        print(f"\nAll Hopfield simulations finished in {total_time:.1f}s")

        # Save raw results
        with open(artifacts_dir / "hopfield_capacity_results.json", "w") as f:
            json.dump(results, f, indent=2)

    # ---------------- Figure 1: Unstable-bit fraction ----------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)
    colors = {100: "#2ca02c", 200: "#ff7f0e", 400: "#1f77b4", 800: "#9467bd"}

    # Theoretical curve (large N)
    alpha_fine = np.linspace(0.04, 0.26, 200)
    pred_large = [crosstalk_prediction_large_n(a) for a in alpha_fine]

    # Linear scale
    ax1.plot(alpha_fine, pred_large, "k-", lw=2, label="$\\Phi(-1/\\sqrt{\\alpha})$, large $N$")
    for N in N_values:
        ax1.errorbar(
            alpha_values,
            results["unstable_means"][str(N)],
            yerr=results["unstable_sems"][str(N)],
            fmt="o-",
            color=colors[N],
            label=f"N = {N}",
            capsize=2,
            ms=4,
            lw=1.2,
        )
    ax1.set_title("Unstable bits at a stored pattern", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Load $\\alpha = P/N$", fontsize=9)
    ax1.set_ylabel("Fraction of bits with $\\xi_i h_i < 0$", fontsize=9)
    ax1.set_xlim(0.04, 0.26)
    ax1.set_ylim(-0.001, 0.026)
    ax1.legend(loc="upper left", frameon=True, fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Logarithmic scale
    ax2.plot(alpha_fine, pred_large, "k-", lw=2, label="$\\Phi(-1/\\sqrt{\\alpha})$, large $N$")
    # Finite-N prediction for N=100
    pred_finite_100 = [crosstalk_prediction_finite_n(100, int(round(a * 100))) for a in alpha_values]
    ax2.plot(alpha_values, pred_finite_100, "k--", lw=1.5, label="exact, $N = 100$")

    for N in N_values:
        ax2.errorbar(
            alpha_values,
            results["unstable_means"][str(N)],
            yerr=results["unstable_sems"][str(N)],
            fmt="o-",
            color=colors[N],
            label=f"N = {N}",
            capsize=2,
            ms=4,
            lw=1.2,
        )
    ax2.set_title("Same data, logarithmic scale", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Load $\\alpha = P/N$", fontsize=9)
    ax2.set_ylabel("Fraction of bits with $\\xi_i h_i < 0$", fontsize=9)
    ax2.set_yscale("log")
    ax2.set_xlim(0.04, 0.26)
    ax2.set_ylim(1e-5, 3e-2)
    ax2.legend(loc="lower right", frameon=True, fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.5)

    unstable_fig_path = evidence_dir / "hopfield-unstable-bits.png"
    plt.tight_layout()
    plt.savefig(unstable_fig_path, bbox_inches="tight")
    plt.close()
    print(f"Saved unstable bits figure to {unstable_fig_path}")

    # ---------------- Figure 2: Storage capacity & recall collapse ----------------
    fig, (ax_ov, ax_succ) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)
    alpha_c = 0.138

    # Left: Mean final overlap
    for N in N_values:
        ax_ov.plot(
            alpha_values,
            results["overlap_means"][str(N)],
            "o-",
            color=colors[N],
            label=f"N = {N}",
            ms=4,
            lw=1.5,
        )
    ax_ov.axvline(alpha_c, color="red", linestyle="--", lw=1.2, label="$\\alpha_c = 0.138$")
    ax_ov.set_title("Recall from a stored pattern", fontsize=10, fontweight="bold")
    ax_ov.set_xlabel("Load $\\alpha = P/N$", fontsize=9)
    ax_ov.set_ylabel("Mean final overlap $m = \\xi \\cdot s / N$", fontsize=9)
    ax_ov.set_xlim(0.04, 0.26)
    ax_ov.set_ylim(0.0, 1.05)
    ax_ov.legend(loc="lower left", frameon=True, fontsize=8)
    ax_ov.grid(True, linestyle="--", alpha=0.5)

    # Right: Retrieval probability (overlap >= 0.9)
    for N in N_values:
        ax_succ.plot(
            alpha_values,
            results["success_fraction"][str(N)],
            "o-",
            color=colors[N],
            label=f"N = {N}",
            ms=4,
            lw=1.5,
        )
    ax_succ.axvline(alpha_c, color="red", linestyle="--", lw=1.2, label="$\\alpha_c = 0.138$")
    ax_succ.set_title("Retrieval probability", fontsize=10, fontweight="bold")
    ax_succ.set_xlabel("Load $\\alpha = P/N$", fontsize=9)
    ax_succ.set_ylabel("Fraction of recalls ending at $m \\geq 0.9$", fontsize=9)
    ax_succ.set_xlim(0.04, 0.26)
    ax_succ.set_ylim(-0.02, 1.05)
    ax_succ.legend(loc="lower left", frameon=True, fontsize=8)
    ax_succ.grid(True, linestyle="--", alpha=0.5)

    capacity_fig_path = evidence_dir / "hopfield-capacity.png"
    plt.tight_layout()
    plt.savefig(capacity_fig_path, bbox_inches="tight")
    plt.close()
    print(f"Saved capacity collapse figure to {capacity_fig_path}")

    # Check crossing points: find where mean overlap falls below 0.9
    print("\nVerification of 0.9 overlap crossings:")
    reference_crossings = {100: 0.186, 200: 0.170, 400: 0.156, 800: 0.150}
    for N in N_values:
        ovs = np.array(results["overlap_means"][str(N)])
        # Interpolate crossing point
        below = np.where(ovs < 0.9)[0]
        if len(below) > 0:
            idx = below[0]
            if idx > 0:
                # linear interpolation
                a0, a1 = alpha_values[idx - 1], alpha_values[idx]
                o0, o1 = ovs[idx - 1], ovs[idx]
                crossing = a0 + (0.9 - o0) * (a1 - a0) / (o1 - o0)
            else:
                crossing = alpha_values[0]
        else:
            crossing = alpha_values[-1]
        ref = reference_crossings[N]
        diff = abs(crossing - ref)
        print(f"N = {N:3d}: measured crossing = {crossing:.3f}, reference = {ref:.3f} (diff = {diff:.3f})")
        assert diff <= 0.035, f"Crossing for N={N} ({crossing:.3f}) must lie close to reference ({ref:.3f})"

    print("Part 3 verification successfully passed!")

if __name__ == "__main__":
    main()
