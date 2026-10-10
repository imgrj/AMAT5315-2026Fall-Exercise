import json
import time
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.datasets import load_mnist, binarize, to_spins
from src.hopfield import create_hebbian_couplings, asynchronous_recall
from src.modern_hopfield import (
    modern_hopfield_update,
    modern_hopfield_attention,
    modern_hopfield_weights,
)

def check_retrieval_success(retrieved_x, target_idx, X_all):
    """
    Success criterion:
    1. Overlap of sign(retrieved_x) with target >= 0.95 (at most 19 wrong pixels of 784).
    2. Target is the nearest stored digit: argmax_mu (sign(retrieved_x) @ X[:, mu]) == target_idx.
    """
    retrieved_spins = np.sign(retrieved_x)
    retrieved_spins[retrieved_spins == 0] = 1
    N, P = X_all.shape

    all_overlaps = (X_all.T @ retrieved_spins) / float(N)
    target_overlap = all_overlaps[target_idx]
    is_nearest = (np.argmax(all_overlaps) == target_idx)
    success = (target_overlap >= 0.95) and is_nearest
    return success, float(target_overlap)

def main():
    root = Path(__file__).resolve().parent.parent
    evidence_dir = root / "evidence"
    artifacts_dir = root / "artifacts" / "modern_hopfield"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("=== Part 4: Retrieve digits with a modern Hopfield network ===")

    # 1. Load MNIST training images and convert to spins {-1, +1}
    train_x, _ = load_mnist(split="train")
    train_bin = binarize(train_x).reshape(-1, 784)
    train_spins = to_spins(train_bin)

    max_P = 1000
    stored_patterns = train_spins[:max_P].T # shape (784, 1000)

    P_values = [3, 10, 30, 100, 300, 1000]
    beta_values = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
    num_queries = 100

    # 2. Verify equivalence between Modern Hopfield update and JAX softmax attention
    print("\nTesting numerical equivalence with jax.nn.dot_product_attention...")
    max_attention_diff = 0.0
    rng = np.random.default_rng(2026)

    # For verification across queries, P, beta
    for P in P_values:
        X_P = jnp.array(stored_patterns[:, :P], dtype=jnp.float32)
        for beta in beta_values:
            for q_idx in range(min(5, P)):
                q_vec = stored_patterns[:, q_idx].copy().astype(np.float32)
                q_vec[392:] = rng.choice([-1.0, 1.0], size=392)
                q_jnp = jnp.array(q_vec)

                x_hop = modern_hopfield_update(X_P, q_jnp, beta)
                x_att = modern_hopfield_attention(X_P, q_jnp, beta)
                diff = float(jnp.max(jnp.abs(x_hop - x_att)))
                if diff > max_attention_diff:
                    max_attention_diff = diff

    print(f"Largest difference between Hopfield update and JAX softmax attention: {max_attention_diff:.2e}")
    assert max_attention_diff < 1e-5, f"Difference {max_attention_diff} must be < 1e-5"

    # ---------------- Figure 1: Softmax weights plot ----------------
    # Plot weights for query 0 at P=100 across beta in {0.003, 0.03, 0.3}
    P_demo = 100
    X_demo = jnp.array(stored_patterns[:, :P_demo], dtype=jnp.float32)
    
    # Query 0: pattern 0 with lower half erased
    q0 = stored_patterns[:, 0].copy().astype(np.float32)
    q0[392:] = rng.choice([-1.0, 1.0], size=392)
    q0_jnp = jnp.array(q0)
    demo_betas = [0.003, 0.03, 0.3]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), dpi=300, sharey=True)
    weights_dict = {}
    for idx, b_val in enumerate(demo_betas):
        ax = axes[idx]
        w = np.array(modern_hopfield_weights(X_demo, q0_jnp, b_val))
        weights_dict[str(b_val)] = w.tolist()
        targ_w = w[0]

        ax.bar(np.arange(P_demo), w, color="#1f77b4", width=1.0)
        ax.bar([0], [targ_w], color="#ff7f0e", width=1.0)
        ax.set_title(f"$\\beta = {b_val}$", fontsize=10, pad=5, fontweight="bold")
        ax.set_xlabel("Stored pattern $\\mu$", fontsize=9)
        if idx == 0:
            ax.set_ylabel("Weight $\\mathrm{softmax}(\\beta X^T x)_\\mu$", fontsize=9)
        ax.set_ylim(0.0, 1.05)
        ax.text(0.5, 0.88, f"target weight {targ_w:.2f}", transform=ax.transAxes, ha="center", fontsize=8, color="#d62728")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.text(0.5, 0.01, f"P = 100 stored digits; query 0 targets pattern 0 (orange). Max |modern update - jax.nn.dot_product_attention|: {max_attention_diff:.2e}",
             ha="center", fontsize=8, color="#555555")
    weights_fig_path = evidence_dir / "modern-hopfield-weights.png"
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(weights_fig_path, bbox_inches="tight")
    plt.close()
    print(f"Saved weights distribution plot to {weights_fig_path}")

    # ---------------- Figure 2: Beta transition gallery ----------------
    print("Generating beta sweep transition gallery...")
    J_demo = create_hebbian_couplings(stored_patterns[:, :P_demo].T)
    demo_queries = [0, 1, 2]
    
    fig, axes = plt.subplots(len(demo_queries), 2 + len(beta_values) + 1, figsize=(12, 4.5), dpi=300)
    col_titles = ["target", "query"] + [f"$\\beta={b}$" for b in beta_values] + ["classical"]

    demo_query_spins = []
    for r_idx, q_i in enumerate(demo_queries):
        target_spins = stored_patterns[:, q_i]
        query_spins = target_spins.copy()
        query_spins[392:] = rng.choice([-1, 1], size=392)
        demo_query_spins.append(query_spins)

        axes[r_idx, 0].imshow(target_spins.reshape(28, 28), cmap="gray", interpolation="nearest")
        axes[r_idx, 1].imshow(query_spins.reshape(28, 28), cmap="gray", interpolation="nearest")

        for b_col, b_val in enumerate(beta_values):
            x_ret = np.array(modern_hopfield_update(X_demo, jnp.array(query_spins, dtype=jnp.float32), b_val))
            axes[r_idx, 2 + b_col].imshow(x_ret.reshape(28, 28), cmap="gray", vmin=-1, vmax=1, interpolation="nearest")

        s_class = asynchronous_recall(J_demo, query_spins, max_sweeps=100, rng=rng)
        axes[r_idx, 2 + len(beta_values)].imshow(s_class.reshape(28, 28), cmap="gray", interpolation="nearest")

        for c_idx in range(axes.shape[1]):
            axes[r_idx, c_idx].set_xticks([])
            axes[r_idx, c_idx].set_yticks([])
            if r_idx == 0:
                axes[0, c_idx].set_title(col_titles[c_idx], fontsize=8, pad=3)

    fig.suptitle("One modern update $x \\leftarrow X \\mathrm{softmax}(\\beta X^T x)$, $P = 100$ stored digits (gray scale: -1 black, +1 white)",
                 fontsize=10, fontweight="bold", y=0.98)
    beta_fig_path = evidence_dir / "modern-hopfield-beta-sweep.png"
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(beta_fig_path, bbox_inches="tight")
    plt.close()
    print(f"Saved beta transition gallery to {beta_fig_path}")

    # ---------------- 3. Systematic benchmark across P and beta ----------------
    print("\nRunning capacity benchmark across P and beta...")
    success_modern = {str(b): [] for b in beta_values}
    success_classical = []

    for P in P_values:
        print(f"Benchmarking storage P = {P}...")
        X_curr = jnp.array(stored_patterns[:, :P], dtype=jnp.float32)
        patterns_curr = stored_patterns[:, :P].T
        J_curr = create_hebbian_couplings(patterns_curr)

        # Generate 100 queries cycling through the stored ones
        queries_P = []
        target_indices = []
        for q_i in range(num_queries):
            target_idx = q_i % P
            targ = stored_patterns[:, target_idx].copy()
            targ[392:] = rng.choice([-1, 1], size=392)
            queries_P.append(targ)
            target_indices.append(target_idx)

        # Classical recall
        succ_class_count = 0
        for q_i in range(num_queries):
            s_rec = asynchronous_recall(J_curr, queries_P[q_i], max_sweeps=100, rng=rng)
            succ, _ = check_retrieval_success(s_rec, target_indices[q_i], stored_patterns[:, :P])
            if succ:
                succ_class_count += 1
        success_classical.append(float(succ_class_count) / num_queries)

        # Modern recall across beta
        for b_val in beta_values:
            succ_mod_count = 0
            for q_i in range(num_queries):
                x_rec = np.array(modern_hopfield_update(X_curr, jnp.array(queries_P[q_i], dtype=jnp.float32), b_val))
                succ, _ = check_retrieval_success(x_rec, target_indices[q_i], stored_patterns[:, :P])
                if succ:
                    succ_mod_count += 1
            success_modern[str(b_val)].append(float(succ_mod_count) / num_queries)

    print(f"Success rates (Classical): {success_classical}")
    print(f"Success rates (Modern beta=0.1): {success_modern['0.1']}")
    print(f"Success rates (Modern beta=0.3): {success_modern['0.3']}")

    # ---------------- Figure 3: Capacity plot comparing Classical vs Modern ----------------
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
    ax.plot(P_values, success_classical, "s--", color="gray", label="classical (Hebbian)", lw=1.5, ms=5)
    ax.plot(P_values, success_modern["0.1"], "^-", color="#1f77b4", label="modern, one step, $\\beta = 0.1$", lw=1.8, ms=5)
    ax.plot(P_values, success_modern["0.3"], "o-", color="#d62728", label="modern, one step, $\\beta = 0.3$", lw=1.8, ms=5)

    ax.set_xscale("log")
    ax.set_title("Retrieval of half-erased digits", fontsize=10, fontweight="bold")
    ax.set_xlabel("Stored digits $P$", fontsize=9)
    ax.set_ylabel("Fraction of queries retrieved\n(target nearest, overlap $\\geq 0.95$)", fontsize=9)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="center right", frameon=True, fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)

    capacity_fig_path = evidence_dir / "modern-hopfield-capacity.png"
    plt.tight_layout()
    plt.savefig(capacity_fig_path, bbox_inches="tight")
    plt.close()
    print(f"Saved capacity comparison to {capacity_fig_path}")

    # ---------------- Figure 4: Gallery at P=100 (queries 0-5) ----------------
    demo_indices = [0, 1, 2, 3, 4, 5]
    fig, axes = plt.subplots(4, len(demo_indices), figsize=(8, 5.5), dpi=300)
    row_titles = ["target", "query", "classical", "modern\n$\\beta = 0.3$"]

    for col, q_i in enumerate(demo_indices):
        target = stored_patterns[:, q_i]
        query = target.copy()
        query[392:] = rng.choice([-1, 1], size=392)

        s_class = asynchronous_recall(J_demo, query, max_sweeps=100, rng=rng)
        x_mod = np.array(modern_hopfield_update(X_demo, jnp.array(query, dtype=jnp.float32), 0.3))
        s_mod = np.sign(x_mod)

        axes[0, col].imshow(target.reshape(28, 28), cmap="gray", interpolation="nearest")
        axes[1, col].imshow(query.reshape(28, 28), cmap="gray", interpolation="nearest")
        axes[2, col].imshow(s_class.reshape(28, 28), cmap="gray", interpolation="nearest")
        axes[3, col].imshow(s_mod.reshape(28, 28), cmap="gray", interpolation="nearest")

        for r in range(4):
            axes[r, col].set_xticks([])
            axes[r, col].set_yticks([])

    for r in range(4):
        axes[r, 0].set_ylabel(row_titles[r], rotation=0, labelpad=30, va="center", fontsize=9)

    fig.suptitle("P = 100, queries 0-5", fontsize=10, fontweight="bold", y=0.98)
    gallery_fig_path = evidence_dir / "modern-hopfield-gallery.png"
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(gallery_fig_path, bbox_inches="tight")
    plt.close()
    print(f"Saved P=100 gallery to {gallery_fig_path}")

    # Summary JSON in artifacts
    summary = {
        "max_attention_difference": max_attention_diff,
        "P_values": P_values,
        "beta_values": beta_values,
        "success_classical": success_classical,
        "success_modern": success_modern,
        "demo_query0_weights": weights_dict,
    }
    with open(artifacts_dir / "modern_hopfield_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Verification: Modern beta=0.3 success >= 0.95 for all P, Classical success == 0
    assert all(s == 0.0 for s in success_classical), "Classical Hebbian recall must fail for all P due to correlated digits"
    assert all(s >= 0.95 for s in success_modern["0.3"]), f"Modern Hopfield at beta=0.3 must achieve >= 0.95 success for all P, got {success_modern['0.3']}"
    print("Part 4 verification successfully passed!")

if __name__ == "__main__":
    main()
