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
from src.rbm import (
    init_rbm_params,
    gibbs_sweep,
    gibbs_k_sweeps,
    rbm_train_step,
    evaluate_reconstruction_error,
    clamped_gibbs_step,
    calculate_ink_overlap,
)

def main():
    root = Path(__file__).resolve().parent.parent
    evidence_dir = root / "evidence"
    artifacts_dir = root / "artifacts" / "rbm"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("=== Part 2: Learn a generative spin model of handwritten digits ===")

    # 1. Load & binarize MNIST
    train_x, train_y = load_mnist(split="train")
    test_x, test_y = load_mnist(split="test")

    train_bin = binarize(train_x).reshape(-1, 784).astype(np.float32)
    test_bin = binarize(test_x).reshape(-1, 784).astype(np.float32)

    # Figure 1: Binarized MNIST & First image as spins
    fig = plt.figure(figsize=(9, 4.5), dpi=300)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.3)

    # First 16 training images
    gs_16 = gs[0].subgridspec(4, 4, wspace=0.05, hspace=0.05)
    for idx in range(16):
        r, c = divmod(idx, 4)
        ax = fig.add_subplot(gs_16[r, c])
        ax.imshow(train_bin[idx].reshape(28, 28), cmap="gray", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.text(0.28, 0.94, "Binarized MNIST\nFirst 16 training images", ha="center", va="top", fontsize=10, fontweight="bold")

    # Enlarged spin configuration
    ax_spins = fig.add_subplot(gs[1])
    first_spins = to_spins(train_bin[0]).reshape(28, 28)
    im = ax_spins.imshow(first_spins, cmap="bone", vmin=-1, vmax=1, interpolation="nearest")
    ax_spins.set_title("First image as spins\n28 × 28 sites", fontsize=10, pad=10, fontweight="bold")
    ax_spins.set_xticks([])
    ax_spins.set_yticks([])
    cbar = plt.colorbar(im, ax=ax_spins, fraction=0.046, pad=0.04, ticks=[-1, 1])
    cbar.set_label("Spin values s_i = 2v_i - 1", fontsize=9)

    spins_fig_path = evidence_dir / "rbm-spins.png"
    plt.savefig(spins_fig_path, bbox_inches="tight")
    plt.close()
    print(f"Saved spin visualization to {spins_fig_path}")

    # Training settings
    num_visible = 784
    num_hidden = 128
    num_epochs = 20
    batch_size = 100
    learning_rate = 0.1
    seed = 42

    num_samples = train_bin.shape[0]
    num_batches = num_samples // batch_size

    # Function to train RBM
    def train_rbm(k_steps):
        print(f"\n--- Training RBM with CD-{k_steps} ---")
        rng_key = jax.random.PRNGKey(seed)
        rng_key, init_key = jax.random.split(rng_key)
        params = init_rbm_params(init_key, num_visible, num_hidden)

        # Baseline evaluation at epoch 0
        rng_key, eval_key = jax.random.split(rng_key)
        rec_err_0 = evaluate_reconstruction_error(params, test_bin, eval_key)
        
        # Initial surrogate on first batch
        rng_key, sample_key = jax.random.split(rng_key)
        v0 = train_bin[:batch_size]
        vk = gibbs_k_sweeps(params, v0, sample_key, k_steps)
        from src.rbm import cd_surrogate_loss
        surr_0 = float(cd_surrogate_loss(params, v0, vk))

        history = {
            "epochs": [0],
            "surrogate": [surr_0],
            "reconstruction_error": [rec_err_0],
            "epoch_runtimes": [0.0],
        }
        print(f"Epoch 0: Surrogate = {surr_0:.4f}, Held-out 1-step Rec Error = {rec_err_0*100:.2f}%")

        t_start_total = time.time()
        for epoch in range(1, num_epochs + 1):
            t_epoch_start = time.time()
            epoch_surr = 0.0

            # Match permutation / data order per epoch across CD-1 and CD-10
            epoch_rng = np.random.default_rng(seed + epoch * 1000)
            perm = epoch_rng.permutation(num_samples)

            for b in range(num_batches):
                batch_indices = perm[b * batch_size:(b + 1) * batch_size]
                v0_batch = train_bin[batch_indices]

                rng_key, step_key = jax.random.split(rng_key)
                vk_batch = gibbs_k_sweeps(params, v0_batch, step_key, k_steps)
                params, loss_val = rbm_train_step(params, v0_batch, vk_batch, lr=learning_rate)
                epoch_surr += float(loss_val)

            epoch_surr /= num_batches
            t_epoch = time.time() - t_epoch_start

            # Held-out 1-step reconstruction error
            rng_key, eval_key = jax.random.split(rng_key)
            rec_err = evaluate_reconstruction_error(params, test_bin, eval_key)

            history["epochs"].append(epoch)
            history["surrogate"].append(epoch_surr)
            history["reconstruction_error"].append(rec_err)
            history["epoch_runtimes"].append(t_epoch)

            print(f"Epoch {epoch:2d}/{num_epochs}: Surrogate = {epoch_surr:7.3f}, Rec Error = {rec_err*100:5.2f}% ({t_epoch:.1f}s)")

        total_runtime = time.time() - t_start_total
        print(f"Completed CD-{k_steps} in {total_runtime:.1f}s")
        return params, history, total_runtime

    # Train CD-1
    params_cd1, hist_cd1, runtime_cd1 = train_rbm(k_steps=1)
    # Train CD-10
    params_cd10, hist_cd10, runtime_cd10 = train_rbm(k_steps=10)

    # Save models and histories to artifacts
    np.savez_compressed(
        artifacts_dir / "cd1_model.npz",
        W=np.array(params_cd1["W"]),
        a=np.array(params_cd1["a"]),
        b=np.array(params_cd1["b"]),
    )
    np.savez_compressed(
        artifacts_dir / "cd10_model.npz",
        W=np.array(params_cd10["W"]),
        a=np.array(params_cd10["a"]),
        b=np.array(params_cd10["b"]),
    )
    with open(artifacts_dir / "training_history.json", "w") as f:
        json.dump({
            "settings": {
                "num_visible": num_visible,
                "num_hidden": num_hidden,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "seed": seed,
            },
            "cd1": hist_cd1,
            "cd10": hist_cd10,
            "runtimes": {"cd1_sec": runtime_cd1, "cd10_sec": runtime_cd10},
        }, f, indent=2)

    # Figure 2: Plot CD surrogate and held-out reconstruction error
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=300)
    
    # CD surrogate
    ax1.plot(hist_cd1["epochs"], hist_cd1["surrogate"], label="CD-1", color="#1f77b4", lw=2)
    ax1.plot(hist_cd10["epochs"], hist_cd10["surrogate"], label="CD-10", color="#d62728", lw=2)
    ax1.set_title("CD surrogate\nNot a likelihood curve", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Training epoch", fontsize=9)
    ax1.set_ylabel("Mean F(data) - F(v_k)", fontsize=9)
    ax1.legend(loc="lower right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Reconstruction error
    ax2.plot(hist_cd1["epochs"], hist_cd1["reconstruction_error"], label="CD-1", color="#1f77b4", lw=2)
    ax2.plot(hist_cd10["epochs"], hist_cd10["reconstruction_error"], label="CD-10", color="#d62728", lw=2)
    ax2.set_title("Held-out reconstruction\nOne Gibbs sweep", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Training epoch", fontsize=9)
    ax2.set_ylabel("Fraction of changed pixels", fontsize=9)
    ax2.set_ylim(0.0, 0.30)
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.5)

    curves_fig_path = evidence_dir / "rbm-learning-curves.png"
    plt.tight_layout()
    plt.savefig(curves_fig_path, bbox_inches="tight")
    plt.close()
    print(f"Saved learning curves to {curves_fig_path}")

    # Check that error dropped by more than half
    assert hist_cd1["reconstruction_error"][-1] < 0.5 * hist_cd1["reconstruction_error"][0], "CD-1 error must drop >50%"
    assert hist_cd10["reconstruction_error"][-1] < 0.5 * hist_cd10["reconstruction_error"][0], "CD-10 error must drop >50%"
    print(f"CD-1 reconstruction error: {hist_cd1['reconstruction_error'][0]*100:.2f}% -> {hist_cd1['reconstruction_error'][-1]*100:.2f}%")
    print(f"CD-10 reconstruction error: {hist_cd10['reconstruction_error'][0]*100:.2f}% -> {hist_cd10['reconstruction_error'][-1]*100:.2f}%")

    # Figure 3: Fresh generation from random spins
    print("\nRunning fresh generation from random spins...")
    steps = [0, 1, 10, 100, 1000]
    num_chains = 2
    gen_rng = jax.random.PRNGKey(101)
    gen_rng, init_key = jax.random.split(gen_rng)
    # Start from shared random spin configurations (50% prob Bernoulli bits)
    v_random_starts = (jax.random.uniform(init_key, (num_chains, 784)) < 0.5).astype(jnp.float32)

    def sample_generation_trajectory(params, starts):
        samples_by_step = {0: np.array(starts)}
        v_curr = starts
        curr_key = gen_rng
        total_steps = 0
        for target_step in steps[1:]:
            steps_to_take = target_step - total_steps
            for _ in range(steps_to_take):
                curr_key, subkey = jax.random.split(curr_key)
                v_curr, _ = gibbs_sweep(params, v_curr, subkey)
            total_steps = target_step
            samples_by_step[target_step] = np.array(v_curr)
        return samples_by_step

    gen_cd1 = sample_generation_trajectory(params_cd1, v_random_starts)
    gen_cd10 = sample_generation_trajectory(params_cd10, v_random_starts)

    fig, axes = plt.subplots(4, 5, figsize=(8, 6.5), dpi=300)
    row_labels = ["CD-1\nChain 1", "CD-1\nChain 2", "CD-10\nChain 1", "CD-10\nChain 2"]
    data_chains = [
        (gen_cd1, 0),
        (gen_cd1, 1),
        (gen_cd10, 0),
        (gen_cd10, 1),
    ]

    for r_idx, (data_dict, c_idx) in enumerate(data_chains):
        for col_idx, s in enumerate(steps):
            ax = axes[r_idx, col_idx]
            img = data_dict[s][c_idx].reshape(28, 28)
            ax.imshow(img, cmap="gray", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if r_idx == 0:
                ax.set_title(str(s), fontsize=10, pad=4)
        axes[r_idx, 0].set_ylabel(row_labels[r_idx], rotation=0, labelpad=35, va="center", fontsize=9)

    fig.suptitle("Generation from shared random starts\nGibbs sweeps", fontsize=11, fontweight="bold", y=0.98)
    gen_fig_path = evidence_dir / "rbm-generation.png"
    plt.tight_layout()
    plt.savefig(gen_fig_path, bbox_inches="tight")
    plt.close()
    print(f"Saved generation evidence to {gen_fig_path}")

    # Figure 4 & 5: Clamped Gibbs sampling (Image completion)
    print("\nRunning clamped Gibbs image completion...")
    # Upper half observed (rows 0..13 -> first 392 pixels), lower half missing (rows 14..27 -> pixels 392..783)
    mask_observed = np.zeros(784, dtype=bool)
    mask_observed[:392] = True
    mask_observed_jax = jnp.array(mask_observed)

    # 4.1 Visual Gallery for first 4 test digits
    test_indices = [0, 1, 2, 3] # test digits
    comp_steps = [1, 10, 100, 1000]
    
    fig = plt.figure(figsize=(10, 8), dpi=300)
    gs = fig.add_gridspec(len(test_indices) * 2, 8, wspace=0.08, hspace=0.15)

    comp_rng = jax.random.PRNGKey(2026)
    for digit_idx, t_idx in enumerate(test_indices):
        true_digit = test_bin[t_idx]
        observed_digit = true_digit.copy()
        observed_digit[~mask_observed] = 0.0

        for m_idx, (model_name, params) in enumerate([("CD-1", params_cd1), ("CD-10", params_cd10)]):
            row = digit_idx * 2 + m_idx
            
            # Col 0: truth (only on first model row of pair)
            ax_truth = fig.add_subplot(gs[row, 0])
            if m_idx == 0:
                ax_truth.imshow(true_digit.reshape(28, 28), cmap="gray", interpolation="nearest")
                ax_truth.set_title("truth", fontsize=8)
            ax_truth.set_xticks([]); ax_truth.set_yticks([])
            if m_idx == 0:
                ax_truth.set_ylabel(f"test #{digit_idx}", fontsize=9, labelpad=5)

            # Col 1: observed
            ax_obs = fig.add_subplot(gs[row, 1])
            if m_idx == 0:
                ax_obs.imshow(observed_digit.reshape(28, 28), cmap="gray", interpolation="nearest")
                ax_obs.set_title("observed", fontsize=8)
            ax_obs.set_xticks([]); ax_obs.set_yticks([])

            # Chain 1 at steps 1, 10, 100, 1000
            comp_rng, start_key = jax.random.split(comp_rng)
            init_v = (jax.random.uniform(start_key, (1, 784)) < 0.5).astype(jnp.float32)
            init_v = jnp.where(mask_observed_jax, jnp.array(true_digit).reshape(1, 784), init_v)

            v_curr = init_v
            tot_s = 0
            for s_col, s_target in enumerate(comp_steps):
                ax_step = fig.add_subplot(gs[row, 2 + s_col])
                for _ in range(s_target - tot_s):
                    comp_rng, subkey = jax.random.split(comp_rng)
                    v_curr = clamped_gibbs_step(params, v_curr, jnp.array(true_digit).reshape(1, 784), mask_observed_jax, subkey)
                tot_s = s_target
                ax_step.imshow(np.array(v_curr[0]).reshape(28, 28), cmap="gray", interpolation="nearest")
                ax_step.set_xticks([]); ax_step.set_yticks([])
                if row == 0:
                    ax_step.set_title(str(s_target), fontsize=8)

            # Extra chains at 1000 sweeps (chains 2, 3)
            for extra_col in range(2):
                ax_extra = fig.add_subplot(gs[row, 6 + extra_col])
                comp_rng, extra_start_key = jax.random.split(comp_rng)
                v_extra = (jax.random.uniform(extra_start_key, (1, 784)) < 0.5).astype(jnp.float32)
                v_extra = jnp.where(mask_observed_jax, jnp.array(true_digit).reshape(1, 784), v_extra)
                for _ in range(1000):
                    comp_rng, subkey = jax.random.split(comp_rng)
                    v_extra = clamped_gibbs_step(params, v_extra, jnp.array(true_digit).reshape(1, 784), mask_observed_jax, subkey)
                ax_extra.imshow(np.array(v_extra[0]).reshape(28, 28), cmap="gray", interpolation="nearest")
                ax_extra.set_xticks([]); ax_extra.set_yticks([])
                if row == 0:
                    ax_extra.set_title(f"chain {extra_col+2}", fontsize=8)
            
            # Label model on rightmost axis
            ax_right = fig.add_subplot(gs[row, 7])
            ax_right.axis("off")
            ax_right.text(0.1, 0.5, model_name, fontsize=8, va="center", color="#d62728" if model_name=="CD-10" else "#1f77b4")

    fig.text(0.43, 0.98, "chain 1 after Gibbs sweeps", ha="center", fontsize=9, fontweight="bold")
    fig.text(0.82, 0.98, "more chains, 1000 sweeps", ha="center", fontsize=9, fontweight="bold")
    comp_gallery_path = evidence_dir / "rbm-completion-gallery.png"
    plt.savefig(comp_gallery_path, bbox_inches="tight")
    plt.close()
    print(f"Saved completion gallery to {comp_gallery_path}")

    # 4.2 Quantitative ink overlap measurement across 64 held-out digits
    print("Evaluating ink overlap across 64 held-out digits with 8 chains each...")
    num_eval_digits = 64
    num_chains_eval = 8
    sweep_eval_steps = [1, 10, 100, 1000]

    eval_digits = test_bin[:num_eval_digits]
    # Wrong upper half: cyclically permuted by 1 digit
    wrong_digits = np.roll(eval_digits, shift=1, axis=0)

    def evaluate_model_overlap(params):
        results_true = {s: [] for s in sweep_eval_steps}
        results_wrong = {s: [] for s in sweep_eval_steps}

        overlap_rng = jax.random.PRNGKey(777)
        for i in range(num_eval_digits):
            targ = eval_digits[i]
            targ_jax = jnp.array(targ)
            wrong_targ = wrong_digits[i]
            wrong_jax = jnp.array(wrong_targ)

            # Chains for true upper half: shape (num_chains_eval, 784)
            overlap_rng, k_start = jax.random.split(overlap_rng)
            v_true = (jax.random.uniform(k_start, (num_chains_eval, 784)) < 0.5).astype(jnp.float32)
            v_true = jnp.where(mask_observed_jax, targ_jax, v_true)

            # Chains for wrong upper half: shape (num_chains_eval, 784)
            overlap_rng, k_start_w = jax.random.split(overlap_rng)
            v_wrong = (jax.random.uniform(k_start_w, (num_chains_eval, 784)) < 0.5).astype(jnp.float32)
            v_wrong = jnp.where(mask_observed_jax, wrong_jax, v_wrong)

            curr_s = 0
            for target_s in sweep_eval_steps:
                for _ in range(target_s - curr_s):
                    overlap_rng, k_s1, k_s2 = jax.random.split(overlap_rng, 3)
                    v_true = clamped_gibbs_step(params, v_true, targ_jax, mask_observed_jax, k_s1)
                    v_wrong = clamped_gibbs_step(params, v_wrong, wrong_jax, mask_observed_jax, k_s2)
                curr_s = target_s

                # Compute ink overlap with the true lower half for each chain
                ov_true = calculate_ink_overlap(np.array(v_true), targ, mask_observed)
                ov_wrong = calculate_ink_overlap(np.array(v_wrong), targ, mask_observed)

                # Average across 8 chains for this digit
                results_true[target_s].append(float(np.mean(ov_true)))
                results_wrong[target_s].append(float(np.mean(ov_wrong)))

        return results_true, results_wrong

    overlap_cd1_true, overlap_cd1_wrong = evaluate_model_overlap(params_cd1)
    overlap_cd10_true, overlap_cd10_wrong = evaluate_model_overlap(params_cd10)

    # Calculate means and standard errors over the 64 digits
    def get_stats(res_dict):
        means = [float(np.mean(res_dict[s])) for s in sweep_eval_steps]
        sems = [float(np.std(res_dict[s], ddof=1) / np.sqrt(num_eval_digits)) for s in sweep_eval_steps]
        return means, sems

    m_cd1_t, sem_cd1_t = get_stats(overlap_cd1_true)
    m_cd1_w, sem_cd1_w = get_stats(overlap_cd1_wrong)
    m_cd10_t, sem_cd10_t = get_stats(overlap_cd10_true)
    m_cd10_w, sem_cd10_w = get_stats(overlap_cd10_wrong)

    print(f"At 1000 sweeps:")
    print(f"CD-1:  True = {m_cd1_t[-1]:.3f} +/- {sem_cd1_t[-1]:.3f}, Wrong = {m_cd1_w[-1]:.3f} +/- {sem_cd1_w[-1]:.3f} (Diff = {m_cd1_t[-1] - m_cd1_w[-1]:.3f})")
    print(f"CD-10: True = {m_cd10_t[-1]:.3f} +/- {sem_cd10_t[-1]:.3f}, Wrong = {m_cd10_w[-1]:.3f} +/- {sem_cd10_w[-1]:.3f} (Diff = {m_cd10_t[-1] - m_cd10_w[-1]:.3f})")

    # Plot Ink Overlap vs Steps
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
    ax.errorbar(sweep_eval_steps, m_cd1_t, yerr=sem_cd1_t, fmt="o-", color="#1f77b4", label="CD-1, true upper half", capsize=3)
    ax.errorbar(sweep_eval_steps, m_cd1_w, yerr=sem_cd1_w, fmt="s--", color="#1f77b4", alpha=0.7, label="CD-1, wrong upper half", capsize=3)
    ax.errorbar(sweep_eval_steps, m_cd10_t, yerr=sem_cd10_t, fmt="o-", color="#d62728", label="CD-10, true upper half", capsize=3)
    ax.errorbar(sweep_eval_steps, m_cd10_w, yerr=sem_cd10_w, fmt="s--", color="#d62728", alpha=0.7, label="CD-10, wrong upper half", capsize=3)

    ax.axhline(0.0, color="gray", linestyle="-.", lw=1, label="blank lower half (0)")
    ax.set_xscale("log")
    ax.set_xticks(sweep_eval_steps)
    ax.set_xticklabels([str(s) for s in sweep_eval_steps])
    ax.set_xlabel("clamped Gibbs sweeps", fontsize=10)
    ax.set_ylabel("ink overlap with the true lower half", fontsize=10)
    ax.set_ylim(-0.02, 0.75)
    ax.legend(loc="upper left", frameon=True, fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)

    overlap_fig_path = evidence_dir / "rbm-ink-overlap.png"
    plt.tight_layout()
    plt.savefig(overlap_fig_path, bbox_inches="tight")
    plt.close()
    print(f"Saved ink overlap plot to {overlap_fig_path}")

    # Summary in artifacts
    with open(artifacts_dir / "ink_overlap_summary.json", "w") as f:
        json.dump({
            "steps": sweep_eval_steps,
            "num_digits": num_eval_digits,
            "num_chains_per_digit": num_chains_eval,
            "cd1_true": {"means": m_cd1_t, "sems": sem_cd1_t},
            "cd1_wrong": {"means": m_cd1_w, "sems": sem_cd1_w},
            "cd10_true": {"means": m_cd10_t, "sems": sem_cd10_t},
            "cd10_wrong": {"means": m_cd10_w, "sems": sem_cd10_w},
        }, f, indent=2)

    # Verification: true-half overlap after 1000 steps should exceed wrong-half overlap by at least 0.1
    assert m_cd1_t[-1] - m_cd1_w[-1] >= 0.10, f"CD-1 margin {m_cd1_t[-1] - m_cd1_w[-1]:.3f} must be >= 0.1"
    assert m_cd10_t[-1] - m_cd10_w[-1] >= 0.10, f"CD-10 margin {m_cd10_t[-1] - m_cd10_w[-1]:.3f} must be >= 0.1"
    print("Part 2 verification successfully passed!")

if __name__ == "__main__":
    main()
