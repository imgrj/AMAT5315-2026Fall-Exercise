import unicodedata
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def main():
    ref_image = np.load("artifacts/adjoint/image.npy")
    ref_l2 = np.linalg.norm(ref_image)
    
    budgets = [1, 3, 5, 10]
    expected_peaks = [2, 4, 6, 11]
    expected_calls_per_shot = [28680, 1695, 990, 642]
    
    print("Checkpointing verification against full history:")
    
    forward_steps_per_shot = []
    peak_bytes_list = []
    
    for b, exp_peak, exp_calls in zip(budgets, expected_peaks, expected_calls_per_shot):
        ckpt_dir = Path(f"artifacts/checkpoint-{b}")
        ckpt_image = np.load(ckpt_dir / "image.npy")
        rel_l2_err = np.linalg.norm(ckpt_image - ref_image) / ref_l2
        
        with open(ckpt_dir / "result.json") as f:
            res = json.load(f)
            
        stats = res["statistics"]
        peak_states = stats["peak_saved_states"]
        peak_bytes = stats["peak_saved_bytes"]
        peak_bytes_list.append(peak_bytes)
        
        # Calls per shot
        shot0_calls = stats["per_shot"][0]["scheduler_forward_calls"]
        forward_steps_per_shot.append(shot0_calls)
        
        # Audit every action file
        grad_diff_count = 0
        invalid_restores = 0
        budget_overruns = 0
        
        for shot_idx in range(len(res["shots"])):
            act_path = ckpt_dir / f"actions-{shot_idx}.json"
            with open(act_path) as f:
                actions = json.load(f)
                
            saved_slots = {0}
            grad_steps = []
            
            for act in actions:
                atype = act["action"]
                step = act["step"]
                scount = act["saved_states"]
                
                if scount > b + 1:
                    budget_overruns += 1
                    
                if atype == "store":
                    saved_slots.add(step)
                elif atype == "fetch":
                    if step in saved_slots:
                        saved_slots.remove(step)
                elif atype == "restore":
                    if step not in saved_slots:
                        invalid_restores += 1
                elif atype == "grad":
                    grad_steps.append(step)
                    
            expected_grad_order = list(range(res["steps"] - 1, -1, -1))
            if grad_steps != expected_grad_order:
                grad_diff_count += sum(1 for g, e in zip(grad_steps, expected_grad_order) if g != e)
                grad_diff_count += abs(len(grad_steps) - len(expected_grad_order))
                
        print(f"Budget delta = {b:2d}:")
        print(f"  Relative L2 error vs full history: {rel_l2_err:.2e} (expected < 1e-9)")
        print(f"  Peak saved states: {peak_states} (expected: {exp_peak})")
        print(f"  Forward steps per shot: {shot0_calls} (expected: {exp_calls})")
        print(f"  Audit: grad discrepancies = {grad_diff_count}, invalid restores = {invalid_restores}, budget overruns = {budget_overruns}")

    # Plot 1: artifacts/checkpoint-actions.png (first shot, budget 5)
    with open("artifacts/checkpoint-5/actions-0.json") as f:
        act5 = json.load(f)
        
    op_indices = np.arange(len(act5))
    steps_act = [a["step"] for a in act5]
    types_act = [a["action"] for a in act5]
    
    # Plot sawtooth line
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(op_indices, steps_act, color="gray", linewidth=0.7, alpha=0.6)
    
    color_map = {
        "store": "C2",    # green
        "restore": "C1",  # orange
        "call": "C0",     # blue
        "grad": "C3",     # red
        "fetch": "C4"     # purple
    }
    
    for atype in ["store", "restore", "call", "grad", "fetch"]:
        idx = [i for i, t in enumerate(types_act) if t == atype]
        s = [steps_act[i] for i in idx]
        ax.scatter(idx, s, label=atype.capitalize(), color=color_map[atype], s=12, alpha=0.9)
        
    ax.set_title("Treeverse schedule; reflector shot 0, budget 5")
    ax.set_xlabel("Operation index")
    ax.set_ylabel("Time step")
    ax.set_ylim(-10, 260)
    ax.legend(loc="upper right", ncol=5, framealpha=0.8, fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.5)
    
    plt.tight_layout()
    act_out = Path("artifacts/checkpoint-actions.png")
    plt.savefig(act_out, dpi=200)
    plt.close()
    print(f"Saved {act_out}")

    # Plot 2: artifacts/checkpoint-work.png (recomputation cost and storage)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
    
    # Left: Recomputation cost
    ax1.plot(budgets, forward_steps_per_shot, 'o-', color="C0", label="Treeverse", linewidth=1.5)
    ax1.axhline(240, color="gray", linestyle=":", label="Full history", linewidth=1.2)
    ax1.set_yscale("log")
    ax1.set_xlabel("Additional checkpoint slots $\\delta$")
    ax1.set_ylabel("Forward steps of the schedule per shot")
    ax1.set_title("Recomputation cost")
    ax1.set_xticks(budgets)
    ax1.legend(loc="upper right", framealpha=0.8)
    ax1.grid(True, which="both", linestyle=":", alpha=0.5)
    
    # Right: Storage in bytes
    ax2.plot(budgets, peak_bytes_list, 's-', color="C0", linewidth=1.5)
    for b_val, byte_val, p_states in zip(budgets, peak_bytes_list, expected_peaks):
        offset = (10, -10) if b_val == 10 else (10, 0)
        ax2.annotate(f"{p_states} states", (b_val, byte_val), textcoords="offset points", xytext=offset, fontsize=8)
        
    full_history_bytes = 241 * (2 * 41 * 41 * 8) # 6,481,936
    ax2.text(0.05, 0.90, f"Full history: {full_history_bytes:,} bytes (241 states)", transform=ax2.transAxes, fontsize=8)
    ax2.set_xlabel("Additional checkpoint slots $\\delta$")
    ax2.set_ylabel("Peak saved-state bytes")
    ax2.set_title("Two wavefields per saved state")
    ax2.set_xticks(budgets)
    ax2.set_ylim(0, 360000)
    ax2.grid(True, linestyle=":", alpha=0.5)
    
    plt.tight_layout()
    work_out = Path("artifacts/checkpoint-work.png")
    plt.savefig(work_out, dpi=200)
    plt.close()
    print(f"Saved {work_out}")

if __name__ == "__main__":
    main()
