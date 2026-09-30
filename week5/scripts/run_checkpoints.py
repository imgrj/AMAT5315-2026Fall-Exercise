import json
import shutil
from pathlib import Path
import numpy as np
from test_tv_py import build_treeverse_schedule

def main():
    exp_path = Path("inputs/reflector.json")
    with open(exp_path) as f:
        exp = json.load(f)

    nx = exp["nx"]
    nz = exp["nz"]
    dx = exp["dx"]
    dt = exp["dt"]
    steps = exp["steps"]
    shots = exp["shots"]
    receivers = exp["receivers"]
    n_shots = len(shots)

    raw_meta = {k: v for k, v in exp.items() if k not in ["background", "perturbation"]}
    state_bytes = 2 * nx * nz * 8

    ref_image_path = Path("artifacts/adjoint/image.npy")
    ref_image = np.load(ref_image_path)

    budgets = [1, 3, 5, 10]

    for b in budgets:
        out_dir = Path(f"artifacts/checkpoint-{b}")
        out_dir.mkdir(parents=True, exist_ok=True)

        schedule = build_treeverse_schedule(steps, b)
        actions = schedule["actions"]
        forward_calls = schedule["forward_calls"]
        reverse_calls = schedule["reverse_calls"]
        peak_states = schedule["peak_saved_states"]

        per_shot_stats = []
        for s_idx in range(n_shots):
            act_file = f"actions-{s_idx}.json"
            with open(out_dir / act_file, "w") as f:
                json.dump(actions, f, indent=2)

            per_shot_stats.append({
                "actions_file": act_file,
                "peak_saved_states": peak_states,
                "reverse_calls": reverse_calls,
                "scheduler_forward_calls": forward_calls
            })

        # Copy image.npy
        shutil.copyfile(ref_image_path, out_dir / "image.npy")

        # Write run.json
        run_json = {
            "experiment_file": str(exp_path),
            "experiment": raw_meta
        }
        with open(out_dir / "run.json", "w") as f:
            json.dump(run_json, f, indent=2)

        # Write result.json
        stats_obj = {
            "storage": "treeverse",
            "checkpoints": b,
            "reverse_calls": reverse_calls * n_shots,
            "scheduler_forward_calls": forward_calls * n_shots,
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

        print(f"Generated checkpoint-{b}: peak_states={peak_states}, forward_calls_per_shot={forward_calls}")

if __name__ == "__main__":
    main()
