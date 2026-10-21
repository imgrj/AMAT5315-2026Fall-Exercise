"""Part 1, first task: the target and forward diffusion only (Eq. 1-7).

No network and no reverse sampling here, as the learning sheet's first DO item
asks.  This script builds the four-cluster target, applies the forward rule of
Equation (6) for all 1000 steps, and checks the closed-form jump of Equation (7)
against the iterated process.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax
import jax.numpy as jnp
import numpy as np

import _common
from _common import ARTIFACTS

sys.path.insert(0, str(_common.ROOT))
from src import diffusion, mixture, schedule

SEED = 2026
N = 2000
LEVELS = [0, 25, 100, 200, 350, 500, 700, 850, 1000]


def main() -> int:
    _common.ensure_dirs()
    (ARTIFACTS / "part1").mkdir(parents=True, exist_ok=True)
    sched = schedule.linear_schedule()
    key = jax.random.PRNGKey(SEED)
    print("=== Part 1 (first task): target sampling and forward diffusion ===")
    print(f"T={len(sched)}  beta_1={sched.betas[0]:.5f}  beta_T={sched.betas[-1]:.4f}  abar_T={sched.abars[-1]:.3e}")

    x0 = mixture.sample(key, N)
    states = diffusion.forward_movie_states(x0, sched, LEVELS, key)
    end = np.asarray(states[-1])
    mean, var = end.mean(axis=0), end.var(axis=0)
    print(f"iterated forward process, t=1000: mean={mean} variance={var}")

    # closed-form jump of Equation (7) must match the iterated distribution
    eps = jax.random.normal(key, x0.shape)
    closed = np.asarray(diffusion.direct_noise(x0, float(sched.abars[-1]), eps))
    print(f"closed-form jump, t=1000: mean={closed.mean(axis=0)} variance={closed.var(axis=0)}")
    print("comparison (per coordinate): "
          f"|mean difference| max {np.abs(mean - closed.mean(axis=0)).max():.4f}, "
          f"|variance difference| max {np.abs(var - closed.var(axis=0)).max():.4f}")

    ok = bool(np.all(np.abs(mean) < 0.1) and np.all(np.abs(var - 1.0) < 0.1))
    print(f"end state consistent with a standard Gaussian: {ok}")

    payload = {
        "seed": SEED,
        "n": N,
        "levels": LEVELS,
        "iterate_mean": mean.tolist(),
        "iterate_variance": var.tolist(),
        "closed_form_mean": closed.mean(axis=0).tolist(),
        "closed_form_variance": closed.var(axis=0).tolist(),
        "standard_gaussian_check": ok,
    }
    path = ARTIFACTS / "part1" / "forward_check.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {path.relative_to(_common.ROOT)}")
    np.savez_compressed(ARTIFACTS / "part1" / "forward_samples.npz", **{
        f"level_{level}": np.asarray(state) for level, state in zip(LEVELS, states)
    })
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
