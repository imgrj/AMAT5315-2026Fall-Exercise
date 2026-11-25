# Week 12: Integer programming and learned branching

A 100-spin spin glass has `2^100 ≈ 10^30` states, yet an integer-programming solver finds
its lowest-energy state in a couple of seconds. This week opens that solver on one
problem: the spin glass. It is written as a mixed-integer linear program, solved exactly
by Branch-and-Bound over HiGHS LP relaxations, and then sped up by a small network that
imitates Strong Branching.

The tool is one Rust binary, `branchopt`, with three modes:

| Mode | What it does | Evidence |
| :--- | :--- | :--- |
| `gap` | LP relaxation vs exact MILP, integrality gap, nearest-rounding excess, the opening square and the diet LP | `artifacts/gap.png` |
| `bnb` | exact Branch-and-Bound with the cheap and the strong branching rule | `artifacts/bnb.png` |
| `eval` | most-frac, Strong Branching, neural and root-neural on unseen spin glasses | `artifacts/neural_bnb.png`, `artifacts/eval.png` |

---

## Environment

Recorded on the machine that produced the evidence:

| Component | Version |
| :--- | :--- |
| Rust | `rustc 1.98.1`, `cargo 1.98.1` (MSVC toolchain) |
| HiGHS | built from source by the `highs-sys` 1.16 / `highs` 2.4 crates (needs CMake and a C++ compiler) |
| CMake | 4.4.5 (Ninja generator) |
| Python | 3.14.4 with numpy 2.4.5 and matplotlib 3.11.1 |
| JAX | `0.12.1`, CPU, 64-bit, installed on demand by `uv` |
| Solver settings | `presolve = false`, `threads = 1` (both recorded in `run.json`) |
| `pred` | `problemreductions-cli 0.6.0` (`cargo install problemreductions-cli@0.6.0 --locked`) |

The build needs `cmake` and a C/C++ toolchain on `PATH`; on Windows that is the Visual
Studio Build Tools environment plus CMake and Ninja. If they are missing, HiGHS cannot
be compiled.

---

## Build and run

From `week12/`:

```bash
cargo install --path . --force        # installs `branchopt` into ~/.cargo/bin
# or, to keep the binary inside the crate:
cargo build --release                 # target/release/branchopt
```

The whole pipeline, in the order of the learning sheet:

```bash
python scripts/run_all.py
```

which runs

```bash
branchopt --mode gap  --spins 22 --edges 52 --count 16 --seed 2026 --out artifacts/gap
python scripts/plot_gap.py

branchopt --mode bnb  --spins 22 --edges 52 --count 48 --policy most-frac --seed 2026 --out artifacts/bnb_most
branchopt --mode bnb  --spins 22 --edges 52 --count 48 --policy strong    --seed 2026 --out artifacts/bnb_strong
python scripts/plot_bnb.py

branchopt --mode bnb  --spins 22 --edges 52 --count 48 --policy strong --export-labels \
  --seed 2026 --out artifacts/bnb_strong
uv run --with jax==0.12.1 python train_branch.py \
  --dataset artifacts/bnb_strong/dataset.json --out artifacts/model.json
branchopt --mode eval --spins 22 --edges 52 --count 16 --seed 3026 \
  --model artifacts/model.json --out artifacts/eval_test
python scripts/plot_neural_bnb.py

branchopt --mode eval --spins 30 --edges 72 --count 8 --seed 4026 \
  --model artifacts/model.json --out artifacts/eval_transfer
python scripts/plot_eval.py
```

The label-export run reuses the Part 2 output directory, as the sheet's command does;
`run_all.py` therefore repeats the plain `--policy strong` run afterwards, so the timing
record behind `bnb.png` stays free of label-export overhead. `dataset.json` survives
that second run.

Verification:

```bash
python -m pytest tests -q                    # 69 independent tests
python scripts/check_acceptance.py           # 42 acceptance checks -> evidence/acceptance.json
python scripts/check_negative_controls.py    # 7 deliberate defects must be caught
```

### The opening spin glasses with `pred`

```bash
mkdir -p artifacts/pred && cd artifacts/pred
pred create SpinGlass --graph 0-1,1-2,2-3,3-0 --couplings=1,1,1,-1 -o square.json
pred solve square.json
pred create SpinGlass --random --num-vertices 100 --edge-prob 0.04 --seed 2026 -o r100.json
pred solve r100.json
```

Measured here (`artifacts/pred/session.txt`): the square gives
`"solution": [0, 0, 1, 0]` with `"evaluation": "Min(-2)"`, and the 100-spin magnet gives
`Min(-138)` — the same energies as the sheet — in 3.3 s instead of 1.9 s. Version 0.6.0
prints a harmless `panicked` thread warning before the answer, as the sheet warns.

---

## The formulation

Each spin becomes a bit `x_i = (1 - s_i)/2`, and each bond a continuous disagreement
`z_e in [0,1]` with `s_i s_j = 1 - 2 z_e`, tied by four inequalities:

```text
z_e <= x_u + x_v,      z_e <= 2 - x_u - x_v,
z_e >= x_u - x_v,      z_e >= x_v - x_u
```

so once the spins are 0/1 every `z_e` is pinned to `x_u xor x_v`. The objective is linear,
`C + sum_i (-2 h_i) x_i + sum_e (-2 J_e) z_e` with `C = sum_e J_e + sum_i h_i`; HiGHS
minimizes the linear part and the constant is added back. Only the `n` spins are integer,
so every Branch-and-Bound split is a split on a spin.

---

## Results

### Part 1 — the relaxation hides frustration, and rounding pays for it

| Result | Measured | Sheet |
| :--- | :--- | :--- |
| Opening square, LP bound at (1/2,1/2,1/2,1/2) | **-4.000000** | -4.000000 |
| Opening square, exact MILP optimum | **-2.000000** | -2.000000 |
| Opening square, nearest rounding | **+2.000000** (excess 4) | +2.000000 |
| Two-food diet LP | **23.071429** = 323/14 at (3.178571, 1.142857) | 23.071429 |
| 16 x 22-spin / 52-bond, every LP bound | **-52.00** | -52.00 |
| Mean exact ground-state energy | **-38.25** | -38.25 |
| Mean integrality gap | **13.75** (36.45 % of `|H*|`) | 13.75 (36.5 %) |
| Fractional spins at the LP optimum | **22 of 22** | all 22 |
| Naive rounding suboptimal on | **16 of 16** | 16 of 16 |
| Mean rounding excess / worst | **34.25** / **60** on instance 11 | 34.25 / 60 on instance 11 |

At `x_i = 1/2` every bond takes its preferred value, so the relaxation reaches
`-m = -52` on every graph: it pretends that all 52 bonds are satisfied at once. The
exact optimum is 13.75 higher on average, and rounding each relaxed spin up gives all
spins `-1`, which is 34.25 above the optimum on average.

### Part 2 — a smaller tree at a much higher price per node

| Rule (48 spin glasses) | Mean nodes | Mean LP solves | LP per node | Mean time | Largest error |
| :--- | :--- | :--- | :--- | :--- | :--- |
| most-frac | **52.79** | 52.79 | 1.00 | 44.12 ms | `6.5e-13` |
| Strong Branching | **16.96** | 307.13 | 18.11 | 205.90 ms | `6.5e-13` |

Strong Branching shrinks the tree **3.11x** but solves **18.1 LP per node** instead of
one, so it is **4.7x** slower in wall clock. Node and LP counts depend only on the
instances and the rule; the times are noisy on this laptop (repeated runs of the same
command ranged from 21 to 44 ms for most-frac and from 186 to 309 ms for Strong
Branching, so the speed ratio is between 4.7x and 7.1x). Our node counts sit 3-6 %
above the sheet's answer key, which uses the same instances and the same rules but a
different HiGHS version and therefore different LP vertices on degenerate nodes.

The four-spin frustrated tetrahedron reproduces the sheet's pruned search exactly:

| Node | Branch | Status | LP bound |
| :--- | :--- | :--- | :--- |
| 1 | root | split `x0` | -6 |
| 2 | `x0=1` | split `x1` | -4 |
| 3 | `x0=1,x1=1` | incumbent | -3 |
| 4 | `x0=1,x1=0` | pruned by bound | -3 |
| 5 | `x0=0` | pruned by bound | -2 |

Five nodes instead of seven: once `x0 = 0` has a lower bound of -2 and the incumbent is
-3, its whole fractional subtree is discarded.

### Part 3 — a network that copies the strong rule for free

| Quantity | Measured | Sheet |
| :--- | :--- | :--- |
| Training / validation nodes | **290 / 93** | 276 / 84 |
| Training / validation spin glasses | **36 / 12** (overlap 0) | 36 / 12 (overlap 0) |
| Validation cross-entropy | 2.789 at epoch 1 -> **1.833** at epoch 120 | ~2.7 -> 1.759 |
| JAX vs Rust probe probability difference | **6.7e-16** | below `1e-4` (`1e-15` in the key) |
| Largest error against HiGHS | **2.0e-12** | below `1e-6` (`3e-13` in the key) |
| LP solves equal nodes (most-frac, neural, root-neural) | **yes** | yes |

On the 16 unseen 22-spin graphs (seed 3026):

| Rule | Mean nodes | Mean LP time | Mean inference | Mean total |
| :--- | :--- | :--- | :--- | :--- |
| most-frac | 50.63 | 15.74 ms | 0.00 ms | 16.04 ms |
| Strong Branching | 15.50 | 90.39 ms | 0.00 ms | 91.30 ms |
| neural | **14.50** | 5.28 ms | 0.14 ms | **5.60 ms** |
| root-neural | 16.50 | 13.45 ms | 0.05 ms | 13.95 ms |

On the instance where the network saves the most (instance 4, ground energy -38),
most-frac splits spins 0 and 1 — every fractional spin sits at 1/2, so the cheap rule has
nothing better to use — and needs 153 nodes; the network splits spins 4 and 11 and
finishes in 17 nodes (6.8 ms against 48.4 ms). That is a 9x smaller tree.

### Part 4 — laptop time and size transfer to 30 spins / 72 bonds

| Rule | 22-spin nodes | 30-spin nodes | 30-spin total | LP time | Inference |
| :--- | :--- | :--- | :--- | :--- | :--- |
| most-frac | 50.6 | 175.3 | 71.6 ms | 70.7 ms | 0.00 ms |
| Strong Branching | 15.5 | 37.8 | 861.7 ms | 850.2 ms | 0.00 ms |
| neural | 14.5 | **44.3** | **55.7 ms** | 52.4 ms | 2.49 ms |
| root-neural | 16.5 | 50.5 | 63.4 ms | 62.5 ms | 0.07 ms |

Without retraining, the 22-spin network still cuts the 30-spin tree by 75 % (175.3 ->
44.3 nodes) and runs **15x** faster than Strong Branching, whose trial LPs cost 850 ms of
the 862 ms. Root-cached scoring pays inference once (0.07 ms) and still removes 71 % of
the tree. Our 30-spin most-frac and root-neural node counts agree with the sheet to
within 2 %; the network is slightly weaker than the sheet's (44.3 against 39.5 nodes),
which matches its slightly higher validation cross-entropy (1.833 against 1.759).

---

## Evidence files and producing commands

| Figure | Command | Description |
| :--- | :--- | :--- |
| `artifacts/gap.png` | `python scripts/plot_gap.py` | (a) LP lower bound beside the exact optimum for the 16 spin glasses; (b) integrality gap beside the nearest-rounding excess |
| `artifacts/bnb.png` | `python scripts/plot_bnb.py` | Sorted explored nodes and sorted LP solves per instance for most-frac and Strong Branching |
| `artifacts/neural_bnb.png` | `python scripts/plot_neural_bnb.py` | Training and validation cross-entropy against epoch; sorted nodes for most-frac, neural and Strong Branching on the 16 unseen graphs |
| `artifacts/eval.png` | `python scripts/plot_eval.py` | Mean explored nodes of all four rules on both evaluation runs; wall-clock split between LP solves and inference on the 30-spin graphs |

Every figure is drawn by a Python script from the saved `run.json` files in
`artifacts/`; nothing is hand-entered. The same figures are published under
`docs/assets/week12/` for the web lab.

## Files

| Path | Role |
| :--- | :--- |
| `branchopt.design.toml` | the task specification the tool implements (all four parts) |
| `train_branch.design.toml` | the trainer's settings |
| `src/rng.rs` | SplitMix64, exactly as the design file writes it |
| `src/instance.rs` | spin-glass instances, the opening square and the tetrahedron |
| `src/milp.rs` | the MILP and its relaxation, the reusable LP, the diet LP |
| `src/bnb.rs` | Branch-and-Bound, the four branching rules, the 12 features |
| `src/net.rs` | the MLP, the masked softmax and the JAX parity check |
| `src/main.rs` | the three modes, the run.json evidence and the printed table |
| `train_branch.py` | JAX imitation learning of Strong Branching |
| `scripts/` | the pipeline, the four figures, the acceptance checks and the negative controls |
| `tests/` | 69 independent tests, including a NumPy generator and full enumeration |
| `evidence/` | `acceptance.json` (42 checks) and `negative-controls.json` (7 defects) |
| `SPEC.md` | the task description and its acceptance criteria |

## Notes and discrepancies

* Our node counts are 3-6 % above the sheet's answer key on the 48 Part 2 instances and
  within 2 % on the Part 4 instances. The instances are identical (the Part 1 numbers
  match to the last digit), so the difference comes from which LP vertex HiGHS returns on
  degenerate, half-integral nodes: the `highs` crate is 2.4 here, the answer key used an
  earlier release.
* Wall-clock times fluctuate by up to 2x between runs on this laptop; node and LP counts
  do not. The reported ratios are computed from one consistent run of each command.
* `pred` 0.6.0 prints a `panicked` thread warning before its answer; the sheet documents
  it as harmless, and the answers match.
