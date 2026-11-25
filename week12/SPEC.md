# Week 12: Integer programming and learned branching

Source: [official learning sheet](https://giggleliu.github.io/AMAT5315-2026Fall/week12-learning-sheet.html),
16-page PDF. The local `week12-learning-sheet.pdf` is the committed copy.

## Common contract

Work in `week12/`. One Rust tool, `branchopt`, solves everything through HiGHS (the
`highs` crate, built from source, presolve off, one thread); Python only draws the
figures, runs the independent tests and builds the web lab. Use 64-bit floating point
throughout, keep every saved value at full precision, and record the exact command,
the seed and the parameters in each `run.json`. Keep `artifacts/` and `target/` out of
Git; the published lab needs its figures under `docs/assets/week12/`, and `evidence/`
keeps only small JSON summaries.

The model is a spin glass on a graph `G = (V, E)` with couplings `J_ij` and fields `h_i`,

    H(s) = sum_{(i,j) in E} J_ij s_i s_j + sum_{i in V} h_i s_i,     s_i in {+1, -1},

written as the mixed-integer linear program of Eq. 5: `x_i = (1 - s_i)/2 in {0,1}`,
one continuous `z_e in [0,1]` per bond with `s_i s_j = 1 - 2 z_e`, and four
inequalities per bond (`z <= x_u + x_v`, `z <= 2 - x_u - x_v`, `z >= x_u - x_v`,
`z >= x_v - x_u`). The objective is linearized as `C + sum_i (-2 h_i) x_i +
sum_e (-2 J_e) z_e` with `C = sum_e J_e + sum_i h_i` (Eq. 6).

The sheet's printed reference values (Part 1 and 2 figures come from the sheet's
Python baseline; the printed solver numbers from Part 2 on quote its Rust answer key):

| Quantity | Sheet reference |
| :--- | :--- |
| Opening four-spin square | LP bound -4, exact -2, nearest-rounded +2 (excess 4) |
| Two-food diet LP | 23.071429 = 323/14 at (89/28, 8/7) |
| 16 x 22-spin, 52-bond, seed 2026 | every LP bound -52.00; mean exact -38.25; mean gap 13.75 (36.5% of \|H*\|) |
| Naive rounding on the same 16 | suboptimal on all 16; mean excess 34.25, up to 60 on instance 11 |
| Frustrated tetrahedron (K4, h_0 = +1) | 5 nodes, LP bounds -6, -4, -3, -3, -2 |
| 48 spin glasses, most-frac | 50.88 mean nodes, 50.88 LP solves, 21.86 ms |
| 48 spin glasses, Strong Branching | 16.00 mean nodes, 287.4 LP solves (18.0 per node), 125.23 ms |
| Imitation learning | 276 training nodes / 36 graphs, 84 validation nodes / 12 graphs, overlap 0, best validation cross-entropy 1.759 |
| 16 unseen, seed 3026 | most-frac 50.9 / neural 15.4 / strong 15.6 mean nodes; neural 7.15 ms against 122.85 ms |
| 8 larger, 30 spins / 72 bonds, seed 4026 | most-frac 177.8, neural 39.5, strong 38.0, root-neural 50.8 mean nodes |

Where the sheet says "yours should be close to ours" (node counts, which depend only on
the instances and the rule) the acceptance tolerance is a percentage; where it says
"ours must match" the tolerance is `1e-3` or exact.

Commit source, `Cargo.lock`, both design files, `scripts/`, `tests/` and `README.md`.
Commit the specification before the implementation and the evidence before the
publication commit, and report the actual measurements, including discrepancies.

## 1. Formulate a spin glass as an integer program (Part 1)

`branchopt --mode gap --spins 22 --edges 52 --count 16 --seed 2026 --out artifacts/gap`

* Reproduce both opening `pred` solutions (`Min(-2)` on the square, `Min(-138)` on the
  100-spin magnet), then formulate the square, the diet LP and 16 seeded spin glasses
  for HiGHS.
* The generator is SplitMix64 exactly as the design file writes it: instance `id` starts
  from the first output of a stream seeded `seed XOR (id * 0xD1B54A32D192ED03)`, a
  minimum degree of 2, a fixed bond count, couplings in `{+1,-1}` on the sorted edge
  list, then fields in `{-1,0,+1}`.
* Report the LP bound, the exact MILP optimum, the integrality gap, the fractional spin
  count, the nearest-rounded energy and its excess for every instance.

Acceptance:

* the square prints LP -4.000000 at (1/2,1/2,1/2,1/2), exact -2.000000 and
  nearest-rounded +2.000000 (excess 4);
* the diet LP prints 23.071429;
* every one of the 16 LP bounds is -52.00, the exact energies average -38.25, the mean
  gap is 13.75 (36.5% of `|H*|`), all 22 spins are fractional, naive rounding is
  suboptimal on all 16 instances with a mean excess of 34.25 and a maximum of 60 on
  instance 11;
* an independent NumPy re-implementation of the generator reproduces the saved edges,
  couplings and fields, and full enumeration of `2^n` states reproduces the MILP
  optimum on a 12-spin family.

Deliverable: `artifacts/gap.png` with (a) the LP bound beside the exact optimum and
(b) the integrality gap beside the rounding excess.

## 2. Exact Branch-and-Bound (Part 2)

`branchopt --mode bnb --spins 22 --edges 52 --count 48 --policy most-frac|strong --seed 2026 --out artifacts/bnb_*`

* Depth-first search, no incumbent, prune infeasible LPs and bounds
  `z_LP >= z_best - 1e-6`, split a free spin into `x_i = 0` and `x_i = 1`, visit the
  child nearer its LP value first (at 1/2, `x_i = 1` first).
* most-frac picks the candidate nearest 1/2 and falls back to the lowest index; Strong
  Branching probes both child LPs per candidate and maximizes
  `max(d0,1e-6) * max(d1,1e-6)` with a gain of `1e6` for a pruned or infeasible child.
  Trial LPs count as LP solves and never change the incumbent.

Acceptance: the tetrahedron trace is five nodes with bounds -6, -4, -3, -3, -2;
both rules match HiGHS on all 48 instances within `1e-9`; most-frac spends exactly one
LP per node; Strong Branching shrinks the tree by about 3x at about 18 LP solves per
node and is several times slower in wall clock.

Deliverable: `artifacts/bnb.png` with the sorted explored nodes and the sorted LP
solves per instance for both rules.

## 3. Learn to imitate Strong Branching (Part 3)

`branchopt --mode bnb ... --policy strong --export-labels`, then
`uv run --with jax==0.12.1 python train_branch.py --dataset ... --out artifacts/model.json`,
then `branchopt --mode eval --spins 22 --edges 52 --count 16 --seed 3026 --model ... --out artifacts/eval_test`

* Record Strong Branching's choices with the 12 per-spin features of the design file, at
  nodes with at least two candidates and at most 80 per instance, split by instance
  (0-35 train, 36-47 validate).
* Train the shared 12-48-1 tanh MLP with a masked softmax, cross-entropy plus a
  `2e-4` weight penalty, Adam at `4e-3`, batch 64, 120 epochs, seed 2026, keeping the
  best validation checkpoint among epoch 1 and every fifth epoch.
* Run the network inside Branch-and-Bound; it only chooses the branching spin, so
  exactness by separation keeps every optimum exact.

Acceptance: no instance appears in both splits; the six validation probes agree with
JAX to better than `1e-4` (ours reach `1e-15`); every optimum matches HiGHS; the neural
and most-frac rules spend exactly one LP per node; neural explores fewer nodes than
most-frac on the 16 unseen graphs.

Deliverable: `artifacts/neural_bnb.png` with the training and validation curves and the
sorted explored nodes of most-frac, neural and Strong Branching.

## 4. Time on a laptop and larger spin glasses (Part 4)

`branchopt --mode eval --spins 30 --edges 72 --count 8 --seed 4026 --model ... --out artifacts/eval_transfer`

* Add root-neural: score every spin once at the root, standardize those scores, and at
  each node maximize `r_i + 3 * 2 min(x_i, 1 - x_i)`; report LP time and
  feature-plus-inference time separately for all four rules.
* Test whether the 22-spin network transfers to 30-spin, 72-bond graphs without
  retraining.

Acceptance: every optimum stays exact and the non-Strong rules keep one LP per node;
both neural rules explore fewer nodes than most-frac on both sizes; both run many times
faster than Strong Branching; all four rules match HiGHS within `1e-6`.

Deliverable: `artifacts/eval.png` with the mean explored nodes of all four rules on both
runs and the wall-clock split between LP solves and inference on the larger run.
