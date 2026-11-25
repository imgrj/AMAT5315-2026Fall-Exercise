# Week 11: Hard problems, reductions, and tensor-network sampling

Source: [official learning sheet](https://giggleliu.github.io/AMAT5315-2026Fall/week11-learning-sheet.html),
15 pages. The local `week11-learning-sheet.pdf` is the committed copy; `SPEC.md` is the task
description derived from it.

Four parts, one tool, three figures:

1. **A round trip through a reduction** (`pred`, no code of ours): factoring 15 becomes a
   104-spin magnet, the magnet's ground state is mapped back to `3 x 5`.
2. **Hard problems as ground states**: every logic gate becomes a small magnet (a *gadget*)
   whose floor holds exactly its truth table, so factoring becomes the search for a magnet's
   lowest energy.
3. **Tropical contraction and sampling**: `tnsample` plans a contraction with `omeco`,
   contracts it with rescaled Boltzmann weights for `log Z`, contracts it again in min-plus
   arithmetic for `(E_min, g_0)`, and draws independent spin states from one stored tree.
4. **N-queens as a tensor network**: one bit per square and four carry tensors per square
   count every board for `N = 4..10` and draw uniform boards.

---

## Quick start

```bash
cd week11
cd tnsample && cargo install --path . --force && cd ..     # the tool
python scripts/run_pred.py                                 # Part 1 (needs pred)
python scripts/run_all.py                                  # every stated run + the figures
python -m pytest -q                                        # 40 acceptance checks
```

`scripts/run_all.py` runs each command the sheet prints, in the sheet's order, then draws
`artifacts/gates.png`, `factoring.png` and `nqueens.png` from the saved runs and writes
`evidence/acceptance.json`. `runs/manifest.json` keeps the 47 commands with their exit codes
and wall times.

## Layout

```text
tnsample.design.toml     the sheet's own design file, kept verbatim
SPEC.md                  the task description
tnsample/                the Rust crate (library + `tnsample` binary)
  src/gates.rs           the seven gate energies and the joined circuit
  src/net.rs             networks, omeco planning, dense/tropical/sparse contraction, sampling,
                         the open-lattice builder and the N-queens carry network
  src/rng.rs             documented splitmix64, one stream for bonds and one for draws
  src/main.rs            the CLI, the stdout table and every `run.json`
  tests/network.rs       4 Rust checks (sparse == dense, board counts, seed purity)
scripts/
  run_pred.py            Part 1: the six `pred` commands and their transcript
  run_all.py             every stated command, the figures and the acceptance summary
  reference.py           independent NumPy references (gates, magnet sums, queen rules, OEIS)
  plot_gates.py          artifacts/gates.png
  plot_factoring.py      artifacts/factoring.png
  plot_nqueens.py        artifacts/nqueens.png
  check_sheet.py         every number against the sheet  ->  evidence/acceptance.json
tests/                   40 pytest checks derived from SPEC.md, not from the implementation
evidence/                acceptance.json, reduction.json, negative-controls.json
runs/                    every raw run (not in Git)
artifacts/               the three figures (not in Git; published copies live in ../docs/assets/week11/)
```

## The tool

Built from `tnsample.design.toml` (the sheet's file) with `cargo install --path .`. Every run
prints a `mode metric value runtime_ms` table and writes `<out>/run.json` with the schema
version, mode, seed, the run parameters, the plan's fingerprint (`plan_hash`), and every
number needed to recompute the result at full precision.

```text
tnsample --mode gadgets --seed 2026 --out runs/gadgets
tnsample --mode plan    --order greedy|treesa --input <bundle> --seed 2026 --out runs/plan
tnsample --mode count   --input <bundle> --beta 2.0 --seed 2026 --out runs/count
tnsample --mode count   --l 10 --p-af 0 --beta 2.0 --seed 2026 --out runs/count_p0
tnsample --mode sample  --input <bundle> --beta 2.0 --samples 2000 --seed 2026 --out runs/sample
tnsample --mode count   --n 8 --seed 2026 --out runs/count_nq8
tnsample --mode sample  --n 6 --samples 200 --seed 2026 --out runs/sample_nq6
```

Nothing is defaulted. `--mode`, `--seed` and `--out` are always required; exactly one of
`--input`, `--l`/`--p-af` or `--n` selects the problem; `--beta` is required for a magnet;
`--samples` belongs to `sample` only; an unknown flag is an error with exit code 2, not a
silent default.

**Two sign conventions, never mixed.** The sheet's lattice magnet is
`H = -sum_{<ij>} J_ij s_i s_j` with `J_ij = +-1`; internally it is `K_ij = -J_ij`, which is
the `pred` bundle convention `E = +sum K_ij s_i s_j + sum h_i s_i`. Boltzmann weights are
`exp(-beta E)` in both cases. The lattice bonds are drawn from the documented splitmix64
stream in row-major site order, horizontal bond first, one uniform per bond.

**The planner.** `--order greedy` calls omeco's greedy planner, `--order treesa` its annealer
with four independent trials (`TreeSA::fast().with_ntrials(4)`), and the better plan is kept.
Both planners break equal-cost ties through hash iteration order, so a plan is one sample of
the planner: `run.json` records `tc`, `sc` and a `plan_hash`, and two runs can agree on every
physical number while using different trees. `--order greedy` and `--order treesa` therefore
give the same `E_min`, `g_0`, `log Z` and board counts, but not necessarily the same sample
sequence.

## Part 1: the factoring round trip (`pred` 0.6.0)

```bash
pred path Factoring SpinGlass
pred create Factoring --target 15 --m 2 --n 3 -o runs/pred/factor_15.json
pred reduce runs/pred/factor_15.json --to SpinGlass -o runs/pred/factor_15.bundle.json
pred solve runs/pred/factor_15.bundle.json
pred evaluate runs/pred/factor_15.json --config 1,1,1,0,1
```

| Target | Bits (lowest first) | Reads | Check | Magnet |
| :--- | :--- | :--- | :--- | :--- |
| 15 | `[1,1,1,0,1]` | 3 x 5 | `Min(0)` | `Min(-299)` on 104 spins |
| 6 | `[0,1,1,1]` | 2 x 3 | `Min(0)` | `Min(-204)` on 72 spins |
| 7 | `[1,1,0,1]` | 3 x 2 = 6 | `Min(1)` | `Min(-202)` on 72 spins |

The route is `Factoring -> CircuitSAT -> SpinGlass`, and the magnet is much larger than the
question: five unknown bits become 104 spins. For 7 no two-bit factor pair exists, so the
magnet's lowest state decodes to a wrong product with `Min(1)`. The sheet's example is
`1 x 3` with `Min(4)`; the sheet already warns that "different ground states can therefore
decode to different wrong products", and this run decodes `3 x 2`.

## Part 2: the gadgets (`--mode gadgets`)

Seeded by nothing 鈥?every number is an enumeration, so it must agree with the sheet:

| Gate | Floor | Forbidden rows |
| :--- | :--- | :--- |
| AND, OR | -3 | +1 and +9 |
| NOT, COPY, SET0, SET1 | -1 | +1 |
| XOR (with helper `w`) | -4 | -2 |

The AND-NOT-SET1 circuit reaches `E_0 = -5` on exactly three of its 16 states 鈥?the inputs
`(a,b) = 00, 01, 10`, for which `NOT(AND(a,b)) = 1` 鈥?and the inconsistent wire
`(a,b,t,out) = (1,0,1,1)` sits at +1. `H_XOR + 4 = 2(x + y - z - 2r)^2` is checked row by row.

`artifacts/gates.png` (a) every truth-table row of every gadget, allowed rows marked, one
line per floor; (b) the 16 states of the joined circuit.

## Part 3: counting, sampling and planning

| Run | Result |
| :--- | :--- |
| `count_p0` (10x10, `p-af = 0`, beta = 2) | `E_min = -180 = -2L(L-1)`, `g_0 = 2`, `log Z = 360.694747` |
| `count_f15` (104 spins, beta = 2) | `E_min = -299`, `g_0 = 1`, `log Z = 598.317945`, `sc = 5` (32 entries), `tc = 11.45` |
| `sample_f15` (2000 draws, beta = 2) | `p_0 = 0.727643`, predicted `1455.29 +- 19.91`, observed **1466** |
| `pred extract` of the lowest sample | `[1,1,1,0,1]` -> `3 x 5 = 15`, `Min(0)` |

The sampled energy histogram is `-299: 1466, -297: 261, -295: 223, -293: 44, -291: 6`, which
sums to 2000. The sheet's reference run draws 1460 ground states and its own prediction is
1455; our 1466 is 0.5 binomial sigma above the same prediction.

`log Z` is cross-checked by an independent NumPy contraction of the same bundle (a greedy
order of its own, in `scripts/reference.py`): both give `598.317945217002`, and the same
reference obtains `(E_min, g_0) = (-299, 1)` from `log Z` at beta = 10 and 14.

Planning the multiplier for `k`-bit factors (`14k^2 + 8k` spins):

| Bits `k` | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Spins | 72 | 150 | 256 | 390 | 552 | 742 | 960 |
| `treesa` sc (sheet) | 4 (4) | 7 (7) | 9 (9) | 11 (11) | 13 (13) | 15 (15) | 17 (17) |
| `treesa` tc (sheet) | 10.62 (10.7) | 12.54 (12.6) | 14.47 (14.5) | 16.57 (16.6) | 19.00 (19.0) | 21.59 (21.5) | 24.08 (24.0) |
| `greedy` sc (sheet) | 4 (5) | 10 (9) | 14 (14) | 22 (23) | 33 (27) | 39 (37) | 55 (58) |

`treesa` finds the multiplier's diagonal seam, `sc = 2k + 1` for `k >= 3`; `greedy` pays for
its speed with memory and is far past 8 GB by `k = 8`. `greedy` breaks ties at random, so its
column differs from the sheet's between runs 鈥?the sheet says so itself.

`artifacts/factoring.png`: (a) `sc(k)` for both orders with the 8 GB line and the `2k+1`
guide; (b) the sampled energy histogram beside `2000 g_0 exp(-beta E_min)/Z`.

## Part 4: N-queens (`--mode count/sample --n`)

| `N` | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Contracted | 2 | 10 | 4 | 40 | 92 | 352 | 724 |
| OEIS A000170 | 2 | 10 | 4 | 40 | 92 | 352 | 724 |
| `sc` / wall time | 11 / 0.06 s | 15 / 0.09 s | 18 / 0.12 s | 20 / 0.19 s | 22 / 0.31 s | 27 / 0.81 s | 30 / 15.7 s |

Every count matches. Sparse contraction keeps `N <= 8` to a third of a second; `N = 10`
took 15.7 s in the recorded run (1.8-16 s across runs, depending on the planned order), and
the sheet's "a few seconds" is met by the sparse scheme rather than by dense arithmetic,
which would need `2^30` entries here.

200 uniform draws for `N = 6` visit all four boards `52, 49, 47, 52` times (chi-square 0.36
with three degrees of freedom; the sheet's reference run gives `56, 43, 52, 49`). 200 draws
for `N = 8` visit **85** of the 92 boards, against the expected
`92 (1 - (91/92)^200) = 81.7` (the sheet's reference run finds 80).

`artifacts/nqueens.png`: (a) the counts against OEIS on a log scale; (b) the 6-queens
frequencies beside the uniform 50.

## Environment

| Component | Version |
| :--- | :--- |
| Rust | 1.98.1 (`cargo install --path tnsample`) |
| Crate dependencies | `omeco` 0.2.6 (planning), `serde` 1, `serde_json` 1 |
| `pred` | `problemreductions-cli` 0.6.0 (`pred 0.6.0`) |
| Python | 3.12.13 with numpy 2.5.5, matplotlib 3.11.2, pypdf 7.1.1, pytest 9.2.0 |
| Machine | Windows 11, Intel Core Ultra 7 356H, 31.5 GB RAM |

`week11/.venv` is the interpreter the commands above use; `uv pip install numpy matplotlib
pypdf pytest` recreates it.

## Checks

`python -m pytest -q` runs 40 checks derived from `SPEC.md`; `python scripts/check_sheet.py`
runs 52 comparisons against the sheet's printed numbers and writes
`evidence/acceptance.json`. `tnsample/tests/network.rs` adds 4 Rust checks. The negative
control in `evidence/negative-controls.json` records a deliberate sabotage of three rules
(a wrong XOR field, a min-plus sign flip and a diagonal end constraint) and the tests that
caught it. `python scripts/check_web.py` runs 11 browser checks (the local file unless
`--url` names the published page) and `python scripts/check_publication.py` compares every
deployed file with `docs/` byte for byte; `evidence/web-validation.json` and
`evidence/publication.json` keep both records.

