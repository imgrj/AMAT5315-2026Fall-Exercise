# Week 11: Hard problems, reductions, and tensor-network sampling

Source: [official learning sheet](https://giggleliu.github.io/AMAT5315-2026Fall/week11-learning-sheet.html),
15-page PDF. The local `week11-learning-sheet.pdf` is the committed copy.

## Common contract

Work in `week11/`. Rust runs every number; Python (numpy/matplotlib) draws every
figure and holds the independent checks in `tests/`. The tool is `tnsample`, built from
`tnsample.design.toml` (the sheet's own file, kept verbatim) in the crate `tnsample/`
with `cargo install --path .`, and the commands are the sheet's, run from `week11/`:

```bash
cd tnsample && cargo install --path . --force && cd ..
tnsample --mode <gadgets|plan|count|sample> ... --seed 2026 --out runs/<name>
```

Nothing is defaulted: a missing `--mode`, `--out` or mode-specific input is an error, and an
unknown flag is rejected instead of ignored. Every run writes `<out>/run.json` with the schema
version, mode, seed, the run parameters and the full-precision numbers needed to recompute
them, and prints one tab-separated table with the header `mode metric value runtime_ms`.
`runs/` and `artifacts/` stay out of Git; the published figures are copied to
`docs/assets/week11/`, and `evidence/` keeps the small JSON summaries.

Two sign conventions live side by side and are never mixed silently:

* the sheet's lattice magnet, `H = -sum_{<i,j>} J_ij s_i s_j`, `s_i = +-1`, `J_ij = +-1`
  (`--l`, `--p-af`), with bonds generated from the splitmix64 generator and `--seed`;
* a `pred` SpinGlass bundle (`--input`), whose energy is
  `E(s) = +sum_{i<j} K_ij s_i s_j + sum_i h_i s_i`.

Boltzmann weights are `exp(-beta E)` in both cases, `Z = sum_s exp(-beta E(s))`, and the
tropical (min-plus) contraction returns `(E_min, g_0)` in the same convention as the input.

Commit source, `Cargo.lock`, `tnsample.design.toml`, tests, scripts and `README.md`; commit
the specification before the tests and the tests before the implementation, record a failing
test run and a deliberate negative control, and commit each completed task. Report the actual
measurements, including discrepancies against the sheet, and never tune a tolerance.

## Part 1 (sheet `DO`, no code): the factoring round trip with `pred`

Run the `problem-reductions` CLI (`pred`) from `week11/` and keep every transcript:

```bash
pred path Factoring SpinGlass
pred create Factoring --target 15 --m 2 --n 3 -o runs/pred/factor_15.json
pred reduce runs/pred/factor_15.json --to SpinGlass -o runs/pred/factor_15.bundle.json
pred solve runs/pred/factor_15.bundle.json
pred evaluate runs/pred/factor_15.json --config 1,1,1,0,1
for N in 6 7; do ... same trip with --m 2 --n 2 ... done
```

Acceptance:

* `pred path` prints `Factoring -> CircuitSAT -> SpinGlass`;
* for 15: `pred solve` returns `Solution: [1, 1, 1, 0, 1]` (3 x 5) with `Min(0)`, its
  SpinGlass intermediate reaching `Min(-299)` on 104 spins, and `pred evaluate` repeats
  `Min(0)` on the original problem;
* for 6: `[0, 1, 1, 1]` reads 2 x 3 with `Min(0)`; for 7: the best state decodes to a wrong
  product with `Min(4)` — a reduction cannot invent an answer, and the check on the original
  problem exposes it;
* the bundle of 15 is the input of Parts 3 and 4; its energy is the bundle convention above.

`scripts/run_pred.py` runs the six commands, saves stdout/stderr and parses the results into
`runs/pred/transcript.json`; `evidence/reduction.json` keeps the distilled answers.

## Part 2 (build prompt 1): `--mode gadgets`

Implement `tnsample --mode gadgets` from `tnsample.design.toml`: with bits `x` as spins
`s = 2x - 1`, enumerate every state of the seven gate energies in the design (for XOR,
minimize over the helper spin `w`) and report the energy of every truth-table row; also
enumerate the 16 states of the circuit `AND(a,b,t) + NOT(t,out) + SET1(out)`.

Acceptance (every number by enumeration, no seed involved, so ours must equal the sheet's):

* floors: AND -3, OR -3, NOT -1, COPY -1, SET0 -1, SET1 -1, XOR -4; on every gate the
  allowed truth-table rows sit exactly on the floor and every forbidden row is at least 2 above
  it;
* XOR reaches its floor only through the helper bit, with `H_XOR + 4 = 2(x + y - z - 2r)^2`;
* the three-gate circuit reaches `E_0 = -5` on exactly three of its 16 states, the inputs
  00, 01, 10 for which `NOT(AND(a,b)) = 1`, and the 11 row sits at -3;
* `artifacts/gates.png` plots the energy of every truth-table row of each gadget and all 16
  states of the joined circuit, marking the allowed rows, drawn by `scripts/plot_gates.py`
  from the saved run.

## Part 3 (build prompt 2): `--mode plan`, `--mode count`, `--mode sample`

Append the `[args.magnet]`, `[args.plan]`, `[args.count]` and `[args.sample]` blocks of
`tnsample.design.toml`. Plan with `omeco` (`--order greedy` or `treesa`), contract with
rescaling and accumulated log scales for `log Z`, contract in min-plus on `(energy, count)`
pairs for `(E_min, g_0)`, and sample by walking the stored tree from the root down.

The stated checks:

```bash
tnsample --mode count --l 10 --p-af 0 --beta 2.0 --seed 2026 --out runs/count_p0
tnsample --mode count --input runs/pred/factor_15.bundle.json --beta 2.0 --seed 2026 --out runs/count_f15
tnsample --mode sample --input runs/pred/factor_15.bundle.json --beta 2.0 --samples 2000 --seed 2026 --out runs/sample_f15
for k in 2 3 4 5 6 7 8; do T=((2^k-1)^2); pred create/reduce; tnsample --mode plan --order greedy|treesa --input ... --out runs/plan_k${k}_$ord; done
```

Acceptance:

* the 10 x 10 ferromagnet (`p-af = 0`): every bond satisfied, `E_min = -2L(L-1) = -180`,
  `g_0 = 2`;
* `factor_15.bundle.json` (104 spins): `E_min = -299`, `g_0 = 1`, `log Z = 598.3179` at
  `beta = 2`, largest intermediate tensor 32 entries (`sc = 5`);
* 2000 samples at `beta = 2`: Boltzmann theory predicts `2000 p_0` ground states with
  `p_0 = g_0 exp(-beta E_min)/Z` (1455 for the reference run, 1460 in the sheet's), every
  sampled energy is `>= E_min`, and the histogram sums to 2000;
* `pred extract` on the lowest-energy sample decodes to `[1, 1, 1, 0, 1]` (3 x 5 = 15, Min(0));
* the plan loop reports the sheet's spin counts `14k^2 + 8k` (72, 150, 256, 390, 552, 742, 960)
  and a `treesa` space complexity that follows the multiplier's `2k + 1` seam for `k >= 3`
  instead of `greedy`'s growth through 8 GB of RAM;
* `artifacts/factoring.png` shows `sc(k)` for both orders beside the sampled energy histogram
  with `2000 g_0 exp(-beta E_min)/Z` marked, drawn by `scripts/plot_factoring.py`.

## Part 4 (build prompt 3): `--mode count/sample --n`

Encode N-queens directly as the carry tensor network of the design: one bit per square, four
carry tensors `T(c_in, x, c_out)` per square equal to 1 at `(0,0,0)`, `(0,1,1)`, `(1,0,1)` and
0 elsewhere; rows and columns start at 0 and must end at 1; diagonals start at 0 and may end at
0 or 1. Contract only nonzero entries.

Acceptance:

* the contracted counts for `N = 4..10` match OEIS A000170 exactly:
  2, 10, 4, 40, 92, 352, 724;
* all 200 sampled boards for `N = 6` and `N = 8` obey the queen rules; for `N = 6` each of the
  4 boards appears about 50 times (+-6), and 200 uniform draws out of 92 boards visit about
  `92 (1 - (91/92)^200) ~= 82` distinct boards for `N = 8`;
* `artifacts/nqueens.png` plots the counts against the known sequence (log scale) and the
  6-queens sample frequencies, drawn by `scripts/plot_nqueens.py`;
* sparse contraction keeps every `N <= 10` to seconds (dense contraction is checked only up to
  `N = 6`, where both agree).

## Presentation

Keep `artifacts/gates.png`, `factoring.png` and `nqueens.png` in `artifacts/` and the run
folders in `runs/` on the machine; copy the three figures into `docs/assets/week11/` for the
published page. `README.md` records the reproduction commands, the environment, the `pred`
version, settings and seeds. `docs/week11-tnsample-lab.html` carries the same upper-right
segmented switch as Weeks 5-10: **Interactive Studio** holds the recorded-number animations
(the gate ladder, the tropical contraction of the 104-spin magnet with its samples, and an
N-queens board player), and **Evidence & Verification** holds the three computed still figures
in a lightbox, the acceptance table and the sheet's reference numbers beside our recount. The
site home page and the weekly navigation gain a Week 11 entry.
