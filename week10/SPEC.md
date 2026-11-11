# Week 10: Approximate tensor contraction for the Ising model

Source: [official learning sheet](https://giggleliu.github.io/AMAT5315-2026Fall/week10-learning-sheet.html),
15-page PDF. The local `week10-learning-sheet.pdf` is the committed copy.

## Common contract

Work in `week10/`. Rust (the toolchain in `setup.txt`) runs the numerics; Python
(numpy/matplotlib) produces the figures and the independent tests in `tests/`. Use 64-bit
floating point throughout, keep every saved value at full precision, and state the boundary
condition of every run: an **open** square has no bond across an edge, a **periodic** square
joins opposite edges into a torus.

The model is the zero-field square-lattice Ising model,
`H = -sum_{(i,j) in E} s_i s_j` with `s_i = +-1`, `J = k_B = 1`, and Boltzmann weights
`exp(K s_i s_j)`, `K = 1/T >= 0`. `Z(K) = sum_s exp(-K H(s))`. We report
`g = log Z / N` with `N = L^2` so that `g -> log 2` as `K -> 0`, and

* `c_V(K; h) = K^2 [g(K+h) - 2 g(K) + g(K-h)] / h^2`,
* a finite lattice has a rounded peak; fit a three-point parabola in `K` through the largest
  `c_V` and its two neighbours, take its vertex `K_peak`, and report `T_peak = 1/K_peak`,
* extrapolate `T_peak(L) = T_c + a/L` and compare the intercept with Onsager's
  `T_c = 2/log(1+sqrt(2)) = 2.2691853...`.

The sheet's printed reference values (reference plots come from an independent NumPy
baseline, printed solver expectations quote the CPU Rust answer key):

| Quantity | Sheet reference |
| :--- | :--- |
| Gaussian matrix, seed 2026 | `e_16 ~ 0.943`, `e_256 ~ 0.326`, first 10% at `chi = 395` (1.54x storage) |
| Open-square peaks (L = 16, 32, 64) | 2.093, 2.180, 2.224 |
| Open-square `T_c`, all sizes / largest two | ~2.268 / ~2.269 |
| Open 4x4 at `K = 0.44` | `c_V ~ 0.439397` |
| Periodic-square peaks (L = 64, 256, 1024) | 2.282, 2.272, 2.270 |
| Periodic `T_c`, all sizes / dropping L = 64 | ~2.2692 / ~2.2693 |
| Periodic 2x2 at `K = 0.44` | `c_V ~ 0.404278` |

Our Gaussian matrix uses our own documented generator, so its spectrum is our own; each
random-matrix result is checked against its own saved spectrum, and we report our numbers
beside the sheet's instead of tuning to them.

Commit source, `Cargo.lock`, `thermo.design.toml`, tests, comparison scripts and `README.md`.
Keep generated data, tensors, scales, figures and archives out of Git (`artifacts/`,
`runs/`, `target/`); the published lab needs its figures under `docs/assets/week10/`, and
`evidence/` keeps only small JSON summaries. Commit the specification before the tests and
the tests before the implementation, record a failing test run and a deliberate negative
control, and commit each completed task. Never tune an acceptance tolerance to fit a result.
Report the actual measurements, including discrepancies.

## 1. Compress a matrix (Part 1, hands-on SVD)

`thermo --method svd --l 512 --seed 2026 --ranks 16,64,128,256,512 --out runs/svd`

* Generate a 512x512 matrix with independent standard-normal entries from a documented
  generator: SplitMix64 (Steele, Lea & Flood 2014) seeded with 2026, uniforms
  `u = ((z >> 11) + 0.5) * 2^-53` in the open interval (0,1), and Box-Muller pairs
  `r = sqrt(-2 ln u1)`, `(r cos 2 pi u2, r sin 2 pi u2)` filled row by row. Record the
  generator, its seed and the realised mean and standard deviation in `run.json`.
* Compute the thin SVD `M = U diag(s) V^T` at full precision and save the matrix, the
  singular values, and the truncated reconstructions `M_chi = U[:, :chi] diag(s[:chi]) V[:, :chi]^T`
  at the requested ranks (`.npy` files plus a manifest in `run.json`).
* Report the relative Frobenius error `e_chi = ||M - M_chi||_F / ||M||_F` for each rank and
  the discarded-spectrum prediction `sqrt(sum_{alpha>chi} s_alpha^2 / sum_alpha s_alpha^2)`.

Acceptance:

* an independent NumPy recomputation from the saved matrices reproduces every `e_chi` to
  `1e-8`, and every `e_chi` agrees with the prediction from the saved singular values to
  `1e-8`;
* `e_512 < 1e-8` (full rank keeps every direction);
* the two-factor storage count `(m+n) chi` equals the original `m n` at `chi = 256`, and the
  saved entries are counted, not the estimate;
* the 10% crossing is read from the saved spectrum; report the smallest `chi` with
  `e_chi <= 0.1` and its storage ratio, and answer from those two numbers whether this
  matrix admits a storage-saving approximation at 10% error.

Deliverable: `artifacts/svd.png` with (a) the singular spectrum and (b) the relative error
against retained rank overlaid with the discarded-spectrum prediction, marked at the storage
crossover and at the 10% error threshold.

## 2. Boundary MPS on open squares (Part 2)

Build `thermo` from `thermo.design.toml` with `cargo install --path .`, then run the stated
case `thermo --method boundary --l 64 --k 0.45 --chi 16 --out runs/boundary64`.

Tensors, exactly as the sheet:

* `W = [[sqrt(cosh K), sqrt(sinh K)], [sqrt(cosh K), -sqrt(sinh K)]]`, rows for `s = +1, -1`,
  so that `sum_a W[s,a] W[t,a] = exp(K s t)`;
* `A[r,u,l,d] = sum_{s=+-1} W[s,r] W[s,u] W[s,l] W[s,d]`;
* `b = [1/sqrt(cosh K), 0]^T` closes every missing bond, since `W b = (1,1)^T`;
* row tensors `T[i][l,u,d,r] = A[r,u,l,d]`, `T[1][1,u,d,r] = sum_l b_l A[r,u,l,d]`,
  `T[L][l,u,d,1] = sum_r A[r,u,l,d] b_r`.

Contract with Algorithm 4 of the sheet: the boundary MPS starts as `reshape(b,(1,2,1))` on
the top edge, each row absorbs as `C[i][(alpha l), d, (beta r)] = sum_u B[i][alpha,u,beta] T[i][l,u,d,r]`,
`Compress` canonicalises right-to-left with exact SVDs and truncates left-to-right to `chi`
(Algorithm 3), the boundary is scaled by `c = ||B[L]||_F` through the last core only with
`ell += log c`, and the bottom edge closes with `b`. Return `g = (ell + log v_1) / L^2`.

`runs/boundary64/run.json` records the model and geometry, the settings, `log Z`, `g`, the
solver time, every row's scale and accumulated `ell`, the bond dimensions before and after
compression, and the per-bond discarded weight; `runs/boundary64/cores.npy` holds the final
cores so the last contraction can be recomputed independently.

Peak study with `scripts/run_scan.py`:

* sweep `L = 16, 32, 64` at `chi = 16` over `K = 0.40 .. 0.50` in steps of `0.0025`;
* from the saved `g` values compute `c_V` and locate each peak on the grid;
* around the nearest grid point to each peak, refine nine points spaced `0.00125` at
  `chi = 16` (`h = 0.0025`) and at `chi = 32` (`h = 0.00125`);
* fit each refined peak with a three-point parabola in `K`, convert its vertex to
  `T_peak = 1/K_peak`, and fit `T_peak(L) = T_c + a/L` at `chi = 32` for all three sizes and
  for `L = 32, 64`;
* at `chi = 32`, repeat the wider-step estimate on every other refined sample
  (`h = 0.0025`, spacing `0.0025`) and compare.

Acceptance:

* `|T_c - 2/log(1+sqrt(2))| / T_c <= 1%` for both fits;
* peak positions agree within 1% between `chi = 16` and `chi = 32`, and between the wide and
  the halved finite-difference step;
* open 4x4 square at `chi = 8`: `|g - g_enumeration| <= 1e-8` at `K = 0, 0.44, 0.7`, where
  enumeration sums all `2^16` spin configurations of the same open geometry;
* at `K = 0.44` the finite-difference `c_V` with `h = 0.0025` matches `K^2 Var(H)/16` from the
  same enumeration within `1e-3`;
* every run states its boundary condition, and `g(K=0) = log 2` to `1e-12` for every
  geometry, since all configurations then have equal weight.

Deliverable: `artifacts/boundary-critical.png` with (a) `c_V` against `T` for the three
sizes and (b) the peak extrapolation against `1/L`, plus the `g(K)` tables, timings and the
full runs supporting the `L = 64` peak fit in `evidence/`.

## 3. TRG on periodic squares (Part 3)

Append `[args.trg]` and `[writes.trg]` to `thermo.design.toml` and implement Algorithm 5 of
the sheet: `n = 2 log2 L` steps, `c_j = max|A|` with `A /= c_j` and `ell += N log c_j`,
`Split(A,(l,u),(r,d),chi)` and `Split(A,(d,l),(u,r),chi)` on alternating sites, the exact
regrouping `A'[r,u,l,d] = sum_{w,a,b,gamma} S1[w,a,r] S2[a,b,u] S3[l,b,gamma] S4[d,gamma,w]`,
and the closing `z = sum_{i,j} A[i,j,i,j]` (two independent sums). Use the sheet's ring-index
table and axis order literally. `runs/trg1024/run.json` saves each step's scale, split
singular values, bond dimensions and `ell`, and `final.npy` holds the final tensor.

Run the stated case `thermo --method trg --l 1024 --k 0.44 --chi 16 --out runs/trg1024`, then
repeat the Part 2 peak study with `L = 64, 256, 1024` and the same `K` grid, refinement,
fits and step/rank comparisons.

Acceptance:

* periodic `T_c` within 1% of Onsager for both fits, and peak stability within 1% under
  `chi = 16 -> 32` and step halving;
* `g` against periodic enumeration at `K = 0.44`: `L = 2`, `chi = 16` within `1e-8`; `L = 4`,
  `chi = 8` within `2e-3` (a documented truncation, not a solver error);
* periodic `L = 2` with `h = 0.0025`: `c_V` matches `K^2 Var(H)/4` within `1e-3`;
* bonds are counted once per site in each positive direction, including the distinct
  parallel bonds at `L = 2` (2N bonds for every periodic square; the direct case is cross
  checked against a 4x4 periodic enumeration).

Deliverable: `artifacts/trg-critical.png` with (a) `c_V` against `T` and (b) the periodic
peak extrapolation, plus `g(K)`, timings and the runs supporting the `L = 1024` peak fit.

## Delivery

`README.md`: environment, the generator and every setting, boundary conditions, data
sources, every figure beside its producing command, the peak-refinement and small-check
tables, timings, and exact reproduction commands.

The GitHub Pages lab is `docs/week10-tensor-lab.html`, with the same upper-right segmented
switch used by Weeks 5-9: **Interactive Studio** on the left and **Evidence & Verification
(N)** on the right. The studio holds the interactive material (an animated MPS row
absorption and a TRG coarse-graining player built from recorded numbers); every computed
still figure and the verification table live only under Evidence. Show one panel at a time,
preserve control state across the switch, support arrow/Home/End keyboard navigation, avoid
horizontal overflow on mobile, and record a browser check in `scripts/check_web.py` writing
`evidence/web-validation.json`.

Push the implementation and its evidence to `origin`, verify the remote HEAD and the Pages
URL, and do not edit the instructor's course website.

## Negative controls

1. Boundary MPS: drop the accumulated normalisation scales (`ell` never advanced) while still
   dividing the last core -> the 4x4 open-square check against enumeration must fail.
2. TRG: close the last tensor with all four indices equal (`sum_i A[i,i,i,i]`) instead of
   `sum_{i,j} A[i,j,i,j]` -> the `L = 2` exact check must fail.
3. SVD: predict the error from the retained instead of the discarded spectrum -> the
   `1e-8` agreement must fail.

Each is recorded as a failing check in `evidence/negative-controls.json` and reverted.
