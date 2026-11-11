# Week 10: Approximate tensor contraction for the Ising model

Locating the two-dimensional Ising transition without sampling a single spin
configuration: compress a Gaussian matrix with a truncated SVD, absorb rows of an
open square into a boundary matrix product state, coarse-grain a periodic square
with the tensor renormalization group, and read the critical temperature off the
drift of the specific-heat peaks.

- **Learning sheet:** [`week10-learning-sheet.pdf`](week10-learning-sheet.pdf)
  ([web](https://giggleliu.github.io/AMAT5315-2026Fall/week10-learning-sheet.html)) —
  **version check 2026-11-11:** 15 pages, 558,278 bytes, SHA256
  `d053e28083dce7e1…`; the local file is byte-identical to a cache-bypassing download of
  the official PDF (`python scripts/check_sheet.py` records both hashes in
  `evidence/sheet.json`).
- **Interactive lab:** [Week 10 Tensor RG Lab](https://imgrj.github.io/AMAT5315-2026Fall-Exercise/week10-tensor-lab.html)
  — *Interactive Studio* (a truncated-SVD rank explorer, a boundary-MPS row player and a
  TRG round player, all driven by recorded numbers) and *Evidence & Verification (7)* with
  every computed figure; each figure opens in a lightbox with an English caption and its
  producing command.
- **Specification:** [`SPEC.md`](SPEC.md) records the acceptance criteria before the code.
- **Tool:** [`thermo`](thermo) — Rust 1,400 lines, no numerical dependencies: its own
  Jacobi SVD, its own Krylov truncated SVD, its own NumPy writer.

| Result | Measured value |
| :--- | :--- |
| Gaussian 512×512, seed 2026 | `e_16 = 0.94229`, `e_256 = 0.32613`, `e_512 = 1.3e-13` |
| Measured error vs the discarded spectrum | worst gap **1.3e-13** (requirement 1e-8) |
| 10% error crossing | **χ = 395**, 1.543× the original storage — no saving |
| Open 4×4 at χ = 8 vs 2¹⁶ enumeration | |Δg| = **2.2e-16**, c_V 0.439393 vs 0.439397 |
| Open peaks (L = 16 / 32 / 64) | **2.09310 / 2.17968 / 2.22422** |
| Open `T_c`, all sizes / largest two | **2.26751 / 2.26877** (0.074% / 0.018% from Onsager) |
| Periodic peaks (L = 64 / 256 / 1024) | **2.28160 / 2.27226 / 2.27001** |
| Periodic `T_c`, all sizes / largest two | **2.26920 / 2.26926** (0.001% / 0.003% from Onsager) |
| Peak stability, χ 16→32 / h halved | **0.001% / 0.001%** open, **0.003% / 0.016%** periodic |
| Acceptance checks | **46 / 46** (`evidence/acceptance.json`) |
| pytest | **41 passed** |
| Negative controls | 3 deliberate defects, all detected |

### The sheet's printed reference beside our recount

| Quantity | Sheet reference | This lab |
| :--- | :--- | :--- |
| Gaussian `e_16` / `e_256` | 0.943 / 0.326 | **0.94229 / 0.32613** |
| First rank reaching 10% | χ = 395, 1.54× storage | **χ = 395, 1.543×** |
| Open peaks, L = 16 / 32 / 64 | 2.093 / 2.180 / 2.224 | **2.09310 / 2.17968 / 2.22422** |
| Open `T_c`, all / largest two | ~2.268 / ~2.269 | **2.26751 / 2.26877** |
| Open 4×4 c_V at K = 0.44 | 0.439397 | **0.439397** (exact sum) / 0.439393 (finite difference) |
| Periodic peaks, L = 64 / 256 / 1024 | 2.282 / 2.272 / 2.270 | **2.28160 / 2.27226 / 2.27001** |
| Periodic `T_c`, all / dropping L = 64 | ~2.2692 / ~2.2693 | **2.26920 / 2.26926** |
| Periodic 2×2 c_V at K = 0.44 | 0.404278 | **0.404278** (exact sum) / 0.404282 (finite difference) |

Our Gaussian matrix comes from our own recorded generator, so its spectrum is our own; it
is checked against its own saved singular values instead of against the sheet's seed. Every
Ising number agrees with the sheet to the printed precision.

## What was built

```
week10/
  thermo.design.toml     the sheet's argument/writes contract plus the svd extension
  thermo/
    src/model.rs         W, b and A[r,u,l,d] exactly as the sheet's (14)-(18)
    src/boundary.rs      Algorithms 3-4: row absorption, canonicalisation, truncation, closure
    src/trg.rs           Algorithm 5: normalisation, two diagonal splits, exact regrouping
    src/linalg.rs        exact Jacobi SVD, Krylov truncated SVD, C-order .npy writer
    src/rng.rs           SplitMix64 + Box-Muller, the recorded Part 1 generator
    src/main.rs          the `thermo` CLI and every saved evidence file
  tests/                 41 independent pytest checks plus the NumPy reference models
  scripts/               drivers, figures, web build, browser checks, publication check
  evidence/              acceptance summary, negative controls, sheet and publication records
  web/lab_template.html  the lab page template (built into ../docs/)
```

### Command line

```text
thermo --method boundary --l <side> --k <K> --chi <chi> --out <dir>
thermo --method trg      --l <side> --k <K> --chi <chi> --out <dir>
thermo --method svd      --l <side> --seed <seed> --ranks 16,64,…  --out <dir>
```

`boundary` and `trg` reject `--seed`/`--ranks`; `svd` rejects `--k`/`--chi`. Nothing is
defaulted: `--l`, `--k`, `--chi`, `--out`, `--seed` and `--ranks` are what their method
requires, and an unknown flag or a missing one is an error rather than a silent default.
`trg` rejects a side length that is not a power of two.

Each run keeps full floating-point precision in its evidence:

| File | Contents |
| :--- | :--- |
| `<out>/run.json` | model (K, T, W, b and the sixteen site-tensor entries), geometry and boundary condition, settings, `log Z`, `g`, solver time, and the per-row or per-round diagnostics |
| `<out>/cores.json` | boundary method: the final MPS cores, the normalisation scales, `ℓ` and the bottom closure — enough to recompute `g` without the tool |
| `<out>/final.npy` | TRG: the final tensor `A[r,u,l,d]` |
| `<out>/splits-<j>.npy` | TRG: that round's four half-tensors `S1,S2,S3,S4` (concatenated, layouts in `run.json`) |
| `<out>/matrix.npy`, `singular-values.npy`, `recon-<χ>.npy` | SVD: the generated matrix, the spectrum and each truncated reconstruction |

## Environment

| Component | Version |
| :--- | :--- |
| Rust | 1.98.1 (`cargo build --release`; `Cargo.lock` is committed) |
| Crate dependencies | `serde` 1, `serde_json` 1 — no numerical libraries |
| Python | 3.12.13 with numpy 2.5.4, matplotlib 3.11.2, pypdf 6.21.4, pytest 9.2.0, playwright 1.64.1 |
| CPU | 16 logical cores; runs are independent processes, the drivers use 8 at a time |

`python -m venv .venv` plus `uv pip install pytest numpy matplotlib pypdf playwright`
recreates the interpreter the commands below use (`week10/.venv`).

## The model and the two geometries

`H = -Σ_{(i,j)∈E} s_i s_j`, `s_i = ±1`, zero field, `J = k_B = 1`; the Boltzmann weight of a
bond is `exp(K s_i s_j)`, so `Z(K) = Σ_s exp(-K H(s))`, `g = log Z / L²` and
`c_V = K² [g(K+h) − 2g(K) + g(K−h)] / h²`. At `K = 0` every configuration has equal weight
and `g = log 2` for every geometry — a check every run reproduces to 1e-12.

**Open squares** have no bond across an edge: `2L(L−1)` bonds. Missing bonds are closed with
`b = (1/√cosh K, 0)ᵀ`, for which `W b = (1,1)ᵀ`, so the absent bond contributes weight one.
**Periodic squares** join opposite edges into a torus: `2L²` bonds, counted once per site in
each positive direction. At `L = 2` that leaves *two parallel* bonds between each neighbouring
pair, which is why the 2×2 torus is `2e^{8K} + 2e^{−8K} + 12` rather than the direct-square
value.

## Reproduction

From `week10/` (PowerShell or bash; CPU only). `$PY` is the interpreter above.

```bash
cargo install --path thermo                      # installs `thermo` (or use cargo build --release)
$PY scripts/run_svd.py                           # runs/svd, the stated Part 1 case
$PY scripts/run_stated.py                        # runs/boundary64 and runs/trg1024
$PY scripts/run_checks.py                        # the small squares checked by enumeration
$PY scripts/run_scan.py --method boundary        # 123 scan + 54 refinement runs (~25 s)
$PY scripts/run_scan.py --method trg             # same for TRG (~80 s)
$PY scripts/plot_svd.py                          # artifacts/svd.png
$PY scripts/plot_critical.py --method boundary   # artifacts/boundary-critical.png
$PY scripts/plot_critical.py --method trg        # artifacts/trg-critical.png
$PY scripts/plot_diagnostics.py                  # four supporting figures
$PY scripts/verify_results.py                    # evidence/acceptance.json (46 checks)
$PY scripts/check_sheet.py                       # evidence/sheet.json
$PY scripts/build_web_lab.py                     # docs/week10-tensor-lab.html + assets
$PY scripts/check_web.py                         # browser checks -> evidence/web-validation.json
$PY scripts/verify_publication.py                # Pages build + deployed files
$PY -m pytest tests -q                           # 41 independent checks (~15 s)
```

`python scripts/run_all.py` runs the computational steps in that order.

The individual commands behind each run:

```bash
thermo --method svd --l 512 --seed 2026 --ranks 16,64,128,256,512 --out runs/svd
thermo --method boundary --l 64 --k 0.45 --chi 16 --out runs/boundary64
thermo --method trg --l 1024 --k 0.44 --chi 16 --out runs/trg1024
thermo --method boundary --l 16 --k 0.450000 --chi 16 --out runs/scan/boundary-l16-chi16-k0.450000
thermo --method trg --l 1024 --k 0.440000 --chi 32 --out runs/refine/trg-l1024-chi32-k0.440000
```

The peak study is exactly the sheet's:

* a coarse sweep `K = 0.40 … 0.50`, step `0.0025`, at `χ = 16` for `L = 16, 32, 64` (open)
  and `L = 64, 256, 1024` (periodic);
* nine refinement points spaced `0.00125` around the nearest grid point of each peak, at
  `χ = 16` and `χ = 32`;
* `c_V` from neighbouring temperatures, `h = 0.0025` on the coarse grid and the `χ = 16`
  refinement, `h = 0.00125` on the `χ = 32` refinement, plus the wider-step estimate on
  every other refined sample;
* a three-point parabola in `K` for each peak, `T_peak = 1/K_peak`, and the fit
  `T_peak(L) = T_c + a/L` for all three sizes and for the largest two.

### Peak-refinement table

| Geometry | L | χ = 16, h = 0.0025 | χ = 32, h = 0.00125 | χ = 32, h = 0.0025 | c_V peak |
| :--- | ---: | ---: | ---: | ---: | ---: |
| open | 16 | 2.093085 | 2.093105 | 2.093085 | 1.1154 |
| open | 32 | 2.179657 | 2.179675 | 2.179657 | 1.3875 |
| open | 64 | 2.224207 | 2.224220 | 2.224208 | 1.6830 |
| periodic | 64 | 2.282005 | 2.281602 | 2.281464 | 2.2490 |
| periodic | 256 | 2.272303 | 2.272260 | 2.272021 | 2.9028 |
| periodic | 1024 | 2.270197 | 2.270012 | 2.269618 | 3.0960 |

| Fit (`T_c + a/L`) | open | periodic |
| :--- | ---: | ---: |
| all three sizes (χ = 32) | **2.267506** (0.074% from Onsager) | **2.269205** (0.001%) |
| largest two (χ = 32) | **2.268766** (0.018%) | **2.269263** (0.003%) |
| slope `a`, all sizes | −2.7934 | +0.8079 |

Onsager's `T_c = 2/log(1+√2) = 2.2691853`. Open squares approach it from below, periodic
squares from above: the surface of the open square lowers every finite-size peak, the torus
has no surface. Both extrapolations point at the same bulk transition.

### Small-square ground checks

| Check | Measured | Requirement |
| :--- | ---: | :--- |
| Open 4×4, χ = 8, K = 0, 0.44, 0.7 vs 2¹⁶ configurations | 2.2e-16 | ≤ 1e-8 |
| Open 4×4 c_V (h = 0.0025) vs `K² Var(H)/16` | 0.43939307 vs 0.43939691 | ≤ 1e-3 |
| Periodic 2×2, χ = 16, K = 0.44 vs 16 configurations | 0.0 | ≤ 1e-8 |
| Periodic 2×2 c_V (h = 0.0025) vs `K² Var(H)/4` | 0.40428194 vs 0.40427823 | ≤ 1e-3 |
| Periodic 4×4, χ = 8, K = 0.44 vs 2¹⁶ configurations | 1.06e-4 (truncation) | ≤ 2e-3 |
| `g(K = 0) = log 2` for L = 2, 3, 4, 8, 16 | 0.0 | ≤ 1e-12 |

## Evidence figures

| Figure | Producing command | What it shows |
| :--- | :--- | :--- |
| `artifacts/svd.png` | `python scripts/plot_svd.py` | singular spectrum, and the relative error against rank with the discarded-spectrum prediction, the storage crossover at χ = 256 and the 10% crossing at χ = 395 |
| `artifacts/boundary-critical.png` | `python scripts/plot_critical.py --method boundary` | open-square `c_V` against `T` for L = 16, 32, 64 (χ = 16 curves plus the χ = 32 refined stars), the `1/L` peak extrapolation with both fits, and the refinement table |
| `artifacts/trg-critical.png` | `python scripts/plot_critical.py --method trg` | the same for the periodic squares up to 1024×1024 |
| `artifacts/g-curves.png` | `python scripts/plot_diagnostics.py` | `g(T)` for every size and both geometries, meeting at `log 2` when `K = 0` |
| `artifacts/phase-compare.png` | `python scripts/plot_diagnostics.py` | both peak families on one axis, open from below and periodic from above |
| `artifacts/boundary-rows.png` | `python scripts/plot_diagnostics.py` | the stated case's per-row scale, accumulated `ℓ`, relative truncation loss and bond dimensions |
| `artifacts/trg-steps.png` | `python scripts/plot_diagnostics.py` | the stated case's per-round scale, split spectra and bond dimensions |

`artifacts/` and `runs/` stay out of Git; the seven figures the published lab needs are
committed under `docs/assets/week10/`.

## Verification

* `python scripts/verify_results.py` recomputes every acceptance number from the saved files
  and writes `evidence/acceptance.json`: 46 checks, all passing. The Gaussian errors are
  recomputed from `matrix.npy` and `recon-*.npy`; `g` for the stated boundary case is
  recomputed from `cores.json` alone; the small squares are compared with a full sum over
  spin configurations; and two independent NumPy implementations of Algorithms 4 and 5
  (`scripts/ising_reference.py`) are compared with the Rust tool.
* `python -m pytest tests -q` holds 41 checks derived from `SPEC.md`: reproducible
  standard-normal generation, the truncated-SVD identity, the enumeration ground truth,
  the `c_V` variance identity, TRG on the 2×2 torus, the saved evidence (cores, split
  factors, final tensor) rebuilding the same answers, and the peak arithmetic on analytic
  curves.
* **Negative controls** (`evidence/negative-controls.json`, produced by the sabotaged commit
  with `python scripts/check_negative_controls.py --record`): dropping the accumulated
  boundary scales moves the 4×4 open `g` by **0.871**, closing TRG with all four indices equal
  moves the 2×2 torus `g` by **0.121**, and predicting the SVD error from the retained
  spectrum moves it by **0.627** — against tolerances of 1e-8. All three are detected.

## Two implementation choices worth stating

* **Right-canonicalisation by LQ instead of an exact SVD.** Algorithm 3 uses `Vᵀ` of an exact
  SVD only to obtain *some* orthonormal basis of the right environment; any exact
  factorisation `B = L R` with orthonormal rows does the same job. We use modified
  Gram–Schmidt with reorthogonalisation, which is the same gauge transformation at about a
  tenth of the cost. The truncating sweep keeps the truncated SVD, so every discarded
  singular value — and therefore every `g` — is unchanged. The committed NumPy reference
  implements the sheet's SVD canonicalisation literally, and the two agree to 2.2e-16.
* **A matrix-free Krylov truncated SVD for the large splits.** A TRG split at χ = 32 is a
  1024×1024 matrix of which 32 directions are kept; forming the Gram matrix would cost
  `m n² = 1.1e9` operations per split. Instead a symmetric Lanczos iteration applies
  `G x = Aᵀ(A x)` (two matrix–vector products per step) with full reorthogonalisation, and
  the singular values are recomputed as `‖A v‖` so they never inherit the squared
  conditioning. A Krylov breakdown means the invariant subspace is exhausted, so degenerate
  and rank-deficient tensors — `K = 0`, or the rank-2 site tensor — are handled exactly
  rather than approximately.

## Publication

`python scripts/verify_publication.py` waits for the GitHub Pages build of the published
commit, downloads every deployed file and compares it with `docs/` byte for byte (line
endings normalised for HTML and SVG), writing `evidence/publication.json`. The lab is
`docs/week10-tensor-lab.html`, featured on the site home page and in the weekly navigation of
Weeks 5, 7, 8 and 9.
