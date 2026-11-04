# Week 9: Probabilistic inference by tensor contraction

Exact inference on the ASIA network and the 414-variable Promedus_14 benchmark, written in
Rust 1.96.0 with [tenferro-rs](https://github.com/tensor4all/tenferro-rs) for tensor storage
and CPU contraction and [omeco](https://github.com/GiggleLiu/omeco) for order search. The
same build then estimates the same probabilities with loopy belief propagation, and every
number is checked against brute-force enumeration or the published UAI 2014 answers.

- **Learning sheet:** [`week9-learning-sheet.pdf`](week9-learning-sheet.pdf)
  ([web](https://giggleliu.github.io/AMAT5315-2026Fall/week9-learning-sheet.html)) —
  Part 1 is the instructor-led reading tutorial and has no deliverable.
  **Version check (2026-11-04):** 15 pages, Typst 0.14.2, 1,025,405 bytes, SHA256
  `27c21a357be350350d3d2ccc3cefdb3fd2be30aaaea45b91764b7b70520c77397`; the local file is
  byte-identical to a cache-bypassing download of the official PDF, and its extracted text is
  identical to the text this implementation was written against — no revision since the first
  reading.
- **Interactive lab:** [Week 9 Inference Lab](https://imgrj.github.io/AMAT5315-2026Fall-Exercise/week9-inference-lab.html)
  — *Interactive Studio* (posterior bars, a contraction-step player, a BP sweep player) and
  *Evidence & Verification (8)* with every computed figure.
- **Specification:** [`SPEC.md`](SPEC.md) records the acceptance criteria before the code.

| Result | Measured value |
| :--- | :--- |
| ASIA contraction vs 256-row enumeration | max marginal error **2.2e-16**, relative evidence error **1.3e-15** |
| Exhaustive pair-order work (prior / AD / ADX) | **44 / 50 / 52** terms (the sheet's values) |
| Saved plans (AD) | random 204 work / 32 entries, greedy 52 / 8 |
| Loopy BP on ASIA (ADX) | **94 sweeps**, residual **8.23e-11**, marginal error **0.01061** |
| Promedus_14 exact | log10 Z = **-7.80674630458244** (published -7.80675, diff 3.7e-6) |
| Promedus_14 marginals (IDs 59/102/139) | MAR within **4.4e-7** |
| Promedus_14 loopy BP | **1000 sweeps, unconverged** (residual 2.17e-10), query error up to **0.3824** |

### The sheet's printed reference beside our recount

| Quantity | Sheet reference | This lab |
| :--- | :--- | :--- |
| AD random plan: work / largest tensor | 408 / 64 | 204 / 32 (own generator, seed 42) |
| AD greedy plan: work / largest tensor | 54 / 8 | 52 / 8 |
| Exhaustive optima, prior / AD / ADX | 44 / 50 / 52 | **44 / 50 / 52** |
| ADX loopy BP: residual / sweeps / error | 8.23e-11 / 93 / 0.01061 | **8.23e-11 / 94 / 0.01061** |
| Promedus greedy: work / largest | 667,111,481 / 4,194,304 | 383,630,739 / 1,048,576 |
| Promedus TreeSA: work / largest | 10,635,220 / 131,072 | 11,415,062 / **131,072** |
| Promedus loopy BP: residual / query error | 2.12e-10 / 0.38235 | **2.17e-10 / 0.38235** |

Only the two plan costs move: the sheet measures its own random generator and greedy
tie-breaking, and says to judge the saved trees and their recounted costs rather than a
prescribed speedup. Every probability, evidence mass, exhaustive optimum, BP residual/error
and Promedus PR/MAR comparison agrees with the sheet. The same table is on the lab page
under **Evidence & Verification**, generated from `artifacts/`.

## What was built

```
week9/
  infer.design.toml      arguments, the files a run writes, and stdout contract
  Cargo.toml, Cargo.lock pin Rust 1.96.0 with tenferro 0.9, omeco 0.2.6, serde, sha2
  src/uai.rs             MARKOV UAI, .evid, .MAR and .PR readers; SHA256 provenance
  src/network.rs         dense tensors, one-hot observation caps, pairwise einsum with log scaling
  src/plan.rs            plan trees, work/space costs, random, greedy, exhaustive and replay
  src/exact.rs           evidence, node marginals, joints, full enumeration
  src/bp.rs              damped synchronous sum-product BP with residual and error traces
  src/checks.rs          AB, tr(AB), the lb network and the three-spin Ising ring
  src/main.rs            the `infer` CLI, JSON reports and stdout summaries
  tests/                 39 independent pytest checks (see below)
  tests/data/tree.uai    a nonuniform tree where BP must be exact
  scripts/               downloads, runs, figures, web build and browser checks
  evidence/              figures, acceptance summary, web validation
  web/lab_template.html  the lab page template (built into ../docs/)
```

### Command line

```text
infer --model asia.uai --evidence prior|AD|ADX --out artifacts/AD [--query ids]
      [--order greedy|random|exhaustive|treesa] [--seed n]
      [--plan order.json | --analyze order.json]
      [--method exact|bp --damping 0.5 --tolerance 1e-10 --max-sweeps 1000]
```

Observations enter as `(1,0)` cap tensors on the observed variable, exactly as drawn in the
sheet; every other label is summed, and the queried labels stay open. For models without an
`--evidence` label the observations come from the sibling `<model>.evid` file; `--evidence
prior` always means "no observations". Every run writes:

| File | Contents |
| :--- | :--- |
| `<out>/run.json` | parsed model, observations, full-precision marginals, joints, dependence, enumeration, checks, timings |
| `<out>/order.json` | the scalar-query plan: method, seed or TreeSA settings, tree, steps with retained labels, work and sizes |
| `<out>/complexity.json` | the independent recount of that plan (work, largest tensor, per-step costs) |

`--analyze` reads a saved plan and writes only `complexity.json`; `--plan` replays a saved
tree and reproduces the same evidence probability.

## Environment

| Component | Version |
| :--- | :--- |
| Rust | 1.96.0 (2026-05-25), `rustup toolchain install 1.96.0 --profile minimal` |
| tenferro | `tenferro-tensor`, `tenferro-cpu`, `tenferro-einsum` 0.9.0 |
| omeco | 0.2.6 |
| Python | 3.12.13 with numpy 2.5.4, matplotlib 3.11.2, pytest 9.2.0, playwright 1.64.1 |

`week9/.cargo/config.toml` keeps build output in `week9/target`; if the machine sets
`CARGO_TARGET_DIR`, Cargo uses that directory instead (both are handled by the tests, which
ask `cargo metadata` for the binary path).

## Data

`python scripts/download_data.py` fetches and SHA256-checks both pinned data sets; the files
stay out of Git (`.gitignore`) but every run records the model hash in `run.json`.

| File | Bytes | SHA256 |
| :--- | ---: | :--- |
| `asia.uai` | 265 | `0d78314c…f3fff32d` |
| `promedus/Promedus_14.uai` | 48170 | `f213b76a…073d71349` |
| `promedus/Promedus_14.uai.evid` | 47 | `e345a4eb…36a403` |
| `promedus/Promedus_14.uai.PR` | 12 | `d52ca345…5f96dfc` |
| `promedus/Promedus_14.uai.MAR` | 8607 | `06e3145b…f96dfc` |

Sources: `TensorBFS/TensorInference.jl` revision `935ed86f` (`examples/asia-network/model.uai`)
and `mroavi/uai-2014-inference-competition` revision `2200584` (`uai2014/PR/Promedus_14.*`,
`uai2014/MAR/Promedus_14.uai.MAR`).

## Reproduction

From `week9/` (PowerShell or bash; CPU only):

```bash
python scripts/download_data.py                 # pinned data + hashes
cargo +1.96.0 install --locked --path .         # installs `infer` (or use the built binary)
python scripts/run_all.py                       # all 19 inference runs into artifacts/ (~1 min)
python scripts/plot_graphs.py                   # evidence/graphs.svg
python scripts/plot_posteriors.py               # evidence/posteriors.png, evidence/joint.png
python scripts/plot_orders.py                   # evidence/orders.png
python scripts/plot_bp.py                       # evidence/bp-convergence.png, evidence/bp-accuracy.png
python scripts/plot_promedus.py                 # evidence/promedus-orders.png, evidence/promedus-bp.png
python scripts/build_web_lab.py                 # docs/week9-inference-lab.html + assets
python scripts/check_web.py                     # browser checks -> evidence/web-validation.json
python scripts/verify_results.py                # acceptance.json from the saved runs
python scripts/verify_publication.py            # Pages build + deployed files -> evidence/publication.json
python -m pytest tests -q                       # 39 independent checks (~20 s)
```

The individual commands behind each run:

```bash
infer --model asia.uai --evidence prior --out artifacts/prior
infer --model asia.uai --evidence AD    --out artifacts/AD
infer --model asia.uai --evidence ADX   --out artifacts/ADX
infer --model asia.uai --evidence AD --order random --seed 42 --out artifacts/runs/random
infer --model asia.uai --evidence AD --order greedy --out artifacts/runs/greedy
infer --model asia.uai --evidence AD --analyze artifacts/runs/random/order.json --out artifacts/runs/analysis
infer --model asia.uai --evidence AD --plan artifacts/runs/greedy/order.json --out artifacts/runs/replay
infer --model asia.uai --evidence AD --order exhaustive --out artifacts/runs/exhaustive
infer --model asia.uai --evidence ADX --order exhaustive --out artifacts/runs/exhaustive-ADX
infer --model asia.uai --evidence prior --method bp --damping 0.5 --tolerance 1e-10 \
      --max-sweeps 1000 --out artifacts/bp-prior
infer --model promedus/Promedus_14.uai --query 59,102,139 --order greedy \
      --out artifacts/promedus/greedy
infer --model promedus/Promedus_14.uai --query 59,102,139 --order treesa \
      --out artifacts/promedus/treesa
infer --model promedus/Promedus_14.uai --query 59,102,139 --order treesa --method bp \
      --damping 0.5 --tolerance 1e-10 --max-sweeps 1000 --out artifacts/promedus/bp
```

## Part 2 — exact inference on ASIA

The parsed tables are printed and stored with each run. The visit/tuberculosis table reads
`p(A=0)=0.01`, `p(T=0|A=0)=0.05`, `p(T=0|A=1)=0.01`; the smoking/lung and smoking/bronchitis
columns each sum to one. Two independent methods then answer every query:

| Case | Evidence mass | `p(T=0)` | `p(L=0)` | `p(B=0)` | Max marginal error |
| :--- | ---: | ---: | ---: | ---: | ---: |
| prior | 1.0000000000000000 | 0.010400 | 0.055000 | 0.450000 | 2.2e-16 |
| AD | 0.0045013750000000 | 0.087751 | 0.099525 | 0.811402 | 2.2e-16 |
| ADX | 0.0009882675000000 | 0.391712 | 0.444271 | 0.628822 | 1.1e-16 |

The library checks the contraction against numpy-equivalent direct sums (both printed):

- `AB = [[4,4],[10,8]]`, `tr(AB) = 12`, sum of all entries 26, errors 0.
- The `s,ls,bs->lb` network gives `p(l,b) = [[0.0315, 0.0235], [0.4185, 0.5265]]`, error 5.6e-17.
- The three-spin Ising ring contracts to `Z = 12.602562098951932`, enumeration 3.6e-15 away,
  which is `2e^{1.5} + 6e^{-0.5}`.

The `T,L` joint keeps a dependence the product of marginals drops:
`p(T=0,L=0) = 0.022214` in ADX while `p(T=0)p(L=0) = 0.174026`, a difference of **-0.151812**;
for AD the difference is -0.003757.

| Figure | Command | Contents |
| :--- | :--- | :--- |
| `evidence/graphs.svg` | `python scripts/plot_graphs.py` | the AD query as a tensor diagram: factor axes, smoking's three occurrences, the `(1,0)` caps, only `T` open |
| `evidence/posteriors.png` | `python scripts/plot_posteriors.py` | prior/AD/ADX disease probabilities, contraction bars with enumeration ticks |
| `evidence/joint.png` | `python scripts/plot_posteriors.py` | each `T,L` joint, the product of its marginals, and the difference labelled as dependence |

## Part 3 — contraction order

A plan is a binary tree of pair steps. An intermediate keeps a label while another tensor or
the output still needs it; work sums the product of the dimensions in each step's input-label
union, and the largest tensor is the maximum entry count over inputs and intermediates. The
saved step list makes both numbers independently recountable.

| Plan (AD query) | Work (terms) | Largest tensor | Note |
| :--- | ---: | ---: | :--- |
| random, seed 42 | 204 | 32 | a six-label first merge, as the sheet describes |
| greedy | 52 | 8 | the largest input already has 8 entries |
| exhaustive (dynamic program over subsets) | **50** | 8 | optimum for these sizes |
| replay of the greedy plan | 52 | 8 | same mass, costs recomputed |
| analysis of the random plan | 204 | 32 | costs only, no contraction |

For the three cases the exhaustive optima are **44, 50 and 52**, matching the sheet. Replaying
`runs/greedy/order.json` reproduces `p(A=0,D=0) = 0.004501375`; all plans agree with
enumeration to better than 1e-15 relative.

The plan search is deliberately robust: greedy takes the cheapest *feasible* plan among a
deterministic "smallest result among connected pairs" pass and 20 seeded `omeco` `GreedyMethod`
runs (a plan is feasible when no step exceeds 3e10 terms). This is because omeco's priority
queue breaks ties in per-process hash order: repeated runs of its greedy on Promedus_14
measured 0.49e9 … 35.6e9 work, and its deterministic "smallest result" cousin can be driven
into a 4.4e12-term final merge. The chosen plan and its recounted costs are saved, so the
evidence is replayable even though the search itself is a heuristic.

| Figure | Command | Contents |
| :--- | :--- | :--- |
| `evidence/orders.png` | `python scripts/plot_orders.py` | both saved trees with per-step work, retained labels, the largest intermediate highlighted, and the cost summary |

## Part 4 — loopy belief propagation

The same factors feed damped synchronous sum-product updates (`damping 0.5`, tolerance 1e-10,
1000 sweeps). Each recorded state reports the maximum **undamped** message change
`max|F(m) - m|` at the same state as the recorded beliefs, plus the cumulative update work
(`prod dims` per factor-to-variable message, `d_v (deg(v)-1)` per variable-to-factor message).

| Case | Sweeps | Final residual | Converged | Max marginal error | Work per sweep |
| :--- | ---: | ---: | :---: | ---: | ---: |
| prior | 54 | 7.03e-11 | yes | 0.003340 | 100 |
| AD | 93 | 9.94e-11 | yes | 0.011407 | 78 |
| ADX | 94 | 8.23e-11 | yes | **0.010612** | 74 |

The sheet's reference for ADX is 8.23e-11 after 93 sweeps with a marginal error of about
0.01061 — our trace lands on the same numbers. The residual is a statement about the
messages, not about the probabilities: ASIA has loops, so the converged beliefs stay about
1% away from the exact marginals. On the committed nonuniform tree (`tests/data/tree.uai`)
the same code is exact: 70 sweeps, marginal error 8.3e-16 against enumeration.

A deliberate negative control (commit `9d10614`, reverted in `0b7a07f`) includes the
recipient's own message in the factor-to-variable update: the tree test then fails to reach
exactness, which is the implementation error the sheet distinguishes from the approximation
error of correct loopy BP.

| Figure | Command | Contents |
| :--- | :--- | :--- |
| `evidence/bp-convergence.png` | `python scripts/plot_bp.py` | undamped residual against sweeps for the three cases with the stop tolerance |
| `evidence/bp-accuracy.png` | `python scripts/plot_bp.py` | max belief-minus-exact error against sweeps: convergence and accuracy side by side |

## Part 5 — the Promedus_14 benchmark

414 binary variables, nine observations, unnormalized MARKOV potentials, and 141 unary and
265 ternary factors. Every intermediate is rescaled by its largest magnitude and the removed
logarithms accumulate, so `log10 Z_evidence` survives the dynamic range.

| Exact plan | Work (terms) | Largest tensor | log10 Z_evidence |
| :--- | ---: | ---: | ---: |
| greedy (cheapest feasible candidate) | 383,630,739 | 1,048,576 | -7.806746304582445 |
| TreeSA (time-only, 2 trials × 10 iterations × 30 betas) | **11,415,062** | **131,072** | -7.806746304582436 |

The two plans agree to 9e-15; the published PR (-7.80675) is 3.7e-6 away (tolerance 1e-4) and
the state-1 marginals of IDs 59, 102 and 139 are 0.1975882582, 0.1497534323 and 0.6175220184,
all within 4.4e-7 of the published MAR (tolerance 1e-5) and within 1e-9 of the sheet's printed
values. The sheet's own reference plans are greedy 667,111,481 / 4,194,304 and TreeSA
10,635,220 / 131,072; our TreeSA settings are lighter, and the greedy candidate set is
different, so only the *answer* is expected to agree. Plan costs move between runs (greedy
measured 180-384e6 and TreeSA 11-22e6 on this machine); the saved plan in `order.json` is
what the tables and figures above report.

Loopy BP with the ASIA settings does not converge here: after 1000 sweeps the residual is
2.17e-10 (tolerance 1e-10) and the state-1 beliefs of the three queried variables are off by
0.1975, 0.1495 and 0.3824. That is reported as **unconverged and inaccurate**, matching the
sheet's 2.12e-10 and 0.38235. Negative control `artifacts/promedus/noevid` drops the `.evid`
file: the run then reports no observations and its log10 Z misses PR by far more than 1e-4.

| Figure | Command | Contents |
| :--- | :--- | :--- |
| `evidence/promedus-orders.png` | `python scripts/plot_promedus.py` | work and peak intermediate for both plans, with the published PR comparison |
| `evidence/promedus-bp.png` | `python scripts/plot_promedus.py` | residual and selected-query error against 1000 sweeps |

## Tests

`python -m pytest tests -q` runs 39 checks derived from `SPEC.md` (not from the
implementation). They build the tool once, run every case, and compare:

| File | Checks |
| :--- | :--- |
| `test_model_parsing.py` | UAI header/scopes, the T/A and S/L/B columns, the tool's parsed model against an independent Python parser |
| `test_tensor_checks.py` | AB, tr(AB), the lb network and the Ising ring against numpy, all at 1e-12 |
| `test_asia_exact.py` | mass, node marginals and `T,L` joints against 256-row enumeration (1e-10), the sheet's printed posteriors, sums-to-one, dependence, evidence as the denominator |
| `test_ordering.py` | random/greedy/exhaustive plans, recount of `order.json` steps, replay, analysis without inference, the sheet's 44/50/52 optima |
| `test_bp.py` | settings recorded, convergence, residual/work traces, ADX 0.0106 error, reproducibility, exactness on a tree |
| `test_promedus.py` | nine observations, `log10 Z` against PR, marginals against MAR, plan agreement, TreeSA cheaper than greedy, unconverged BP, the dropped-evidence negative control |

`evidence/acceptance.json` records the same measurements in one machine-readable file and
`evidence/web-validation.json` records the browser checks (upper-right switch, mutually
exclusive panels, metrics only under Evidence, state preserved across the switch,
arrow/Home/End navigation, no horizontal overflow at 390/700/1024 px, eight figures loaded,
lightbox opens and closes).

## Figures

| Figure | Producing command |
| :--- | :--- |
| `evidence/graphs.svg` | `python scripts/plot_graphs.py` |
| `evidence/posteriors.png` | `python scripts/plot_posteriors.py` |
| `evidence/joint.png` | `python scripts/plot_posteriors.py` |
| `evidence/orders.png` | `python scripts/plot_orders.py` |
| `evidence/bp-convergence.png` | `python scripts/plot_bp.py` |
| `evidence/bp-accuracy.png` | `python scripts/plot_bp.py` |
| `evidence/promedus-orders.png` | `python scripts/plot_promedus.py` |
| `evidence/promedus-bp.png` | `python scripts/plot_promedus.py` |

## Notes and limits

- The exhaustive order search is a dynamic program over factor subsets and is limited to 16
  factors (ASIA uses 10-11); larger networks use greedy or TreeSA.
- Dense contraction refuses a pair step above 3e10 terms and reports it, instead of
  attempting an allocation that cannot succeed. `INFER_PLAN_DEBUG=1` prints the analysed plan
  and its widest steps.
- The observed variables enter as cap tensors (as drawn in the sheet), so the plan for AD has
  ten leaves and nine steps; the sheet's exhaustive optima 44/50/52 are reproduced exactly
  with this convention (setting a `(1,1)` cap on every single-occurrence index would charge
  two extra entries per merge and shift the optima).
- omeco's TreeSA seeds its trials (`42 + trial`), but its internal label map is a `HashMap`,
  so two TreeSA runs can return different-but-equally-valid plans (12–22e6 work here). The
  saved plan is authoritative, and every cost in this README was recounted from it.
- The browser check needs Microsoft Edge (`channel='msedge'`); screenshots stay in the
  ignored `artifacts/web-qa/`.
