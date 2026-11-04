# Week 9: Probabilistic inference by tensor contraction

Source: [official learning sheet](https://giggleliu.github.io/AMAT5315-2026Fall/week9-learning-sheet.html),
15-page PDF. The local `week9-learning-sheet.pdf` is the committed copy.
In-class Part 1 (reading a tensor network) is instructor-led and has no deliverable.

## Common contract

Work in `week9/`. Rust 1.96.0 runs the inference; `tenferro-tensor`, `tenferro-cpu` and
`tenferro-einsum` supply tensor storage and CPU contraction (no GPU, no autodiff);
`omeco` supplies contraction-order search. Python (numpy/matplotlib) produces the plots in
`evidence/` and the independent tests in `tests/`. Use 64-bit floating point arithmetic
throughout, keep the UAI tables unchanged, and read every table with its declared scope
axis order, last scope variable varying fastest.

Commit source, `Cargo.lock`, `infer.design.toml`, tests, small summaries and figures in
`evidence/`. Exclude downloads (`asia.uai`, `promedus/`), build output (`target/`) and raw
runs (`artifacts/`, `runs/`) from Git; `scripts/download_data.py` fetches the two pinned
data sets and checks their SHA256. Commit the specification before the tests and the tests
before the implementation, record failing tests and a deliberate negative control, and
commit each completed task. Never tune an acceptance tolerance to fit a result. Report the
actual measurements, including discrepancies.

Pinned data (sha256):

| File | Bytes | SHA256 |
| :--- | ---: | :--- |
| `asia.uai` | 265 | `0d78314c96b1cbbfd94682f0e99f73461cb50a935a1da85cbc814c70e3fff32d` |
| `promedus/Promedus_14.uai` | 48170 | `f213b76a7f145a06f39daa9061408981dbd525e4ce2e5156edb83a7073d71349` |
| `promedus/Promedus_14.uai.evid` | 47 | `e345a4ebecf02cdd9fdaf42e7456afa5f6ecafa989b6a3c15ef525abe736a403` |
| `promedus/Promedus_14.uai.PR` | 12 | `d52ca345f79e79f0385f7bd91a3c2e3521eff165b3dcb7b29be1e56ca5f96dfc` |
| `promedus/Promedus_14.uai.MAR` | 8607 | `06e3145bb59fd0a22af235af43bc7c45bee71a764647e152314bdded4f0d256f` |

Sources: `TensorBFS/TensorInference.jl` revision `935ed86f` (`examples/asia-network/model.uai`)
and `mroavi/uai-2014-inference-competition` revision `2200584` (`uai2014/PR/Promedus_14.*`,
`uai2014/MAR/Promedus_14.uai.MAR`).

## 1. Build `infer` (Part 2, hands-on exact inference)

Build the binary from `infer.design.toml` with `cargo +1.96.0 install --locked --path .`
(equivalently `cargo +1.96.0 build --release` for local runs). The design file records the
arguments `model`, `evidence`, `out`, `query`, `ordering`, `bp` and the files each run
writes. IDs 0-7 are A,T,S,L,B,E,X,D; state 0 means yes/present/positive. `--query` lists the
variable IDs kept open as marginals; it is required above 16 variables (the benchmark asks
for IDs 59, 102, 139) and defaults to every node otherwise. Every `run.json` also carries the
parsed model (scope, dimensions and table entries per factor) so the table reading can be
checked against an independent parser.

* Parse MARKOV UAI files: variable count, dimensions, scopes, and table entry counts in
  scope order (last axis varies fastest). Do not reorder tables or normalize the factors.
* Apply observations by selecting the observed state in every factor that uses the
  variable (a fixed one-hot cap), keep any resulting scalar factor, and remember the
  observations in `run.json`.
* Two independent methods for ASIA: enumeration over all 256 joint rows (filtering rows
  that contradict the observations) and tensor contraction. Both keep full precision.
* One network per query: fix observed indices, sum unknown indices, keep the query indices
  open. Node marginals are normalized over the open index; `p(T,L)` is a joint marginal.
  The evidence mass is the closed contraction; dividing joint values by it gives posteriors.
* Library checks, printed by the run and stored in `run.json`:
  * `AB` for A=[[1,2],[3,4]], B=[[2,0],[1,2]] via einsum and via direct sums, plus
    `tr(AB)` and the sum of all entries.
  * The `s,ls,bs->lb` network of the sheet's three-factor example, giving
    `p(l,b)` = [[0.0315, 0.0235], [0.4185, 0.5265]] (rows L, columns B).
  * The three-spin Ising ring, `beta = 0.5`, `J = 1`, no field: contraction versus
    enumeration of the same eight bond weights.
  All checks use 1e-12 absolute tolerance.
* The `T,A` table must be inspected on its own (`p(T=0|A=0)=0.05`,
  `p(T=0|A=1)=0.01`, `p(A)=[0.01,0.99]`), because enumeration and contraction can agree
  while sharing a misread input.

Reference ASIA values (prior, `A=0,D=0` = AD, `A=0,D=0,X=0` = ADX):

| Quantity | Value |
| :--- | :--- |
| evidence mass, prior / AD / ADX | 1 / 0.004501375 / 0.00098822675 |
| `p(T=0)` prior / AD / ADX | 0.0104 / 0.0877509649829 / 0.391711720008 |
| `p(L=0)` prior / AD / ADX | 0.055 / 0.0995251450946 / 0.444270507755 |
| `p(B=0)` prior / AD / ADX | 0.45 / 0.811402071589 / 0.628821775974 |
| ADX `p(T=0,L=0)` joint | 0.02221353 |
| ADX product of the two marginals | 0.17402596 |
| AD `p(T=0,L=0)` joint / product | 0.00497626 / 0.00873343 |

Acceptance: every node and joint marginal sums to one; maximum absolute node-marginal
error between contraction and enumeration below 1e-10; relative evidence-mass error below
1e-10 for all three cases; the ADX joint keeps the dependence (joint minus product about
-0.15181, not 0).

Deliverables: `evidence/graphs.svg` (the AD tuberculosis-query tensor diagram, one axis
per UAI scope axis, smoking's three occurrences joined, observations capped, only `T`
open), `evidence/posteriors.png` (prior and posterior disease probabilities, enumeration
and contraction overlaid) and `evidence/joint.png` (per-case `T,L` joint versus the product
of its marginals, with the difference labelled as dependence). Saved runs in
`artifacts/prior`, `artifacts/AD`, `artifacts/ADX`.

## 2. Contraction order (Part 3)

Extend `infer` with `[args.ordering]`: `--order random|greedy|exhaustive` (default greedy),
`--seed` (required with random, absent otherwise), `--plan` (replay a saved scalar-query plan)
and `--analyze` (analyse a saved plan without contraction). `order`, `plan` and `analyze` are
mutually exclusive; a normal exact run also saves its own scalar plan.

A plan is a binary tree of pair steps over the factor leaves. Each intermediate keeps
every label that another tensor or the requested output still needs and sums the rest.
Costs depend only on index sizes:

* work = sum over steps of the product of the dimensions in the union of the two inputs'
  labels,
* largest tensor = the maximum entry count among all inputs and intermediates,
* per-step retained labels and costs are saved so the totals can be recounted
  independently.

Greedy planning keeps the cheapest *feasible* plan among a deterministic "smallest result
among connected pairs" pass and 20 `omeco` `GreedyMethod` runs, because omeco's priority queue
breaks ties in per-process hash order (repeated runs on Promedus_14 measured 0.49e9-35.6e9
work). A plan is feasible when no pair step exceeds 3e10 terms, the largest step dense
tensors can evaluate in reasonable time; an infeasible plan is reported rather than attempted.
Exhaustive search is a dynamic program over factor subsets limited to 16 factors (ASIA uses
10-11 leaves), and TreeSA reuses omeco's time-only score with the settings recorded in
`order.json`.

`<out>/order.json` records the planning method, random seed (with random) or the omeco
TreeSA settings, the tree, and the reconstructed step list. `<out>/complexity.json` records
work, largest tensor, per-step costs and retained labels for the scalar query.

Acceptance: random and greedy plans give the same evidence probability at different costs
(relative error below 1e-10 against enumeration); replaying `runs/greedy/order.json`
reproduces the value; an independent recount of the saved plan reproduces `work` and
`largest_entries` exactly; analysis without contraction writes no probabilities. The sheet's
exhaustive optimum for ASIA is 44, 50 and 52 work for prior, AD and ADX; report our own
greedy and random numbers rather than forcing those values.

Deliverables: `artifacts/runs/random`, `artifacts/runs/greedy`, `artifacts/runs/analysis`,
`artifacts/runs/replay`, `evidence/orders.png` (both saved contraction trees, work and
largest-tensor comparison, largest intermediate marked).

## 3. Loopy belief propagation (Part 4)

Append `[args.bp]` and `[writes.bp]` as the sheet specifies: `--method exact|bp` (default
exact), `--damping`, `--tolerance`, `--max-sweeps` (all required with bp only). Apply the
observations to the factor tables, initialize messages uniformly, and update
synchronously with Algorithm 1:

* variable to factor: normalize the product of the other incoming messages,
* factor to variable: normalize the sum of the factor times the other incoming messages,
* belief: normalize the product of all incoming factor messages.

Damp each message with `m <- (1-eta) m + eta F(m)`. Every proposal is normalized. Record
per sweep: node beliefs, the maximum absolute undamped message change `max|F(m)-m|`
evaluated at the same state as those beliefs, and the cumulative update work. The returned
state also reports this residual. The stop test uses the undamped residual, not the damped
step size; reaching the sweep limit is not convergence. Work is the number of table
entries looped over: `prod dims` per factor-to-variable message and
`d_v * (deg(v)-1)` per variable-to-factor message, summed over sweeps.

Reference: ADX reaches residual 8.23e-11 after 93 sweeps but its maximum node-marginal
error against exact contraction is about 0.01061. Our own trace is reported as measured.
A synthetic nonuniform tree must match enumeration to below 1e-12 with BP: that separates a
correct implementation from the approximation error of correct loopy BP. A negative
control includes the recipient's own message in a factor-to-variable update, which then
fails even on the tree.

Acceptance: all three traces converge below 1e-10 residual within 1000 sweeps, posterior
beliefs are reproducible across two runs, the tree factor graph is exact, and loopy
marginal errors are reported against the exact values with the known ADX gap (order 1e-2)
not hidden.

Deliverables: `artifacts/bp-prior`, `artifacts/bp-AD`, `artifacts/bp-ADX`,
`evidence/bp-convergence.png` (undamped residual against sweeps per case) and
`evidence/bp-accuracy.png` (maximum node-marginal error against sweeps per case).

## 4. Promedus_14 benchmark (Part 5)

Use the downloaded model with its nine observations. 414 binary variables, 2^414 assignments,
unnormalized MARKOV potentials. Extend exact inference to this model with omeco greedy and
seeded time-only TreeSA (`ScoreFunction::time_optimized()`), saving plans and search
settings. Observations come from the sibling `<model>.evid` file when `--evidence` is not
given; an explicit `--evidence prior` always means "no observations". Keep the tables unchanged and retain a logarithmic scale whenever intermediate
tensors are rescaled, so the evidence partition sum does not underflow. Compute
`log10 Z_evidence` and the state-ordered marginals of IDs 59, 102 and 139. Run BP with the
ASIA settings (damping 0.5, tolerance 1e-10, max sweeps 1000).

Reference values: `log10 Z_evidence = -7.806746304582435`; state-1 marginals
0.1975882582, 0.1497534323, 0.6175220184; published PR `-7.80675` (tolerance 1e-4) and MAR
(tolerance 1e-5). Reference plans: greedy work 667,111,481 with a 4,194,304-entry largest
intermediate, TreeSA 10,635,220 with 131,072. Reference BP at 1000 sweeps is unconverged
(residual 2.12e-10) and inaccurate for the queried marginals (maximum error about 0.38235).

Acceptance: greedy and TreeSA log10 evidence agree within 1e-10 absolute on the same data;
both match PR within 1e-4; the three marginals match MAR within 1e-5 and the two plans
agree; TreeSA uses less work than greedy in our run; runtime and work are recorded; the BP
result is reported as unconverged/inaccurate rather than being tuned. Negative controls:
dropping the log-scale restoration changes the evidence sum; comparing without applying the
observations fails PR.

Deliverables: `artifacts/promedus/greedy`, `artifacts/promedus/treesa`,
`artifacts/promedus/bp`, `evidence/promedus-orders.png` (plan work and largest tensor),
`evidence/promedus-bp.png` (residual and query error against sweeps), plus a comparison
table in the README.

## Delivery

`README.md`: data sources and hashes, environment (Rust 1.96.0, crate versions, Python),
every figure beside its producing command, the ASIA and Promedus tables, BP work accounting,
and exact reproduction commands. Add Week 9 to the root `README.md` and to the GitHub Pages
lab navigation without touching other weeks.

The GitHub Pages lab is `docs/week9-inference-lab.html`, with the same upper-right segmented
switch used by Weeks 5-8: **Interactive Studio** on the left and
**Evidence & Verification (N)** on the right. The studio holds the interactive material (an
animated tensor-contraction / message-passing demo built from recorded numbers); every
computed still figure and the verification table live only under Evidence. Show one panel at
a time, preserve control state across the switch, support arrow/Home/End keyboard navigation,
avoid horizontal overflow on mobile, and record a browser check in
`scripts/check_web.py` writing `evidence/web-validation.json`.

Push the implementation and its evidence to `origin`, verify the remote HEAD and the Pages
URL, and do not edit the instructor's course website.
