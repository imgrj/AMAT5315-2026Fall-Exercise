# Week 5: Automatic Differentiation and Checkpointing

Implementation and verification of 2D acoustic wave simulation, hand-written JAX forward and reverse automatic differentiation (AD), Born forward modeling, adjoint reverse-time migration (RTM), and optimal binomial checkpointing using the Treeverse algorithm.

---

## Architecture & Components

- `seismic/src/experiment.rs`: Experiment JSON metadata parser, survey geometry loader, and smooth quadratic sponge absorbing boundary calculator.
- `seismic/src/npy.rs`: Float32 and Float64 NumPy `.npy` array reader and writer (with Fortran/C-order header parser).
- `seismic/src/solver.rs`: 2D finite-difference acoustic wave propagation engine:
  - `run_forward_shot`: Direct acoustic wave propagation with Ricker wavelet source injection and surface receiver recording.
  - `run_born_shot`: Linearized Born modeling $d_{\mathrm{born}} = J m$ evaluating scattered waves off velocity perturbation $m(x, z)$.
  - `run_adjoint_shot_full`: Adjoint migration $m_{\mathrm{img}} = J^T d$ under full state history storage.
  - `run_adjoint_shot_treeverse`: Checkpointed adjoint migration executing action schedules within a strict memory budget $\delta$.
- `seismic/src/treeverse.rs`: Exact port of [GiggleLiu/TreeverseAlgorithm.jl](https://github.com/GiggleLiu/TreeverseAlgorithm.jl) for optimal binomial checkpoint scheduling.
- `seismic/src/runner.rs`: Unified CLI driver implementing the specification in `seismic.design.toml`.
- `seismic/src/main.rs`: Binary entrypoint.
- `scripts/`: Python verification and evidence generation suite:
  - `run_ad_derivatives.py` & `plot_ad_modes.py`: Hand-written JAX dual-number tangent and cotangent backpropagation vs finite differences.
  - `plot_ad_graphs.py`: Graphviz tape extraction for forward and reverse AD graphs.
  - `run_ad_scaling.py`: High-dimensional cluster parameter scaling ($P = 1 \dots 3072$).
  - `plot_inputs.py`: Visual survey layout, velocity profiles, and sponge damping.
  - `plot_forward.py`: Synthetic receiver gathers and isolated reflector echo.
  - `plot_adjoint.py`: Adjoint migration image, dot-product transpose identity check, and depth profile.
  - `verify_checkpoint.py`: Checkpoint audit suite across budgets $\delta \in \{1, 3, 5, 10\}$.
  - `plot_marmousi.py`: Marmousi 4-panel grand synthesis benchmark.

---

## Installation & Requirements

From `week5/seismic/`:
```bash
cargo build --release
```

Python virtual environment:
```bash
uv pip install numpy matplotlib scipy jax
```

---

## Core Pipelines

### 1. Forward Wave Simulation
```bash
seismic --experiment inputs/reflector.json --mode forward --every 3 --out artifacts/forward
```
Output checks:
- Shot 0 L2 norm: `6.726588`
- Shot 1 L2 norm: `6.564757`
- Shot 2 L2 norm: `6.726588`
- Total traces L2 norm: `11.574770`

### 2. Born Forward Modeling
```bash
seismic --experiment inputs/reflector.json --mode born --out artifacts/born
```
Output checks:
- Shot 0 Born L2: `0.068420`
- Shot 1 Born L2: `0.066770`
- Shot 2 Born L2: `0.068420`
- Total Born data L2 norm: `0.117962`

### 3. Full-History Adjoint Migration & Transpose Check
```bash
seismic --experiment inputs/reflector.json --mode adjoint --data artifacts/born/born_data.npy --storage full --out artifacts/adjoint
```
Output checks:
- Transpose dot-product identity:
  $$\|d_{\mathrm{born}}\|^2 = \langle m, J^T d_{\mathrm{born}} \rangle = 0.01391515$$
  Relative error: $3.98 \times 10^{-16}$ (machine precision).
- Peak image profile depth: $z = 2.1\text{ km}$ (exact reflector position, $\Delta z = 0.0\text{ km}$).

### 4. Treeverse Binomial Checkpointing
```bash
for b in 1 3 5 10; do
  seismic --experiment inputs/reflector.json --mode adjoint --data artifacts/born/born_data.npy --storage treeverse --checkpoints $b --out artifacts/checkpoint-$b
done
```
Verification checks:
- Budget $\delta = 1$: Peak states = 2, Forward steps/shot = 28,680, Relative L2 error = $0.00$
- Budget $\delta = 3$: Peak states = 4, Forward steps/shot = 1,695, Relative L2 error = $0.00$
- Budget $\delta = 5$: Peak states = 6, Forward steps/shot = 990, Relative L2 error = $0.00$
- Budget $\delta = 10$: Peak states = 11, Forward steps/shot = 642, Relative L2 error = $0.00$

### 5. Marmousi Migration Benchmark
```bash
seismic --experiment inputs/marmousi.json --mode born --out artifacts/marmousi-born
seismic --experiment inputs/marmousi.json --mode adjoint --data artifacts/marmousi-born/born_data.npy --storage treeverse --checkpoints 5 --out artifacts/marmousi-image
```
Output checks:
- Model size: $805 \times 269$ grid points, 1,200 timesteps, 9 shots, 91 receivers.
- Memory: 6 saved states (20.8 MB) replace 4.16 GB per shot of full history ($200\times$ reduction).
- Migrated raw image L2 norm: $1.2854061 \times 10^{-4}$.

---

## Reproduction Guide (Ordered Execution)

Run each script from `week5/` in sequence:

### Part 1: Hand-Written JAX AD & Scaling
```bash
uv run python scripts/run_ad_derivatives.py
uv run python scripts/plot_ad_modes.py
uv run python scripts/plot_ad_graphs.py
uv run python scripts/run_ad_scaling.py
```

### Part 2: Forward Acoustic Modeling & Echoes
```bash
python scripts/plot_inputs.py
# (seismic forward run)
python scripts/plot_forward.py
```

### Part 3: Born Scattering & Transpose Check
```bash
# (seismic born and adjoint runs)
python scripts/plot_adjoint.py
```

### Part 4: Binomial Checkpointing & Marmousi
```bash
uv run python scripts/run_checkpoints.py
uv run python scripts/verify_checkpoint.py
uv run python scripts/run_marmousi_migration.py
uv run python scripts/plot_marmousi.py
```

---

## Evidence Files & Producing Commands

| Evidence File | Producing Command | Description |
| :--- | :--- | :--- |
| `artifacts/inputs.png` | `python scripts/plot_inputs.py` | Physical survey layout, background wave speed, reflector perturbation, and sponge damping |
| `artifacts/ad/modes.png` | `python scripts/plot_ad_modes.py` | Forward vs reverse AD precision comparison against finite differences ($10^{-14}$ vs $10^{-9}$) |
| `artifacts/ad/graph.png` | `python scripts/plot_ad_graphs.py` | JAX primal forward computational tape graph |
| `artifacts/ad/grad-graph.png` | `python scripts/plot_ad_graphs.py` | JAX reverse-mode cotangent backpropagation graph |
| `artifacts/ad/scaling.png` | `python scripts/run_ad_scaling.py` | Forward vs reverse AD runtime scaling across $P \in [1, 3072]$ ($>1000\times$ speedup) |
| `artifacts/forward/gathers.png` | `python scripts/plot_forward.py` | 3-shot receiver gathers with hyperbolic reflection arrivals ($\|d\|_{L_2} = 11.574770$) |
| `artifacts/forward/echo.png` | `python scripts/plot_forward.py` | Subtracted reflection echo slice $\Delta u = u(c_0 + m) - u(c_0)$ at step 120 |
| `artifacts/adjoint/image.png` | `python scripts/plot_adjoint.py` | Adjoint migrated image, transpose dot-product check ($3.98 \times 10^{-16}$), and depth peak at $z = 2.1\text{ km}$ |
| `artifacts/checkpoint-actions.png` | `python scripts/verify_checkpoint.py` | Treeverse sawtooth action schedule (budget $\delta = 5$, shot 0) |
| `artifacts/checkpoint-work.png` | `python scripts/verify_checkpoint.py` | Storage–recomputation Pareto tradeoff curve across budgets $\delta \in \{1, 3, 5, 10\}$ |
| `artifacts/marmousi.png` | `python scripts/plot_marmousi.py` | 4-panel Marmousi benchmark: background velocity $c_0$, perturbation $m$, Born gather, and raw migrated image |

---

## Interactive Web Lab

The interactive laboratory for this week is published at:
[https://imgrj.github.io/AMAT5315-2026Fall-Exercise/week5-seismic-lab.html](https://imgrj.github.io/AMAT5315-2026Fall-Exercise/week5-seismic-lab.html)

It includes an interactive Treeverse binomial checkpointing schedule animator, real-time memory slot racks, acoustic wavefield propagation simulator, and an 11-item evidence lightbox.
