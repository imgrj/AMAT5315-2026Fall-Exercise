# Week 4: Continuum Fluid Dynamics

Implementation and verification of 2D incompressible fluid dynamics using the vorticity-streamfunction formulation, Fourier pseudospectral differentiation with 2/3-rule dealiasing, and explicit time integrators (Forward Euler, Explicit Midpoint, Classical RK4).

## Architecture & Components

- `src/integrator.rs`: `Integrator` trait and implementations (`ForwardEuler`, `ExplicitMidpoint`, `Rk4`, `Rk4EqualWeights`).
- `src/line.rs`: 1D advection-diffusion rate functions (Fourier spectral multipliers with stationary Nyquist mode $k = -N/2$, and 2nd-order centred finite differences).
- `src/fluid_solver.rs`: 2D pseudospectral Navier–Stokes solver on periodic $[0, 2\pi)^2$, 2/3 dealiasing rule, Poisson solver for streamfunction, velocity recovery, and kinetic energy/enstrophy diagnostics.
- `src/rng.rs`: PRNG for phase generation independent of grid size $n$.
- `src/bin/field.rs`: CLI tool generating velocity fields (`taylor-green` and `random`).
- `src/bin/fluid.rs`: CLI tool integrating velocity fields piped on stdin.
- `src/bin/line_runner.rs`: Driver binary for Part 1 1D simulations.

---

## Installation

From `week4/`:

```bash
cargo install --path . --quiet
```

This installs `field` and `fluid` into your Cargo bin directory.

Python requirements:
```bash
python -m pip install numpy matplotlib
```

---

## Core Pipelines

### 1. Taylor–Green Decay Pipeline
```bash
mkdir -p artifacts/taylor-green
field taylor-green --n 64 | fluid --method rk4 --nu 0.1 --dt 0.01 --t-end 1 --every 0.1 --out artifacts/taylor-green > artifacts/taylor-green.tsv
field taylor-green --n 64 --nu 0.1 --t 1 > artifacts/taylor-green/exact-t1.json
```
Output checks:
- $t = 0.0$: $E = 0.250000$, $Z = 0.500000$
- $t = 1.0$: $E = 0.167580$, $Z = 0.335160$

### 2. Random Flow Decay Pipeline
```bash
mkdir -p artifacts/random
field random --n 128 --seed 2026 --k-min 2 --k-max 6 | fluid --method rk4 --nu 0.004 --dt 0.01 --t-end 10 --every 0.1 --out artifacts/random > artifacts/random.tsv
```
Output checks:
- $E$ decays from $0.50$ to $0.29$
- $Z$ decays from $6.66$ to $0.92$

---

## Reproduction Guide (Ordered Execution)

Run each script from `week4/` in the following sequence:

### Part 1: Integrators on a Line
```bash
# Generate 1D simulation datasets
cargo run --release --bin line_runner -- stability --dt 0.045 --t-end 6 --out artifacts/stability_0.045.json
cargo run --release --bin line_runner -- stability --dt 0.056 --t-end 6 --out artifacts/stability_0.056.json
cargo run --release --bin line_runner -- pulse-profiles --out artifacts/pulse_profiles.json
cargo run --release --bin line_runner -- convergence-line --out artifacts/line_convergence.json

# Plot stability map and pulse profiles / error slopes
python scripts/plot_line_stability.py
python scripts/plot_line_accuracy.py
```

### Part 2: Discretize the Flow and Build the Solver
```bash
# Check Fourier vs Finite Difference derivatives on g(x,y) = sin(3x)cos(2y)
python scripts/check_derivatives.py

# Run Taylor-Green pipeline and generate field comparison
python scripts/plot_taylor_green.py
```

### Part 3: Stability Limits & Perturbation Sensitivity
```bash
# 4-frame random flow decay visualization
python scripts/plot_random.py

# Run stability scans (diffusive and advective limits) and plot blow-up
python scripts/run_stability_scans.py
python scripts/plot_blowup.py

# Run physical perturbation sensitivity study and plot divergence
python scripts/run_sensitivity.py
```

### Part 4: Accuracy Order & Richardson Step Selection
```bash
# Taylor-Green order study on coarse grid (N = 8, nu = 0.5)
python scripts/run_order.py

# Random flow self-convergence (N = 128, nu = 0.004) and Richardson estimation
python scripts/run_convergence.py
```

---

## Evidence Files & Producing Commands

| Evidence File | Producing Command | Description |
| :--- | :--- | :--- |
| `evidence/line-stability.png` | `python scripts/plot_line_stability.py` | Complex stability region and wave evolution at $h = 0.045, 0.056$ |
| `evidence/line-accuracy.png` | `python scripts/plot_line_accuracy.py` | Final pulse profiles at $t = 2\pi$ and temporal convergence slopes (1, 2, 4, 2) |
| `evidence/taylor-green.png` | `python scripts/plot_taylor_green.py` | Vorticity and velocity quiver comparison at $t = 0$ and $t = 1$ |
| `evidence/random.png` | `python scripts/plot_random.py` | 4-panel vorticity decay at $t = 0, 2, 5, 10$ on shared $[-10.97, 10.97]$ scale |
| `evidence/blowup.png` | `python scripts/plot_blowup.py` | Log energy blow-up plots for Taylor–Green and Random flow against theoretical limits |
| `evidence/sensitivity.png` | `python scripts/run_sensitivity.py` | Perturbation sensitivity distance $\|\omega_1 - \omega_2\| / \|\omega_1\|$ over time |
| `evidence/order.png` | `python scripts/run_order.py` | RK4 convergence on fluid: relative velocity error vs $\Delta t$, fitted slope $\approx 4.10$ |
| `evidence/convergence.png` | `python scripts/run_convergence.py` | Self-convergence plot, Richardson prediction, and chosen step $\Delta t = 0.0125$ |
| `evidence/convergence.json` | `python scripts/run_convergence.py` | JSON record of time steps, errors, fitted slope (4.028), and chosen step |
