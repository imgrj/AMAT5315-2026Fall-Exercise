# Week 3: Monte Carlo simulation (2D Ising Model)

Implementation and numerical analysis of the two-dimensional ferromagnetic Ising model on an $L \times L$ torus with periodic boundary conditions ($J = 1, h = 0$):
$$E(s) = -J \sum_{\langle ij \rangle} s_i s_j$$

This project includes:
- A high-performance Monte Carlo CLI tool `ising` with both **single-spin Metropolis updates** and the **Wolff cluster algorithm**.
- Automated verification against Onsager's exact 1944 infinite-lattice solution ($T_c \approx 2.26919$).
- Finite-size scaling analysis canceling the leading $1/L$ peak drift ($T_c \approx 2 T_{\text{peak}}(64) - T_{\text{peak}}(32)$).
- Rigorous statistical error analysis: autocorrelation functions $\rho(t)$, integrated autocorrelation time $\tau_{\text{int}}$ (with $6\tau_{\text{int}}$ cutoff rule), binning analysis, and block bootstrap.
- Quantitative demonstration of overcoming critical slowing down via Wolff cluster flips, achieving a **~1000x reduction in spin-update work per independent sample** at $T_c$.

---

## 1. Directory Structure

```text
week3/
├── Cargo.toml
├── ising.design.toml
├── README.md
├── src/
│   ├── lib.rs
│   ├── main.rs
│   ├── lattice.rs
│   ├── metropolis.rs
│   ├── wolff.rs
│   └── rng.rs
├── tests/
│   └── test_ising.rs
├── scripts/
│   ├── ising_runner.jl
│   ├── plot_boltzmann.py
│   ├── render_frames.py
│   ├── plot_magnetization.py
│   ├── peaks.py
│   ├── errors.py
│   ├── plot_part3.py
│   └── compare.py
├── spins.jsonl
└── evidence/
    ├── boltzmann.png
    ├── magnetization.png
    ├── susceptibility.png
    ├── trace.png
    ├── acf-binning.png
    ├── tau.png
    ├── chi-bootstrap.png
    ├── magnetization-compare.png
    ├── tau-compare.png
    ├── viewer-T1.8.png
    ├── viewer-T2.3.png
    ├── viewer-T3.0.png
    ├── peaks.txt
    └── errors.txt
```

---

## 2. Installation & Reproduction Guide

All reproduction commands can be executed from clean clone inside `week3/`.

### Step 0: Install the Engine
```bash
cargo build --release
cargo install --path . --quiet
```
*(On Windows systems where Smart App Control restricts newly built unsigned binaries, the runner automatically redirects through `scripts/ising_runner.jl` ensuring 100% contract and data compatibility).*

---

### Step 1: Part 1 Verification & Visual Evidence
```bash
# 1. Magnetization at cold and hot temperatures
ising --update metropolis --l 64 --t-from 1.8 --t-to 1.8 --t-step 0.1 --discard 2000 --measure 2000 --seed 2026 --out runs/T1.8
ising --update metropolis --l 64 --t-from 3.0 --t-to 3.0 --t-step 0.1 --discard 2000 --measure 2000 --seed 2026 --out runs/T3.0

# 2. Boltzmann energy ratio test
ising --update metropolis --l 64 --t-from 3.1 --t-to 3.1 --t-step 0.1 --discard 2000 --measure 2000 --seed 2026 --out runs/T3.1
python scripts/plot_boltzmann.py

# 3. Seed repeatability
ising --update metropolis --l 64 --t-from 1.8 --t-to 1.8 --t-step 0.1 --discard 2000 --measure 2000 --seed 2026 --out runs/a
ising --update metropolis --l 64 --t-from 1.8 --t-to 1.8 --t-step 0.1 --discard 2000 --measure 2000 --seed 2026 --out runs/b
diff runs/a/series.jsonl runs/b/series.jsonl && echo IDENTICAL
ising --update metropolis --l 64 --t-from 1.8 --t-to 1.8 --t-step 0.1 --discard 2000 --measure 2000 --seed 2027 --out runs/c

# 4. Heating ramp recording and frame extraction
ising --update metropolis --l 64 --t-from 1.5 --t-to 3.5 --t-step 0.05 --discard 2000 --measure 200 --every 20 --seed 2026 --out runs/ramp
cp runs/ramp/spins.jsonl spins.jsonl
python scripts/render_frames.py
```

---

### Step 2: Part 2 Multi-Run Simulation & Finite-Size Extrapolation
```bash
# Execute the four production Metropolis ramps
ising --update metropolis --l 32 --t-from 1.5 --t-to 3.5 --t-step 0.1 --discard 2000 --measure 5000 --seed 1042 --out artifacts/coarse-l32
ising --update metropolis --l 64 --t-from 1.5 --t-to 3.5 --t-step 0.1 --discard 2000 --measure 5000 --seed 42 --out artifacts/coarse-l64
ising --update metropolis --l 32 --t-from 2.0 --t-to 2.6 --t-step 0.05 --discard 2000 --measure 100000 --seed 1042 --out artifacts/window-l32
ising --update metropolis --l 64 --t-from 2.0 --t-to 2.6 --t-step 0.05 --discard 2000 --measure 100000 --seed 42 --out artifacts/window-l64

# Generate magnetization plot, susceptibility peak fitting, and Tc estimate
python scripts/plot_magnetization.py
python scripts/peaks.py
```

---

### Step 3: Part 3 Error Analysis & Autocorrelation Dynamics
```bash
# Calculate integrated autocorrelation time and block binning table
python scripts/errors.py

# Generate trace, acf-binning, tau spike, and block bootstrap plots
python scripts/plot_part3.py
```

---

### Step 4: Part 4 Wolff Cluster Sampling & Beating Critical Slowing Down
```bash
# Execute the Wolff cluster production window runs
ising --update wolff --l 64 --t-from 2.0 --t-to 2.6 --t-step 0.05 --discard 20000 --measure 100000 --seed 42 --out artifacts/wolff-l64
ising --update wolff --l 32 --t-from 2.0 --t-to 2.6 --t-step 0.05 --discard 20000 --measure 100000 --seed 1042 --out artifacts/wolff-l32

# Verify two-sampler agreement and calculate work-normalized speedup
python scripts/compare.py
```

---

## 3. Evidence Mapping

| Evidence File | Generating Command | Description & Benchmark Result |
| :--- | :--- | :--- |
| `evidence/boltzmann.png` | `python scripts/plot_boltzmann.py` | Energy histograms at $T=3.0$ and $3.1$; verified slope $\Delta(1/T) = 0.01075$. |
| `evidence/viewer-T1.8.png` | `python scripts/render_frames.py` | Low temperature lattice frame: ordered ferromagnetic single domain with sparse defects. |
| `evidence/viewer-T2.3.png` | `python scripts/render_frames.py` | Critical temperature lattice frame: fractal domains of all sizes coexisting at $T \approx T_c$. |
| `evidence/viewer-T3.0.png` | `python scripts/render_frames.py` | High temperature lattice frame: fine-grained thermal noise with vanishing correlation length. |
| `evidence/magnetization.png` | `python scripts/plot_magnetization.py` | Mean $|M|$ vs $T$ for $L=64$ alongside Onsager exact curve showing finite-size rounding. |
| `evidence/susceptibility.png` | `python scripts/peaks.py` | Susceptibility $\chi(T)$ for $L=32$ and $64$ with 5-point quadratic peak fits. |
| `evidence/peaks.txt` | `python scripts/peaks.py` | Extrapolated $T_c = 2.2723$ (deviation $+0.14\% < 2\%$, Onsager $2.26919$). Cold $|M| \ge 0.986$. |
| `evidence/errors.txt` | `python scripts/errors.py` | Autocorrelation times and error inflation ratios across sizes and temperatures. |
| `evidence/trace.png` | `python scripts/plot_part3.py` | Time series of $|m|$ over first 2000 sweeps comparing critical memory ($T=2.3$) vs hot noise ($T=3.0$). |
| `evidence/acf-binning.png` | `python scripts/plot_part3.py` | Autocorrelation $\rho(t)$ decay and binning error plateau analysis at $L=64, T=2.3$. |
| `evidence/tau.png` | `python scripts/plot_part3.py` | $\tau_{\text{int}}$ vs $T$ on logarithmic scale demonstrating critical slowing down ($z \approx 2.2$). |
| `evidence/chi-bootstrap.png` | `python scripts/plot_part3.py` | 500 block bootstrap parabola fits and confidence envelope around susceptibility peak. |
| `evidence/magnetization-compare.png` | `python scripts/compare.py` | Agreement between Metropolis and Wolff ($d = 0.63 \le 3$) and Wolff susceptibility peak fit. |
| `evidence/tau-compare.png` | `python scripts/compare.py` | Work-normalized autocorrelation $\tau_{\text{work}}$ showing **>1000x speedup** at $T_c$. |

---

## 4. Extension: How Long is Long Enough?

At $L = 64$ and $T = 2.3$, the single-spin Metropolis run reports an integrated autocorrelation time of $\tau_{\text{int}} \approx 1039$ sweeps. 

1. **Effective Sample Count ($n_{\text{eff}}$)**:
   For the 100,000 measured sweeps:
   $$n_{\text{eff}} = \frac{n}{2\tau_{\text{int}}} = \frac{100{,}000}{2 \times 1039} \approx 48.1 \text{ independent configurations.}$$

2. **Run Length for Honest Error to Match Naive Error**:
   To achieve an honest standard error $\sigma_{\text{honest}} = \sigma_{\text{naive}} \sqrt{2\tau_{\text{int}}}$ that equals the naive error of 100,000 samples, the simulation must be extended by a factor of $2\tau_{\text{int}} \approx 2078$:
   $$n' = n \times (2\tau_{\text{int}}) \approx 2.08 \times 10^8 \text{ sweeps.}$$
   At a measured single-core sweep rate of approximately 15,000 sweeps/sec, this required run length translates to:
   $$t_{\text{run}} \approx \frac{2.08 \times 10^8}{15{,}000} \approx 13{,}860 \text{ seconds} \approx \mathbf{3.85 \text{ hours}}$$
   for a **single** temperature point.

In contrast, the Wolff cluster algorithm reduces $\tau_{\text{work}}$ to **0.987**, generating the same number of independent configurations in **under 10 seconds** of compute time.
