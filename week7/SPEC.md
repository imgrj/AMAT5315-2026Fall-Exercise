# Week 7 — Diffusion and Flow Matching (Specification)

Source: `week7-learning-sheet.pdf` (revised version, 19 pages), *Week 7: Diffusion
and flow matching* (Jin-Guo Liu, Xi-Wei Pan). This file restates the required
work as a specification with measurable acceptance criteria and names the
algorithm of the sheet that implements each piece.

## 0. Repository rules

- Work in `week7/` with `uv`, CPU as the baseline.
- Code, small metric summaries and figures are committed; `inputs/` (downloads)
  and `artifacts/` (raw runs) are **not**.
- Figures go to `evidence/`, raw runs with settings to `artifacts/`.
- Commit after each task.

## 1. Shared objects

### 1.1 Target (Eq. 1–2)

Equally weighted mixture with centres `(3,0), (0,3), (-3,0), (0,-3)`,
covariance `0.09 I`: draw `k` uniformly, then `x0 = μ_k + 0.3 (z1, z2)` with
independent standard Gaussians.

### 1.2 Schedule (Eq. 6)

`T = 1000`, `β_t` linear from `1e-4` to `0.02`, `α_t = 1 - β_t`,
`ᾱ_t = Π_{k≤t} α_k`, `ᾱ_0 = 1`. At `t = 1000`, `ᾱ_t ≈ 4e-5`.

### 1.3 Direct noising (Eq. 7)

`x_t = √ᾱ_t x_0 + √(1 - ᾱ_t) ε`, `ε ~ N(0, I)`.

### 1.4 Time features (Eq. 8–9)

`ω_j = 10000^(-j/16)`, `j = 0..15`, and
`r(t) = [sin((t-1)ω_0), ..., sin((t-1)ω_15), cos((t-1)ω_0), ..., cos((t-1)ω_15)]`,
appended to the two coordinates (`h0 ∈ R^34`). The flow network uses `1000 s`
in place of `t - 1`.

### 1.5 Predictor (Eq. 11)

Three hidden layers of 128 SiLU units, linear 2-coordinate output. Weights are
normal with standard deviation `1/√(input width)`, biases zero.

### 1.6 Loss (Eq. 12 / Algorithm 1, line 13)

`ℓ(θ) = (1/2)‖ε_θ(x_t,t) - ε‖²` per example, averaged over the batch and over
`d = 2` coordinates; for two coordinates this equals `(1/d)‖·‖²`.

### 1.7 Exact conditional quantities (mixture only)

With `v_t = 0.09 ᾱ_t + (1 - ᾱ_t)`:

```
E[x_0|x_t,k] = μ_k + (0.09√ᾱ_t/v_t)(x_t - √ᾱ_t μ_k)
r_k(x_t) ∝ exp(-‖x_t - √ᾱ_t μ_k‖²/(2 v_t))
E[ε|x_t] = (x_t - √ᾱ_t E[x_0|x_t]) / √(1 - ᾱ_t)
```

`E[ε|x_t]` is the optimal noise prediction and removes network error from the
DDIM comparison.

### 1.8 Exact marginal velocity (Eq. 15)

With `x_s = (1-s)x_0 + s z`, `u_s = (1-s)²·0.09 + s²`,
`E[x_0|x_s=x,k] = μ_k + ((1-s)·0.09/u_s)(x - (1-s)μ_k)`,
`v_k(x,s) = (x - E[x_0|x_s=x,k])/s`, and
`v(x,s) = Σ_k r_k(x_s) v_k` with `r_k ∝ exp(-‖x - (1-s)μ_k‖²/(2u_s))`.
At `s = 1`, `v(x,1) = x`.

## 2. Part 1 — Diffusion on the Gaussian mixture

### 2.1 Required implementation

1. **First task**: target sampling and forward diffusion in JAX only (Eq. 2, 6,
   7). No network, no reverse sampling.
2. **Then**: train the noise predictor and generate points with DDPM.
   - `week7/diffusion.toml` fixes the architecture and the optimizer:
     `[128,128,128]` SiLU, 32 time features with base `10000` and offset `-1`,
     normal weights `1/√n`, zero biases; Adam `(0.9, 0.999, 1e-8)`, learning
     rate `1e-3` decaying with a cosine to zero, batch 512, 25,000 updates,
     `float32`, seed 2026, checkpoints
     `[0, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 25000]`.
   - **Algorithm 1** is the training loop; **Algorithm 2** is the sampler:
     variance `β_t`, no clipping, and `z ← 0` on the final step.

### 2.2 Acceptance criteria

1. **Forward diffusion merges the modes.** Movie of the same samples on one
   fixed scale; at `t = 1000` each coordinate's mean within 0.1 of 0 and its
   variance within 0.1 of 1, with at least 2000 samples.
2. **Training moves the endpoints onto the clusters.** Movie using the
   checkpoints above, 10 DDPM paths each, **shared starts and step noise**,
   loss history so far, each stage labelled with its update. Initialization
   paths leave the square; the final stage puts all ten endpoints on the four
   clusters. The loss levels off near 0.25; a positive plateau alone does not
   establish failure (Eq. 14).
3. **Generated samples recover populations, centres and widths.** 2000 samples,
   1000 DDPM steps: populations within **0.04** of 0.25, spreads within **0.03**
   of 0.30, centres within **0.05**.

## 3. Part 2 — Deterministic DDIM (Algorithm 3)

Implement DDIM over any selectable decreasing grid, plus the exact conditional
mean noise for comparison. One random draw at level `T`, then
`x̂0 = (x_t - √(1-ᾱ_t) ε̂)/√ᾱ_t`, `x_{t'} = √ᾱ_{t'} x̂0 + √(1-ᾱ_{t'}) ε̂`.

**Acceptance.** Two DDIM runs over 1000 levels from the same noise differ by
**exactly 0**; two DDPM runs from the same start agree on the final cluster only
about a quarter of the time (chance). Both still reproduce the target: DDIM
cluster fractions within **0.015** of 0.25 and widths between **0.29 and 0.32**
(controls with the exact conditional-mean noise and with exact target draws
isolate finite-sample variation).

## 4. Part 3 — Flow matching and fewer steps (Algorithms 4–5)

`week7/flow.toml` fixes the same architecture with `time_scale = 1000`,
25,000 updates, batch 512, seed 2029, cosine decay. Train on
`x_s = (1-s)x_0 + s z`, target `z - x_0`, `s ~ U[0,1]`; sample with Euler steps
from `s = 1` to `s = 0`. Compare DDPM, DDIM and flow at **1, 2, 5, 10, 20, 50,
100 steps** and 1000 levels for DDPM/DDIM, 4000 samples per run, with the energy
distance (Eq. 19) and a null scale from 20 pairs of exact sets.

**Acceptance.** Learned vs exact velocity relative RMS **< 0.1** at
`s = 0.1, 0.5, 0.9`; by 20 steps flow reaches energy distance **< 0.005** and
widths within **0.03** of 0.30; DDPM needs about fifty levels.

## 5. Part 4 — The pretrained predictor on photographs

Download `google/ddpm-cifar10-32` at revision
`267b167dc01f0e4e61923ea244e8b988f84deb80` into `inputs/cifar-model`; load the
weights and configuration as a fixed JAX noise predictor matching diffusers
v0.40.0; apply Algorithm 2 unchanged with 1000 steps, `float32`, no clipping,
model index `t - 1`; generate four images and compare with real CIFAR-10
images. The weights are fixed; the sampler computes the reverse updates.

## 6. Part 5 — Inpainting with RePaint (Algorithm 6)

Load `inputs/images.npz`, hide rows and columns `6:26` in all channels, and
inpaint the four test images with seeds 7201–7204 using RePaint: levels
`1000, 980, ..., 20, 0`, two attempts per jump except the final one, forward
noise between attempts, observations replaced by forward-noised observations
after every merge, Tweedie estimate clipped to `[-1, 1]`, merge with the mask.

**Acceptance.** Every observed pixel changes by **< 1e-5**; outside the hole the
four completions are identical; inside they differ; report each completion's
hidden-pixel MSE and the pixelwise variance across completions (reference
average 0.24).

## 7. Documentation

`week7/README.md` places each evidence figure beside its producing command,
explains the transport comparison, the generated photographs and the inpainting
result, records the environment, the two `.toml` settings and the pinned model
download, keeps downloads out of git, and is pushed to GitHub.
