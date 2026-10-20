# Week 7 — Diffusion and Flow Matching (Specification)

Source: `week7-learning-sheet.pdf`, *Week 7: Diffusion and flow matching*
(Jin-Guo Liu, Xi-Wei Pan). This file restates the required work as a
specification with measurable acceptance criteria. Work happens in `week7/`,
CPU first, JAX for every model.

## 0. Repository rules

- Code, figures and small metric summaries are committed.
- Downloaded assets (`inputs/`) and raw runs (`artifacts/`) are **not** committed.
- Figures go to `evidence/`, raw runs (weights, arrays, trajectories) to `artifacts/`.
- Every number quoted below is a pass/fail condition taken from the sheet.

## 1. Shared objects

### 1.1 Target: equally weighted 2-D Gaussian mixture

`p_data(x) = (1/4) Σ_k N(x; μ_k, Σ)` with `Σ = 0.09 I` and

```
μ_1 = ( 3,  0),  μ_2 = (0,  3),  μ_3 = (-3,  0),  μ_4 = (0, -3)
```

Exact samples choose a component uniformly, then `x = μ_k + 0.3 z`,
`z ~ N(0, I)`. Exact draws are the reference for every learned quantity.

### 1.2 Variance schedule

`T = 1000`, per-step variance `β_t` **linear** from `β_1 = 1e-4` to `β_T = 0.02`,
`α_t = 1 - β_t`, `ᾱ_t = Π_{s≤t} α_s`, `ᾱ_0 = 1`.
The sheet states `ᾱ_1000 ≈ 4e-5`; the implementation must land within 25% of it.

### 1.3 Direct noising (closed form)

`x_t = sqrt(ᾱ_t) x_0 + sqrt(1 - ᾱ_t) ε`, `ε ~ N(0, I)` (Eq. 5).

### 1.4 Exact conditional means (mixture only)

For component `k`, `v_t = 0.09 ᾱ_t + (1 - ᾱ_t)`,

```
E[x_0 | x_t, k] = μ_k + (0.09 sqrt(ᾱ_t) / v_t) (x_t - sqrt(ᾱ_t) μ_k)
r_k(x_t) ∝ exp(-||x_t - sqrt(ᾱ_t) μ_k||² / (2 v_t))          (normalised)
E[ε | x_t] = Σ_k r_k (x_t - sqrt(ᾱ_t) E[x_0|x_t,k]) / sqrt(1 - ᾱ_t)
E[x_0 | x_t] = Σ_k r_k E[x_0|x_t,k]
```

`E[ε|x_t]` is the optimal noise-prediction target; `E[x_0|x_t]` is the exact
denoiser and the reference for the optional denoiser check.

### 1.5 Exact marginal velocity (flow matching)

With `x_s = (1-s) x_0 + s z`, `u_s = (1-s)² · 0.09 + s²` and
`r_k(x_s) ∝ exp(-||x_s - (1-s) μ_k||² / (2 u_s))`,

```
v_k(x, s) = (x - E[x_0 | x_s = x, k]) / s
E[x_0 | x_s = x, k] = μ_k + (1-s)·0.09/u_s · (x - (1-s) μ_k)
v(x, s) = Σ_k r_k v_k(x, s),   s > 0
```

At `s = 1` this must approach `v(x, 1) = x` (zero mixture mean).

## 2. Part 1 — DDPM on the Gaussian mixture

### 2.1 Required implementation

- Forward diffusion, Eq. (4), for the mixture above.
- **DDPM sampler** with variance `β_t`, **no clipping**, **no added noise in the
  final step** (`σ_1 = 0`):
  `μ_θ(x_t,t) = (x_t - β_t/sqrt(1-ᾱ_t) ε_θ(x_t,t)) / sqrt(α_t)`, Eq. (8–9).
- Small **time-conditioned** fully connected noise predictor with three hidden
  layers, output in R²; `t` enters as numerical time features.
- Train by the mean squared noise-prediction loss, Eq. (6), with noise levels
  drawn uniformly, batch of 512 freshly drawn mixture samples per update,
  **25000 updates**, learning rate **decayed to zero** by the end.
- Record training settings and the loss history; retain the model at **10
  training stages including initialization and the final model**.

### 2.2 Acceptance criteria (VERIFY)

1. **Forward diffusion merges the modes.** Movie of the same 2000 points at
   increasing levels on one fixed scale; at `t = 1000` each coordinate's mean
   lies within 0.1 of 0 and each coordinate's variance within 0.1 of 1.
2. **Training moves generated endpoints onto the clusters.** Movie over the 10
   saved stages, 10 DDPM paths per stage, **same initial noise and same
   reverse-step noise across stages**, plus the loss history so far. At
   initialization paths leave the plot (some reaching `|x| ≈ 1500`); at the
   final stage all ten endpoints lie on the four clusters. Loss levels off near
   0.25 rather than 0.
3. **Generated samples recover populations, centres and widths.** 2000 samples,
   1000 DDPM steps, assign each point to its nearest target centre:
   - every population within **0.04** of 0.25,
   - every per-coordinate spread within **0.03** of 0.30,
   - every component centre within **0.05** of its target mean.

### 2.3 Optional denoiser check (cheap, included)

With the trained predictor, compare the Tweedie denoiser
`x̂_0 = (x_t - sqrt(1-ᾱ_t) ε_θ) / sqrt(ᾱ_t)` against `E[x_0|x_t]` at
`t = 100, 300, 600, 900`, reporting RMS error divided by the exact posterior
standard deviation (`< 0.1` at t ≤ 600 is the reference behaviour; t = 900 is
outside the pass condition and grows to ≈0.24).

## 3. Part 2 — Fewer-step sampling: DDIM and flow matching

### 3.1 Required implementation

- Deterministic **DDIM** over any selectable subset of levels, using the same
  trained noise predictor (Eq. 10–11), plus an option to substitute the exact
  conditional-mean noise for comparison.
- **DDPM with skipped levels** for the same coarse grid (Eq. 12), i.e.
  `a = ᾱ_t/ᾱ_{t'}`, mean `(x_t - (1-a)/sqrt(1-ᾱ_t) ε)/sqrt(a)`, added noise
  variance `1-a` when `t' > 0` and none at `t' = 0`.
- **Velocity network** trained from scratch on the same mixture by flow
  matching: path `x_s = (1-s)x_0 + s z`, target `z - x_0`, `s ~ U[0,1]`,
  same architecture as the noise predictor, learning rate decaying to zero.
  Sampling by **Euler steps from s = 1 to s = 0**.
- Exact marginal velocity of §1.5 for comparison.
- Compare DDPM, DDIM and flow matching at **1, 2, 5, 10, 20, 50, 100 steps**
  (and 1000 levels for DDPM and DDIM), evenly spaced levels, **4000 samples per
  run**; record cluster fractions, centres, widths and the **energy distance**
  to 4000 exact samples; estimate the null scale from **20 pairs** of
  independent exact sample sets.

### 3.2 Acceptance criteria (VERIFY)

1. **DDIM is deterministic, DDPM is not.** Two DDIM runs over 1000 levels from
   the same starting noise differ by **exactly 0**; two DDPM runs from the same
   start with different step noise agree in their final cluster only about
   chance (≈1/4). Both still reproduce the target: DDIM cluster fractions
   within **0.015** of 0.25, and both samplers' per-coordinate widths within
   **0.03** of 0.30. The 0.03 allowance is the sheet's own Part 1 allowance for
   the same quantity (three sampling fluctuations of exact data at 2000
   samples); the reference's observed band `[0.29, 0.32]` is only about ±2 such
   fluctuations. The evidence therefore also reports the widths of the same
   starting noise propagated with the **exact** conditional-mean noise and of
   an exact 2000-point target draw, which isolates finite-sample variation from
   predictor error.
2. **Learned velocity matches the exact marginal velocity.** Fields at
   `s = 0.1, 0.5, 0.9`; relative RMS error (RMS of the difference divided by
   RMS of the exact velocity) measured over 4000 exact path samples at each
   `s`, i.e. over the region the field must transport, **< 0.1 at every s**.
   The figure draws the arrows on a coarse grid; grid points far outside the
   sample support are extrapolation and are reported separately in the metrics.
3. **Flow matching and DDIM need tens of steps; DDPM needs more.** Plot energy
   distance, largest cluster-fraction error and largest width error against the
   number of steps, with the null scale shaded. By 20 steps the flow sampler
   reaches energy distance **< 0.005** and widths within **0.03** of 0.30.

## 4. Part 3 — The pretrained predictor on photographs

### 4.1 Required implementation

- Download `google/ddpm-cifar10-32` (`config.json`,
  `diffusion_pytorch_model.safetensors`) at revision
  `267b167dc01f0e4e61923ea244e8b988f84deb80` into `inputs/cifar-model`
  (not committed), with the pinned `hf download` command recorded.
- Load the weights as a **fixed noise predictor in JAX**, matching the diffusers
  v0.40.0 `UNet2DModel` architecture described by the configuration.
- Use Part 1's **1000-step DDPM sampler unchanged**, no clipping during
  sampling, `float32`, and pass our level `t` as model index `t-1`.
- Generate **four images** from Gaussian noise.

### 4.2 Acceptance criteria (VERIFY)

Downloaded weights plus Part 1's update produce **recognizable photographs**
without further training. Visualize stages of the reverse process; compare with
real CIFAR-10 images. Structure appears late: at `t = 200` still mostly noise,
objects recognizable between `t = 200` and `t = 50`. Explain which part was
learned before the download and which part the sampler computes now.

## 5. Part 4 — Inpainting with the downloaded model

### 5.1 Required implementation

- Load the prepared images `inputs/images.npz` (four test, four validation, in
  `[-1, 1]`).
- **RePaint**: reverse jumps over levels `1000, 980, …, 20, 0` (50 jumps), **two
  attempts per jump except the final jump to zero**, forward noising between
  attempts (`sqrt(a) X + sqrt(1-a) f`), and observed pixels replaced by
  **forward-noised observations** after every merge (Eq. 19–21).
  Proposal uses the clipped Tweedie estimate with conditional mean (Eq. 21) and
  conditional variance (Eq. 22); the final merge restores the clean observation.
- Hide rows and columns `6:26` in all three channels; inpaint each of the four
  test images with seeds **7201–7204**; record each completion and its
  **hidden-pixel MSE**.

### 5.2 Acceptance criteria (VERIFY)

- Every observed pixel changes by **< 1e-5**; outside the hole the four
  completions are identical; inside they differ.
- Completions are plausible but not equal to the truth; the pixelwise variance
  across the four completions is shown and interpreted.
- 16 completions × 99 network evaluations, reported as such.

## 6. Documentation

`week7/README.md` must place each evidence figure beside its producing command,
explain the transport comparison (DDPM/DDIM/flow), the generated photographs and
the inpainting result, and record training settings, environment and the pinned
model download. `README.md` at the repository root gains the Week 7 entry.
Finally the work is pushed to GitHub.
