# AMAT5315: Scientific Computing for Physicists - Weekly Exercises

This repository stores weekly exercises for AMAT5315 (*Scientific Computing for Physicists*). Each week contains complete source code, verification test suites, evidence figures, and interactive web laboratories.

- **Weekly Exercise Directories**:
  - [`week1/`](week1/): Monte Carlo $\pi$ Estimator & AI Agent Workflows
  - [`week2/`](week2/): 2D Lennard-Jones Molecular Dynamics & Spatial Partitioning
  - [`week3/`](week3/): 2D Ising Lattice Model & Zero-Temperature Metropolis MCMC
  - [`week4/`](week4/README.md): Pseudospectral Continuum Fluid Dynamics (Navier–Stokes & Taylor–Green)
  - [`week5/`](week5/README.md): Acoustic Wave Simulation, Automatic Differentiation & Binomial Checkpointing
  - [`week6/`](week6/README.md): Energy-Based Models (RBM, Classical Hopfield & Softmax Attention)
  - [**`week7/` (Latest)**](week7/README.md): **Diffusion Models & Flow Matching (DDPM, DDIM, pretrained CIFAR-10 U-Net, RePaint)**

- **Interactive GitHub Pages Web Labs**:
  - [Week 1: AI Agents & Monte Carlo $\pi$ Lab](https://imgrj.github.io/AMAT5315-2026Fall-Exercise/week1-pi-lab.html)
  - [Week 2: Molecular Dynamics Simulation Studio](https://imgrj.github.io/AMAT5315-2026Fall-Exercise/index.html)
  - [Week 3: Ising Lattice Lab](https://imgrj.github.io/AMAT5315-2026Fall-Exercise/week3-ising-lab.html)
  - [Week 4: Continuum Fluid Dynamics Lab](https://imgrj.github.io/AMAT5315-2026Fall-Exercise/week4-fluid-lab.html)
  - [Week 5: Seismic & Automatic Differentiation Lab](https://imgrj.github.io/AMAT5315-2026Fall-Exercise/week5-seismic-lab.html)
  - [Week 6: Energy-Based Models & Softmax Attention Lab](https://imgrj.github.io/AMAT5315-2026Fall-Exercise/week6-energy-lab.html)
  - [**Week 7: Diffusion & Flow Matching Lab**](https://imgrj.github.io/AMAT5315-2026Fall-Exercise/week7-diffusion-lab.html)

---

## Week 7: Diffusion and Flow Matching (Latest Exercise)

Generation by transporting fresh Gaussian noise: DDPM and DDIM on a four-cluster mixture, flow matching on straight paths, a JAX port of a pretrained CIFAR-10 noise predictor, and RePaint inpainting.

- **Full Documentation & Reproduction Guide**: [**`week7/README.md`**](week7/README.md)
- **Live Interactive Lab (runs the trained network in the browser)**: [**Week 7 Diffusion Lab**](https://imgrj.github.io/AMAT5315-2026Fall-Exercise/week7-diffusion-lab.html)

### Key Results & Evidence Gallery

#### 1. DDPM on the Gaussian mixture
- **Forward diffusion**: at `t = 1000` the per-coordinate mean is `(-0.007, -0.003)` and the variance `(1.046, 0.990)`, both within the required 0.1 of the standard Gaussian.
- **Training**: 25000 updates, batch 512, cosine decay to zero; the loss plateaus at **0.274**, not zero, because the added noise is still uncertain given the noisy point. Ten stages are saved, including initialization (endpoints up to `|x| ≈ 8.9e3`) and the final model.
- **Generation** (2000 samples, 1000 steps): populations 0.2405/0.2590/0.2420/0.2585 (allowance 0.04), spreads 0.285–0.310 (allowance 0.03), largest centre error **0.019** (allowance 0.05).
- **Denoiser check**: the Tweedie estimate matches the exact posterior mean to 0.056/0.052/0.046 posterior standard deviations at `t = 100, 300, 600`, degrading to 0.377 at `t = 900`, where Tweedie's formula amplifies predictor error.

#### 2. Fewer steps: DDIM and flow matching
- **Determinism**: two DDIM runs over 1000 levels differ by **exactly 0**; two DDPM runs share their final cluster 27.3% of the time, the chance rate for four clusters.
- **Learned marginal velocity**: relative RMS error **0.022, 0.053, 0.063** at `s = 0.1, 0.5, 0.9` (requirement < 0.1).
- **Step-count sweep** (4000 samples per run, null 95% quantile 0.0021): flow matching reaches energy distance **0.0026 at 20 steps** (requirement < 0.005) with width error 0.029; DDIM reaches 0.0043 at 20 levels; DDPM needs about fifty levels (0.0017).

#### 3. The downloaded predictor on photographs
- **Port verification**: the JAX re-implementation of the diffusers v0.40.0 `UNet2DModel` matches the reference predictions to **7.15e-6** on six fixed inputs; 35,746,307 parameters (about 35.7M).
- **Generation**: four images from Gaussian noise with Part 1's 1000-step DDPM update, unclipped; structure appears between `t = 200` and `t = 50`. The weights were learned before the download; the reverse updates are computed here.

#### 4. RePaint inpainting
- **Observed pixels**: hidden square rows/columns 6:26, 99 network evaluations per completion; the largest change of an observed pixel is **0.0** (requirement < 1e-5).
- **Hidden content**: mean hidden-pixel MSE **0.221** over 16 completions (reference 0.24), differing between seeds inside the hole; the variance map shows where the observation leaves the content undetermined.

| Trained DDPM recovers the four clusters | Flow matching and DDIM in tens of steps |
| :---: | :---: |
| ![Week 7 generation](week7/evidence/part1-generation.png) | ![Week 7 steps](week7/evidence/part2-steps.png) |

| Downloaded CIFAR-10 predictor | RePaint keeps every observed pixel |
| :---: | :---: |
| ![Week 7 CIFAR](week7/evidence/part3-cifar-samples.png) | ![Week 7 inpainting](week7/evidence/part4-inpainting.png) |

### Quick Reproducibility Commands

From `week7/` (CPU):
```powershell
python scripts/part1_diffusion.py        # forward diffusion, training, DDPM samples (~4 min)
python scripts/part2_fewer_steps.py      # DDIM, exact noise, flow matching, step sweep (~7 min)
python scripts/part3_cifar.py            # pretrained CIFAR-10 DDPM, four images (~13 min)
python scripts/part4_inpaint.py          # RePaint inpainting (~4.5 min)
python -m pytest tests -q                # 30 verification tests
```

---

## Week 6: Energy-Based Models

Comprehensive implementation, verification, and empirical study of energy-based probabilistic generative models and associative memories:

- **Full Documentation & Reproduction Guide**: [**`week6/README.md`**](week6/README.md)
- **Live Interactive Simulator & Evidence Lightbox**: [**Week 6 Interactive Web Lab**](https://imgrj.github.io/AMAT5315-2026Fall-Exercise/week6-energy-lab.html)

### Key Results & Evidence Gallery

#### 1. Datasets & Native Pixels
- **MNIST**: $28 \times 28$ grayscale pixels ($60k$ train, $10k$ test). Digit `1` is most frequent ($11.24\%$), digit `5` least frequent ($9.03\%$).
- **CIFAR-10**: $32 \times 32$ RGB images ($50k$ train, $10k$ test). Exactly $5,000$ train and $1,000$ test images per class (balanced $0.1000$ fraction).

| MNIST Native Pixels & Class Frequencies | CIFAR-10 Native Pixels & Balanced Splits |
| :---: | :---: |
| ![MNIST Dataset](week6/evidence/mnist-dataset.png) | ![CIFAR-10 Dataset](week6/evidence/cifar-dataset.png) |

#### 2. Restricted Boltzmann Machine (RBM)
- **Visible Spins**: Pixels binarized at half intensity ($v_i \in \{0, 1\} \implies s_i = 2v_i - 1 \in \{-1, +1\}$).
- **CD-1 vs CD-10 Training**: Trained with automatic differentiation of free-energy differences and `stop_gradient` on 128 hidden units across 20 epochs.
  - Held-out one-step reconstruction error dropped from $49.96\%$ to **$4.77\%$** (CD-1) and **$4.95\%$** (CD-10).
- **Generation from Random Spins**: Pure noise evolves through Gibbs sweeps into connected strokes and coherent digit contours.
- **Clamped Image Completion**: Upper half clamped, lower half sampled; ink overlap $2|A \cap B|/(|A|+|B|)$ on 64 test digits exceeds wrong-upper-half conditioning by **$+0.167$** (CD-1) and **$+0.202$** (CD-10), far exceeding the required $0.10$ margin.

| RBM Training Learning Curves | Fresh Generation from Random Spins |
| :---: | :---: |
| ![RBM Curves](week6/evidence/rbm-learning-curves.png) | ![RBM Generation](week6/evidence/rbm-generation.png) |

| Clamped Gibbs Image Completion | Quantitative Ink Overlap vs Sweeps |
| :---: | :---: |
| ![RBM Completion](week6/evidence/rbm-completion-gallery.png) | ![RBM Ink Overlap](week6/evidence/rbm-ink-overlap.png) |

#### 3. Classical Hopfield Network & Capacity Collapse
- **Hebbian Couplings & Crosstalk**: Stored $P$ random patterns with $J_{ij} = \frac{1}{N}\sum_\mu \xi_i^\mu \xi_j^\mu$ ($J_{ii}=0$).
  - Unstable bit fraction follows Gaussian crosstalk prediction $\Phi(-1/\sqrt{\alpha})$ with finite-$N$ correction $\Phi(-\sqrt{\frac{N-1}{P-1}})$.
- **Storage Capacity Phase Transition**: Zero-temperature asynchronous recall sweeps demonstrate storage capacity collapse toward Amit-Gutfreund-Sompolinsky limit $\alpha_c \approx 0.138$.
  - Crossing points ($m < 0.90$): $N=100 \implies 0.175$, $N=200 \implies 0.166$, $N=400 \implies 0.154$, $N=800 \implies 0.146$ (all within $0.011$ of reference).

| Hopfield Unstable Bits vs Crosstalk | Storage Capacity & Recall Collapse |
| :---: | :---: |
| ![Hopfield Unstable Bits](week6/evidence/hopfield-unstable-bits.png) | ![Hopfield Capacity](week6/evidence/hopfield-capacity.png) |

#### 4. Modern Hopfield Network & Softmax Attention Equivalence
- **Exact Transformer Attention Identity**: The modern update $x \leftarrow X \mathrm{softmax}(\beta X^T x)$ mathematically matches scaled dot-product attention:
  $$\mathrm{Attention}(q, K, V) = V^T \mathrm{softmax}(\beta K q)$$
  Measured maximum absolute difference with `jax.nn.dot_product_attention` is **$1.31 \times 10^{-6}$** ($< 10^{-5}$).
- **Temperature-Controlled Retrieval**:
  - $\beta \le 0.01$: diffuse average of all stored digits.
  - $\beta = 0.03$: target digit features emerge from background.
  - $\beta \ge 0.1$: sharp single-step memory retrieval.
- **Correlated Digit Capacity**:
  - Classical Hebbian recall achieves **$0.0\%$** success for all $P \in \{3, 10, 30, 100, 300, 1000\}$ due to high background correlations ($87\%$ background pixels).
  - Modern Hopfield ($\beta = 0.3$) achieves **$95\% \sim 100\%$** retrieval success across all $P$ up to $1000$.

| Softmax Weights across Temperature $\beta$ | Retrieval Transition Gallery across $\beta$ |
| :---: | :---: |
| ![Modern Weights](week6/evidence/modern-hopfield-weights.png) | ![Modern Sweep](week6/evidence/modern-hopfield-beta-sweep.png) |

| Capacity: Modern vs Classical on Digits | Modern Hopfield Retrieval Gallery ($P=100$) |
| :---: | :---: |
| ![Modern Capacity](week6/evidence/modern-hopfield-capacity.png) | ![Modern Gallery](week6/evidence/modern-hopfield-gallery.png) |

### Quick Reproducibility Commands

From `week6/`:
```bash
# Part 1: Datasets inspection
python scripts/part1_datasets.py

# Part 2: RBM CD-1 / CD-10 training, generation & completion
python scripts/part2_rbm.py

# Part 3: Classical Hopfield crosstalk & capacity sweeps
python scripts/part3_hopfield.py

# Part 4: Modern Hopfield attention equivalence & correlated digit benchmark
python scripts/part4_modern_hopfield.py
```

---

## Week 1: AI Agents & Monte Carlo $\pi$ Estimator

Week 1 contains `pi.py`, a Monte Carlo estimator of $\pi$ using pseudo-random dart throws in the unit square, along with `test_pi.py`.

```bash
python3 -m pytest week1/
```

Expected output:
```
estimate=3.146604, absolute error=0.005011
```

![pi estimate and absolute error](week1/result.png)
