# AMAT5315: Scientific Computing for Physicists - Weekly Exercises

This repository stores weekly exercises for AMAT5315 (*Scientific Computing for Physicists*). Each week contains complete source code, verification test suites, evidence figures, and interactive web laboratories.

- **Weekly Exercise Directories**:
  - [`week1/`](week1/): Monte Carlo $\pi$ Estimator & AI Agent Workflows
  - [`week2/`](week2/): 2D Lennard-Jones Molecular Dynamics & Spatial Partitioning
  - [`week3/`](week3/): 2D Ising Lattice Model & Zero-Temperature Metropolis MCMC
  - [`week4/`](week4/README.md): Pseudospectral Continuum Fluid Dynamics (Navier–Stokes & Taylor–Green)
  - [`week5/`](week5/README.md): Acoustic Wave Simulation, Automatic Differentiation & Binomial Checkpointing
  - [**`week6/` (Latest)**](week6/README.md): **Energy-Based Models (RBM, Classical Hopfield & Softmax Attention)**

- **Interactive GitHub Pages Web Labs**:
  - [Week 1: AI Agents & Monte Carlo $\pi$ Lab](https://imgrj.github.io/AMAT5315-2026Fall-Exercise/week1-pi-lab.html)
  - [Week 2: Molecular Dynamics Simulation Studio](https://imgrj.github.io/AMAT5315-2026Fall-Exercise/index.html)
  - [Week 3: Ising Lattice Lab](https://imgrj.github.io/AMAT5315-2026Fall-Exercise/week3-ising-lab.html)
  - [Week 4: Continuum Fluid Dynamics Lab](https://imgrj.github.io/AMAT5315-2026Fall-Exercise/week4-fluid-lab.html)
  - [Week 5: Seismic & Automatic Differentiation Lab](https://imgrj.github.io/AMAT5315-2026Fall-Exercise/week5-seismic-lab.html)
  - [**Week 6: Energy-Based Models & Softmax Attention Lab**](https://imgrj.github.io/AMAT5315-2026Fall-Exercise/week6-energy-lab.html)

---

## Week 6: Energy-Based Models (Latest Exercise)

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
