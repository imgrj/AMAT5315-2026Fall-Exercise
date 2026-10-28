# Week 8: Language models and the transformer

Source: [official learning sheet](https://giggleliu.github.io/AMAT5315-2026Fall/week8-learning-sheet.html),
24-page PDF, SHA256 `4e33bea929459c01511efc0298308a93d08d99ee7df38372bbf52e89be69cc35`.
The local PDF matches the official PDF byte for byte (checked 2026-10-28).

## Common contract

Work in `week8/`, use uv, CPU, JAX models and independent NumPy checks.
Commit source, tests, small summaries, figures in `evidence/`, and this source PDF.
Exclude downloads (`data/`, `inputs/`) and raw runs/settings (`artifacts/`).
Commit specifications before tests, tests before implementation; record failing tests
and deliberate negative controls. Commit each completed task. Never tune acceptance
tolerances to fit results. Report actual measurements, including discrepancies.

## 1. Embedding geometry

Load `minishlab/potion-base-8M` revision `bf8b056` (`model.safetensors`,
`tokenizer.json`) without PyTorch. Each word must map to one token. Normalize rows
before taking offsets. Use the exact nine pairs in each of the sheet's three lists
(country/capital, male/female, adjective/comparative). For each offset compare with
the mean of the other eight. Each relation's mean cosine >= 0.3; row shuffle seed
1730 must give |mean| <= 0.15. Show nine unit arrows per relation, trained above
and shuffled below the horizontal reference. Also record 20 shuffled controls.

## 2. Counts, normalization and sampling

Download TinyStoriesV2-GPT4-valid.txt from `roneneldan/TinyStories` revision
`f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`. Split at `<|endoftext|>`, strip, remove
empty stories, first 90% training: exactly 24867 / 2763 stories. Use SentencePiece
`stories260K/tok512.model` from `karpathy/tinyllamas` revision `0bd21da`.
Vocabulary 512; EOS separates stories. Report characters/token (~2.05), round-trip
exceptions (three held-out double spaces), and split/token counts.

Fit orders 1..6. Choose add-k from 0.0001, 0.001, 0.01, 0.1, 1 using the last 10%
of training tokens, then refit counts on all training tokens. No validation tuning.
Report nats/token, exp(loss), bits/character, unseen n-gram fractions.
Reference uniform/unigram/bigram/trigram nats 6.238/4.839/3.038/2.094,
perplexities 512/126.4/20.9/8.12, bits/char 4.386/3.403/2.136/1.473:
loss and bits/char within 0.03, perplexity within 3% (uniform exact).
Best order must be 4; 4/5/6 bits/char within 0.03 of 1.310/1.46/1.79.
Unseen fractions orders 3..6 within 0.02 of 0.006/0.042/0.122/0.235.

Explicitly enumerate all 512^l bigram token sequences for l=1,2,3 from EOS:
total probability error < 1e-12. Sample 250000 ancestral bigram tokens; pool
chi-square statistics/df across the 20 most frequent training predecessor tokens,
pooling rare bins until expected counts >=5; p > 0.001. Negative controls:
unnormalized scores fail normalization; two-token-lag sampler fails chi-square.

## 3. Memory in weights and one-token MLP

NumPy stores: outer-product, least-squares, ReLU powers 1,2,3; independent unit
keys and values. Sweep d=16,32,64, M=8..65536 powers of two, three draws. Query
stored keys; report query sample size explicitly if estimating across keys.
For least squares use the diagonal projection coefficient as the signal weight,
not the original value with weight one. Median over draws; interpolate R=1 in
log-log coordinates. At d=64 outer capacity 50..85, least-squares 100..170 and
R<1e-10 through M=64. Slopes in d within 0.1 of 1 for outer/LS/ReLU, within
0.15 of 2 for ReLU^2 and >2.5 for ReLU^3. Show R versus M and capacity versus d.
Include correlated-key control (mean similarity 0.3). Discuss sqrt(M) growth
and the LS threshold rather than treating its near-threshold curve as a power law.

JAX MLP: token embedding 64, hidden 512 tanh, output 512; 4000 AdamW steps,
16384 examples/step, peak LR 0.006, warmup/cosine. Held-out loss within 0.01 of
measured bigram loss (reference 3.038). Record seed, warmup, decay and weight decay.
Optional extension: positive/negative hidden-unit top-five keys and values,
activation concentration before/after training.

## 4. Memory in context

32x32 character picture (`算`, identifiable in Fig. 13 though the PDF text font is
missing), +1 ink/-1 background. Other values are random +/-1 pictures; keys unit
random vectors, d=64. M=64,256,1024,4096. Outer/ReLU/ReLU^2/softmax beta=8.
Plot recalled dots scaled by the matching memory's own weight. Report every R.
At M=1024 outer/ReLU median over 20 draws within 15% of 4/2.83; at M=4096
ReLU^2 R>1, softmax R<0.1 at every M. Show identity of normalized versus
unnormalized exponential R and permutation invariance of context memory.

## 5. Transformer and samplers

JAX pre-norm decoder: learned token/position embeddings, 2 blocks, width 96,
4 heads, context 64, biased Q/K/V/O maps, causal mask, residuals, biased GELU
MLP width 384, final layer norm and biased untied readout. Train 6000 AdamW steps,
32 sequences of 65 tokens (64 inputs/shifted targets), peak LR 0.003, 300 warmup
steps, cosine to 0.0003. Record initialization, seed, weight decay and timing.
No future-token leakage; masked weights exactly zero, rows normalized.

Held-out bits/char below measured best count model AND 1.310. Show uniform,
unigram, bigram, trigram, MLP and transformer comparison; 60-token openings at
T=1, seed 1848. Evaluate consecutive 64-input windows, per-position losses and
2-SE bands; mean loss after >=32 tokens below 2.09 nats/token. Generate stories
at T=0.8, top-k=20/T=0.8, top-p=0.9/T=0.8, min-p=0.1/T=0.7.

Fixed broad context, 100000 next-token draws per sampler; independent target
probabilities and grouped chi-square p>0.001. Top-p includes the threshold-crossing
token. T=0.0001 always chooses argmax. Negative control drops crossing token.
Generate 128 independent sequences of 128 tokens at T=1 and at T=0.7. Score both
by original T=1 model; z=mean(sequence surprise-entropy)/SE across sequences:
|z(T=1)|<4, |z(T=0.7)|>40. Optional extension: eight attention heat maps.

## 6. Chinchilla refit

Download `epoch-research/analyzing-chinchilla` revision
`92258837425e1b5f2851d624287f0120583a3d0e/data/svg_extracted_data.csv`.
245 rows, D=C/(6N), remove the five highest-loss rows. Fit
L=E+A/N^alpha+B/D^beta using Huber(log(prediction)-log(observed)), delta=0.001,
L-BFGS from a grid; keep lowest objective. Refit 400 bootstrap resamples starting
from best fit; standard errors for all five parameters. Require alpha 0.32..0.38,
beta 0.32..0.42, E 1.75..1.90. Plot losses versus N at C=1e19/1e20/1e21 and
predicted versus measured. Mean log residual within +/-0.005; original parameters
(E=1.69,A=406.4,B=410.7,alpha=0.34,beta=0.28) mean residual >0.01.
Derive and plot N*,D* for C=1e18..1e25. At C=5.76e23, D*/N* in 10..30, report
bootstrap 95% interval and whether it includes 20. Mark the published Chinchilla
70B/1.4T point; explain its nominal 6ND budget (5.88e23) versus specified 5.76e23.

## Delivery

`README.md`: every figure alongside its producing command, tokenizer/provenance,
memory picture explanation, bits/char model comparison, scaling derivation and
fit, versions, exact reproduction commands. Add Week 8 to root README and existing
GitHub Pages lab navigation, preserving other weeks' exercises. Review with a
separate reviewer as required by requesting-code-review. Push source/evidence to
origin, verify remote HEAD and Pages. Do not edit the instructor's course website.
