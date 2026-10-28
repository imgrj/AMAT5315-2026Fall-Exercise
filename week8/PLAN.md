# Week 8 implementation plan

**Goal:** Execute all six parts of SPEC.md, publish reproducible evidence and push.
**Architecture:** small numerical modules in `src/`, executable part scripts in
`scripts/`, independent checks in `tests/`, figures/JSON in `evidence/`.
**Tech stack:** uv, Python, JAX/Optax CPU, NumPy/SciPy, Matplotlib, SentencePiece,
safetensors and tokenizers; static HTML/JavaScript for the existing Pages site.
**Spec:** [SPEC.md](SPEC.md). All tolerances/settings there apply unchanged.
**Execution:** inline, with independent reviewer before publication. The user has
explicitly authorized execution, commits and push. No additional design approval.

## Tasks and verification

- [x] 1. `scripts/download_data.py`, `src/data.py`, `src/embeddings.py`,
  `scripts/part1_embeddings.py`. First test split order and leave-one-out cosine
  on hand-made vectors; run red; implement. Download pinned sources with SHA256
  manifest. Run Part 1 and commit nine-arrow figures and measured controls.
- [x] 2. `src/ngrams.py`, `scripts/part2_counts.py`: `fit(tokens,n,vocab,k)`,
  `probabilities(context)`, `score(tokens)`. Tests: explicit add-k fixture
  `[0,1,0,1,1]`, unseen context uniform, probabilities sum to one. Run red before
  implementation. Use packed integer n-grams and sorted sparse counts. Execute
  full held-out comparisons, 512^3 enumeration and 250k samples. Commit results.
- [x] 3. `src/memory.py`, `src/models.py`, `src/training.py`,
  `scripts/part3_memory.py`, `scripts/part3_mlp.py`. Test orthogonal recall and LS
  signal decomposition before stores; test MLP loss/gradient against NumPy before
  training. Sweep capacities; train the specified 4000-step model and compare
  with bigram, save checkpoints/settings locally and curves/metrics in Git.
- [x] 4. `scripts/part4_attention.py`: reuse tested store functions. Test softmax
  recall against explicit exponential weights, normalized R invariance and key
  permutation. Render the 32x32 picture with a CJK font; run 20 random draws,
  save dots and numeric R; commit.
- [x] 5. `src/models.py`, `src/sampling.py`, `scripts/part5_transformer.py`,
  `scripts/part5_evaluate.py`: test causal invariance, masked weights, independent
  small attention oracle, parameter count and hand-derived truncation fixtures
  before implementation. Train exactly 6000 steps. Evaluate all held-out windows,
  samplers, 128-sequence self-consistency, samples and eight attention maps.
- [x] 6. `src/scaling.py`, `scripts/part6_scaling.py`: test Huber gradient via
  finite differences and optimum via derivative/budget identity. Run grid fit,
  400 bootstraps, print residual comparison and interval; commit figures/metrics.
- [x] 7. `scripts/verify_results.py` and `tests/`: audit each SPEC threshold;
  execute and record normalization/lag/top-p negative controls. Review source and
  evidence independently; repair substantive findings and rerun affected checks.
- [ ] 8. `README.md`, root README, `docs/week8-language-lab.html`, site navigation:
  add measured figures, producing commands, explanations and interactive memory,
  token sampling and compute-budget views consistent with prior lab style.
  Inspect rendered figures/site. Commit, push main without force, verify remote
  SHA and deployed page/assets; mark goal complete only after full audit.

Run tests with `.venv/Scripts/python -m pytest tests -q` from week8. Run each
experiment with `.venv/Scripts/python scripts/partN_*.py`. Scripts fail loudly if
prerequisite data/checkpoints are missing. Heavy runs emit progress and are not
repeated after successful verification unless a related implementation changes.

## Final review

Independent numerical review approved after fixing shared sampling-path coverage,
exact uniform acceptance, NumPy gradient regression, and initial trigram history.
All 34 unit tests and 49 full-run checks pass. Local desktop/mobile browser QA
passes, including 64 independent sampler comparisons. Publication is the final
remaining action in Task 8.
