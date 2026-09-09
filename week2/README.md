# md: Lennard-Jones fluid in 2D (week 2)

`md` simulates a 2D Lennard-Jones fluid: `md run` writes a trajectory,
`md check` verifies its physics from the raw frames, `md video` renders an
MP4 of the motion with the radial distribution function.

- `md run --n N --temperature T [--ramp-to T2] [--dt D] [--steps S]
  [--equil E] [--sample-every K] [--force naive|cells] [--out DIR] [--seed S]`
  writes into the output directory `DIR/`: a per-frame text `trajectory.txt`
  and a `run.json` with the run settings (including `ramp_to` when a linear
  temperature ramp was requested). One frame is recorded every
  `--sample-every` steps (default 1). `--temp` is accepted as an alias for
  `--temperature`.
- `md check FILE-OR-DIR` recomputes temperature, energy drift, and the speed
  distribution from the recorded frames and reports PASS/FAIL. Pass a run
  output directory or a trajectory file.
- `md video FILE-OR-DIR --out OUT.mp4` renders atoms (periodically wrapped)
  and a building `g(r)` panel.

Physics: shifted-force Lennard-Jones cutoff at `rc = 2.5`, density 0.8,
periodic box `L = sqrt(N / 0.8)`, reduced units.

## Force engine

Two pair-scan strategies, selectable with `--force`:

- `naive`: every pair `i < j`, minimum-image distance (original loop).
- `cells` (default): neighbour search through a cell grid of side >= `rc`.

The two produce identical physics up to round-off (see
`force-compare.png`), and `cells` is the faster engine for large `N`.

## Benchmark: naive vs cells

Reproduce with (from `week2/`, after `cargo build --release`):

```bash
python3 benchmark.py
```

Method: `md run -n N --temperature 1.0 --dt 0.005 --steps 500 --equil 100
--force naive|cells --seed 1`, three runs per cell; wall time measured for
the whole `md run` process. Times are median seconds with the observed
min-max range. Machine: Apple Silicon (release build, debug info on).

| N | naive (s) median (min-max) | cells (s) median (min-max) | speedup |
|---|---|---|---|
| 100 | 0.030 (0.026-0.034) | 0.023 (0.023-0.031) | 1.3x |
| 400 | 0.184 (0.179-0.189) | 0.097 (0.094-0.101) | 1.9x |
| 1600 | 2.033 (2.009-2.181) | 0.381 (0.379-0.458) | 5.3x |

![Scaling of seconds per step against N for the naive and cell-list force
engines.](scaling.png)

Seconds per step in `scaling.png` = wall time / (equil + steps) = wall / 600,
so fixed per-run overhead (process start, lattice, trajectory formatting) is
included in both lines. That overhead is why the ratio is ~1 at N = 100 and
why the asymptote (naive O(N^2) vs cells O(N)) grows with N.
