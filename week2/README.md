# md: Lennard-Jones fluid in 2D (week 2)

`md` simulates a 2D Lennard-Jones fluid: `md run` writes a trajectory,
`md check` verifies its physics from the raw frames, `md video` renders an
MP4 of the motion with the radial distribution function.

- `md run -n N --temp T --dt DT --steps S --equil E [--ramp-to T2] [--force naive|cells] ...`
  writes a per-frame text trajectory and `run.json` (run settings, including
  `ramp_to` when a linear temperature ramp was requested).
- `md check FILE` recomputes temperature, energy drift, and the speed
  distribution from the file and reports PASS/FAIL.
- `md video FILE --out video.mp4` renders atoms (periodically wrapped) and a
  building `g(r)` panel.

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

Method: `md run -n N --temp 1.0 --dt 0.005 --steps 500 --equil 100
--force naive|cells --seed 1`, three runs per cell; wall time measured for
the whole `md run` process. Times are median seconds with the observed
min-max range. Machine: Apple Silicon (release build, debug info on).

| N | naive (s) median (min-max) | cells (s) median (min-max) | speedup |
|---|---|---|---|
| 100 | 0.027 (0.024-0.031) | 0.027 (0.023-0.031) | 1.0x |
| 400 | 0.194 (0.187-0.207) | 0.112 (0.105-0.114) | 1.7x |
| 1600 | 2.056 (2.010-2.072) | 0.447 (0.388-0.457) | 4.6x |

![Scaling of seconds per step against N for the naive and cell-list force
engines.](scaling.png)

Seconds per step in `scaling.png` = wall time / (equil + steps) = wall / 600,
so fixed per-run overhead (process start, lattice, trajectory formatting) is
included in both lines. That overhead is why the ratio is ~1 at N = 100 and
why the asymptote (naive O(N^2) vs cells O(N)) grows with N.
