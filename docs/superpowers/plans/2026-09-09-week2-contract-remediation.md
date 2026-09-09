# Week 2 learning-sheet contract remediation plan

## Context

The repository contains a working prototype, but it predates the final Week 2
learning sheet. Its crate lives directly in week2/, writes a custom
trajectory.txt, uses a square box and a shifted-force cutoff, and requires
non-default physics flags. The final sheet requires a crate in week2/md/,
JSON/JSONL artifacts, a rectangular triangular-lattice box, a shifted-potential
cutoff, and a no-argument contract run.

This plan preserves the useful prototype and its Git history while correcting
the public interfaces and physics contract.

## Correctness criteria

1. week2/md/Cargo.toml builds a crate named md; cargo run prints
   Hello, world!, and all unit/integration tests pass.
2. Plain Lennard-Jones energy and force agree through an independent central
   difference, and the dimer comparison routes Euler and velocity-Verlet
   through one Integrator trait.
3. A default run is exactly N=100, rho=0.8, T=0.5, dt=0.01, 2000
   equilibration steps, 10000 production steps, sampling every 50, seed 2026.
4. The lattice uses a=sqrt(2/(sqrt(3)*rho)), h=sqrt(3)*a/2, with box
   [n_side*a, n_side*h]; periodic distances use the minimum image separately
   on each axis.
5. The cutoff energy is U(r)-U(rc) below rc=2.5 and zero otherwise; the
   force is the unshifted Lennard-Jones force below the cutoff and zero above.
6. run writes run.json and one JSON object per line in traj.jsonl, with
   all fields and frame-count rules from the learning sheet.
7. check rejects malformed/inconsistent files, recomputes kinetic and
   shifted-potential energy from raw frames, and reports the stated drift,
   temperature, and 24-bin speed-shape checks.
8. Naive and cell-list forces/energies agree on perturbed, boundary, cutoff,
   and two-cell-box cases; --force cells is the default.
9. --ramp-to records its metadata and applies the linear production
   temperature schedule. Video output shows positions beside a running RDF
   and remains below 2 MB for required runs.
10. README tables contain measurements made on this Windows machine, all
    figures/data are reproducible, and the final fresh-clone verification
    succeeds.

## Execution order

1. Move the Rust crate beneath week2/md/ and update the Makefile/examples.
2. Add/adjust tests for the final geometry, cutoff, JSON schema, frame count,
   energy cross-checks, CLI defaults, force equality, and heating schedule.
3. Implement the final contract without weakening any tolerance.
4. Run release tests and the default physics gate; fix failures from measured
   evidence.
5. Regenerate field, dimer, scaling, trajectories, and videos; benchmark on
   Windows and update README.
6. Review from a clean clone, record findings in week2/REVIEW.md, then
   publish docs/ after authenticating the student's GitHub account.
