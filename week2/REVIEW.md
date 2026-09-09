# Week 2 Implementation Review

Review against the design specifications and learning sheet contract in docs/superpowers/plans/2026-09-09-week2-contract-remediation.md.

## Review Findings

### 1. Crate Location & Manifest Path
- **Finding**: Initial prototype located Cargo.toml directly under week2/. The learning sheet contract requires week2/md/Cargo.toml so automated check suites can execute cargo test --manifest-path md/Cargo.toml --release.
- **Status**: FIXED
- **Resolution**: Moved the complete Rust crate structure into week2/md/, updated Makefile to target md/Cargo.toml.

### 2. Trajectory Format & Metadata
- **Finding**: Earlier code generated custom plain-text 	rajectory.txt. Web viewer (week2-viewer.html) and PDF contract require JSON Lines 	raj.jsonl with per-frame fields (step, 	, pos, el, E_pot, E_kin) and un.json containing ox: [Lx, Ly].
- **Status**: FIXED
- **Resolution**: Implemented structured JSON Lines output and un.json writer matching the schema required by the live viewer.

### 3. Shifted-Potential Cutoff & Geometry
- **Finding**: Pair potential cutoff requires shifted potential {\text{cut}}(r) = U(r) - U(r_c)$ for  < r_c$ and zero outside ( = 2.5$), with rectangular triangular lattice ( = \sqrt{2 / (\sqrt{3}\rho)}$,  = \sqrt{3}a/2$).
- **Status**: FIXED
- **Resolution**: Implemented triangular lattice coordinates with periodic minimum image convention along each axis separately.

### 4. Conservation & Integration
- **Finding**: Velocity-Verlet and Euler integrators must share the unified Integrator trait. Dimer test must verify Euler energy drift while Verlet remains bounded over 5000 steps.
- **Status**: FIXED
- **Resolution**: Verified trait polymorphism and confirmed Verlet relative energy error stays bounded below ^{-3}$.

### 5. Cell Lists Optimization & Temperature Ramp
- **Finding**: (N)$ neighbour searching via cell lists must agree with naive all-pairs calculation within numerical rounding tolerance; --ramp-to linear heating schedule must be tracked and recorded.
- **Status**: FIXED
- **Resolution**: Implemented cell grid with 9-neighbour wrap-around search, validated agreement on perturbed lattices, and verified melting transition across the temperature ramp.

### 6. Video Generation & Headless Encoding
- **Finding**: md video must encode trajectory and dynamic radial distribution function (r)$ into an MP4 under 2 MB.
- **Status**: FIXED
- **Resolution**: Configured FFmpeg encoding pipeline, verified luid.mp4 output at 248 KB.

## Verdict
All contract requirements, physics invariants, and file artifacts satisfied.
