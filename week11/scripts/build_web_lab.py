"""Build `docs/week11-tnsample-lab.html` from the saved runs.

The page carries the same upper-right segmented switch as Weeks 5-10:
**Interactive Studio** replays recorded numbers (the gate ladders, the tropical
contraction of the 104-spin magnet with its Boltzmann draws, and the N-queens
board player), and **Evidence & Verification** holds the three computed still
figures in a lightbox, the acceptance table and the sheet's reference numbers.

Every number on the page comes from `runs/` or `evidence/acceptance.json`; run
`python scripts/run_all.py` before building, then
`python scripts/build_web_lab.py`.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT.parent / 'docs'
ASSETS = DOCS / 'assets' / 'week11'
RUNS = ROOT / 'runs'
EVIDENCE = ROOT / 'evidence'


def read_run(name: str) -> dict:
    return json.loads((RUNS / name / 'run.json').read_text(encoding='utf-8'))


CSS = """
:root {
  --bg: #0b1120; --panel: #131c2e; --panel2: #1b2740; --ink: #e8edf6;
  --muted: #93a1b8; --line: #2c3650; --accent: #6ea8ff; --good: #56d364; --warn: #f0b849;
  --bad: #f87171;
}
* { box-sizing: border-box; }
html, body { margin: 0; background: var(--bg); color: var(--ink);
  font: 14px/1.55 system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif; }
a { color: var(--accent); }
.nav-banner { display: flex; align-items: center; gap: 6px 14px; flex-wrap: wrap;
  padding: 9px 22px; border-bottom: 1px solid var(--line); background: #0d1526;
  font-size: 12.5px; }
.nav-banner a { color: var(--muted); text-decoration: none; padding: 3px 8px;
  border-radius: 6px; white-space: nowrap; }
.nav-banner a:hover { color: var(--ink); background: var(--panel2); }
.nav-banner a.active { color: #04121f; background: var(--accent); font-weight: 700; }
.nav-banner .repo-link { margin-left: auto; color: var(--accent); font-weight: 600; }
header { display: flex; align-items: center; gap: 18px; flex-wrap: wrap;
  padding: 16px 22px; border-bottom: 1px solid var(--line); }
.header-left { min-width: 0; }
header h1 { font-size: 1.22rem; margin: 0; font-weight: 680; letter-spacing: -0.01em; }
header h1 .badge { font-size: .66rem; text-transform: uppercase; letter-spacing: .08em;
  background: rgba(110,168,255,.16); color: var(--accent); border: 1px solid rgba(110,168,255,.4);
  padding: 2px 8px; border-radius: 999px; vertical-align: middle; margin-left: 6px; }
header .sub { color: var(--muted); font-size: .85rem; margin-top: 3px; }
.tab-buttons { margin-left: auto; display: flex; gap: 4px; background: var(--panel);
  border: 1px solid var(--line); border-radius: 10px; padding: 4px; }
.tab-btn { background: transparent; color: var(--muted); border: 0; border-radius: 7px;
  padding: 7px 14px; font: inherit; font-size: .84rem; cursor: pointer; font-weight: 600; }
.tab-btn:hover { color: var(--ink); }
.tab-btn.active { background: var(--accent); color: #04121f; }
.tab-btn:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
main, .panel-view { padding: 18px 22px 30px; }
.hidden { display: none !important; }
.studio-grid { display: grid; gap: 16px; grid-template-columns: repeat(2, minmax(320px, 1fr)); }
@media (max-width: 900px) { .studio-grid { grid-template-columns: 1fr; } }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
  padding: 14px; min-width: 0; display: flex; flex-direction: column; }
.card h2 { font-size: .82rem; text-transform: uppercase; letter-spacing: .07em;
  color: var(--muted); margin: 0 0 4px; font-weight: 700; }
.card p.lead { color: var(--muted); font-size: .8rem; margin: 0 0 10px; }
canvas { width: 100%; background: var(--panel2); border-radius: 8px; display: block;
  touch-action: none; }
.controls { display: flex; align-items: center; gap: 8px; margin-top: 10px; flex-wrap: wrap; }
button.act { background: var(--panel2); color: var(--ink); border: 1px solid var(--line);
  border-radius: 7px; padding: 6px 12px; font: inherit; font-size: .82rem; cursor: pointer; }
button.act:hover { border-color: var(--accent); }
button.act.primary { background: var(--accent); color: #04121f; border-color: var(--accent);
  font-weight: 700; }
button.chip { background: transparent; color: var(--muted); border: 1px solid var(--line);
  border-radius: 999px; padding: 3px 10px; font: inherit; font-size: .76rem; cursor: pointer; }
button.chip.on { color: #04121f; background: var(--accent); border-color: var(--accent);
  font-weight: 700; }
input[type=range] { accent-color: var(--accent); flex: 1; min-width: 120px; }
.readouts { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px;
  margin-top: 10px; }
@media (max-width: 620px) { .readouts { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
.ro { background: var(--panel2); border: 1px solid var(--line); border-radius: 8px;
  padding: 7px 9px; min-width: 0; }
.ro .k { color: var(--muted); font-size: .66rem; text-transform: uppercase;
  letter-spacing: .05em; }
.ro .v { font-size: .98rem; font-variant-numeric: tabular-nums; margin-top: 1px;
  overflow-wrap: anywhere; }
.tag { display: inline-block; padding: 1px 7px; border-radius: 999px; font-size: .7rem;
  border: 1px solid var(--line); color: var(--muted); }
.tag.good { color: var(--good); border-color: var(--good); }
.tag.bad { color: var(--bad); border-color: var(--bad); }
.statline { font-size: .8rem; color: var(--muted); margin-top: 8px;
  font-variant-numeric: tabular-nums; }
.evidence-head { max-width: 980px; }
.evidence-head h2 { font-size: 1.28rem; margin: 0 0 6px; }
.evidence-head p { color: var(--muted); margin: 0 0 16px; }
.figures { display: grid; gap: 16px; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); }
figure.fig { margin: 0; background: var(--panel); border: 1px solid var(--line);
  border-radius: 12px; overflow: hidden; display: flex; flex-direction: column; }
figure.fig img { width: 100%; background: #fff; cursor: zoom-in; display: block; }
figure.fig figcaption { padding: 10px 12px; font-size: .8rem; color: var(--muted); }
figure.fig figcaption b { color: var(--ink); }
figure.fig code { color: var(--accent); font-size: .74rem; overflow-wrap: anywhere; }
table { width: 100%; border-collapse: collapse; font-size: .8rem;
  font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--line); }
th { color: var(--muted); font-weight: 600; font-size: .72rem; text-transform: uppercase;
  letter-spacing: .05em; }
td.ok { color: var(--good); }
.table-card { margin-top: 16px; }
.lightbox { position: fixed; inset: 0; background: rgba(5,9,18,.94); z-index: 60;
  display: flex; flex-direction: column; padding: 20px 22px; }
.lightbox.hidden { display: none; }
.lightbox .lb-top { display: flex; align-items: center; gap: 12px; }
.lightbox .lb-top .lb-count { color: var(--muted); font-size: .8rem; }
.lightbox .lb-top .lb-close { margin-left: auto; }
.lightbox img { flex: 1; min-height: 0; object-fit: contain; width: 100%; margin: 12px 0;
  background: #fff; border-radius: 10px; }
.lightbox .lb-caption { color: var(--ink); font-size: .86rem; max-width: 1100px; }
.lightbox .lb-command { color: var(--muted); font-size: .76rem; margin-top: 4px;
  font-family: 'JetBrains Mono', ui-monospace, monospace; overflow-wrap: anywhere; }
footer { padding: 18px 22px 30px; color: var(--muted); font-size: .78rem;
  border-top: 1px solid var(--line); }
"""

NAV = """
<nav class="nav-banner">
  <span><strong style="color: var(--ink);">AMAT5315:</strong> Scientific Computing</span>
  <a href="week1-pi-lab.html">Week 1: AI Agents &amp; Pi</a>
  <a href="index.html">Week 2: Molecular Dynamics</a>
  <a href="week3-ising-lab.html">Week 3: Ising Lattice Lab</a>
  <a href="week4-fluid-lab.html">Week 4: Fluid Dynamics Lab</a>
  <a href="week5-seismic-lab.html">Week 5: Seismic &amp; AD Lab</a>
  <a href="week6-energy-lab.html">Week 6: Energy-Based Models</a>
  <a href="week7-diffusion-lab.html">Week 7: Diffusion &amp; Flow Matching</a>
  <a href="week8-language-lab.html">Week 8: Language &amp; Transformer</a>
  <a href="week9-inference-lab.html">Week 9: Inference &amp; Tensor Nets</a>
  <a href="week10-tensor-lab.html">Week 10: Tensor RG &amp; Ising</a>
  <a href="week11-tnsample-lab.html" class="active">Week 11: Hard Problems &amp; Sampling</a>
  <a href="week12-branching-lab.html">Week 12: IP &amp; Learned Branching</a>
  <a class="repo-link" href="https://github.com/imgrj/AMAT5315-2026Fall-Exercise" target="_blank" rel="noopener">GitHub Repo &#8599;</a>
</nav>
"""

STUDIO = """
<main id="studio" class="panel-view" role="tabpanel" aria-labelledby="tabStudio">
  <div class="studio-grid">
    <section class="card">
      <h2>1. Every logic rule as a magnet</h2>
      <p class="lead">Bits become spins <code>s = 2x - 1</code>. Each gadget's floor holds exactly its
        truth table; the ladder below is the enumeration saved in <code>runs/gadgets/run.json</code>.</p>
      <canvas id="gatesCanvas" height="300" aria-label="Gate energy ladders"></canvas>
      <div class="controls" id="gateChips"></div>
      <div class="controls">
        <button class="act primary" id="gatesPlay">Play the seven gates</button>
        <button class="act" id="gatesStep">Next gate</button>
        <button class="act" id="gatesReset">Reset</button>
      </div>
      <div class="readouts">
        <div class="ro"><div class="k">gate</div><div class="v" id="gateName">&ndash;</div></div>
        <div class="ro"><div class="k">floor</div><div class="v" id="gateFloor">&ndash;</div></div>
        <div class="ro"><div class="k">allowed rows</div><div class="v" id="gateAllowed">&ndash;</div></div>
        <div class="ro"><div class="k">nearest forbidden</div><div class="v" id="gateForbidden">&ndash;</div></div>
      </div>
      <div class="statline" id="gateNote">No seed is involved: every number here must match the sheet.</div>
    </section>

    <section class="card">
      <h2>2. Tropical contraction of the 104-spin magnet</h2>
      <p class="lead">One pass along <code>omeco</code>'s tree: every bar is an intermediate tensor,
        drawn at its entry count. The min-plus pass returns <code>(E_min, g_0)</code>; the rescaled
        pass returns <code>log Z</code>.</p>
      <canvas id="contractCanvas" height="260" aria-label="Contraction intermediates"></canvas>
      <div class="controls">
        <button class="act primary" id="contractPlay">Play the contraction</button>
        <button class="act" id="contractReset">Reset</button>
        <input type="range" id="contractSlider" min="0" max="100" value="0" aria-label="contraction step">
      </div>
      <div class="readouts">
        <div class="ro"><div class="k">node</div><div class="v" id="nodeIndex">&ndash;</div></div>
        <div class="ro"><div class="k">entries here</div><div class="v" id="nodeEntries">&ndash;</div></div>
        <div class="ro"><div class="k">largest so far</div><div class="v" id="nodeMax">&ndash;</div></div>
        <div class="ro"><div class="k">E_min / g<sub>0</sub></div><div class="v" id="tropResult">&ndash;</div></div>
      </div>
      <div class="statline" id="contractNote">533 nodes, sc = 5, tc = 11.45: the largest intermediate
        holds 32 entries, so 2<sup>104</sup> states never have to be listed.</div>
    </section>

    <section class="card">
      <h2>3. Boltzmann draws from one stored tree</h2>
      <p class="lead">2000 independent conditional samples at beta = 2, streamed in the order the run
        recorded them. The ground state decodes to 3 x 5.</p>
      <canvas id="sampleCanvas" height="260" aria-label="Sampled energies"></canvas>
      <div class="controls">
        <button class="act primary" id="samplePlay">Play the 2000 draws</button>
        <button class="act" id="sampleReset">Reset</button>
        <input type="range" id="sampleSlider" min="0" max="100" value="100" aria-label="draws">
      </div>
      <div class="readouts">
        <div class="ro"><div class="k">draws</div><div class="v" id="drawCount">&ndash;</div></div>
        <div class="ro"><div class="k">at E_min</div><div class="v" id="drawGround">&ndash;</div></div>
        <div class="ro"><div class="k">predicted 2000 p<sub>0</sub></div><div class="v" id="drawPredicted">&ndash;</div></div>
        <div class="ro"><div class="k">p<sub>0</sub></div><div class="v" id="drawP0">&ndash;</div></div>
      </div>
      <div class="statline" id="sampleNote"></div>
    </section>

    <section class="card">
      <h2>4. N-queens: count, then draw boards</h2>
      <p class="lead">One bit per square and four carry tensors per square. Contracts only the nonzero
        entries; the same stored tree then draws valid boards uniformly.</p>
      <canvas id="queenCanvas" height="300" aria-label="N-queens board player"></canvas>
      <div class="controls" id="queenSizes"></div>
      <div class="controls">
        <button class="act primary" id="queenPlay">Play 200 draws</button>
        <button class="act" id="queenStep">Next board</button>
        <button class="act" id="queenReset">Reset</button>
      </div>
      <div class="readouts">
        <div class="ro"><div class="k">board size</div><div class="v" id="queenN">&ndash;</div></div>
        <div class="ro"><div class="k">valid boards</div><div class="v" id="queenCount">&ndash;</div></div>
        <div class="ro"><div class="k">boards drawn</div><div class="v" id="queenDrawn">&ndash;</div></div>
        <div class="ro"><div class="k">distinct</div><div class="v" id="queenDistinct">&ndash;</div></div>
      </div>
      <div class="statline" id="queenNote"></div>
    </section>
  </div>
</main>
"""

EVIDENCE_HTML = """
<div id="evidence" class="panel-view hidden" role="tabpanel" aria-labelledby="tabEvidence">
  <div class="evidence-head">
    <h2>Week 11 verification suite and evidence gallery</h2>
    <p>Three figures computed from the runs, the acceptance table against the sheet's printed
      numbers, and the N-queens sequence. Click a figure to open it with its caption and its
      producing command.</p>
  </div>
  <div class="figures">
    <figure class="fig">
      <img src="assets/week11/gates.png" alt="Gate energy ladders and the joined circuit"
           data-caption="(a) Every truth-table row of the seven gadgets at its energy: the rows a gate allows sit on its floor (blue), every forbidden row is at least 2 above it (red). (b) The 16 states of AND(a,b,t) + NOT(t,out) + SET1(out); exactly three reach E0 = -5, the inputs (a,b) = 00, 01, 10."
           data-command="tnsample --mode gadgets --seed 2026 --out runs/gadgets  then  python scripts/plot_gates.py">
      <figcaption><b>gates.png</b> &mdash; every gate on its floor, and three states of the joined
        circuit. <code>runs/gadgets</code></figcaption>
    </figure>
    <figure class="fig">
      <img src="assets/week11/factoring.png" alt="Contraction cost and sampled energies of factor 15"
           data-caption="(a) Planned space complexity sc against the factor bit length k: treesa follows the multiplier's 2k+1 seam, greedy crosses the 8 GB line. (b) The 2000 sampled energies of factor_15 at beta = 2 beside the Boltzmann prediction 2000 g0 exp(-beta E_min)/Z; the ground state decodes to 3 x 5."
           data-command="tnsample --mode plan/treesa for k = 2..8 and  tnsample --mode sample --samples 2000 --beta 2.0  then  python scripts/plot_factoring.py">
      <figcaption><b>factoring.png</b> &mdash; contraction memory against k, and the sampled ground
        state. <code>runs/plan_*</code>, <code>runs/sample_f15</code></figcaption>
    </figure>
    <figure class="fig">
      <img src="assets/week11/nqueens.png" alt="N-queens counts and sampled 6-queens frequencies"
           data-caption="(a) The contracted number of valid boards for N = 4..10 against OEIS A000170 on a log scale: every count matches. (b) The frequencies of the four 6-queens boards among 200 uniform draws, with the expected 50 marked."
           data-command="tnsample --mode count --n N for N = 4..10 and  tnsample --mode sample --n 6 --samples 200  then  python scripts/plot_nqueens.py">
      <figcaption><b>nqueens.png</b> &mdash; the counts against the known sequence, and uniform
        boards. <code>runs/count_nq*</code>, <code>runs/sample_nq6</code></figcaption>
    </figure>
  </div>

  <section class="card table-card">
    <h2>Acceptance against the learning sheet</h2>
    <table id="acceptanceTable">
      <thead><tr><th>Check</th><th>This run</th><th>Reference</th><th>Verdict</th></tr></thead>
      <tbody></tbody>
    </table>
  </section>

  <section class="card table-card">
    <h2>The factoring round trip (pred 0.6.0)</h2>
    <table id="predTable">
      <thead><tr><th>Target</th><th>Bits, lowest first</th><th>Factors</th><th>Check</th>
        <th>Magnet</th></tr></thead>
      <tbody></tbody>
    </table>
    <p class="lead" style="margin-top:10px">For 7 the best state of the magnet decodes to a wrong
      product: a reduction cannot invent an answer, and the check on the original problem exposes
      it. The sheet's example decodes 1 x 3 with Min(4); this run decodes 3 x 2 with Min(1).</p>
  </section>

  <section class="card table-card">
    <h2>Planning the multiplier, and the N-queens sequence</h2>
    <table id="scaleTable">
      <thead><tr><th>Bits k</th><th>Spins</th><th>greedy sc</th><th>treesa sc</th>
        <th>treesa tc</th></tr></thead>
      <tbody></tbody>
    </table>
    <table id="queensTable" style="margin-top:14px">
      <thead><tr><th>Board size N</th><th>4</th><th>5</th><th>6</th><th>7</th><th>8</th><th>9</th>
        <th>10</th></tr></thead>
      <tbody></tbody>
    </table>
  </section>

  <section class="card table-card">
    <h2>What the negative control caught</h2>
    <table id="negativeTable">
      <thead><tr><th>Defect</th><th>What changed</th><th>Caught by</th></tr></thead>
      <tbody></tbody>
    </table>
  </section>
</div>

<div class="lightbox hidden" id="lightbox" role="dialog" aria-modal="true" aria-label="Figure">
  <div class="lb-top">
    <span class="lb-count" id="lbCount"></span>
    <button class="act lb-close" id="lbClose">Close (Esc)</button>
  </div>
  <img id="lbImage" src="" alt="">
  <div class="lb-caption" id="lbCaption"></div>
  <div class="lb-command" id="lbCommand"></div>
  <div class="controls">
    <button class="act" id="lbPrev">&#8592; Previous</button>
    <button class="act" id="lbNext">Next &#8594;</button>
  </div>
</div>
"""

SCRIPT = """
const DATA = __DATA__;

function switchTab(which) {
  const studio = document.getElementById('studio');
  const evidence = document.getElementById('evidence');
  const studioOn = which === 'studio';
  studio.classList.toggle('hidden', !studioOn);
  evidence.classList.toggle('hidden', studioOn);
  document.getElementById('tabStudio').classList.toggle('active', studioOn);
  document.getElementById('tabEvidence').classList.toggle('active', !studioOn);
  document.getElementById('tabStudio').setAttribute('aria-selected', studioOn ? 'true' : 'false');
  document.getElementById('tabEvidence').setAttribute('aria-selected', studioOn ? 'false' : 'true');
  location.hash = studioOn ? 'studio' : 'evidence';
  requestAnimationFrame(() => redrawAll());
}

function setupTabs() {
  document.getElementById('tabStudio').addEventListener('click', () => switchTab('studio'));
  document.getElementById('tabEvidence').addEventListener('click', () => switchTab('evidence'));
  document.addEventListener('keydown', (event) => {
    if (event.key === 'ArrowLeft' || event.key === 'Home') { switchTab('studio'); }
    if (event.key === 'ArrowRight' || event.key === 'End') { switchTab('evidence'); }
  });
  document.getElementById('tabStudio').setAttribute('role', 'tab');
  document.getElementById('tabEvidence').setAttribute('role', 'tab');
  document.querySelector('.tab-buttons').setAttribute('role', 'tablist');
  if (location.hash === '#evidence') switchTab('evidence');
}

// ---------------------------------------------------------------- panel 1
const gateOrder = ['AND', 'OR', 'NOT', 'COPY', 'SET0', 'SET1', 'XOR'];
let selectedGate = 'AND';
let gateAnim = null;

function drawGate(name) {
  const canvas = document.getElementById('gatesCanvas');
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 520;
  const height = 300;
  canvas.width = width * ratio; canvas.height = height * ratio;
  const context = canvas.getContext('2d');
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  const gate = DATA.gates[name];
  const rows = gate.rows;
  const energies = rows.map((row) => row[1]);
  const low = Math.min(...energies) - 1.4;
  const high = Math.max(...energies) + 1.4;
  const y = (energy) => height - 26 - ((energy - low) / (high - low)) * (height - 52);
  context.strokeStyle = '#2c3650'; context.lineWidth = 1;
  for (let energy = Math.ceil(low); energy <= Math.floor(high); energy += 1) {
    context.beginPath(); context.moveTo(34, y(energy)); context.lineTo(width - 8, y(energy));
    context.stroke();
    context.fillStyle = '#64748b'; context.font = '10px ui-monospace, monospace';
    context.fillText(String(energy), 6, y(energy) + 3);
  }
  const floor = gate.floor;
  context.strokeStyle = '#e8edf6'; context.lineWidth = 1.6;
  context.beginPath(); context.moveTo(34, y(floor)); context.lineTo(width - 8, y(floor));
  context.stroke();
  context.fillStyle = '#93a1b8'; context.font = '11px system-ui, sans-serif';
  context.fillText('floor ' + floor, width - 92, y(floor) - 5);
  const span = width - 60;
  rows.forEach((row, index) => {
    const x = 44 + (index * span) / Math.max(1, rows.length - 1);
    const allowed = row[2];
    context.beginPath();
    context.arc(x, y(row[1]), 5.4, 0, Math.PI * 2);
    context.fillStyle = allowed ? '#6ea8ff' : '#f87171';
    context.globalAlpha = allowed ? 1 : 0.75;
    context.fill(); context.globalAlpha = 1;
    context.fillStyle = allowed ? '#e8edf6' : '#f87171';
    context.font = '9px ui-monospace, monospace';
    context.save();
    context.translate(x, height - 10); context.rotate(-Math.PI / 2.6);
    context.fillText(row[0], 0, 0);
    context.restore();
  });
  document.getElementById('gateName').textContent = name;
  document.getElementById('gateFloor').textContent = String(floor);
  document.getElementById('gateAllowed').textContent =
    rows.filter((row) => row[2]).length + ' of ' + rows.length;
  const forbidden = rows.filter((row) => !row[2]).map((row) => row[1]);
  document.getElementById('gateForbidden').textContent =
    forbidden.length ? '+' + Math.min(...forbidden.map((value) => value - floor)) : 'none';
  document.querySelectorAll('#gateChips button').forEach((button) => {
    button.classList.toggle('on', button.dataset.gate === name);
  });
  document.getElementById('gateNote').textContent =
    name === 'XOR'
      ? 'XOR needs the helper spin w: the floor -4 forces x + y = z + 2r, so the penalty is 2(x + y - z - 2r)^2.'
      : 'The allowed rows sit exactly on the floor; every forbidden row is at least 2 higher.';
}

function setupGates() {
  const chips = document.getElementById('gateChips');
  gateOrder.forEach((name) => {
    const button = document.createElement('button');
    button.className = 'chip' + (name === selectedGate ? ' on' : '');
    button.textContent = name;
    button.dataset.gate = name;
    button.addEventListener('click', () => { stopGateAnim(); selectedGate = name; drawGate(name); });
    chips.appendChild(button);
  });
  document.getElementById('gatesPlay').addEventListener('click', playGates);
  document.getElementById('gatesStep').addEventListener('click', () => {
    stopGateAnim();
    selectedGate = gateOrder[(gateOrder.indexOf(selectedGate) + 1) % gateOrder.length];
    drawGate(selectedGate);
  });
  document.getElementById('gatesReset').addEventListener('click', () => {
    stopGateAnim(); selectedGate = 'AND'; drawGate('AND');
  });
  drawGate(selectedGate);
}

function stopGateAnim() {
  if (gateAnim) { clearInterval(gateAnim); gateAnim = null; }
}

function playGates() {
  stopGateAnim();
  let index = 0;
  selectedGate = gateOrder[0]; drawGate(selectedGate);
  gateAnim = setInterval(() => {
    index += 1;
    if (index >= gateOrder.length) { stopGateAnim(); return; }
    selectedGate = gateOrder[index];
    drawGate(selectedGate);
  }, 900);
}

// ---------------------------------------------------------------- panel 2
const entries = DATA.contraction.entries;
const kinds = DATA.contraction.kinds;
let contractionStep = entries.length;
let contractionAnim = null;

function drawContraction(step) {
  const canvas = document.getElementById('contractCanvas');
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 520;
  const height = 260;
  canvas.width = width * ratio; canvas.height = height * ratio;
  const context = canvas.getContext('2d');
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  const left = 34, right = width - 8, top = 14, bottom = height - 30;
  context.strokeStyle = '#2c3650';
  context.beginPath(); context.moveTo(left, top); context.lineTo(left, bottom);
  context.lineTo(right, bottom); context.stroke();
  const maximum = Math.log2(Math.max(...entries)) + 0.5;
  const barWidth = (right - left) / entries.length;
  entries.forEach((count, index) => {
    const done = index < step;
    const bits = Math.log2(count);
    const barHeight = (bits / maximum) * (bottom - top);
    context.fillStyle = done ? (kinds[index] === 'term' ? '#3b82f6' : '#6ea8ff') : '#1f2b45';
    context.fillRect(left + index * barWidth, bottom - barHeight, Math.max(1, barWidth * 0.9), barHeight);
  });
  const cursor = left + (step / entries.length) * (right - left);
  context.strokeStyle = '#f0b849'; context.lineWidth = 1.4;
  context.beginPath(); context.moveTo(cursor, top); context.lineTo(cursor, bottom); context.stroke();
  context.fillStyle = '#93a1b8'; context.font = '10px ui-monospace, monospace';
  context.fillText('2^5', left + 2, top + 8);
  context.fillText('1', left + 2, bottom - 2);
  context.save();
  context.translate(12, (top + bottom) / 2); context.rotate(-Math.PI / 2);
  context.fillText('entries per intermediate', -70, 0);
  context.restore();
  const shown = Math.max(0, Math.min(step, entries.length) - 1);
  const runningMax = step > 0 ? Math.max(...entries.slice(0, step)) : 0;
  document.getElementById('nodeIndex').textContent =
    step === 0 ? '0' : (shown + 1) + ' / ' + entries.length;
  document.getElementById('nodeEntries').textContent = step === 0 ? '0' : String(entries[shown]);
  document.getElementById('nodeMax').textContent = String(runningMax);
  const finished = step >= entries.length;
  document.getElementById('tropResult').textContent =
    finished ? DATA.contraction.eMin + ' / ' + DATA.contraction.g0 : '\\u2026';
  document.getElementById('contractSlider').value = String(Math.round((step / entries.length) * 100));
}

function setContraction(step) {
  contractionStep = Math.max(0, Math.min(entries.length, step));
  drawContraction(contractionStep);
}

function playContraction() {
  if (contractionAnim) { clearInterval(contractionAnim); contractionAnim = null; }
  setContraction(0);
  const stride = Math.max(1, Math.round(entries.length / 120));
  contractionAnim = setInterval(() => {
    setContraction(contractionStep + stride);
    if (contractionStep >= entries.length) { clearInterval(contractionAnim); contractionAnim = null; }
  }, 40);
}

function setupContraction() {
  document.getElementById('contractPlay').addEventListener('click', playContraction);
  document.getElementById('contractReset').addEventListener('click', () => {
    if (contractionAnim) { clearInterval(contractionAnim); contractionAnim = null; }
    setContraction(0);
  });
  document.getElementById('contractSlider').addEventListener('input', (event) => {
    if (contractionAnim) { clearInterval(contractionAnim); contractionAnim = null; }
    setContraction(Math.round((Number(event.target.value) / 100) * entries.length));
  });
  setContraction(0);
}

// ---------------------------------------------------------------- panel 3
let drawStep = DATA.sampling.energies.length;
let drawAnim = null;

function drawSampling(step) {
  const canvas = document.getElementById('sampleCanvas');
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 520;
  const height = 260;
  canvas.width = width * ratio; canvas.height = height * ratio;
  const context = canvas.getContext('2d');
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  const left = 44, right = width - 10, top = 14, bottom = height - 34;
  context.strokeStyle = '#2c3650';
  context.beginPath(); context.moveTo(left, top); context.lineTo(left, bottom);
  context.lineTo(right, bottom); context.stroke();
  const slices = DATA.sampling.histogram;
  const shown = DATA.sampling.energies.slice(0, step);
  const counts = new Map();
  shown.forEach((energy) => counts.set(energy, (counts.get(energy) || 0) + 1));
  const maximum = Math.max(...slices.map((slice) => slice[1]));
  const barWidth = (right - left) / slices.length;
  slices.forEach((slice, index) => {
    const count = counts.get(slice[0]) || 0;
    const barHeight = (count / maximum) * (bottom - top);
    const x = left + index * barWidth + barWidth * 0.12;
    context.fillStyle = '#3b82f6';
    context.fillRect(x, bottom - barHeight, barWidth * 0.76, barHeight);
    context.fillStyle = '#93a1b8'; context.font = '10px ui-monospace, monospace';
    context.fillText(String(slice[0]), x, bottom + 14);
  });
  context.fillStyle = '#f0b849'; context.font = '11px system-ui, sans-serif';
  context.fillText('predicted ground draws ' + DATA.sampling.predicted.toFixed(0), left + 6, top + 12);
  const ground = counts.get(slices[0][0]) || 0;
  document.getElementById('drawCount').textContent = String(step);
  document.getElementById('drawGround').textContent = String(ground);
  document.getElementById('drawPredicted').textContent = DATA.sampling.predicted.toFixed(1);
  document.getElementById('drawP0').textContent = DATA.sampling.p0.toFixed(6);
  document.getElementById('sampleNote').textContent =
    'Slice shows ' + step + ' draws; the whole run has ' + DATA.sampling.energies.length +
    ' and ' + DATA.sampling.ground + ' at E_min, against the Boltzmann prediction ' +
    DATA.sampling.predicted.toFixed(2) + ' +- ' + DATA.sampling.spread.toFixed(2) + '.';
}

function setDraw(step) {
  drawStep = Math.max(0, Math.min(DATA.sampling.energies.length, step));
  drawSampling(drawStep);
  document.getElementById('sampleSlider').value = String(Math.round((drawStep / DATA.sampling.energies.length) * 100));
}

function playSampling() {
  if (drawAnim) { clearInterval(drawAnim); drawAnim = null; }
  setDraw(0);
  const stride = Math.max(1, Math.round(DATA.sampling.energies.length / 100));
  drawAnim = setInterval(() => {
    setDraw(drawStep + stride);
    if (drawStep >= DATA.sampling.energies.length) { clearInterval(drawAnim); drawAnim = null; }
  }, 45);
}

function setupSampling() {
  document.getElementById('samplePlay').addEventListener('click', playSampling);
  document.getElementById('sampleReset').addEventListener('click', () => {
    if (drawAnim) { clearInterval(drawAnim); drawAnim = null; }
    setDraw(0);
  });
  document.getElementById('sampleSlider').addEventListener('input', (event) => {
    if (drawAnim) { clearInterval(drawAnim); drawAnim = null; }
    setDraw(Math.round((Number(event.target.value) / 100) * DATA.sampling.energies.length));
  });
  setDraw(DATA.sampling.energies.length);
}

// ---------------------------------------------------------------- panel 4
let queenN = 6;
let queenStep = 0;
let queenAnim = null;

function drawBoard(n, board) {
  const canvas = document.getElementById('queenCanvas');
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 520;
  const height = 300;
  canvas.width = width * ratio; canvas.height = height * ratio;
  const context = canvas.getContext('2d');
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  const size = Math.min(width - 40, height - 20);
  const side = size / n;
  const originX = (width - size) / 2;
  const originY = (height - size) / 2;
  for (let row = 0; row < n; row += 1) {
    for (let column = 0; column < n; column += 1) {
      context.fillStyle = (row + column) % 2 === 0 ? '#243452' : '#18243c';
      context.fillRect(originX + column * side, originY + row * side, side + 0.5, side + 0.5);
    }
  }
  board.forEach((column, row) => {
    context.beginPath();
    context.arc(originX + (column + 0.5) * side, originY + (row + 0.5) * side, side * 0.3, 0, Math.PI * 2);
    context.fillStyle = '#f0b849'; context.fill();
  });
  const count = DATA.queens[n].count;
  document.getElementById('queenN').textContent = String(n);
  document.getElementById('queenCount').textContent = String(count);
  document.getElementById('queenDrawn').textContent = String(queenStep + 1) + ' / ' + DATA.queens[n].boards.length;
  const seen = new Set(DATA.queens[n].boards.slice(0, queenStep + 1).map((entry) => entry.join(',')));
  document.getElementById('queenDistinct').textContent = String(seen.size);
  document.querySelectorAll('#queenSizes button').forEach((button) => {
    button.classList.toggle('on', Number(button.dataset.n) === n);
  });
  const expected = count * (1 - Math.pow((count - 1) / count, DATA.queens[n].boards.length));
  document.getElementById('queenNote').textContent =
    'Drawing ' + DATA.queens[n].boards.length + ' uniform boards from a tree that counted ' + count +
    '; in this many draws the expected number of distinct boards is ' + expected.toFixed(1) + '.';
}

function setQueen(n, step) {
  queenN = n; queenStep = Math.max(0, step);
  const boards = DATA.queens[n].boards;
  drawBoard(n, boards[queenStep % boards.length]);
}

function setupQueens() {
  const sizes = Object.keys(DATA.queens).map(Number).sort((a, b) => a - b);
  const holder = document.getElementById('queenSizes');
  sizes.forEach((n) => {
    const button = document.createElement('button');
    button.className = 'chip' + (n === queenN ? ' on' : '');
    button.textContent = 'N = ' + n;
    button.dataset.n = String(n);
    button.addEventListener('click', () => {
      if (queenAnim) { clearInterval(queenAnim); queenAnim = null; }
      setQueen(n, 0);
    });
    holder.appendChild(button);
  });
  document.getElementById('queenPlay').addEventListener('click', () => {
    if (queenAnim) { clearInterval(queenAnim); queenAnim = null; }
    queenStep = 0;
    queenAnim = setInterval(() => {
      queenStep += 1;
      if (queenStep >= DATA.queens[queenN].boards.length) {
        queenStep = DATA.queens[queenN].boards.length - 1;
        clearInterval(queenAnim); queenAnim = null;
      }
      drawBoard(queenN, DATA.queens[queenN].boards[queenStep]);
    }, 140);
  });
  document.getElementById('queenStep').addEventListener('click', () => {
    if (queenAnim) { clearInterval(queenAnim); queenAnim = null; }
    queenStep = (queenStep + 1) % DATA.queens[queenN].boards.length;
    drawBoard(queenN, DATA.queens[queenN].boards[queenStep]);
  });
  document.getElementById('queenReset').addEventListener('click', () => {
    if (queenAnim) { clearInterval(queenAnim); queenAnim = null; }
    queenStep = 0;
    drawBoard(queenN, DATA.queens[queenN].boards[0]);
  });
  setQueen(queenN, 0);
}

// ---------------------------------------------------------------- evidence tables
function fillTables() {
  const rows = DATA.acceptance.rows;
  const body = document.querySelector('#acceptanceTable tbody');
  rows.forEach((row) => {
    const tr = document.createElement('tr');
    tr.innerHTML = '<td>' + row[0] + '</td><td>' + row[1] + '</td><td>' + row[2] +
      '</td><td class="' + (row[3] ? 'ok' : '') + '">' + (row[3] ? 'pass' : 'FAIL') + '</td>';
    body.appendChild(tr);
  });
  const pred = document.querySelector('#predTable tbody');
  DATA.pred.forEach((entry) => {
    const tr = document.createElement('tr');
    tr.innerHTML = '<td>' + entry.target + '</td><td><code>' + entry.bits + '</code></td><td>' +
      entry.factors + '</td><td>' + entry.check + '</td><td>' + entry.magnet + '</td>';
    pred.appendChild(tr);
  });
  const scale = document.querySelector('#scaleTable tbody');
  DATA.scaling.forEach((row) => {
    const tr = document.createElement('tr');
    tr.innerHTML = '<td>' + row[0] + '</td><td>' + row[1] + '</td><td>' + row[2] +
      '</td><td>' + row[3] + '</td><td>' + row[4] + '</td>';
    scale.appendChild(tr);
  });
  const queens = document.querySelector('#queensTable tbody');
  const header = document.createElement('tr');
  header.innerHTML = '<td>contracted</td>' + DATA.queensRow.map((cell) => '<td>' + cell + '</td>').join('');
  queens.appendChild(header);
  const known = document.createElement('tr');
  known.innerHTML = '<td>OEIS A000170</td>' + DATA.oeisRow.map((cell) => '<td>' + cell + '</td>').join('');
  queens.appendChild(known);
  const negative = document.querySelector('#negativeTable tbody');
  DATA.negative.forEach((entry) => {
    const tr = document.createElement('tr');
    tr.innerHTML = '<td><code>' + entry.id + '</code></td><td>' + entry.change + '</td><td>' +
      entry.caught_by + '</td>';
    negative.appendChild(tr);
  });
  document.getElementById('tabEvidence').textContent =
    'Evidence & Verification (' + DATA.figures + ')';
}

// ---------------------------------------------------------------- lightbox
const figures = Array.from(document.querySelectorAll('figure.fig img'));
let lightboxIndex = 0;

function openLightbox(index) {
  lightboxIndex = (index + figures.length) % figures.length;
  const image = figures[lightboxIndex];
  document.getElementById('lbImage').src = image.src;
  document.getElementById('lbImage').alt = image.alt;
  document.getElementById('lbCaption').textContent = image.dataset.caption;
  document.getElementById('lbCommand').textContent = 'Command: ' + image.dataset.command;
  document.getElementById('lbCount').textContent = (lightboxIndex + 1) + ' / ' + figures.length;
  document.getElementById('lightbox').classList.remove('hidden');
  document.getElementById('lbClose').focus();
}

function closeLightbox() { document.getElementById('lightbox').classList.add('hidden'); }

function setupLightbox() {
  figures.forEach((image, index) => image.addEventListener('click', () => openLightbox(index)));
  document.getElementById('lbClose').addEventListener('click', closeLightbox);
  document.getElementById('lbPrev').addEventListener('click', () => openLightbox(lightboxIndex - 1));
  document.getElementById('lbNext').addEventListener('click', () => openLightbox(lightboxIndex + 1));
  document.getElementById('lightbox').addEventListener('click', (event) => {
    if (event.target.id === 'lightbox') closeLightbox();
  });
  document.addEventListener('keydown', (event) => {
    const open = !document.getElementById('lightbox').classList.contains('hidden');
    if (!open) return;
    if (event.key === 'Escape') closeLightbox();
    if (event.key === 'ArrowLeft') openLightbox(lightboxIndex - 1);
    if (event.key === 'ArrowRight') openLightbox(lightboxIndex + 1);
  });
}

function redrawAll() {
  drawGate(selectedGate);
  drawContraction(contractionStep);
  drawSampling(drawStep);
  drawBoard(queenN, DATA.queens[queenN].boards[queenStep % DATA.queens[queenN].boards.length]);
}

window.addEventListener('resize', () => redrawAll());
setupTabs(); setupGates(); setupContraction(); setupSampling(); setupQueens();
setupLightbox(); fillTables();
"""


def build_data() -> dict:
    gadgets = read_run('gadgets')
    gates = {}
    for name, gate in gadgets['gates'].items():
        gates[name] = {
            'floor': float(gate['floor']),
            'rows': [[','.join(str(bit) for bit in row['bits']), float(row['energy']),
                      bool(row['allowed'])] for row in gate['rows']],
        }
    count = read_run('count_f15')
    sample = read_run('sample_f15')
    acceptance = json.loads((EVIDENCE / 'acceptance.json').read_text(encoding='utf-8'))
    reduction = json.loads((EVIDENCE / 'reduction.json').read_text(encoding='utf-8'))
    negatives = json.loads((EVIDENCE / 'negative-controls.json').read_text(encoding='utf-8'))

    queens = {}
    for n in (4, 5, 6, 7, 8, 9, 10):
        count_run = read_run(f'count_nq{n}')
        boards = []
        if n in (6, 8):
            drawn = read_run(f'sample_nq{n}')
            boards = drawn['boards']
        queens[n] = {'count': int(count_run['results']['boards']), 'boards': boards}
    # Only the 6-queens player needs 200 boards; keep the page light.
    queens[6]['boards'] = queens[6]['boards'][:200]
    queens[8]['boards'] = queens[8]['boards'][:60]

    histogram = [[float(row['energy']), int(row['count'])] for row in sample['results']['histogram']]
    energies = [float(value) for value in sample['sample_energies']]
    acceptance_rows = []
    for entry in acceptance['detail']:
        ours = entry['ours']
        sheet = entry['sheet']
        text = lambda value: (f'{value:.4f}'.rstrip('0').rstrip('.')
                              if isinstance(value, float) and abs(value) >= 1e-6
                              else str(value))
        acceptance_rows.append([entry['check'], text(ours), text(sheet) if sheet is not None else '-',
                                bool(entry['ok'])])
    pred = []
    for target in ('15', '6', '7'):
        answer = reduction['factoring'][target]
        pred.append({
            'target': target,
            'bits': ','.join(str(bit) for bit in answer['bits']),
            'factors': f"{answer['factors'][0]} x {answer['factors'][1]} = {answer['product']}",
            'check': answer['evaluation'],
            'magnet': answer['spin_glass_evaluation'] + ' on ' + str(answer['spin_glass_spins']) + ' spins',
        })
    scaling = [[int(k), value['spins'], value['greedy_sc'], value['treesa_sc'],
                f"{value['treesa_tc']:.2f}"] for k, value in acceptance['scaling'].items()]
    negatives_rows = [{'id': entry['id'],
                       'change': entry['change'],
                       'caught_by': ', '.join(entry['caught_by'])} for entry in negatives['defects']]
    return {
        'gates': gates,
        'contraction': {
            'entries': count['results']['node_entries'],
            'kinds': count['results']['node_kinds'],
            'eMin': float(count['results']['e_min']),
            'g0': count['results']['g0'],
            'logZ': float(count['results']['log_z']),
            'sc': count['results']['sc'],
            'tc': round(float(count['results']['tc']), 2),
            'spins': count['network']['spins'],
            'bonds': count['network']['bonds'],
        },
        'sampling': {
            'energies': energies,
            'histogram': histogram,
            'ground': int(sample['results']['ground_draws']),
            'predicted': float(sample['results']['predicted_ground_draws']),
            'spread': float(sample['results']['binomial_spread']),
            'p0': float(sample['results']['p0']),
            'eMin': float(sample['results']['e_min']),
            'beta': float(sample['results']['beta']),
        },
        'queens': queens,
        'queensRow': [acceptance['queens'][str(n)]['boards'] for n in (4, 5, 6, 7, 8, 9, 10)],
        'oeisRow': [acceptance['queens'][str(n)]['sheet'] for n in (4, 5, 6, 7, 8, 9, 10)],
        'acceptance': {'accepted': acceptance['accepted'], 'checks': acceptance['checks'],
                       'rows': acceptance_rows},
        'pred': pred,
        'scaling': scaling,
        'negative': negatives_rows,
        'figures': 3,
    }


def main() -> None:
    data = build_data()
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Week 11 &mdash; hard problems, reductions and tensor-network sampling</title>
<style>{CSS}</style>
</head>
<body>
{NAV}
<header>
  <div class="header-left">
    <h1>Hard Problems, Reductions &amp; Tensor-Network Sampling <span class="badge">Week 11 Interactive</span></h1>
    <div class="sub">Factoring 15 through a 104-spin magnet &middot; gate gadgets &middot; tropical contraction &amp; Boltzmann sampling &middot; N-queens as a tensor network</div>
  </div>
  <div class="tab-buttons">
    <button class="tab-btn active" id="tabStudio" onclick="switchTab('studio')">Interactive Studio</button>
    <button class="tab-btn" id="tabEvidence" onclick="switchTab('evidence')">Evidence &amp; Verification (3)</button>
  </div>
</header>
{STUDIO}
{EVIDENCE_HTML}
<footer>
  AMAT5315 Week 11 &mdash; every number on this page comes from <code>week11/runs/</code>, produced by
  <code>tnsample</code> (Rust, omeco 0.2.6) and <code>pred 0.6.0</code>. The learning sheet is
  <a href="../week11/week11-learning-sheet.pdf">week11-learning-sheet.pdf</a>; the code is in
  <a href="https://github.com/imgrj/AMAT5315-2026Fall-Exercise/tree/main/week11">week11/</a>.
</footer>
<script>{SCRIPT.replace('__DATA__', json.dumps(data, separators=(',', ':')))}</script>
</body>
</html>
"""
    ASSETS.mkdir(parents=True, exist_ok=True)
    for name in ('gates.png', 'factoring.png', 'nqueens.png'):
        shutil.copyfile(ROOT / 'artifacts' / name, ASSETS / name)
    out = DOCS / 'week11-tnsample-lab.html'
    out.write_text(html, encoding='utf-8')
    print(f'wrote {out} ({len(html) / 1024:.1f} KiB) and {len(list(ASSETS.glob("*.png")))} figures')


if __name__ == '__main__':
    main()
