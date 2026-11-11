"""Shared helpers for the Week 10 drivers, evidence plots and web build."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CRATE = ROOT / 'thermo'
RUNS = ROOT / 'runs'
ARTIFACTS = ROOT / 'artifacts'
TABLES = ARTIFACTS / 'tables'
EVIDENCE = ROOT / 'evidence'

COLORS = {
    'L16': '#7aa2f7',
    'L32': '#f6a45c',
    'L64': '#4ade80',
    'L256': '#38bdf8',
    'L1024': '#e06c75',
    'measured': '#38bdf8',
    'prediction': '#f6a45c',
    'threshold': '#e0af68',
    'onsager': '#a78bfa',
}


def style() -> None:
    plt.rcParams.update({
        'figure.dpi': 130,
        'savefig.dpi': 150,
        'font.size': 9.5,
        'axes.titlesize': 10.5,
        'axes.labelsize': 9.5,
        'axes.facecolor': '#0f172a',
        'figure.facecolor': '#080c16',
        'savefig.facecolor': '#080c16',
        'axes.edgecolor': '#3b4a63',
        'axes.labelcolor': '#dce6f2',
        'text.color': '#dce6f2',
        'xtick.color': '#9aaec5',
        'ytick.color': '#9aaec5',
        'grid.color': '#26344c',
        'legend.facecolor': '#0f172a',
        'legend.edgecolor': '#3b4a63',
    })


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


BINARY: Path | None = None


def binary() -> Path:
    """Path of the installed `thermo` binary, building it if necessary."""
    global BINARY
    if BINARY is not None and BINARY.exists():
        return BINARY
    subprocess.run(['cargo', 'build', '--release'], cwd=CRATE, check=True, capture_output=True)
    metadata = json.loads(subprocess.check_output(
        ['cargo', 'metadata', '--format-version', '1', '--no-deps'], cwd=CRATE, text=True))
    target = Path(metadata['target_directory']) / 'release'
    exe = target / ('thermo.exe' if sys.platform == 'win32' else 'thermo')
    assert exe.exists(), exe
    BINARY = exe
    return exe


def command(method: str, out: Path, **kwargs) -> list[str]:
    args = [str(binary()), '--method', method]
    for name, value in kwargs.items():
        args += [f'--{name}', str(value)]
    args += ['--out', str(out)]
    return args


def run(args: list[str]) -> dict:
    completed = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"{' '.join(args)}\n{completed.stdout}\n{completed.stderr}")
    return read_json(Path(args[-1]) / 'run.json')


def run_many(commands: list[list[str]], workers: int | None = None) -> list[dict]:
    """Run independent `thermo` invocations concurrently; return their run.json."""
    if workers is None:
        workers = max(1, min(8, (os.cpu_count() or 4) - 2))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run, commands))


def save(figure, path: Path, tight: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    print(f'wrote {path.relative_to(ROOT)}')


def run_path(group: str, method: str, length: int, chi: int, k: float) -> Path:
    return RUNS / group / f'{method}-l{length}-chi{chi}-k{k:.6f}'
