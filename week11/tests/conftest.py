"""Shared fixtures: the built `tnsample` tool, its runner and the stated runs."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
ARTIFACTS = ROOT / 'artifacts'
PRED = ROOT / 'runs' / 'pred'


@pytest.fixture(scope='session')
def binary() -> Path:
    crate = ROOT / 'tnsample'
    subprocess.run(['cargo', 'build', '--release'], cwd=crate, check=True, capture_output=True)
    metadata = json.loads(subprocess.check_output(
        ['cargo', 'metadata', '--format-version', '1', '--no-deps'], cwd=crate, text=True))
    exe = Path(metadata['target_directory']) / 'release'
    exe = exe / ('tnsample.exe' if sys.platform == 'win32' else 'tnsample')
    assert exe.exists(), exe
    return exe


class Runner:
    """Run `tnsample`, keep the command line, and load the run's JSON evidence."""

    def __init__(self, binary: Path) -> None:
        self.binary = binary
        self.commands: list[list[str]] = []

    def __call__(self, *args: str) -> subprocess.CompletedProcess:
        command = [str(self.binary), *args]
        self.commands.append(command)
        return subprocess.run(command, cwd=ROOT, capture_output=True, text=True)

    def run(self, mode: str, out: Path, **flags) -> dict:
        args = ['--mode', mode]
        for name, value in flags.items():
            args += [f'--{name.replace("_", "-")}', str(value)]
        out.mkdir(parents=True, exist_ok=True)
        completed = self(*args, '--out', str(out), '--seed', '2026')
        assert completed.returncode == 0, f'{completed.args}\n{completed.stdout}\n{completed.stderr}'
        return json.loads((out / 'run.json').read_text(encoding='utf-8'))


@pytest.fixture(scope='session')
def tnsample(binary: Path) -> Runner:
    return Runner(binary)


@pytest.fixture(scope='session')
def gadgets_run(tnsample: Runner) -> dict:
    out = ARTIFACTS / 'tests' / 'gadgets'
    out.mkdir(parents=True, exist_ok=True)
    completed = tnsample('--mode', 'gadgets', '--seed', '2026', '--out', str(out))
    assert completed.returncode == 0, completed.stderr
    return json.loads((out / 'run.json').read_text(encoding='utf-8'))


@pytest.fixture(scope='session')
def bundle() -> Path:
    """The `pred` bundle of the sheet's Part 1 (built by scripts/run_pred.py)."""
    path = PRED / 'factor_15.bundle.json'
    if not path.exists():
        pytest.skip(f'{path} is missing: run python scripts/run_pred.py')
    return path
