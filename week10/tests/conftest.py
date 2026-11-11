"""Shared fixtures: the built `thermo` tool, its runner and cached runs."""
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


@pytest.fixture(scope='session')
def binary() -> Path:
    crate = ROOT / 'thermo'
    subprocess.run(['cargo', 'build', '--release'], cwd=crate, check=True, capture_output=True)
    metadata = json.loads(subprocess.check_output(
        ['cargo', 'metadata', '--format-version', '1', '--no-deps'], cwd=crate, text=True))
    exe = Path(metadata['target_directory']) / 'release'
    exe = exe / ('thermo.exe' if sys.platform == 'win32' else 'thermo')
    assert exe.exists(), exe
    return exe


class Runner:
    """Run `thermo`, keep the command line, and load the run's JSON evidence."""

    def __init__(self, binary: Path) -> None:
        self.binary = binary
        self.commands: list[list[str]] = []

    def __call__(self, *args: str) -> subprocess.CompletedProcess:
        command = [str(self.binary), *args]
        self.commands.append(command)
        return subprocess.run(command, cwd=ROOT, capture_output=True, text=True)

    def run(self, method: str, out: Path, **kwargs) -> dict:
        self.binary.exists()
        args = ['--method', method]
        for name, value in kwargs.items():
            args += [f'--{name}', str(value)]
        out.mkdir(parents=True, exist_ok=True)
        completed = self(*args, '--out', str(out))
        assert completed.returncode == 0, f'{completed.args}\n{completed.stdout}\n{completed.stderr}'
        return json.loads((out / 'run.json').read_text(encoding='utf-8'))


@pytest.fixture(scope='session')
def thermo(binary: Path) -> Runner:
    return Runner(binary)


@pytest.fixture(scope='session')
def svd_run(thermo: Runner) -> tuple[Path, dict]:
    out = ARTIFACTS / 'tests' / 'svd'
    record = thermo.run('svd', out, l=512, seed=2026, ranks='16,64,128,256,512')
    return out, record
