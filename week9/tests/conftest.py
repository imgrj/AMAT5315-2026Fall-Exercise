"""Shared fixtures: pinned data, the built `infer` tool and cached runs."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from reference import ROOT, read_uai

ARTIFACTS = ROOT / 'artifacts'

CASES = {
    'prior': {},
    'AD': {0: 0, 7: 0},
    'ADX': {0: 0, 7: 0, 6: 0},
}


def ensure_data() -> None:
    required = [ROOT / 'asia.uai',
                ROOT / 'promedus' / 'Promedus_14.uai',
                ROOT / 'promedus' / 'Promedus_14.uai.evid',
                ROOT / 'promedus' / 'Promedus_14.uai.PR',
                ROOT / 'promedus' / 'Promedus_14.uai.MAR']
    if all(path.exists() for path in required):
        return
    subprocess.run([sys.executable, 'scripts/download_data.py'], cwd=ROOT, check=True)


@pytest.fixture(scope='session')
def data() -> Path:
    ensure_data()
    return ROOT


@pytest.fixture(scope='session')
def binary(data: Path) -> Path:
    subprocess.run(['cargo', '+1.96.0', 'build', '--release'], cwd=ROOT, check=True)
    metadata = subprocess.run(
        ['cargo', '+1.96.0', 'metadata', '--format-version', '1', '--no-deps'],
        cwd=ROOT, check=True, capture_output=True, text=True)
    target = Path(json.loads(metadata.stdout)['target_directory'])
    exe = target / 'release' / ('infer.exe' if sys.platform == 'win32' else 'infer')
    assert exe.exists(), exe
    return exe


class Runner:
    """Run `infer`, keep the command line, and load the run's JSON outputs."""

    def __init__(self, binary: Path):
        self.binary = binary
        self.commands: list[list[str]] = []

    def __call__(self, *args: str) -> subprocess.CompletedProcess:
        command = [str(self.binary), *args]
        self.commands.append(command)
        return subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)

    def json(self, out: Path, name: str = 'run.json') -> dict:
        return json.loads((out / name).read_text())


@pytest.fixture(scope='session')
def infer(binary: Path) -> Runner:
    return Runner(binary)


@pytest.fixture(scope='session')
def asia(infer: Runner) -> dict[str, dict]:
    runs = {}
    for case in CASES:
        out = ARTIFACTS / case
        infer('--model', 'asia.uai', '--evidence', case, '--out', str(out))
        runs[case] = infer.json(out)
    return runs


@pytest.fixture(scope='session')
def asia_model(data: Path):
    return read_uai(ROOT / 'asia.uai')


@pytest.fixture(scope='session')
def tree_model() -> Path:
    """A small nonuniform chain: BP must be exact on a tree."""
    path = ROOT / 'tests' / 'data' / 'tree.uai'
    assert path.exists(), path
    return path
