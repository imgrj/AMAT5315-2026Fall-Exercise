"""The learning sheet's acceptance numbers, run under pytest.

The checks live in `scripts/accept.py` so the JSON evidence and the test suite
cannot drift apart.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import accept  # noqa: E402

ROWS = accept.checks()


def test_every_check_ran():
    assert len(ROWS) >= 40


@pytest.mark.parametrize("row", ROWS, ids=[row["check"] for row in ROWS])
def test_acceptance(row):
    assert row["pass"], f"{row['check']}: sheet {row['sheet']}, measured {row['measured']}"
