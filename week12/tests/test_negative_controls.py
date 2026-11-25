"""The seven deliberate defects of `scripts/check_negative_controls.py`, run as
tests: a green acceptance record must be able to fail.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import accept  # noqa: E402
from check_negative_controls import DEFECTS  # noqa: E402

RAW = accept.load()
BASELINE = {row["check"]: row["pass"] for row in accept.checks(copy.deepcopy(RAW))}


def test_baseline_passes():
    assert all(BASELINE.values())


@pytest.mark.parametrize("defect", DEFECTS, ids=[d[0] for d in DEFECTS])
def test_known_defect_is_caught(defect):
    name, mutate, check = defect
    assert BASELINE[check], f"{check} does not pass on the undamaged evidence"
    data = copy.deepcopy(RAW)
    mutate(data)
    rows = {row["check"]: row for row in accept.checks(data)}
    assert not rows[check]["pass"], f"{name} was not caught by {check}"
