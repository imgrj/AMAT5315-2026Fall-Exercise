"""Deliberate defects: each one must turn a passing acceptance check into a
failing one.

Run from week12/:  python scripts/check_negative_controls.py

The defects are applied to a copy of the saved evidence, never to the artifacts:
they test the checks, which is the only thing a green run can prove. The recorded
result goes to `evidence/negative-controls.json`.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import accept  # noqa: E402

EVIDENCE = Path(__file__).resolve().parents[1] / "evidence"


def flip_offset_sign(data: dict) -> None:
    """Read the square's energy with the constant offset of the wrong sign."""
    square = data["gap"]["square"]
    square["lp_bound"] = -square["lp_bound"]


def disable_bound_pruning(data: dict) -> None:
    """Explore the two children a correct search prunes away."""
    tetra = data["most"]["tetrahedron"]
    tetra["nodes"] = 7
    tetra["lp_solves"] = 7


def wrong_tie_break(data: dict) -> None:
    """Split on a random spin instead of the lowest index: the tree grows."""
    data["most"]["summary"]["mean_nodes"] *= 1.6


def leak_the_validation_split(data: dict) -> None:
    """Train on nodes of the validation graphs."""
    data["model"]["overlap"] = 3


def hide_strong_branching_lookahead(data: dict) -> None:
    """Charge the neural rule for trial LPs it never runs."""
    for inst in data["test"]["instances"]:
        neural = inst["policies"]["neural"]
        neural["lp_solves"] = 2 * neural["nodes"]


def break_the_rust_port(data: dict) -> None:
    """Shift one probe probability past the 1e-4 parity bound."""
    data["test"]["model"]["probe_probability_max_diff"] = 1e-3


def keep_the_relaxed_bonds(data: dict) -> None:
    """Report the LP's bond variables as the nearest-rounded spin state."""
    square = data["gap"]["square"]
    square["rounded_energy"] = square["lp_bound"]


DEFECTS = [
    ("square offset sign flipped", flip_offset_sign, "square LP bound"),
    ("no pruning by bound", disable_bound_pruning, "tetrahedron nodes"),
    ("wrong branching tie-break", wrong_tie_break, "most-frac mean nodes"),
    ("validation instances leak into training", leak_the_validation_split, "instance overlap"),
    ("hidden lookahead in the neural rule", hide_strong_branching_lookahead,
     "LP solves equal nodes (non-Strong)"),
    ("Rust no longer agrees with JAX", break_the_rust_port, "JAX/Rust probe difference"),
    ("rounded state keeps the LP bonds", keep_the_relaxed_bonds,
     "square nearest-rounded energy"),
]


def main() -> int:
    raw = accept.load()
    baseline = {row["check"]: row["pass"] for row in accept.checks(copy.deepcopy(raw))}
    if not all(baseline.values()):
        print("the undamaged evidence already fails; run scripts/run_all.py first")
        return 1

    rows = []
    for name, mutate, check in DEFECTS:
        data = copy.deepcopy(raw)
        mutate(data)
        measured = {row["check"]: row for row in accept.checks(data)}
        caught = not measured[check]["pass"]
        rows.append(
            {
                "defect": name,
                "check": check,
                "baseline": "pass",
                "with_defect": "pass" if measured[check]["pass"] else "fail",
                "caught": caught,
            }
        )
        print(f"{name:<42} -> {check:<36} {'caught' if caught else 'MISSED'}")

    payload = {
        "schema": "week12-negative-controls-v1",
        "summary": {
            "defects": len(rows),
            "caught": sum(1 for row in rows if row["caught"]),
        },
        "controls": rows,
    }
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    (EVIDENCE / "negative-controls.json").write_text(json.dumps(payload, indent=2) + "\n",
                                                     encoding="utf-8")
    missed = [row["defect"] for row in rows if not row["caught"]]
    print(f"\n{payload['summary']['caught']}/{len(rows)} defects caught")
    return 0 if not missed else 1


if __name__ == "__main__":
    sys.exit(main())
