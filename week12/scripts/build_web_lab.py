"""Build the Week 12 web lab from the saved runs.

From `week12/`:  python scripts/build_web_lab.py

Reads `artifacts/*/run.json`, `artifacts/model.json` and
`evidence/acceptance.json`, injects them into `web/lab_template.html`, copies the
four figures to `docs/assets/week12/`, and writes `docs/week12-branching-lab.html`.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common as common  # noqa: E402

ROOT = common.ROOT
TEMPLATE = ROOT / "web" / "lab_template.html"
DOCS = ROOT.parent / "docs"
ASSETS = DOCS / "assets" / "week12"
TARGET = DOCS / "week12-branching-lab.html"

FIGURES = ["gap.png", "bnb.png", "neural_bnb.png", "eval.png"]


def payload() -> dict:
    gap = common.read_json(common.ARTIFACTS / "gap" / "run.json")
    most = common.read_json(common.ARTIFACTS / "bnb_most" / "run.json")
    strong = common.read_json(common.ARTIFACTS / "bnb_strong" / "run.json")
    model = common.read_json(common.ARTIFACTS / "model.json")
    test = common.read_json(common.ARTIFACTS / "eval_test" / "run.json")
    transfer = common.read_json(common.ARTIFACTS / "eval_transfer" / "run.json")
    acceptance = common.read_json(ROOT / "evidence" / "acceptance.json")
    controls = common.read_json(ROOT / "evidence" / "negative-controls.json")

    def per_instance(run: dict) -> dict:
        return {
            "ids": [inst["id"] for inst in run["instances"]],
            "highs": [inst["highs_energy"] for inst in run["instances"]],
            "policies": {
                name: {
                    "nodes": [inst["policies"][name]["nodes"] for inst in run["instances"]],
                    "lp_solves": [inst["policies"][name]["lp_solves"] for inst in run["instances"]],
                    "total_ms": [inst["policies"][name]["total_ms"] for inst in run["instances"]],
                    "infer_ms": [inst["policies"][name]["infer_ms"] for inst in run["instances"]],
                    "first_splits": [
                        inst["policies"][name]["first_splits"] for inst in run["instances"]
                    ],
                }
                for name in ("most-frac", "strong", "neural", "root-neural")
            },
        }

    return {
        "square": gap["square"],
        "diet": gap["diet"],
        "gap": {
            "ids": [inst["id"] for inst in gap["instances"]],
            "lp": [inst["lp_bound"] for inst in gap["instances"]],
            "exact": [inst["exact_energy"] for inst in gap["instances"]],
            "gap": [inst["integrality_gap"] for inst in gap["instances"]],
            "excess": [inst["rounding_excess"] for inst in gap["instances"]],
            "summary": gap["summary"],
        },
        "tetrahedron": most["tetrahedron"],
        "bnb": {
            "most": {
                "nodes": common.sorted_nodes_bnb(most),
                "lp": sorted(inst["lp_solves"] for inst in most["instances"]),
                "summary": most["summary"],
            },
            "strong": {
                "nodes": common.sorted_nodes_bnb(strong),
                "lp": sorted(inst["lp_solves"] for inst in strong["instances"]),
                "summary": strong["summary"],
            },
        },
        "model": {
            "epoch": model["history"]["epoch"],
            "train_ce": model["history"]["train_ce"],
            "val_ce": model["history"]["val_ce"],
            "best_epoch": model["best_epoch"],
            "best_val_ce": model["best_val_ce"],
            "train": model["train"],
            "validation": model["validation"],
            "overlap": model["overlap"],
        },
        "test": {"summary": test["summary"], **per_instance(test)},
        "transfer": {"summary": transfer["summary"], **per_instance(transfer)},
        "acceptance": {
            "checks": acceptance["summary"]["checks"],
            "passed": acceptance["summary"]["passed"],
            "rows": [
                {"check": row["check"], "sheet": row["sheet"], "measured": row["measured"],
                 "pass": row["pass"]}
                for row in acceptance["checks"]
            ],
        },
        "controls": controls["summary"],
    }


def main() -> None:
    data = payload()
    template = TEMPLATE.read_text(encoding="utf-8")
    html = template.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    TARGET.write_text(html, encoding="utf-8")

    ASSETS.mkdir(parents=True, exist_ok=True)
    for name in FIGURES:
        shutil.copyfile(common.ARTIFACTS / name, ASSETS / name)
    print(f"wrote {TARGET.relative_to(DOCS.parent)}")
    print(f"copied {len(FIGURES)} figures to docs/assets/week12/")


if __name__ == "__main__":
    main()
