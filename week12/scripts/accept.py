"""The acceptance checks for Week 12, in one place.

Every check compares a saved measurement with a number printed in the learning
sheet. Where the sheet says "yours should be close to ours" (node counts, which
depend only on the instances and the rule) the tolerance is a percentage; where
the sheet says "yours must match" (the square, the diet, the bounds of the
seeded 16-instance benchmark) the tolerance is `1e-3` or exact.

`scripts/check_acceptance.py` writes `evidence/acceptance.json` from these
checks, and `tests/test_sheet.py` runs the same list under pytest.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load() -> dict:
    return {
        "gap": read_json(ARTIFACTS / "gap" / "run.json"),
        "most": read_json(ARTIFACTS / "bnb_most" / "run.json"),
        "strong": read_json(ARTIFACTS / "bnb_strong" / "run.json"),
        "model": read_json(ARTIFACTS / "model.json"),
        "test": read_json(ARTIFACTS / "eval_test" / "run.json"),
        "transfer": read_json(ARTIFACTS / "eval_transfer" / "run.json"),
    }


def near(value: float, target: float, tol: float) -> bool:
    return abs(value - target) <= tol


def policy(run: dict, name: str, field: str = "mean_nodes") -> float:
    return float(run["summary"]["policies"][name][field])


def checks(data: dict | None = None) -> list[dict]:
    d = load() if data is None else data
    gap, most, strong = d["gap"], d["most"], d["strong"]
    model, test, transfer = d["model"], d["test"], d["transfer"]

    square = gap["square"]
    worst_gap = max(inst["integrality_gap"] for inst in gap["instances"])
    worst_excess = max(gap["instances"], key=lambda inst: inst["rounding_excess"])
    exact_errors = [
        (run["policy"], inst["id"], inst["error"])
        for run in (most, strong)
        for inst in run["instances"]
    ]
    largest_error = max(err[2] for err in exact_errors)
    policies_test = {k: policy(test, k) for k in ("most-frac", "strong", "neural", "root-neural")}
    policies_transfer = {
        k: policy(transfer, k) for k in ("most-frac", "strong", "neural", "root-neural")
    }

    out: list[dict] = []

    def add(name: str, sheet: str, measured, ok: bool, unit: str = "") -> None:
        out.append({"check": name, "sheet": sheet, "measured": measured, "pass": bool(ok), "unit": unit})

    add("square LP bound", "-4.000000", round(square["lp_bound"], 6),
        near(square["lp_bound"], -4.0, 1e-6))
    add("square LP spins", "(1/2,1/2,1/2,1/2)",
        [round(v, 6) for v in square["lp_spins"]],
        all(near(v, 0.5, 1e-6) for v in square["lp_spins"]))
    add("square exact energy", "-2.000000", round(square["exact_energy"], 6),
        near(square["exact_energy"], -2.0, 1e-6))
    add("square nearest-rounded energy", "+2.000000", round(square["rounded_energy"], 6),
        near(square["rounded_energy"], 2.0, 1e-6))
    add("square rounding excess", "4", round(square["rounded_excess"], 6),
        near(square["rounded_excess"], 4.0, 1e-6))
    add("diet LP optimum", "23.071429 (323/14)", round(gap["diet"]["cost"], 6),
        near(gap["diet"]["cost"], 323.0 / 14.0, 1e-6))

    add("16 x LP bound", "every bound -52.00", round(gap["summary"]["mean_lp_bound"], 4),
        near(gap["summary"]["mean_lp_bound"], -52.0, 1e-3))
    add("16 x exact energy", "mean -38.25", round(gap["summary"]["mean_exact_energy"], 4),
        near(gap["summary"]["mean_exact_energy"], -38.25, 1e-3))
    add("16 x integrality gap", "mean 13.75", round(gap["summary"]["mean_integrality_gap"], 4),
        near(gap["summary"]["mean_integrality_gap"], 13.75, 1e-3))
    add("16 x relative gap", "36.5% of |H*|",
        f"{100 * gap['summary']['mean_relative_gap']:.2f}%",
        near(100 * gap["summary"]["mean_relative_gap"], 36.5, 0.5))
    add("16 x fractional spins", "all 22 spins at 1/2",
        round(gap["summary"]["mean_fractional_spins"], 3),
        near(gap["summary"]["mean_fractional_spins"], 22.0, 1e-9))
    add("16 x suboptimal roundings", "16 of 16", gap["summary"]["suboptimal_roundings"],
        gap["summary"]["suboptimal_roundings"] == 16)
    add("16 x mean rounding excess", "34.25", round(gap["summary"]["mean_rounding_excess"], 4),
        near(gap["summary"]["mean_rounding_excess"], 34.25, 1e-3))
    add("16 x largest rounding excess", "60 on instance 11",
        f"{worst_excess['rounding_excess']:.0f} on instance {worst_excess['id']}",
        near(worst_excess["rounding_excess"], 60.0, 1e-6) and worst_excess["id"] == 11)
    add("worst single integrality gap", "every gap is 52 + H* > 0",
        round(worst_gap, 4),
        all(inst["integrality_gap"] > 0.0 for inst in gap["instances"])
        and near(worst_gap, 52.0 + max(inst["exact_energy"] for inst in gap["instances"]), 1e-9))

    tetra = most["tetrahedron"]
    add("tetrahedron nodes", "5", tetra["nodes"], tetra["nodes"] == 5)
    add("tetrahedron LP bounds", "-6, -4, -3, -3, -2",
        [round(row["bound"], 4) for row in tetra["trace"]],
        all(
            near(row["bound"], want, 1e-6)
            for row, want in zip(tetra["trace"], [-6.0, -4.0, -3.0, -3.0, -2.0])
        ))

    add("48 x 2 exactness", "all match HiGHS (ours < 1e-12)", f"{largest_error:.2e}",
        largest_error < 1e-9, "energy")
    add("most-frac mean nodes", "50.88", round(most["summary"]["mean_nodes"], 4),
        near(most["summary"]["mean_nodes"], 50.88, 0.10 * 50.88))
    add("strong mean nodes", "16.00", round(strong["summary"]["mean_nodes"], 4),
        near(strong["summary"]["mean_nodes"], 16.00, 0.15 * 16.00))
    add("tree shrink", "3.18x smaller",
        round(most["summary"]["mean_nodes"] / strong["summary"]["mean_nodes"], 3),
        near(most["summary"]["mean_nodes"] / strong["summary"]["mean_nodes"], 3.18, 0.8))
    add("strong LP solves per node", "18.0",
        round(strong["summary"]["mean_lp_solves"] / strong["summary"]["mean_nodes"], 3),
        near(strong["summary"]["mean_lp_solves"] / strong["summary"]["mean_nodes"], 18.0, 3.0))
    add("most-frac LP solves per node", "1.0 (one LP per node)",
        round(most["summary"]["mean_lp_solves"] / most["summary"]["mean_nodes"], 3),
        near(most["summary"]["mean_lp_solves"] / most["summary"]["mean_nodes"], 1.0, 1e-9))

    add("training instances", "36", len(model["train"]["instances"]),
        len(model["train"]["instances"]) == 36)
    add("validation instances", "12", len(model["validation"]["instances"]),
        len(model["validation"]["instances"]) == 12)
    add("instance overlap", "0", model["overlap"], model["overlap"] == 0)
    add("training/validation nodes", "276 / 84",
        f"{model['train']['nodes']} / {model['validation']['nodes']}",
        model["train"]["nodes"] >= 200 and model["validation"]["nodes"] >= 60)
    add("best validation cross-entropy", "1.759", round(model["best_val_ce"], 3),
        model["best_val_ce"] < 2.0)
    add("JAX/Rust probe difference", "< 1e-4",
        f"{test['model']['probe_probability_max_diff']:.2e}",
        test["model"]["probe_probability_max_diff"] < 1e-4)
    add("largest HiGHS error (eval)", "< 1e-6 (ours < 3e-13)",
        f"{test['summary']['largest_highs_error']:.2e}",
        test["summary"]["largest_highs_error"] < 1e-6, "energy")
    add("LP solves equal nodes (non-Strong)", "yes",
        "every instance" if all(
            inst["policies"][name]["lp_solves"] == inst["policies"][name]["nodes"]
            for run in (test, transfer)
            for inst in run["instances"]
            for name in ("most-frac", "neural", "root-neural")
        ) else "a rule probes extra LPs",
        all(
            inst["policies"][name]["lp_solves"] == inst["policies"][name]["nodes"]
            for run in (test, transfer)
            for inst in run["instances"]
            for name in ("most-frac", "neural", "root-neural")
        ))
    add("16 unseen: most-frac mean nodes", "50.9", round(policies_test["most-frac"], 3),
        near(policies_test["most-frac"], 50.9, 0.10 * 50.9))
    add("16 unseen: neural mean nodes", "15.4", round(policies_test["neural"], 3),
        near(policies_test["neural"], 15.4, 0.25 * 15.4))
    add("16 unseen: strong mean nodes", "15.6", round(policies_test["strong"], 3),
        near(policies_test["strong"], 15.6, 0.25 * 15.6))
    add("16 unseen: neural beats most-frac", "fewer nodes",
        f"{policies_test['neural']:.2f} < {policies_test['most-frac']:.2f}",
        policies_test["neural"] < policies_test["most-frac"])
    add("16 unseen: neural faster than Strong", "7.15 ms against 122.85 ms",
        f"{policy(test, 'neural', 'mean_total_ms'):.2f} ms against "
        f"{policy(test, 'strong', 'mean_total_ms'):.2f} ms",
        policy(test, "neural", "mean_total_ms") < policy(test, "strong", "mean_total_ms"))

    add("30 spins: most-frac mean nodes", "177.8", round(policies_transfer["most-frac"], 3),
        near(policies_transfer["most-frac"], 177.8, 0.10 * 177.8))
    add("30 spins: strong mean nodes", "38.0", round(policies_transfer["strong"], 3),
        near(policies_transfer["strong"], 38.0, 0.25 * 38.0))
    add("30 spins: neural mean nodes", "39.5", round(policies_transfer["neural"], 3),
        near(policies_transfer["neural"], 39.5, 0.30 * 39.5))
    add("30 spins: root-neural mean nodes", "50.8", round(policies_transfer["root-neural"], 3),
        near(policies_transfer["root-neural"], 50.8, 0.25 * 50.8))
    add("30 spins: both neural rules beat most-frac", "fewer nodes",
        f"neural {policies_transfer['neural']:.1f}, root-neural "
        f"{policies_transfer['root-neural']:.1f} against {policies_transfer['most-frac']:.1f}",
        policies_transfer["neural"] < policies_transfer["most-frac"]
        and policies_transfer["root-neural"] < policies_transfer["most-frac"])
    add("30 spins: Strong Branching is the slow one", "558.31 ms against 106.62 ms",
        f"{policy(transfer, 'strong', 'mean_total_ms'):.2f} ms against "
        f"{policy(transfer, 'most-frac', 'mean_total_ms'):.2f} ms",
        policy(transfer, "strong", "mean_total_ms") > policy(transfer, "most-frac", "mean_total_ms"))
    return out


def summary(rows: list[dict]) -> dict:
    return {
        "checks": len(rows),
        "passed": sum(1 for row in rows if row["pass"]),
        "failed": [row["check"] for row in rows if not row["pass"]],
    }
