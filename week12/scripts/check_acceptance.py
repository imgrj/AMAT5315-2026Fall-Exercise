"""Write `evidence/acceptance.json` and print the acceptance table.

Run from week12/:  python scripts/check_acceptance.py

Exits non-zero when a check fails, so the script doubles as the acceptance gate.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import accept  # noqa: E402

EVIDENCE = Path(__file__).resolve().parents[1] / "evidence"


def main() -> int:
    rows = accept.checks()
    summary = accept.summary(rows)
    payload = {"schema": "week12-acceptance-v1", "summary": summary, "checks": rows}
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    path = EVIDENCE / "acceptance.json"
    path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")

    width = max(len(row["check"]) for row in rows)
    print(f"{'check'.ljust(width)}  {'sheet':<28}  measured  ok")
    for row in rows:
        mark = "PASS" if row["pass"] else "FAIL"
        print(f"{row['check'].ljust(width)}  {str(row['sheet']):<28}  {row['measured']}  {mark}")
    print(f"\n{summary['passed']}/{summary['checks']} checks pass")
    if summary["failed"]:
        print("failed: " + ", ".join(summary["failed"]))
    print(f"wrote {path}")
    return 0 if not summary["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
