"""
Evaluate English-to-Portuguese translation quality on sense-sensitive cases.

This gate is intentionally separate from slang normalization. It checks the
Portuguese output after English is already literal or normalized.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.translator import get_translator


DEFAULT_CASES_PATH = BACKEND_ROOT / "data" / "translation_gold_cases.json"
DEFAULT_REPORT_PATH = BACKEND_ROOT / "reports" / "evaluation" / "latest_translation_gold_report.json"


def normalize_for_match(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"[.!?]+$", "", text)
    return " ".join(text.split())


def load_cases(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a JSON list.")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Case {index} must be an object.")
        for key in ("input", "expected_pt", "kind"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Case {index} needs a non-empty {key!r}.")
    return rows


def evaluate_cases(cases_path: Path) -> dict[str, Any]:
    translator = get_translator()
    cases = load_cases(cases_path)
    rows = []
    by_kind: Counter[str] = Counter()
    by_kind_passed: Counter[str] = Counter()

    for case in cases:
        predicted = translator.translate(case["input"])
        ok = normalize_for_match(predicted) == normalize_for_match(case["expected_pt"])
        kind = case["kind"]
        by_kind[kind] += 1
        by_kind_passed[kind] += int(ok)
        rows.append({
            "input": case["input"],
            "expected_pt": case["expected_pt"],
            "predicted_pt": predicted,
            "kind": kind,
            "ok": ok,
        })

    total = len(rows)
    passed = sum(1 for row in rows if row["ok"])
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "accuracy": float(passed / total) if total else 0.0,
        "by_kind": {
            kind: {
                "passed": by_kind_passed[kind],
                "failed": by_kind[kind] - by_kind_passed[kind],
                "total": by_kind[kind],
            }
            for kind in sorted(by_kind)
        },
        "failures": [row for row in rows if not row["ok"]],
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate translation gold cases.")
    parser.add_argument("--cases", default=str(DEFAULT_CASES_PATH))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--min-accuracy", type=float, default=1.0)
    args = parser.parse_args()

    results = evaluate_cases(Path(args.cases))
    promotion_passed = results["accuracy"] >= args.min_accuracy
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "cases_path": args.cases,
        "promotion_gates": {
            "min_accuracy": args.min_accuracy,
            "passed": promotion_passed,
        },
        **results,
    }

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"SUMMARY pass={results['passed']}/{results['total']} accuracy={results['accuracy']:.3f}")
    print(f"Saved translation report to {report_path}")
    if not promotion_passed:
        for failure in results["failures"]:
            print(
                "FAIL "
                f"kind={failure['kind']} input={failure['input']!r} "
                f"expected={failure['expected_pt']!r} predicted={failure['predicted_pt']!r}"
            )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
