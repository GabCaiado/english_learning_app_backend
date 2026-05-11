"""
Evaluate the production slang normalization decision path.

This gate measures the hybrid path the app actually uses:
dictionary candidates -> context guardrails -> sense classifier -> safety rewrites.
It intentionally does not score the raw sense classifier in isolation.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.context_resolver import ContextResolver
from app.ml.slang_detector import SlangDetector
from app.ml.slang_dictionary import SlangDictionary
from scripts.smoke_test_slang_pipeline import (
    DEFAULT_CASES_PATH,
    load_cases,
    normalize_for_match,
    normalize_sentence_with_trace,
)


DEFAULT_REPORT_PATH = BACKEND_ROOT / "reports" / "evaluation" / "latest_slang_pipeline_production_report.json"


def _rate(passed: int, total: int) -> float:
    return float(passed / total) if total else 0.0


def evaluate_cases(cases_path: Path) -> dict[str, Any]:
    dictionary = SlangDictionary()
    detector = SlangDetector()
    resolver = ContextResolver()
    cases = load_cases(cases_path)

    rows = []
    by_kind: Counter[str] = Counter()
    by_kind_passed: Counter[str] = Counter()

    for case in cases:
        trace = normalize_sentence_with_trace(case.text, dictionary, detector, resolver)
        predicted = trace["normalized"]
        ok = normalize_for_match(predicted) == normalize_for_match(case.expected)
        by_kind[case.kind] += 1
        by_kind_passed[case.kind] += int(ok)
        rows.append(
            {
                "input": case.text,
                "expected": case.expected,
                "predicted": predicted,
                "kind": case.kind,
                "ok": ok,
                "detector_score": trace["detector_score"],
                "slangs_found": trace["slangs_found"],
            }
        )

    total = len(rows)
    passed = sum(1 for row in rows if row["ok"])
    slang_total = by_kind["slang"]
    literal_total = by_kind["literal"]
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "accuracy": _rate(passed, total),
        "slang_recall": _rate(by_kind_passed["slang"], slang_total),
        "literal_safety": _rate(by_kind_passed["literal"], literal_total),
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
    parser = argparse.ArgumentParser(description="Evaluate production slang normalization.")
    parser.add_argument("--cases", default=str(DEFAULT_CASES_PATH))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--min-accuracy", type=float, default=0.98)
    parser.add_argument("--min-slang-recall", type=float, default=0.98)
    parser.add_argument("--min-literal-safety", type=float, default=0.98)
    args = parser.parse_args()

    results = evaluate_cases(Path(args.cases))
    promotion_passed = (
        results["accuracy"] >= args.min_accuracy
        and results["slang_recall"] >= args.min_slang_recall
        and results["literal_safety"] >= args.min_literal_safety
    )
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "cases_path": args.cases,
        "promotion_gates": {
            "min_accuracy": args.min_accuracy,
            "min_slang_recall": args.min_slang_recall,
            "min_literal_safety": args.min_literal_safety,
            "passed": promotion_passed,
        },
        **results,
    }

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(
        "SUMMARY "
        f"pass={results['passed']}/{results['total']} "
        f"accuracy={results['accuracy']:.3f} "
        f"slang_recall={results['slang_recall']:.3f} "
        f"literal_safety={results['literal_safety']:.3f}"
    )
    print(f"Saved production report to {report_path}")
    if not promotion_passed:
        for failure in results["failures"]:
            print(
                "FAIL "
                f"kind={failure['kind']} input={failure['input']!r} "
                f"expected={failure['expected']!r} predicted={failure['predicted']!r}"
            )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
