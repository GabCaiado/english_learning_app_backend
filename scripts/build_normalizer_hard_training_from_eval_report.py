"""
Build training-only hard cases from the latest targeted evaluation report.

This uses the terms with the most evaluation failures, then copies rows for
those terms from the approved targeted eval file into a separate training file.
The dataset builder can include this file in train without changing the locked
approved eval file.

Run:
  python scripts/build_normalizer_hard_training_from_eval_report.py --top-terms 60 --repeat 12
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPORT_PATH = Path("reports/evaluation/latest_normalizer_v3_evaluation.json")
APPROVED_PATH = Path("data/targeted_normalizer_eval_approved.json")
OUTPUT_PATH = Path("data/targeted_normalizer_training_hard_cases.json")


def clean(text: Any) -> str:
    return " ".join(str(text or "").strip().split())


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def normalize_row(row: dict[str, Any], repeat: int) -> dict[str, Any]:
    return {
        "input": clean(row.get("input", "")),
        "target": clean(row.get("target", "")),
        "term": clean(row.get("term", "")).lower(),
        "sense": clean(row.get("sense", "")).lower(),
        "source": "targeted_eval_failure_training",
        "repeat": repeat,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build hard normalizer training cases from eval report.")
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--approved", type=Path, default=APPROVED_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--top-terms", type=int, default=60)
    parser.add_argument("--min-errors", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=12)
    args = parser.parse_args()

    report = read_json(args.report)
    approved = read_json(args.approved)
    if not isinstance(approved, list):
        raise SystemExit(f"{args.approved} must contain a JSON list.")

    top_terms = [
        item["term"]
        for item in report.get("top_error_terms", [])
        if item.get("term") and int(item.get("errors", 0)) >= args.min_errors
    ][: args.top_terms]
    top_terms_set = set(top_terms)

    rows = []
    seen: set[tuple[str, str, str]] = set()
    for item in approved:
        if not isinstance(item, dict):
            continue
        term = clean(item.get("term", "")).lower()
        if term not in top_terms_set:
            continue
        normalized = normalize_row(item, args.repeat)
        if not normalized["input"] or not normalized["target"] or normalized["sense"] not in {"literal", "slang"}:
            continue
        key = (normalized["input"].lower(), normalized["target"].lower(), normalized["sense"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(normalized)

    write_json(args.output, rows)
    print(f"Selected {len(top_terms)} weak terms from {args.report}:")
    print(", ".join(top_terms))
    print(f"Wrote {len(rows)} training rows to {args.output}")
    print(f"Each row repeat count: {args.repeat}")


if __name__ == "__main__":
    main()
