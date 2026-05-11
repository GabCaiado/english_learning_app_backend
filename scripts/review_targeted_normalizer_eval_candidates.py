"""
Review generated targeted normalizer eval candidates.

This is a structural review pass, not a semantic oracle. It approves rows that
are well-formed and flags rows that need a human look before they become locked
evaluation cases.

Run:
  python scripts/review_targeted_normalizer_eval_candidates.py

Then inspect:
  data/targeted_normalizer_eval_review_report.json

If the report looks reasonable, promote the expanded approved file:
  copy data\targeted_normalizer_eval_approved_expanded.json data\targeted_normalizer_eval_approved.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DATA_DIR = Path("data")
DEFAULT_INPUTS = [
    DATA_DIR / "targeted_normalizer_eval_approved.json",
    DATA_DIR / "targeted_normalizer_eval_candidates.json",
    DATA_DIR / "targeted_normalizer_eval_candidates_batch2.json",
    DATA_DIR / "targeted_normalizer_eval_candidates_batch3.json",
]
DEFAULT_APPROVED_OUTPUT = DATA_DIR / "targeted_normalizer_eval_approved_expanded.json"
DEFAULT_REJECTED_OUTPUT = DATA_DIR / "targeted_normalizer_eval_rejected.json"
DEFAULT_REPORT_OUTPUT = DATA_DIR / "targeted_normalizer_eval_review_report.json"

BAD_WRAPPER_PREFIXES = (
    "honestly, ",
    "honestly ",
    "i think ",
    "everyone said ",
    "people online said ",
    "the comments agreed",
)

BAD_WRAPPER_FRAGMENTS = (
    "the comments agreed",
    "people online said",
    "everyone said",
)


def clean(text: Any) -> str:
    return " ".join(str(text or "").strip().split())


def normalize_for_match(text: Any) -> str:
    return clean(text).lower()


def read_json_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list.")
    return [item for item in data if isinstance(item, dict)]


def has_bad_wrapper(text: str) -> bool:
    lower = normalize_for_match(text)
    return lower.startswith(BAD_WRAPPER_PREFIXES) or any(fragment in lower for fragment in BAD_WRAPPER_FRAGMENTS)


def classify_row(row: dict[str, Any], repeat: int = 1) -> tuple[dict[str, Any] | None, list[str]]:
    source = clean(row.get("input", ""))
    target = clean(row.get("target", ""))
    term = clean(row.get("term", "")).lower()
    sense = clean(row.get("sense", "")).lower()
    source_type = clean(row.get("source", "openai_targeted_eval")) or "openai_targeted_eval"
    reasons: list[str] = []

    if not source or not target:
        reasons.append("missing input or target")
    if not term:
        reasons.append("missing term")
    if sense not in {"literal", "slang"}:
        reasons.append("invalid sense")
    if source and target and has_bad_wrapper(source):
        reasons.append("wrapper artifact in input")
    if source and target and has_bad_wrapper(target):
        reasons.append("wrapper artifact in target")
    if sense == "literal" and normalize_for_match(source) != normalize_for_match(target):
        reasons.append("literal row rewrites target")
    if sense == "slang" and normalize_for_match(source) == normalize_for_match(target):
        reasons.append("slang row is identity")
    if source and target and len(target) > max(60, int(len(source) * 2.0)):
        reasons.append("target too long")
    if "(" in term or ")" in term:
        reasons.append("term label contains parentheses")

    normalized = {
        "input": source,
        "target": target,
        "term": term,
        "sense": sense,
        "source": source_type,
    }
    if repeat > 1:
        normalized["repeat"] = repeat
    return normalized, reasons


def dedupe_and_review(rows: list[dict[str, Any]], repeat: int = 1) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    approved: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    targets_by_input_and_sense: dict[tuple[str, str], str] = {}
    seen_rows: set[tuple[str, str, str]] = set()

    for row in rows:
        normalized, reasons = classify_row(row, repeat)
        if normalized is None:
            continue

        row_key = (
            normalize_for_match(normalized["input"]),
            normalize_for_match(normalized["target"]),
            normalized["sense"],
        )
        input_key = (normalize_for_match(normalized["input"]), normalized["sense"])
        existing_target = targets_by_input_and_sense.get(input_key)
        if existing_target is not None and existing_target != normalize_for_match(normalized["target"]):
            reasons.append("conflicting target for same input and sense")

        if row_key in seen_rows:
            reasons.append("duplicate row")

        if reasons:
            rejected.append({**normalized, "reasons": reasons})
            continue

        targets_by_input_and_sense[input_key] = normalize_for_match(normalized["target"])
        seen_rows.add(row_key)
        approved.append(normalized)

    return approved, rejected


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def build_report(input_paths: list[Path], approved: list[dict[str, Any]], rejected: list[dict[str, Any]]) -> dict[str, Any]:
    approved_by_sense = Counter(row["sense"] for row in approved)
    rejected_reasons = Counter(reason for row in rejected for reason in row["reasons"])
    terms = sorted({row["term"] for row in approved})
    term_counts = Counter(row["term"] for row in approved)
    sense_by_term: dict[str, Counter[str]] = defaultdict(Counter)
    for row in approved:
        sense_by_term[row["term"]][row["sense"]] += 1

    thin_terms = [
        {
            "term": term,
            "literal": sense_by_term[term]["literal"],
            "slang": sense_by_term[term]["slang"],
        }
        for term in terms
        if sense_by_term[term]["literal"] == 0 or sense_by_term[term]["slang"] == 0
    ]

    return {
        "input_files": [str(path) for path in input_paths if path.exists()],
        "approved_rows": len(approved),
        "rejected_rows": len(rejected),
        "approved_by_sense": dict(sorted(approved_by_sense.items())),
        "unique_terms": len(terms),
        "top_terms": term_counts.most_common(25),
        "thin_terms": thin_terms[:100],
        "rejected_reasons": dict(rejected_reasons.most_common()),
        "rejected_preview": rejected[:50],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Review targeted normalizer eval candidates.")
    parser.add_argument("--input", nargs="*", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--approved-output", type=Path, default=DEFAULT_APPROVED_OUTPUT)
    parser.add_argument("--rejected-output", type=Path, default=DEFAULT_REJECTED_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument("--repeat", type=int, default=1, help="Add this repeat count to approved rows for training-only files.")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for path in args.input:
        rows.extend(read_json_rows(path))

    approved, rejected = dedupe_and_review(rows, max(1, args.repeat))
    report = build_report(args.input, approved, rejected)

    write_json(args.approved_output, approved)
    write_json(args.rejected_output, rejected)
    write_json(args.report_output, report)

    print(f"Read {len(rows)} rows.")
    print(f"Approved {len(approved)} rows -> {args.approved_output}")
    print(f"Rejected {len(rejected)} rows -> {args.rejected_output}")
    print(f"Wrote report -> {args.report_output}")
    print(json.dumps({
        "approved_by_sense": report["approved_by_sense"],
        "unique_terms": report["unique_terms"],
        "rejected_reasons": report["rejected_reasons"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
