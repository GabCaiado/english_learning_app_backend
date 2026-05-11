"""
Generates a term-spec file for the weakest normalizer terms from the eval report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DATA_DIR = Path("data")
REPORT_PATH = Path("reports/evaluation/latest_normalizer_v3_evaluation.json")
APPROVED_PATH = DATA_DIR / "targeted_normalizer_eval_approved.json"
OUTPUT_PATH = DATA_DIR / "targeted_normalizer_weak_terms.json"
TERM_SPEC_PATHS = [
    DATA_DIR / "targeted_normalizer_discovered_terms.json",
    DATA_DIR / "targeted_normalizer_discovered_terms_batch2.json",
    DATA_DIR / "targeted_normalizer_discovered_terms_batch3.json",
]


def clean(text: Any) -> str:
    return " ".join(str(text or "").strip().split())


def read_json(path: Path) -> Any:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def normalize_example(example: Any) -> dict[str, str] | None:
    if isinstance(example, dict):
        source = clean(example.get("input", ""))
        target = clean(example.get("target", ""))
        sense = clean(example.get("sense", "")).lower()
    elif isinstance(example, list) and len(example) == 3:
        source = clean(example[0])
        target = clean(example[1])
        sense = clean(example[2]).lower()
    else:
        return None
    if not source or not target or sense not in {"literal", "slang"}:
        return None
    return {"input": source, "target": target, "sense": sense}


def load_discovered_specs() -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    for path in TERM_SPEC_PATHS:
        data = read_json(path)
        if isinstance(data, dict):
            items = data.get("terms", [])
        elif isinstance(data, list):
            items = data
        else:
            items = []
        for item in items:
            if not isinstance(item, dict):
                continue
            term = clean(item.get("term", "")).lower()
            examples = [ex for ex in (normalize_example(raw) for raw in item.get("examples", [])) if ex]
            if term and examples:
                specs[term] = {
                    "term": term,
                    "notes": clean(item.get("notes", "")) or "ambiguous literal vs slang or idiomatic use",
                    "examples": examples,
                }
    return specs


def examples_from_approved(term: str, approved_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    examples = []
    seen_senses: set[str] = set()
    for row in approved_rows:
        if clean(row.get("term", "")).lower() != term:
            continue
        example = normalize_example(row)
        if not example or example["sense"] in seen_senses:
            continue
        examples.append(example)
        seen_senses.add(example["sense"])
        if seen_senses == {"literal", "slang"}:
            break
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Build weak normalizer term specs from eval report.")
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--approved", type=Path, default=APPROVED_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--top-terms", type=int, default=60)
    parser.add_argument("--min-errors", type=int, default=4)
    args = parser.parse_args()

    report = read_json(args.report)
    approved = read_json(args.approved)
    if not isinstance(report, dict):
        raise SystemExit(f"{args.report} must contain a JSON object.")
    if not isinstance(approved, list):
        raise SystemExit(f"{args.approved} must contain a JSON list.")

    discovered_specs = load_discovered_specs()
    weak_terms = [
        clean(item.get("term", "")).lower()
        for item in report.get("top_error_terms", [])
        if clean(item.get("term", "")) and int(item.get("errors", 0)) >= args.min_errors
    ][: args.top_terms]

    output_specs = []
    for term in weak_terms:
        spec = discovered_specs.get(term)
        if spec:
            output_specs.append(spec)
            continue
        examples = examples_from_approved(term, approved)
        senses = {example["sense"] for example in examples}
        if senses != {"literal", "slang"}:
            continue
        output_specs.append({
            "term": term,
            "notes": "ambiguous literal vs slang or idiomatic use; generate fresh variants, not copies of the seed examples",
            "examples": examples,
        })

    write_json(args.output, output_specs)
    print(f"Wrote {len(output_specs)} weak term specs to {args.output}")
    print(", ".join(spec["term"] for spec in output_specs))


if __name__ == "__main__":
    main()
