"""
Merge reviewed OpenAI normalizer variants into a new training file.

This script does not mutate the canonical v4 train file. It creates a candidate
training set for the next model experiment, with conflict checks and gold input
leakage protection.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DATA_DIR = Path("data")
DEFAULT_BASE_TRAIN = DATA_DIR / "slang_normalizer_v4_train.json"
DEFAULT_GOLD = DATA_DIR / "slang_normalizer_gold_eval_v4.json"
DEFAULT_OPENAI_APPROVED = DATA_DIR / "feedback_normalizer_openai_approved.json"
DEFAULT_OUTPUT = DATA_DIR / "slang_normalizer_v4_openai_augmented_train.json"
DEFAULT_REPORT = DATA_DIR / "slang_normalizer_v4_openai_augmented_train_report.json"


def clean(text: Any) -> str:
    return " ".join(str(text or "").strip().split())


def norm(text: Any) -> str:
    return clean(text).lower().strip(" .!?")


def read_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    return [row for row in data if isinstance(row, dict)]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def normalize_row(row: dict[str, Any], source: str) -> dict[str, Any] | None:
    input_text = clean(row.get("input", ""))
    target = clean(row.get("target") or row.get("expected") or "")
    term = clean(row.get("term") or row.get("target_term") or "")
    sense = clean(row.get("sense") or row.get("kind") or "").lower()
    if not input_text or not target or sense not in {"literal", "slang"}:
        return None

    normalized: dict[str, Any] = {
        "input": input_text,
        "target": target,
        "term": term,
        "sense": sense,
        "source": source,
    }
    if row.get("source_feedback_id"):
        normalized["source_feedback_id"] = row["source_feedback_id"]
    if row.get("failure_type"):
        normalized["failure_type"] = row["failure_type"]
    return normalized


def validate_training_row(row: dict[str, Any]) -> str | None:
    input_text = clean(row.get("input", ""))
    target = clean(row.get("target", ""))
    sense = row.get("sense", "")
    if not input_text or not target:
        return "missing input or target"
    if sense not in {"literal", "slang"}:
        return "invalid sense"
    if sense == "literal" and norm(input_text) != norm(target):
        return "literal row rewrites target"
    if sense == "slang" and norm(input_text) == norm(target):
        return "slang row is identity"
    if len(target) > max(80, int(len(input_text) * 2.2)):
        return "target too long"
    return None


def add_rows(
    merged: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    repeat: int,
    gold_inputs: set[str],
    targets_by_input: dict[str, str],
) -> tuple[int, Counter[str]]:
    added = 0
    skipped: Counter[str] = Counter()

    for row in rows:
        reason = validate_training_row(row)
        if reason:
            skipped[reason] += 1
            continue

        input_key = norm(row["input"])
        target_key = norm(row["target"])
        if input_key in gold_inputs:
            skipped["gold_input_leakage"] += 1
            continue
        if input_key in targets_by_input and targets_by_input[input_key] != target_key:
            skipped["conflicting_target"] += 1
            continue

        targets_by_input[input_key] = target_key
        for _ in range(max(1, repeat)):
            merged.append(dict(row))
            added += 1

    return added, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an OpenAI-augmented normalizer train file.")
    parser.add_argument("--base-train", type=Path, default=DEFAULT_BASE_TRAIN)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--openai-approved", type=Path, default=DEFAULT_OPENAI_APPROVED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--openai-repeat", type=int, default=1)
    args = parser.parse_args()

    base_rows_raw = read_json_list(args.base_train)
    openai_rows_raw = read_json_list(args.openai_approved)
    gold_inputs = {norm(row.get("input", "")) for row in read_json_list(args.gold)}

    merged: list[dict[str, Any]] = []
    targets_by_input: dict[str, str] = {}
    skipped_total: Counter[str] = Counter()

    base_rows = [
        normalize_row(row, clean(row.get("source", "base_train")) or "base_train")
        for row in base_rows_raw
    ]
    base_rows = [row for row in base_rows if row]
    base_added, base_skipped = add_rows(
        merged,
        base_rows,
        repeat=1,
        gold_inputs=gold_inputs,
        targets_by_input=targets_by_input,
    )
    skipped_total.update({f"base:{key}": value for key, value in base_skipped.items()})

    openai_rows = [
        normalize_row(row, "openai_feedback_variant")
        for row in openai_rows_raw
    ]
    openai_rows = [row for row in openai_rows if row]
    openai_added, openai_skipped = add_rows(
        merged,
        openai_rows,
        repeat=args.openai_repeat,
        gold_inputs=gold_inputs,
        targets_by_input=targets_by_input,
    )
    skipped_total.update({f"openai:{key}": value for key, value in openai_skipped.items()})

    source_counts = Counter(row.get("source", "") for row in merged)
    sense_counts = Counter(row.get("sense", "") for row in merged)
    term_counts = Counter(row.get("term", "") for row in merged)

    report = {
        "base_train": str(args.base_train),
        "openai_approved": str(args.openai_approved),
        "output": str(args.output),
        "base_rows_read": len(base_rows_raw),
        "openai_rows_read": len(openai_rows_raw),
        "base_rows_added": base_added,
        "openai_rows_added": openai_added,
        "total_rows": len(merged),
        "unique_inputs": len(targets_by_input),
        "sense_counts": dict(sorted(sense_counts.items())),
        "top_sources": source_counts.most_common(30),
        "top_terms": term_counts.most_common(40),
        "skipped": dict(skipped_total.most_common()),
        "openai_repeat": max(1, args.openai_repeat),
    }

    write_json(args.output, merged)
    write_json(args.report, report)
    print(f"Wrote {len(merged)} rows to {args.output}")
    print(f"OpenAI rows added: {openai_added}")
    print(f"Report: {args.report}")
    print(json.dumps({
        "sense_counts": report["sense_counts"],
        "skipped": report["skipped"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
