"""
Colab-friendly gold evaluator for one or more slang normalizer model folders.

This version only depends on transformers/torch/numpy and does not import the
backend app package. Use it after training v4 in Colab.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


GOLD_PATH = Path("data/slang_normalizer_gold_eval_v4.json")
REPORT_PATH = Path("reports/evaluation/normalizer_model_gold_v4_report.json")
TASK_PREFIX = "normalize slang to standard English: "
MAX_INPUT_LENGTH = 160
MAX_TARGET_LENGTH = 160
MIN_ACCURACY = 0.92
MIN_SLANG_RECALL = 0.92
MIN_LITERAL_SAFETY = 0.98
MAX_BAD_OUTPUTS = 0

BAD_FRAGMENTS = [
    "blackjack",
    "bookmark",
    "clinician",
    "dealers",
    "jerusalem",
    "marketers",
    "synchronous",
    "did extremely well after the party",
    "doing extremely well about",
]


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} must contain a non-empty JSON list.")
    return rows


def normalize_for_match(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"[.!?]+$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def is_match(predicted: str, row: dict[str, Any]) -> bool:
    predicted_norm = normalize_for_match(predicted)
    expected = normalize_for_match(row["expected"])
    acceptable = {
        normalize_for_match(item)
        for item in row.get("acceptable", [])
        if isinstance(item, str)
    }
    return predicted_norm == expected or predicted_norm in acceptable


def has_bad_fragment(predicted: str) -> bool:
    lower = predicted.lower()
    return any(fragment in lower for fragment in BAD_FRAGMENTS)


def has_repeated_token_run(predicted: str, max_run: int = 3) -> bool:
    tokens = normalize_for_match(predicted).split()
    if not tokens:
        return False
    current = tokens[0]
    run = 1
    for token in tokens[1:]:
        if token == current:
            run += 1
            if run >= max_run:
                return True
        else:
            current = token
            run = 1
    return False


def failure_type(predicted: str, row: dict[str, Any]) -> str:
    source = normalize_for_match(row["input"])
    expected = normalize_for_match(row["expected"])
    predicted_norm = normalize_for_match(predicted)
    if has_bad_fragment(predicted) or has_repeated_token_run(predicted):
        return "bad_output"
    if source == expected and predicted_norm != expected:
        return "over_normalized_literal"
    if source != expected and predicted_norm == source:
        return "missed_slang"
    return "wrong_rewrite"


def load_tokenizer(model_dir: str):
    try:
        return AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    except Exception:
        return AutoTokenizer.from_pretrained(model_dir, use_fast=True, extra_special_tokens={})


def generate_predictions(model_dir: str, rows: list[dict[str, Any]], batch_size: int) -> list[str]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)
    model.eval()

    predictions: list[str] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        inputs = tokenizer(
            [TASK_PREFIX + item["input"] for item in batch],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_INPUT_LENGTH,
        ).to(device)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_length=MAX_TARGET_LENGTH,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2,
            )
        predictions.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    return predictions


def evaluate_model(model_dir: str, rows: list[dict[str, Any]], batch_size: int) -> dict[str, Any]:
    predictions = generate_predictions(model_dir, rows, batch_size)
    failures = []
    counts_by_sense: dict[str, Counter] = defaultdict(Counter)
    counts_by_category: dict[str, Counter] = defaultdict(Counter)
    failure_types = Counter()

    for row, predicted in zip(rows, predictions):
        ok = is_match(predicted, row)
        bucket = "passed" if ok else "failed"
        counts_by_sense[row["sense"]][bucket] += 1
        counts_by_category[row["category"]][bucket] += 1
        if not ok:
            kind = failure_type(predicted, row)
            failure_types[kind] += 1
            failures.append(
                {
                    "input": row["input"],
                    "expected": row["expected"],
                    "predicted": predicted,
                    "term": row["term"],
                    "sense": row["sense"],
                    "category": row["category"],
                    "failure_type": kind,
                }
            )

    total = len(rows)
    passed = total - len(failures)
    slang_total = sum(1 for row in rows if row["sense"] == "slang")
    literal_total = sum(1 for row in rows if row["sense"] == "literal")
    slang_passed = counts_by_sense["slang"]["passed"]
    literal_passed = counts_by_sense["literal"]["passed"]

    bad_outputs = failure_types.get("bad_output", 0)
    accuracy = passed / total if total else 0.0
    slang_recall = slang_passed / slang_total if slang_total else 0.0
    literal_safety = literal_passed / literal_total if literal_total else 0.0

    return {
        "name": model_dir,
        "total": total,
        "passed": passed,
        "failed": len(failures),
        "accuracy": accuracy,
        "slang_recall": slang_recall,
        "literal_safety": literal_safety,
        "promotion_passed": (
            accuracy >= MIN_ACCURACY
            and slang_recall >= MIN_SLANG_RECALL
            and literal_safety >= MIN_LITERAL_SAFETY
            and bad_outputs <= MAX_BAD_OUTPUTS
        ),
        "failure_types": dict(failure_types.most_common()),
        "by_sense": {
            key: {
                "passed": value["passed"],
                "failed": value["failed"],
                "total": value["passed"] + value["failed"],
            }
            for key, value in sorted(counts_by_sense.items())
        },
        "by_category": {
            key: {
                "passed": value["passed"],
                "failed": value["failed"],
                "total": value["passed"] + value["failed"],
            }
            for key, value in sorted(counts_by_category.items())
        },
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate normalizer model folders on v4 gold set.")
    parser.add_argument("--gold-path", default=str(GOLD_PATH))
    parser.add_argument("--report-path", default=str(REPORT_PATH))
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    rows = read_rows(Path(args.gold_path))
    results = []
    for model_dir in args.models:
        print(f"Evaluating {model_dir}...")
        results.append(evaluate_model(model_dir, rows, args.batch_size))

    report = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "gold_path": args.gold_path,
        "num_examples": len(rows),
        "promotion_gates": {
            "min_accuracy": MIN_ACCURACY,
            "min_slang_recall": MIN_SLANG_RECALL,
            "min_literal_safety": MIN_LITERAL_SAFETY,
            "max_bad_outputs": MAX_BAD_OUTPUTS,
        },
        "results": results,
    }

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print("\nSUMMARY")
    for result in results:
        print(
            f"{result['name']}: "
            f"accuracy={result['accuracy']:.3f} "
            f"slang_recall={result['slang_recall']:.3f} "
            f"literal_safety={result['literal_safety']:.3f} "
            f"failures={result['failed']} "
            f"promotion_passed={result['promotion_passed']} "
            f"failure_types={result['failure_types']}"
        )
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
