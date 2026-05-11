"""
Evaluate slang normalizer candidates on the locked v4 gold set.

This script is for model selection. Do not train on the gold eval file.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.context_resolver import ContextResolver
from app.ml.normalizer import SlangNormalizer
from app.ml.slang_detector import AMBIGUOUS_SLANG, SlangDetector
from app.ml.slang_dictionary import SlangDictionary


GOLD_PATH = Path("data/slang_normalizer_gold_eval_v4.json")
REPORT_PATH = Path("reports/evaluation/normalizer_gold_v4_report.json")
DEFAULT_MODELS = [
    "models/slang_normalizer",
    "models/slang_normalizer_v2",
    "models/slang_normalizer_v3_1",
    "models/slang_normalizer_v3_2",
]

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
    required = {"input", "expected", "term", "sense", "category"}
    for idx, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise ValueError(f"{path} row {idx} missing required keys: {sorted(missing)}")
        if row["sense"] not in {"slang", "literal"}:
            raise ValueError(f"{path} row {idx} has invalid sense: {row['sense']!r}")
    return rows


def normalize_for_match(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"[.!?]+$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def is_match(predicted: str, row: dict[str, Any]) -> bool:
    expected = normalize_for_match(row["expected"])
    predicted_norm = normalize_for_match(predicted)
    if predicted_norm == expected:
        return True
    return predicted_norm in {
        normalize_for_match(item)
        for item in row.get("acceptable", [])
        if isinstance(item, str)
    }


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


class ProductionPipelineNormalizer:
    """
    Isolated production normalization path without Portuguese translation.
    """

    name = "production_pipeline"

    def __init__(self) -> None:
        self.dictionary = SlangDictionary()
        self.detector = SlangDetector()
        self.resolver = ContextResolver()

    def normalize_sentence(self, sentence: str) -> str:
        slangs_found: list[dict[str, Any]] = []
        detector_score = self.detector.predict_score(sentence)
        all_slangs = sorted(self.dictionary.get_all_slangs(), key=len, reverse=True)

        for slang in all_slangs:
            pattern = r"\b" + re.escape(slang) + r"(?:ing|ed|es|s|er)?\b"
            for match in re.finditer(pattern, sentence, flags=re.IGNORECASE):
                slang_info = self.dictionary.lookup(slang)
                if not slang_info:
                    continue

                context_decision = None
                should_normalize = True
                if slang in AMBIGUOUS_SLANG:
                    context_decision = self.resolver.resolve(
                        term=slang,
                        sentence=sentence,
                        detector_score=detector_score,
                        dictionary_has_entry=True,
                        slang_meaning=slang_info.meaning_en or slang_info.normalized,
                    )
                    should_normalize = context_decision.should_normalize

                overlaps_existing = any(
                    match.start() < item["end"] and match.end() > item["start"]
                    for item in slangs_found
                )
                if should_normalize and not overlaps_existing:
                    slangs_found.append(
                        {
                            "start": match.start(),
                            "end": match.end(),
                            "normalized": slang_info.normalized or slang,
                            "original": match.group(),
                            "base_slang": slang,
                            "reason": context_decision.reason if context_decision else "dictionary match",
                        }
                    )

        normalized = sentence
        for item in sorted(slangs_found, key=lambda value: value["start"], reverse=True):
            normalized = normalized[: item["start"]] + item["normalized"] + normalized[item["end"] :]
        return normalized


def evaluate_candidate(name: str, normalizer: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    failures = []
    counts_by_sense: dict[str, Counter] = defaultdict(Counter)
    counts_by_category: dict[str, Counter] = defaultdict(Counter)
    counts_by_term: dict[str, Counter] = defaultdict(Counter)
    failure_types = Counter()

    for row in rows:
        predicted = normalizer.normalize_sentence(row["input"])
        ok = is_match(predicted, row)
        bucket = "passed" if ok else "failed"
        counts_by_sense[row["sense"]][bucket] += 1
        counts_by_category[row["category"]][bucket] += 1
        counts_by_term[row["term"]][bucket] += 1

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

    return {
        "name": name,
        "total": total,
        "passed": passed,
        "failed": len(failures),
        "accuracy": passed / total if total else 0.0,
        "slang_recall": slang_passed / slang_total if slang_total else 0.0,
        "literal_safety": literal_passed / literal_total if literal_total else 0.0,
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
        "top_failed_terms": [
            {
                "term": key,
                "failed": value["failed"],
                "total": value["passed"] + value["failed"],
            }
            for key, value in sorted(
                counts_by_term.items(),
                key=lambda item: (-item[1]["failed"], item[0]),
            )
            if value["failed"]
        ][:30],
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate normalizer candidates on v4 gold set.")
    parser.add_argument("--gold-path", default=str(GOLD_PATH))
    parser.add_argument("--report-path", default=str(REPORT_PATH))
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--skip-pipeline", action="store_true")
    args = parser.parse_args()

    rows = read_rows(Path(args.gold_path))
    results = []

    for model_path in args.models:
        path = Path(model_path)
        if not path.exists():
            print(f"SKIP {model_path}: missing")
            continue
        print(f"Evaluating {model_path}...")
        results.append(evaluate_candidate(model_path, SlangNormalizer(model_path), rows))

    if not args.skip_pipeline:
        print("Evaluating production_pipeline...")
        results.append(evaluate_candidate("production_pipeline", ProductionPipelineNormalizer(), rows))

    report = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "gold_path": args.gold_path,
        "num_examples": len(rows),
        "promotion_gates": {
            "min_accuracy": 0.90,
            "min_slang_recall": 0.90,
            "min_literal_safety": 0.97,
            "max_bad_outputs": 0,
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
            f"failures={result['failed']}"
        )
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
