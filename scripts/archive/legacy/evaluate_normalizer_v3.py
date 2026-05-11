"""
Evaluate slang normalizer V3.1.

Run after training:
  python evaluate_normalizer_v3.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


MODEL_DIR = Path("models/slang_normalizer_v3_1")
TEST_DATA_PATH = Path("data/slang_normalizer_v3_1_test.json")
TARGETED_APPROVED_PATH = Path("data/targeted_normalizer_eval_approved.json")
REPORT_DIR = Path("reports/evaluation")
TASK_PREFIX = "normalize slang to standard English: "
MAX_INPUT_LENGTH = 160
MAX_TARGET_LENGTH = 160


HARD_CASES = [
    ("the concert was lit", "the concert was exciting"),
    ("she threw shade at him.", "she subtly insulted him."),
    ("spill the tea.", "share the gossip."),
    ("spill the tea", "share the gossip"),
    ("he spilled tea on the table", "he spilled tea on the table"),
    ("the knight slayed the dragon.", "the knight slayed the dragon."),
    ("can i crash at your place?", "can i sleep at your place?"),
    ("this song is my jam.", "this song is my favorite."),
    ("i'm cooked for the exam.", "i am in trouble for the exam."),
    ("the room has low-key lighting.", "the room has low-key lighting."),
    ("he is a chill guy.", "he is a relaxed guy."),
    ("she is a chill friend", "she is a relaxed friend"),
    ("the weather is chill today.", "the weather is chill today."),
    ("i crashed at my friend's house.", "i slept at my friend's house."),
    ("that beat drop was nasty.", "that beat drop was amazing."),
    ("she ate and left no crumbs.", "she did extremely well."),
    ("i ship those two characters.", "i support those two characters as a couple."),
    ("that last-minute goal was clutch.", "that last-minute goal was decisive."),
    ("i feel sick today", "i feel sick today"),
    ("flex your arm slowly", "flex your arm slowly"),
    ("that trick was sick", "that trick was excellent"),
    ("that car is a flex", "that car is showing off"),
    ("the website is legit", "the website is legit"),
    ("the food was nasty", "the food was nasty"),
    ("she ate and left no crumbs on the plate", "she ate and left no crumbs on the plate"),
    ("that album was mid", "that album was mediocre"),
    ("that performance was mid", "that performance was mediocre"),
    ("facts.", "that is true."),
    ("he said facts", "he said something true"),
    ("they have beef with the coworker", "they have a conflict with the coworker"),
    ("i put jam on my bread.", "i put jam on my bread."),
    ("this song is my jam.", "this song is my favorite."),
    ("the fish was hooked.", "the fish was hooked."),
    ("he is cracked at fortnite.", "he is very good at fortnite."),
    ("that player is washed.", "that player is no longer good."),
    ("the weather is chill today.", "the weather is chill today."),
    ("he is a chill guy.", "he is a relaxed guy."),
    ("these shoes are tight.", "these shoes are tight."),
    ("we're tight.", "we are close friends."),
    ("you look sharp today.", "you look stylish today."),
    ("that guitar solo was nasty.", "that guitar solo was amazing."),
    ("the bathroom smells nasty.", "the bathroom smells nasty."),
    ("i'm cooked for the exam.", "i am in trouble for the exam."),
    ("she ate that performance.", "she did extremely well in that performance."),
    ("the bus is coming.", "the bus is coming."),
]


def read_targeted_hard_cases(path: Path = TARGETED_APPROVED_PATH) -> list[tuple[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        return []
    return [
        (row["input"], row["target"])
        for row in rows
        if isinstance(row, dict) and row.get("input") and row.get("target")
    ]


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python scripts/build_normalizer_v3_dataset.py")
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} must contain a non-empty JSON list.")
    return rows


def normalize_for_match(text: str) -> str:
    return " ".join((text or "").lower().strip().split())


def exact_match_rate(predictions: list[str], references: list[str]) -> float:
    if not predictions:
        return 0.0
    return float(np.mean([normalize_for_match(p) == normalize_for_match(r) for p, r in zip(predictions, references)]))


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_for_match(prediction).split()
    ref_tokens = normalize_for_match(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    ref_counts: dict[str, int] = {}
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    common = 0
    for token in pred_tokens:
        if ref_counts.get(token, 0) > 0:
            common += 1
            ref_counts[token] -= 1
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def generate(tokenizer, model, texts: list[str], device: torch.device, batch_size: int = 16) -> list[str]:
    outputs: list[str] = []
    model.eval()
    for i in range(0, len(texts), batch_size):
        batch = [TASK_PREFIX + text for text in texts[i : i + batch_size]]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_INPUT_LENGTH,
        ).to(device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_length=MAX_TARGET_LENGTH,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2,
            )
        outputs.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate normalizer v3.")
    parser.add_argument("--model-dir", default=str(MODEL_DIR))
    parser.add_argument("--test-data", default=str(TEST_DATA_PATH))
    parser.add_argument("--report-path", default=str(REPORT_DIR / "latest_normalizer_v3_evaluation.json"))
    parser.add_argument("--min-exact-match", type=float, default=0.80)
    parser.add_argument("--max-over-normalization-rate", type=float, default=0.05)
    parser.add_argument("--max-sample-errors", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir).to(device)

    rows = read_rows(Path(args.test_data))
    inputs = [row["input"] for row in rows]
    references = [row["target"] for row in rows]
    predictions = generate(tokenizer, model, inputs, device)

    neutral_indices = [
        idx for idx, (source, reference) in enumerate(zip(inputs, references))
        if normalize_for_match(source) == normalize_for_match(reference)
    ]
    over_normalized = [
        idx for idx in neutral_indices
        if normalize_for_match(predictions[idx]) != normalize_for_match(references[idx])
    ]

    hard_cases = HARD_CASES + read_targeted_hard_cases()
    hard_inputs = [case[0] for case in hard_cases]
    hard_refs = [case[1] for case in hard_cases]
    hard_preds = generate(tokenizer, model, hard_inputs, device, batch_size=8)
    test_exact_match = exact_match_rate(predictions, references)
    test_token_f1 = float(np.mean([token_f1(p, r) for p, r in zip(predictions, references)]))
    test_over_normalization_rate = len(over_normalized) / len(neutral_indices) if neutral_indices else 0.0
    quality_passed = (
        test_exact_match >= args.min_exact_match
        and test_over_normalization_rate <= args.max_over_normalization_rate
    )
    error_indices = [
        idx
        for idx in range(len(rows))
        if normalize_for_match(predictions[idx]) != normalize_for_match(references[idx])
    ]
    errors_by_term = Counter(str(rows[idx].get("term", "")) for idx in error_indices)
    errors_by_sense = Counter(str(rows[idx].get("sense", "")) for idx in error_indices)
    total_by_term = Counter(str(row.get("term", "")) for row in rows)
    term_error_rates = [
        {
            "term": term,
            "errors": errors,
            "total": total_by_term[term],
            "error_rate": errors / total_by_term[term] if total_by_term[term] else 0.0,
        }
        for term, errors in errors_by_term.most_common()
    ]

    report = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "model_dir": args.model_dir,
        "quality_gates": {
            "min_exact_match": args.min_exact_match,
            "max_over_normalization_rate": args.max_over_normalization_rate,
            "passed": quality_passed,
        },
        "test_exact_match": test_exact_match,
        "test_token_f1": test_token_f1,
        "test_over_normalization_rate": test_over_normalization_rate,
        "num_examples": len(rows),
        "num_errors": len(error_indices),
        "errors_by_sense": dict(errors_by_sense.most_common()),
        "top_error_terms": term_error_rates[:50],
        "hard_cases": [
            {
                "input": source,
                "expected": expected,
                "predicted": predicted,
                "ok": normalize_for_match(predicted) == normalize_for_match(expected),
            }
            for source, expected, predicted in zip(hard_inputs, hard_refs, hard_preds)
        ],
        "sample_errors": [
            {
                "input": inputs[idx],
                "expected": references[idx],
                "predicted": predictions[idx],
                "term": rows[idx].get("term"),
                "sense": rows[idx].get("sense"),
            }
            for idx in error_indices
        ][: args.max_sample_errors],
    }

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved evaluation report to {report_path}")


if __name__ == "__main__":
    main()
