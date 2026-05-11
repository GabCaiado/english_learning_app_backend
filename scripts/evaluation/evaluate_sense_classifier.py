"""
Evaluate the slang sense classifier.

Run after training:
  python scripts/evaluation/evaluate_sense_classifier.py
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_DIR = Path("models/slang_sense_classifier")
TEST_DATA_PATH = Path("data/slang_sense_test.json")
REPORT_DIR = Path("reports/evaluation")
MAX_LENGTH = 160
MIN_MACRO_F1 = 0.95
MIN_SLANG_RECALL = 0.93
MAX_LITERAL_FALSE_POSITIVE_RATE = 0.03


HARD_CASES = [
    {"term": "tea", "sentence": "i wanna drink some tea", "slang_meaning": "gossip", "label": 0},
    {"term": "tea", "sentence": "what's the tea", "slang_meaning": "gossip", "label": 1},
    {"term": "fire", "sentence": "the house is on fire", "slang_meaning": "excellent or impressive", "label": 0},
    {"term": "fire", "sentence": "this beat is fire", "slang_meaning": "excellent or impressive", "label": 1},
    {"term": "sick", "sentence": "i feel sick today", "slang_meaning": "excellent or impressive", "label": 0},
    {"term": "sick", "sentence": "that trick was sick", "slang_meaning": "excellent or impressive", "label": 1},
    {"term": "legit", "sentence": "this app is legit", "slang_meaning": "excellent, real, or credible depending on context", "label": 0},
    {"term": "legit", "sentence": "that's legit", "slang_meaning": "excellent, real, or credible depending on context", "label": 1},
    {"term": "chill", "sentence": "the weather is chill today", "slang_meaning": "relaxed or easygoing", "label": 0},
    {"term": "chill", "sentence": "she's super chill", "slang_meaning": "relaxed or easygoing", "label": 1},
    {"term": "cooked", "sentence": "the pasta was cooked well", "slang_meaning": "in serious trouble or likely to fail", "label": 0},
    {"term": "cooked", "sentence": "we're cooked if we miss the deadline", "slang_meaning": "in serious trouble or likely to fail", "label": 1},
    {"term": "serving", "sentence": "the waiter is serving dinner", "slang_meaning": "projecting or giving off a strong vibe", "label": 0},
    {"term": "serving", "sentence": "this look is serving confidence", "slang_meaning": "projecting or giving off a strong vibe", "label": 1},
    {"term": "snatched", "sentence": "the thief snatched her bag", "slang_meaning": "very stylish, flattering, or well put together", "label": 0},
    {"term": "snatched", "sentence": "her outfit looks snatched", "slang_meaning": "very stylish, flattering, or well put together", "label": 1},
    {"term": "slayed", "sentence": "the knight slayed the dragon", "slang_meaning": "did very well", "label": 0},
    {"term": "slayed", "sentence": "she slayed that presentation", "slang_meaning": "did very well", "label": 1},
]


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python scripts/build_slang_sense_dataset.py")
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} must contain a non-empty JSON list.")
    return rows


def format_input(row: dict[str, Any]) -> str:
    return (
        f"term: {row['term']} [SEP] "
        f"meaning: {row['slang_meaning']} [SEP] "
        f"sentence: {row['sentence']}"
    )


def predict_rows(model, tokenizer, rows: list[dict[str, Any]], device: torch.device, batch_size: int = 64):
    probabilities: list[float] = []
    texts = [format_input(row) for row in rows]
    model.eval()
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[:, 1]
        probabilities.extend(probs.detach().cpu().numpy().tolist())
    probs_np = np.array(probabilities)
    return (probs_np >= 0.5).astype(int), probs_np


def metrics_for_rows(rows: list[dict[str, Any]], preds: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    labels = np.array([int(row["label"]) for row in rows])
    report = classification_report(labels, preds, output_dict=True, zero_division=0)

    term_metrics = {}
    for term in sorted({row["term"] for row in rows}):
        indices = np.array([idx for idx, row in enumerate(rows) if row["term"] == term])
        term_labels = labels[indices]
        term_preds = preds[indices]
        literal_indices = term_labels == 0
        slang_indices = term_labels == 1
        term_metrics[term] = {
            "term_sense_accuracy": float((term_preds == term_labels).mean()),
            "false_positive_rate_on_literal": float((term_preds[literal_indices] == 1).mean()) if literal_indices.any() else 0.0,
            "false_negative_rate_on_slang": float((term_preds[slang_indices] == 0).mean()) if slang_indices.any() else 0.0,
            "num_examples": int(len(indices)),
        }

    return {
        "term_sense_accuracy": float((preds == labels).mean()),
        "macro_f1": float(report.get("macro avg", {}).get("f1-score", 0.0)),
        "false_positive_rate_on_literal": float((preds[labels == 0] == 1).mean()) if (labels == 0).any() else 0.0,
        "false_negative_rate_on_slang": float((preds[labels == 1] == 0).mean()) if (labels == 1).any() else 0.0,
        "term_metrics": term_metrics,
        "sample_errors": [
            {
                "term": rows[idx]["term"],
                "sentence": rows[idx]["sentence"],
                "expected": int(labels[idx]),
                "predicted": int(preds[idx]),
                "slang_probability": float(probs[idx]),
                "example_type": rows[idx].get("example_type"),
            }
            for idx in np.where(preds != labels)[0][:25]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate slang sense classifier.")
    parser.add_argument("--model-dir", default=str(MODEL_DIR))
    parser.add_argument("--test-data", default=str(TEST_DATA_PATH))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        use_fast=True,
        extra_special_tokens={},
    )
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)

    test_rows = read_rows(Path(args.test_data))
    test_preds, test_probs = predict_rows(model, tokenizer, test_rows, device)
    hard_preds, hard_probs = predict_rows(model, tokenizer, HARD_CASES, device, batch_size=8)
    test_metrics = metrics_for_rows(test_rows, test_preds, test_probs)
    hard_passed = bool(np.all(hard_preds == np.array([int(row["label"]) for row in HARD_CASES])))
    slang_recall = 1.0 - test_metrics["false_negative_rate_on_slang"]
    promotion_passed = (
        test_metrics["macro_f1"] >= MIN_MACRO_F1
        and slang_recall >= MIN_SLANG_RECALL
        and test_metrics["false_positive_rate_on_literal"] <= MAX_LITERAL_FALSE_POSITIVE_RATE
        and hard_passed
    )

    report = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "model_dir": args.model_dir,
        "promotion_gates": {
            "min_macro_f1": MIN_MACRO_F1,
            "min_slang_recall": MIN_SLANG_RECALL,
            "max_literal_false_positive_rate": MAX_LITERAL_FALSE_POSITIVE_RATE,
            "hard_cases_must_all_pass": True,
            "passed": promotion_passed,
        },
        "test": test_metrics,
        "hard_cases": [
            {
                "term": row["term"],
                "sentence": row["sentence"],
                "expected": int(row["label"]),
                "predicted": int(hard_preds[idx]),
                "slang_probability": float(hard_probs[idx]),
                "ok": int(hard_preds[idx]) == int(row["label"]),
            }
            for idx, row in enumerate(HARD_CASES)
        ],
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "latest_slang_sense_evaluation.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved evaluation report to {report_path}")


if __name__ == "__main__":
    main()
