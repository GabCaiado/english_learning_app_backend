"""
Robust evaluation for the slang detector and normalizer.

Run after building canonical datasets and training models:
  python scripts/build_training_datasets.py
  python evaluate_models.py
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import evaluate
import numpy as np
import torch
from sklearn.metrics import classification_report, precision_recall_fscore_support
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


DETECTOR_DIR = Path("models/slang_detector")
NORMALIZER_DIR = Path("models/slang_normalizer")
DETECTOR_TEST_PATH = Path("data/master_detector_test.json")
NORMALIZER_TEST_PATH = Path("data/master_normalizer_test.json")
GOLDEN_EVAL_PATH = Path("data/golden_eval.json")
REPORT_DIR = Path("reports/evaluation")
TASK_PREFIX = "normalize slang: "

MANUAL_CASES = [
    {
        "text": "This beat is fire.",
        "expected_slang": True,
        "expected_normalized": "This beat is excellent.",
    },
    {
        "text": "The house is on fire.",
        "expected_slang": False,
        "expected_normalized": "The house is on fire.",
    },
    {
        "text": "That trick was sick.",
        "expected_slang": True,
        "expected_normalized": "That trick was excellent.",
    },
    {
        "text": "I feel sick today.",
        "expected_slang": False,
        "expected_normalized": "I feel sick today.",
    },
    {
        "text": "Spill the tea.",
        "expected_slang": True,
        "expected_normalized": "Share the gossip.",
    },
    {
        "text": "I made tea.",
        "expected_slang": False,
        "expected_normalized": "I made tea.",
    },
]


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    return rows if isinstance(rows, list) else []


def normalize_for_match(text: str) -> str:
    return " ".join((text or "").lower().strip().split())


def exact_match_rate(predictions: list[str], references: list[str]) -> float:
    if not predictions:
        return 0.0
    return float(
        np.mean([
            normalize_for_match(pred) == normalize_for_match(ref)
            for pred, ref in zip(predictions, references)
        ])
    )


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


def load_detector(device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(DETECTOR_DIR, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(DETECTOR_DIR).to(device)
    model.eval()
    return tokenizer, model


def load_normalizer(device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(NORMALIZER_DIR, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(NORMALIZER_DIR).to(device)
    model.eval()
    return tokenizer, model


def predict_detector(tokenizer, model, texts: list[str], device: torch.device, batch_size: int = 64):
    probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            batch_probs = torch.softmax(logits, dim=-1)[:, 1]
        probs.extend(batch_probs.detach().cpu().numpy().tolist())
    predictions = [1 if prob >= 0.5 else 0 for prob in probs]
    return predictions, probs


def evaluate_detector(rows: list[dict], tokenizer, model, device: torch.device) -> dict:
    if not rows:
        return {"error": "no detector rows found"}

    labels = np.array([int(row.get("label", row.get("is_slang", 0))) for row in rows])
    predictions, probabilities = predict_detector(tokenizer, model, [row["text"] for row in rows], device)
    preds_np = np.array(predictions)
    report = classification_report(labels, preds_np, output_dict=True, zero_division=0)

    ambiguous_indices = [
        idx for idx, row in enumerate(rows) if row.get("target_term") not in {None, ""}
    ]
    hard_negative_indices = [
        idx for idx, row in enumerate(rows) if row.get("is_hard_negative") is True
    ]
    neutral_indices = [idx for idx, label in enumerate(labels) if label == 0]

    def macro_f1(indices: list[int]) -> float | None:
        if not indices:
            return None
        _, _, f1, _ = precision_recall_fscore_support(
            labels[indices], preds_np[indices], average="macro", zero_division=0
        )
        return float(f1)

    def false_positive_rate(indices: list[int]) -> float | None:
        if not indices:
            return None
        return float((preds_np[indices] == 1).mean())

    return {
        "accuracy": float(report.get("accuracy", 0.0)),
        "macro_precision": float(report.get("macro avg", {}).get("precision", 0.0)),
        "macro_recall": float(report.get("macro avg", {}).get("recall", 0.0)),
        "macro_f1": float(report.get("macro avg", {}).get("f1-score", 0.0)),
        "weighted_f1": float(report.get("weighted avg", {}).get("f1-score", 0.0)),
        "ambiguous_macro_f1": macro_f1(ambiguous_indices),
        "hard_negative_false_positive_rate": false_positive_rate(hard_negative_indices),
        "neutral_false_positive_rate": false_positive_rate(neutral_indices),
        "num_examples": len(rows),
        "num_ambiguous_examples": len(ambiguous_indices),
        "num_hard_negatives": len(hard_negative_indices),
        "sample_predictions": [
            {
                "text": rows[idx]["text"],
                "expected": int(labels[idx]),
                "predicted": int(preds_np[idx]),
                "slang_probability": float(probabilities[idx]),
            }
            for idx in range(min(10, len(rows)))
        ],
    }


def generate_normalizations(tokenizer, model, texts: list[str], device: torch.device, batch_size: int = 16) -> list[str]:
    outputs = []
    for i in range(0, len(texts), batch_size):
        batch = [TASK_PREFIX + text for text in texts[i : i + batch_size]]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        ).to(device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_length=128,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2,
            )
        outputs.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return outputs


def evaluate_normalizer(rows: list[dict], tokenizer, model, device: torch.device, max_bertscore: int) -> dict:
    if not rows:
        return {"error": "no normalizer rows found"}

    inputs = [row.get("input") or row.get("slang") for row in rows]
    references = [row.get("target") or row.get("formal") for row in rows]
    predictions = generate_normalizations(tokenizer, model, inputs, device)

    neutral_indices = [
        idx for idx, (source, reference) in enumerate(zip(inputs, references))
        if normalize_for_match(source) == normalize_for_match(reference)
    ]
    over_normalized = [
        idx for idx in neutral_indices
        if normalize_for_match(predictions[idx]) != normalize_for_match(references[idx])
    ]

    result = {
        "exact_match": exact_match_rate(predictions, references),
        "token_f1": float(np.mean([token_f1(p, r) for p, r in zip(predictions, references)])),
        "over_normalization_rate": len(over_normalized) / len(neutral_indices) if neutral_indices else 0.0,
        "num_examples": len(rows),
        "num_neutral_examples": len(neutral_indices),
        "sample_predictions": [
            {
                "input": inputs[idx],
                "expected": references[idx],
                "predicted": predictions[idx],
            }
            for idx in range(min(10, len(rows)))
        ],
    }

    if max_bertscore > 0:
        try:
            bertscore = evaluate.load("bertscore")
            sample_predictions = predictions[:max_bertscore]
            sample_references = references[:max_bertscore]
            bert_results = bertscore.compute(
                predictions=sample_predictions,
                references=sample_references,
                model_type="distilbert-base-uncased",
                batch_size=16,
            )
            result["bertscore_f1_sample"] = float(np.mean(bert_results["f1"]))
            result["bertscore_sample_size"] = len(sample_predictions)
        except Exception as exc:
            result["bertscore_error"] = str(exc)

    return result


def evaluate_manual_cases(det_tokenizer, det_model, norm_tokenizer, norm_model, device: torch.device) -> list[dict]:
    texts = [case["text"] for case in MANUAL_CASES]
    det_preds, det_probs = predict_detector(det_tokenizer, det_model, texts, device, batch_size=8)
    norm_preds = generate_normalizations(norm_tokenizer, norm_model, texts, device, batch_size=8)
    results = []
    for idx, case in enumerate(MANUAL_CASES):
        results.append(
            {
                "text": case["text"],
                "expected_slang": case["expected_slang"],
                "predicted_slang": bool(det_preds[idx]),
                "slang_probability": float(det_probs[idx]),
                "expected_normalized": case["expected_normalized"],
                "predicted_normalized": norm_preds[idx],
                "detector_ok": bool(det_preds[idx]) == case["expected_slang"],
                "normalizer_exact": normalize_for_match(norm_preds[idx]) == normalize_for_match(case["expected_normalized"]),
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate slang detector and normalizer.")
    parser.add_argument("--max-rows", type=int, default=0, help="Limit rows for a quick local smoke test.")
    parser.add_argument("--max-bertscore", type=int, default=500, help="Normalizer rows to score with BERTScore. Use 0 to skip.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector_rows = read_rows(DETECTOR_TEST_PATH)
    normalizer_rows = read_rows(NORMALIZER_TEST_PATH)
    golden_rows = read_rows(GOLDEN_EVAL_PATH)
    if args.max_rows > 0:
        detector_rows = detector_rows[: args.max_rows]
        normalizer_rows = normalizer_rows[: args.max_rows]
        golden_rows = golden_rows[: args.max_rows]

    report = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "device": str(device),
        "detector_dir": str(DETECTOR_DIR),
        "normalizer_dir": str(NORMALIZER_DIR),
    }

    det_tokenizer, det_model = load_detector(device)
    norm_tokenizer, norm_model = load_normalizer(device)

    report["detector_test"] = evaluate_detector(detector_rows, det_tokenizer, det_model, device)
    report["detector_golden"] = evaluate_detector(golden_rows, det_tokenizer, det_model, device)
    report["normalizer_test"] = evaluate_normalizer(
        normalizer_rows, norm_tokenizer, norm_model, device, args.max_bertscore
    )
    report["manual_cases"] = evaluate_manual_cases(
        det_tokenizer, det_model, norm_tokenizer, norm_model, device
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "latest_model_evaluation.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved evaluation report to {report_path}")


if __name__ == "__main__":
    main()
