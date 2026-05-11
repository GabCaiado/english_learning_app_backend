"""
Train the slang sense classifier.
Developed for Google Colab T4.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


TRAIN_DATA_PATH = Path("data/slang_sense_train.json")
TEST_DATA_PATH = Path("data/slang_sense_test.json")
OUTPUT_DIR = Path("models/slang_sense_classifier")
REPORT_DIR = Path("reports/training_runs")
SEED = 42
MAX_LENGTH = 160


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python scripts/build_slang_sense_dataset.py")
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} must contain a non-empty JSON list.")
    return rows


def format_input(row: dict[str, Any]) -> str:
    term = str(row["term"]).strip()
    meaning = str(row["slang_meaning"]).strip()
    sentence = str(row["sentence"]).strip()
    return f"term: {term} [SEP] meaning: {meaning} [SEP] sentence: {sentence}"


class SenseDataset(Dataset):
    def __init__(self, encodings, labels: list[int]):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self) -> int:
        return len(self.labels)


class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        if self.class_weights is None:
            loss = outputs.get("loss")
        else:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(model.device))
            loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def latest_checkpoint(results_dir: Path) -> str | None:
    if not results_dir.exists():
        return None
    checkpoints = [
        path
        for path in results_dir.iterdir()
        if path.is_dir() and path.name.startswith("checkpoint-") and path.name.split("-")[-1].isdigit()
    ]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda path: int(path.name.split("-")[-1]))
    return str(checkpoints[-1])


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    slang_precision, slang_recall, slang_f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    literal_indices = labels == 0
    slang_indices = labels == 1
    false_positive_rate = float((preds[literal_indices] == 1).mean()) if literal_indices.any() else 0.0
    false_negative_rate = float((preds[slang_indices] == 0).mean()) if slang_indices.any() else 0.0
    return {
        "accuracy": float((preds == labels).mean()),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "slang_precision": float(slang_precision),
        "slang_recall": float(slang_recall),
        "slang_f1": float(slang_f1),
        "false_positive_rate_on_literal": false_positive_rate,
        "false_negative_rate_on_slang": false_negative_rate,
    }


def predict_probabilities(model, tokenizer, rows: list[dict[str, Any]], device: torch.device, batch_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    texts = [format_input(row) for row in rows]
    probs: list[float] = []
    model.eval()
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            batch_probs = torch.softmax(logits, dim=-1)[:, 1]
        probs.extend(batch_probs.detach().cpu().numpy().tolist())
    probs_np = np.array(probs)
    return (probs_np >= 0.5).astype(int), probs_np


def evaluate_rows(model, tokenizer, rows: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    labels = np.array([int(row["label"]) for row in rows])
    preds, probs = predict_probabilities(model, tokenizer, rows, device)
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
        "accuracy": float(report.get("accuracy", 0.0)),
        "macro_f1": float(report.get("macro avg", {}).get("f1-score", 0.0)),
        "false_positive_rate_on_literal": float((preds[labels == 0] == 1).mean()) if (labels == 0).any() else 0.0,
        "false_negative_rate_on_slang": float((preds[labels == 1] == 0).mean()) if (labels == 1).any() else 0.0,
        "term_metrics": term_metrics,
        "sample_predictions": [
            {
                "term": rows[idx]["term"],
                "sentence": rows[idx]["sentence"],
                "expected": int(labels[idx]),
                "predicted": int(preds[idx]),
                "slang_probability": float(probs[idx]),
            }
            for idx in range(min(20, len(rows)))
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train slang sense classifier.")
    parser.add_argument("--model-name", default="microsoft/deberta-v3-small")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--results-dir", default="results_slang_sense_classifier")
    parser.add_argument("--report-path", default=str(REPORT_DIR / "slang_sense_classifier_metrics.json"))
    parser.add_argument("--epochs", type=float, default=5)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--save-total-limit", type=int, default=4)
    parser.add_argument("--min-macro-f1", type=float, default=0.95)
    parser.add_argument("--min-slang-recall", type=float, default=0.93)
    parser.add_argument("--max-literal-false-positive-rate", type=float, default=0.03)
    parser.add_argument(
        "--resume-from-checkpoint",
        default="auto",
        help="Use 'auto' to resume from the latest checkpoint in --results-dir, a checkpoint path, or 'none'.",
    )
    parser.add_argument(
        "--allow-failed-save",
        action="store_true",
        help="Save the model even if quality gates fail. Use only for debugging, not deployment.",
    )
    args = parser.parse_args()

    set_seed()
    train_rows = read_rows(TRAIN_DATA_PATH)
    test_rows = read_rows(TEST_DATA_PATH)
    train_labels = [int(row["label"]) for row in train_rows]
    test_labels = [int(row["label"]) for row in test_rows]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    train_encodings = tokenizer([format_input(row) for row in train_rows], truncation=True, padding=True, max_length=MAX_LENGTH)
    test_encodings = tokenizer([format_input(row) for row in test_rows], truncation=True, padding=True, max_length=MAX_LENGTH)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "LITERAL", 1: "SLANG"},
        label2id={"LITERAL": 0, "SLANG": 1},
    )

    class_weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=np.array(train_labels))
    class_weights_pt = torch.tensor(class_weights, dtype=torch.float32)

    training_args = TrainingArguments(
        output_dir=args.results_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=args.save_total_limit,
        fp16=torch.cuda.is_available(),
        seed=SEED,
        report_to="none",
    )

    trainer = WeightedTrainer(
        class_weights=class_weights_pt,
        model=model,
        args=training_args,
        train_dataset=SenseDataset(train_encodings, train_labels),
        eval_dataset=SenseDataset(test_encodings, test_labels),
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    print(f"Training {args.model_name} on {len(train_rows)} examples.")
    resume_arg: str | bool | None
    if args.resume_from_checkpoint.lower() == "none":
        resume_arg = None
    elif args.resume_from_checkpoint.lower() == "auto":
        resume_arg = latest_checkpoint(Path(args.results_dir))
    else:
        resume_arg = args.resume_from_checkpoint
    if resume_arg:
        print(f"Resuming from checkpoint: {resume_arg}")
    trainer.train(resume_from_checkpoint=resume_arg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    metrics = evaluate_rows(model, tokenizer, test_rows, device)
    quality_passed = (
        metrics["macro_f1"] >= args.min_macro_f1
        and 1.0 - metrics["false_negative_rate_on_slang"] >= args.min_slang_recall
        and metrics["false_positive_rate_on_literal"] <= args.max_literal_false_positive_rate
    )

    if quality_passed or args.allow_failed_save:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        saved_model = True
    else:
        saved_model = False

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "model_name": args.model_name,
        "output_dir": str(args.output_dir),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "train_examples": len(train_rows),
        "test_examples": len(test_rows),
        "quality_gates": {
            "min_macro_f1": args.min_macro_f1,
            "min_slang_recall": args.min_slang_recall,
            "max_literal_false_positive_rate": args.max_literal_false_positive_rate,
            "passed": quality_passed,
        },
        "training_config": {
            "epochs": args.epochs,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "eval_steps": args.eval_steps,
            "early_stopping_patience": args.early_stopping_patience,
            "resume_from_checkpoint": resume_arg,
        },
        "test": metrics,
        "saved_model": saved_model,
    }
    if not saved_model:
        report["blocked_reason"] = "Quality gates failed; final model was not saved."
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if saved_model:
        print(f"Saved model to {args.output_dir}")
    else:
        print("Quality gates failed. Model was not saved and should not be zipped or deployed.")
    print(f"Saved metrics to {report_path}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
