"""
Train the production slang detector with DeBERTa-v3-small. 
Developed for Google Colab T4.
"""

from __future__ import annotations

import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


MODEL_NAME = "microsoft/deberta-v3-small"
TRAIN_DATA_PATH = Path("data/master_detector_train.json")
TEST_DATA_PATH = Path("data/master_detector_test.json")
GOLDEN_DATA_PATH = Path("data/golden_eval.json")
OUTPUT_DIR = Path("models/slang_detector")
REPORT_DIR = Path("reports/training_runs")

SEED = 42
MAX_LENGTH = 128


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SlangDataset(Dataset):
    def __init__(self, encodings, labels, metadata):
        self.encodings = encodings
        self.labels = labels
        self.metadata = metadata

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
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


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python scripts/build_training_datasets.py"
        )
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} must contain a non-empty JSON list.")
    return rows


def labels_from_rows(rows: list[dict]) -> list[int]:
    return [int(row.get("label", row.get("is_slang", 0))) for row in rows]


def compute_basic_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    return {
        "accuracy": float((preds == labels).mean()),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
    }


def score_rows(model, tokenizer, rows: list[dict], device: torch.device) -> dict:
    texts = [row["text"] for row in rows]
    labels = np.array(labels_from_rows(rows))
    preds = []
    probs = []

    model.eval()
    for i in range(0, len(texts), 64):
        batch = texts[i : i + 64]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            batch_probs = torch.softmax(logits, dim=-1)[:, 1]
        probs.extend(batch_probs.detach().cpu().numpy().tolist())
        preds.extend((batch_probs >= 0.5).long().detach().cpu().numpy().tolist())

    preds_np = np.array(preds)
    report = classification_report(labels, preds_np, output_dict=True, zero_division=0)

    ambiguous_indices = [
        idx for idx, row in enumerate(rows) if row.get("target_term") and row.get("target_term") != ""
    ]
    hard_negative_indices = [
        idx for idx, row in enumerate(rows) if row.get("is_hard_negative") is True
    ]
    neutral_indices = [idx for idx, label in enumerate(labels) if label == 0]

    def subset_f1(indices: list[int]) -> float | None:
        if not indices:
            return None
        _, _, f1, _ = precision_recall_fscore_support(
            labels[indices], preds_np[indices], average="macro", zero_division=0
        )
        return float(f1)

    def false_positive_rate(indices: list[int]) -> float | None:
        if not indices:
            return None
        selected = preds_np[indices]
        return float((selected == 1).mean())

    return {
        "accuracy": float(report.get("accuracy", 0.0)),
        "macro_f1": float(report.get("macro avg", {}).get("f1-score", 0.0)),
        "weighted_f1": float(report.get("weighted avg", {}).get("f1-score", 0.0)),
        "ambiguous_macro_f1": subset_f1(ambiguous_indices),
        "hard_negative_false_positive_rate": false_positive_rate(hard_negative_indices),
        "neutral_false_positive_rate": false_positive_rate(neutral_indices),
        "num_examples": len(rows),
        "num_ambiguous_examples": len(ambiguous_indices),
        "num_hard_negatives": len(hard_negative_indices),
    }


def main() -> None:
    set_seed()
    train_rows = read_rows(TRAIN_DATA_PATH)
    test_rows = read_rows(TEST_DATA_PATH)

    train_labels = labels_from_rows(train_rows)
    test_labels = labels_from_rows(test_rows)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    train_encodings = tokenizer(
        [row["text"] for row in train_rows],
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )
    test_encodings = tokenizer(
        [row["text"] for row in test_rows],
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "FORMAL", 1: "SLANG"},
        label2id={"FORMAL": 0, "SLANG": 1},
    )

    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.array([0, 1]), y=np.array(train_labels)
    )
    class_weights_pt = torch.tensor(class_weights, dtype=torch.float32)

    training_args = TrainingArguments(
        output_dir="results_detector_deberta",
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=250,
        save_strategy="steps",
        save_steps=250,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        seed=SEED,
        report_to="none",
    )

    trainer = WeightedTrainer(
        class_weights=class_weights_pt,
        model=model,
        args=training_args,
        train_dataset=SlangDataset(train_encodings, train_labels, train_rows),
        eval_dataset=SlangDataset(test_encodings, test_labels, test_rows),
        compute_metrics=compute_basic_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
    )

    print(f"Training {MODEL_NAME} on {len(train_rows)} examples.")
    trainer.train()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    test_report = score_rows(model, tokenizer, test_rows, device)
    golden_report = score_rows(model, tokenizer, read_rows(GOLDEN_DATA_PATH), device)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "model_name": MODEL_NAME,
        "output_dir": str(OUTPUT_DIR),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "train_examples": len(train_rows),
        "test": test_report,
        "golden": golden_report,
    }
    report_path = REPORT_DIR / "detector_deberta_metrics.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(json.dumps(report, indent=2))
    print(f"Saved model to {OUTPUT_DIR}")
    print(f"Saved metrics to {report_path}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
