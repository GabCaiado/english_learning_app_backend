"""
Train the production slang normalizer with FLAN-T5-base.
Developed for Google Colab T4.
"""

from __future__ import annotations

import json
import os
import random
from datetime import datetime
from pathlib import Path

import evaluate
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    GenerationConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


MODEL_NAME = "google/flan-t5-base"
TRAIN_DATA_PATH = Path("data/master_normalizer_train.json")
TEST_DATA_PATH = Path("data/master_normalizer_test.json")
OUTPUT_DIR = Path("models/slang_normalizer")
REPORT_DIR = Path("reports/training_runs")

TASK_PREFIX = "normalize slang: "
SEED = 42
MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 128


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def normalize_for_match(text: str) -> str:
    return " ".join(text.lower().strip().split())


def exact_match_rate(predictions: list[str], references: list[str]) -> float:
    if not predictions:
        return 0.0
    matches = [
        normalize_for_match(pred) == normalize_for_match(ref)
        for pred, ref in zip(predictions, references)
    ]
    return float(np.mean(matches))


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_for_match(prediction).split()
    ref_tokens = normalize_for_match(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = 0
    ref_counts: dict[str, int] = {}
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    for token in pred_tokens:
        if ref_counts.get(token, 0) > 0:
            common += 1
            ref_counts[token] -= 1
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def build_dataset(rows: list[dict]) -> Dataset:
    return Dataset.from_list(
        [
            {
                "input": row.get("input") or row.get("slang"),
                "target": row.get("target") or row.get("formal"),
                "sense": row.get("sense"),
            }
            for row in rows
            if (row.get("input") or row.get("slang")) and (row.get("target") or row.get("formal"))
        ]
    )


def main() -> None:
    set_seed()
    train_rows = read_rows(TRAIN_DATA_PATH)
    test_rows = read_rows(TEST_DATA_PATH)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    train_dataset = build_dataset(train_rows)
    eval_dataset = build_dataset(test_rows)

    def preprocess(examples):
        inputs = [TASK_PREFIX + text for text in examples["input"]]
        targets = examples["target"]
        model_inputs = tokenizer(
            inputs,
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            padding="max_length",
        )
        labels = tokenizer(
            text_target=targets,
            max_length=MAX_TARGET_LENGTH,
            truncation=True,
            padding="max_length",
        )
        model_inputs["labels"] = [
            [token if token != tokenizer.pad_token_id else -100 for token in seq]
            for seq in labels["input_ids"]
        ]
        return model_inputs

    tokenized_train = train_dataset.map(preprocess, batched=True, remove_columns=train_dataset.column_names)
    tokenized_eval = eval_dataset.map(preprocess, batched=True, remove_columns=eval_dataset.column_names)

    gen_config = GenerationConfig(
        max_length=MAX_TARGET_LENGTH,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.pad_token_id,
    )
    model.generation_config = gen_config

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        return {
            "exact_match": exact_match_rate(decoded_preds, decoded_labels),
            "token_f1": float(np.mean([token_f1(p, r) for p, r in zip(decoded_preds, decoded_labels)])),
        }

    training_args = Seq2SeqTrainingArguments(
        output_dir="results_normalizer_flan",
        num_train_epochs=6,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,
        generation_config=gen_config,
        load_best_model_at_end=True,
        metric_for_best_model="token_f1",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        seed=SEED,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print(f"Training {MODEL_NAME} on {len(train_dataset)} examples.")
    trainer.train()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    predictions = trainer.predict(tokenized_eval)
    metrics = predictions.metrics
    decoded_preds = tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)
    references = [row.get("target") or row.get("formal") for row in test_rows]
    inputs = [row.get("input") or row.get("slang") for row in test_rows]

    neutral_indices = [
        idx for idx, (src, ref) in enumerate(zip(inputs, references))
        if normalize_for_match(src) == normalize_for_match(ref)
    ]
    over_normalized = [
        idx for idx in neutral_indices
        if normalize_for_match(decoded_preds[idx]) != normalize_for_match(references[idx])
    ]
    metrics["test_over_normalization_rate"] = (
        len(over_normalized) / len(neutral_indices) if neutral_indices else 0.0
    )

    try:
        bertscore = evaluate.load("bertscore")
        bert_results = bertscore.compute(
            predictions=decoded_preds[:500],
            references=references[:500],
            model_type="distilbert-base-uncased",
            batch_size=16,
        )
        metrics["test_bertscore_f1_sample"] = float(np.mean(bert_results["f1"]))
    except Exception as exc:
        metrics["test_bertscore_error"] = str(exc)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "model_name": MODEL_NAME,
        "output_dir": str(OUTPUT_DIR),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "train_examples": len(train_dataset),
        "test_examples": len(eval_dataset),
        "metrics": {key: float(value) if isinstance(value, np.floating) else value for key, value in metrics.items()},
    }
    report_path = REPORT_DIR / "normalizer_flan_metrics.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(json.dumps(report, indent=2))
    print(f"Saved model to {OUTPUT_DIR}")
    print(f"Saved metrics to {report_path}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
