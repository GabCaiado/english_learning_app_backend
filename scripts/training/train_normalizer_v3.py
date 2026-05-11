"""
Train slang normalizer V4.2.

Designed for Colab T4:
  python scripts/build_normalizer_v4_dataset.py
  python scripts/training/train_normalizer_v3.py

The model is saved to:
  models/slang_normalizer_v4_2_base
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


TRAIN_DATA_PATH = Path("data/slang_normalizer_v4_train.json")
TEST_DATA_PATH = Path("data/slang_normalizer_v4_test.json")
OUTPUT_DIR = Path("models/slang_normalizer_v4_2_base")
REPORT_DIR = Path("reports/training_runs")
TASK_PREFIX = "normalize slang to standard English: "
SEED = 42
MAX_INPUT_LENGTH = 160
MAX_TARGET_LENGTH = 160


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python scripts/build_normalizer_v4_dataset.py")
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


def build_dataset(rows: list[dict[str, Any]]) -> Dataset:
    return Dataset.from_list(
        [
            {
                "input": row["input"],
                "target": row["target"],
                "term": row.get("term", ""),
                "sense": row.get("sense", ""),
            }
            for row in rows
            if row.get("input") and row.get("target")
        ]
    )


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train slang normalizer V4.2.")
    parser.add_argument("--model-name", default="google/flan-t5-base")
    parser.add_argument("--train-data", default=str(TRAIN_DATA_PATH))
    parser.add_argument("--test-data", default=str(TEST_DATA_PATH))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--results-dir", default="results_normalizer_v4_2_base")
    parser.add_argument("--report-path", default=str(REPORT_DIR / "normalizer_v4_2_base_metrics.json"))
    parser.add_argument("--epochs", type=float, default=6)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.08)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--min-exact-match", type=float, default=0.85)
    parser.add_argument("--max-over-normalization-rate", type=float, default=0.03)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--save-total-limit", type=int, default=4)
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
    parser.add_argument("--fp16", action="store_true", default=torch.cuda.is_available(), help="Enable fp16.")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false", help="Disable fp16 if Colab dtype errors appear.")
    parser.add_argument("--bf16", action="store_true", help="Enable bf16 on supported GPUs.")
    args = parser.parse_args()

    set_seed()
    train_rows = read_rows(Path(args.train_data))
    test_rows = read_rows(Path(args.test_data))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

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

    generation_config = GenerationConfig(
        max_length=MAX_TARGET_LENGTH,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.pad_token_id,
    )
    model.generation_config = generation_config

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
        output_dir=args.results_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,
        generation_config=generation_config,
        load_best_model_at_end=True,
        metric_for_best_model="token_f1",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=args.save_total_limit,
        fp16=args.fp16 and not args.bf16,
        bf16=args.bf16,
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    print(f"Training {args.model_name} on {len(train_dataset)} examples.")
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

    predictions = trainer.predict(tokenized_eval)
    decoded_preds = tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)
    references = [row["target"] for row in test_rows]
    inputs = [row["input"] for row in test_rows]

    neutral_indices = [
        idx for idx, (source, reference) in enumerate(zip(inputs, references))
        if normalize_for_match(source) == normalize_for_match(reference)
    ]
    over_normalized = [
        idx for idx in neutral_indices
        if normalize_for_match(decoded_preds[idx]) != normalize_for_match(references[idx])
    ]
    test_exact_match = exact_match_rate(decoded_preds, references)
    test_token_f1 = float(np.mean([token_f1(p, r) for p, r in zip(decoded_preds, references)]))
    test_over_normalization_rate = len(over_normalized) / len(neutral_indices) if neutral_indices else 0.0
    quality_passed = (
        test_exact_match >= args.min_exact_match
        and test_over_normalization_rate <= args.max_over_normalization_rate
    )

    report = {
        "model_name": args.model_name,
        "output_dir": args.output_dir,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "train_examples": len(train_dataset),
        "test_examples": len(eval_dataset),
        "metrics": {
            key: float(value) if isinstance(value, np.floating) else value
            for key, value in predictions.metrics.items()
        },
        "quality_gates": {
            "min_exact_match": args.min_exact_match,
            "max_over_normalization_rate": args.max_over_normalization_rate,
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
            "early_stopping_patience": args.early_stopping_patience,
            "fp16": args.fp16 and not args.bf16,
            "bf16": args.bf16,
            "resume_from_checkpoint": resume_arg,
        },
        "test_exact_match": test_exact_match,
        "test_token_f1": test_token_f1,
        "test_over_normalization_rate": test_over_normalization_rate,
        "sample_predictions": [
            {
                "input": inputs[idx],
                "expected": references[idx],
                "predicted": decoded_preds[idx],
                "ok": normalize_for_match(decoded_preds[idx]) == normalize_for_match(references[idx]),
            }
            for idx in range(min(30, len(inputs)))
        ],
    }

    if quality_passed or args.allow_failed_save:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        report["saved_model"] = True
    else:
        report["saved_model"] = False
        report["blocked_reason"] = (
            "Quality gates failed; final model was not saved. "
            "Do not zip or deploy this training run."
        )

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["saved_model"]:
        print(f"Saved model to {args.output_dir}")
    else:
        print("Quality gates failed. Model was not saved and should not be zipped or deployed.")
    print(f"Saved metrics to {report_path}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
