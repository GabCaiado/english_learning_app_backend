"""
Model wrapper for deciding whether an ambiguous term is slang in context.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


logger = logging.getLogger(__name__)

DEFAULT_SENSE_MODEL_PATH = "models/slang_sense_classifier"


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %.2f", name, value, default)
        return default


@dataclass
class SenseClassifierPrediction:
    term: str
    slang_probability: float
    confidence: float
    is_slang: bool | None
    reason: str


class SlangSenseClassifier:
    """
    Context classifier for ambiguous slang terms.

    It answers: in this exact sentence, does this candidate term express the
    dictionary slang meaning? If the model is missing or uncertain, callers
    should abstain instead of guessing from FLAN output.
    """

    def __init__(
        self,
        model_path: str | None = None,
        slang_threshold: float | None = None,
        literal_threshold: float | None = None,
    ):
        self.model_path = model_path or os.getenv("SLANG_SENSE_MODEL_PATH", DEFAULT_SENSE_MODEL_PATH)
        self.slang_threshold = slang_threshold if slang_threshold is not None else _env_float("SLANG_SENSE_SLANG_THRESHOLD", 0.75)
        self.literal_threshold = literal_threshold if literal_threshold is not None else _env_float("SLANG_SENSE_LITERAL_THRESHOLD", 0.35)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None

        model_dir = Path(self.model_path)
        if model_dir.exists():
            try:
                logger.info("Loading Slang Sense Classifier from %s on %s", model_dir, self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_dir,
                    use_fast=True,
                    extra_special_tokens={},
                )
                self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
                self.model.eval()
            except Exception as exc:
                logger.warning("Slang Sense Classifier unavailable: %s", exc)
                self.model = None
                self.tokenizer = None
        else:
            logger.info("Slang Sense Classifier not found at %s; using rule fallback", model_dir)

    @property
    def is_available(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def predict(self, term: str, sentence: str, slang_meaning: str | None) -> SenseClassifierPrediction | None:
        if not self.is_available or not slang_meaning:
            return None

        text = (
            f"term: {term.strip().lower()} [SEP] "
            f"meaning: {slang_meaning.strip()} [SEP] "
            f"sentence: {sentence.strip()}"
        )
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=160,
        ).to(self.device)

        with torch.inference_mode():
            logits = self.model(**inputs).logits
            slang_probability = torch.softmax(logits, dim=-1)[0][1].item()

        if slang_probability >= self.slang_threshold:
            return SenseClassifierPrediction(
                term=term,
                slang_probability=slang_probability,
                confidence=slang_probability,
                is_slang=True,
                reason="sense classifier confirmed slang",
            )

        if slang_probability <= self.literal_threshold:
            return SenseClassifierPrediction(
                term=term,
                slang_probability=slang_probability,
                confidence=1.0 - slang_probability,
                is_slang=False,
                reason="sense classifier confirmed literal",
            )

        return SenseClassifierPrediction(
            term=term,
            slang_probability=slang_probability,
            confidence=max(slang_probability, 1.0 - slang_probability),
            is_slang=None,
            reason="sense classifier uncertain",
        )
