"""
Context resolver for ambiguous slang.

This layer makes a conservative decision before normalization. The detector can
still provide probabilities, but literal safety rules get the first chance to
block over-normalization for polysemous words such as "fire", "tea", and "cap".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from app.ml.context_rules import ContextRuleSet, load_context_rules
from app.ml.slang_sense_classifier import SlangSenseClassifier

Sense = Literal["slang", "literal", "unknown"]


@dataclass
class ContextDecision:
    term: str
    is_slang: bool
    confidence: float
    sense: Sense
    reason: str
    should_normalize: bool


class ContextResolver:
    def __init__(
        self,
        ambiguous_terms: set[str] | None = None,
        sense_classifier: SlangSenseClassifier | None = None,
        rule_set: ContextRuleSet | None = None,
    ):
        self.rule_set = rule_set or load_context_rules()
        self.ambiguous_terms = ambiguous_terms or self.rule_set.ambiguous_terms
        self.sense_classifier = sense_classifier or SlangSenseClassifier()

    def resolve(
        self,
        term: str,
        sentence: str,
        detector_score: float,
        dictionary_has_entry: bool = False,
        slang_meaning: str | None = None,
    ) -> ContextDecision:
        term_lower = term.lower().strip()
        text = sentence.lower().strip()

        if not term_lower:
            return ContextDecision(term, False, 1.0, "literal", "empty term", False)

        if term_lower not in self.ambiguous_terms:
            confidence = max(detector_score, 0.90 if dictionary_has_entry else detector_score)
            is_slang = dictionary_has_entry or detector_score >= 0.75
            return ContextDecision(
                term=term_lower,
                is_slang=is_slang,
                confidence=confidence if is_slang else 1.0 - detector_score,
                sense="slang" if is_slang else "literal",
                reason="non-ambiguous dictionary/model decision",
                should_normalize=is_slang,
            )

        for pattern in self.rule_set.literal_regexes.get(term_lower, []):
            if re.search(pattern, text):
                return ContextDecision(
                    term=term_lower,
                    is_slang=False,
                    confidence=0.99,
                    sense="literal",
                    reason=f"literal regex matched: {pattern}",
                    should_normalize=False,
                )

        for phrase in self.rule_set.literal_patterns.get(term_lower, []):
            if phrase in text:
                return ContextDecision(
                    term=term_lower,
                    is_slang=False,
                    confidence=0.98,
                    sense="literal",
                    reason=f"literal context matched: {phrase}",
                    should_normalize=False,
                )

        for pattern in self.rule_set.slang_regexes.get(term_lower, []):
            if re.search(pattern, text):
                return ContextDecision(
                    term=term_lower,
                    is_slang=True,
                    confidence=max(detector_score, 0.90),
                    sense="slang",
                    reason=f"slang regex matched: {pattern}",
                    should_normalize=True,
                )

        sense_prediction = self.sense_classifier.predict(term_lower, sentence, slang_meaning)
        if sense_prediction and sense_prediction.is_slang is True:
            return ContextDecision(
                term=term_lower,
                is_slang=True,
                confidence=sense_prediction.confidence,
                sense="slang",
                reason=(
                    f"{sense_prediction.reason}: "
                    f"p_slang={sense_prediction.slang_probability:.3f}"
                ),
                should_normalize=True,
            )

        if sense_prediction and sense_prediction.is_slang is False:
            return ContextDecision(
                term=term_lower,
                is_slang=False,
                confidence=sense_prediction.confidence,
                sense="literal",
                reason=(
                    f"{sense_prediction.reason}: "
                    f"p_slang={sense_prediction.slang_probability:.3f}"
                ),
                should_normalize=False,
            )

        if sense_prediction and sense_prediction.is_slang is None:
            return ContextDecision(
                term=term_lower,
                is_slang=False,
                confidence=sense_prediction.confidence,
                sense="unknown",
                reason=(
                    f"{sense_prediction.reason}; abstaining: "
                    f"p_slang={sense_prediction.slang_probability:.3f}"
                ),
                should_normalize=False,
            )

        for phrase in self.rule_set.slang_contexts.get(term_lower, []):
            if phrase in text:
                return ContextDecision(
                    term=term_lower,
                    is_slang=True,
                    confidence=max(detector_score, 0.86),
                    sense="slang",
                    reason=f"slang context matched: {phrase}",
                    should_normalize=True,
                )

        if re.search(rf"\bno\s+{re.escape(term_lower)}\b", text):
            return ContextDecision(term_lower, True, 0.95, "slang", "slang phrase pattern", True)

        if detector_score >= 0.85:
            return ContextDecision(
                term_lower,
                True,
                detector_score,
                "slang",
                "high detector confidence",
                True,
            )

        if detector_score <= 0.25:
            return ContextDecision(
                term_lower,
                False,
                1.0 - detector_score,
                "literal",
                "low detector confidence",
                False,
            )

        return ContextDecision(
            term_lower,
            False,
            max(1.0 - detector_score, 0.50),
            "unknown",
            "ambiguous context; abstaining from normalization",
            False,
        )
