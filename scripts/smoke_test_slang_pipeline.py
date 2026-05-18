"""
Smoke test the production slang normalization path without loading translation.

This exercises the dictionary + detector + context resolver path used by the
app before the Portuguese translator sees the sentence.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.context_resolver import ContextResolver
from app.ml.normalizer import SlangNormalizer
from app.ml.slang_detector import AMBIGUOUS_SLANG, SlangDetector
from app.ml.slang_dictionary import SlangDictionary


@dataclass(frozen=True)
class Case:
    text: str
    expected: str
    kind: str


DEFAULT_CASES_PATH = BACKEND_ROOT / "data" / "slang_pipeline_gold_cases.json"
_normalizer: SlangNormalizer | None = None


def get_normalizer() -> SlangNormalizer:
    global _normalizer
    if _normalizer is None:
        _normalizer = SlangNormalizer()
    return _normalizer


def load_cases(path: Path = DEFAULT_CASES_PATH) -> list[Case]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    return [
        Case(text=row["input"], expected=row["expected"], kind=row["kind"])
        for row in rows
    ]


def normalize_for_match(text: str) -> str:
    return " ".join((text or "").lower().strip(" .!?").split())


def normalize_sentence_with_trace(
    sentence: str,
    dictionary: SlangDictionary,
    detector: SlangDetector,
    resolver: ContextResolver,
    normalizer: SlangNormalizer | None = None,
) -> dict:
    normalizer = normalizer or get_normalizer()
    slangs_found = []
    pre_rewritten_sentence = normalizer.apply_safety_rewrites(sentence)
    sentence_for_detection = pre_rewritten_sentence if pre_rewritten_sentence != sentence else sentence
    detector_score = detector.predict_score(sentence_for_detection)
    all_slangs = sorted(dictionary.get_all_slangs(), key=len, reverse=True)

    for slang in all_slangs:
        pattern = r"\b" + re.escape(slang) + r"(?:ing|ed|es|s|er)?\b"
        for match in re.finditer(pattern, sentence_for_detection, flags=re.IGNORECASE):
            slang_info = dictionary.lookup(slang)
            if not slang_info:
                continue

            context_decision = None
            is_really_slang = True
            if slang in AMBIGUOUS_SLANG:
                context_decision = resolver.resolve(
                    term=slang,
                    sentence=sentence_for_detection,
                    detector_score=detector_score,
                    dictionary_has_entry=True,
                    slang_meaning=slang_info.meaning_en or slang_info.normalized,
                )
                is_really_slang = context_decision.should_normalize

            overlaps_existing = any(
                match.start() < slang["end"] and match.end() > slang["start"]
                for slang in slangs_found
            )
            if is_really_slang and not overlaps_existing:
                slangs_found.append(
                    {
                        "start": match.start(),
                        "end": match.end(),
                        "normalized": slang_info.normalized or slang,
                        "original": match.group(),
                        "reason": context_decision.reason if context_decision else "dictionary match",
                    }
                )

    use_sentence_model_first = os.getenv("USE_SENTENCE_NORMALIZER_FIRST", "").lower() in {"1", "true", "yes"}
    if use_sentence_model_first:
        normalized, normalization_source = normalizer.normalize_with_detected_spans(sentence_for_detection, slangs_found)
    else:
        normalized = sentence_for_detection
        for slang in sorted(slangs_found, key=lambda item: item["start"], reverse=True):
            normalized = normalized[: slang["start"]] + slang["normalized"] + normalized[slang["end"] :]
        normalized = SlangNormalizer.apply_safety_rewrites(normalized)
        normalization_source = (
            "dictionary_fallback"
            if slangs_found
            else "safety_rewrites"
            if normalized != sentence
            else "direct"
        )
    return {
        "original": sentence,
        "normalized": normalized,
        "normalization_source": normalization_source,
        "detector_score": detector_score,
        "slangs_found": slangs_found,
    }


def normalize_sentence(sentence: str, dictionary: SlangDictionary, detector: SlangDetector, resolver: ContextResolver) -> str:
    return normalize_sentence_with_trace(sentence, dictionary, detector, resolver)["normalized"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test production slang normalization.")
    parser.parse_args()

    dictionary = SlangDictionary()
    detector = SlangDetector()
    resolver = ContextResolver()
    normalizer = SlangNormalizer()

    passed = 0
    cases = load_cases()

    for case in cases:
        predicted = normalize_sentence_with_trace(case.text, dictionary, detector, resolver, normalizer)["normalized"]
        ok = normalize_for_match(predicted) == normalize_for_match(case.expected)
        passed += int(ok)
        status = "PASS" if ok else "FAIL"
        print(
            f"{status:4} {case.kind:7} input={case.text!r} "
            f"expected={case.expected!r} predicted={predicted!r}"
        )

    print(f"SUMMARY pass={passed}/{len(cases)}")
    if passed != len(cases):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
