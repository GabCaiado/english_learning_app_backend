"""
Focused regression checks for sentence patterns that must generalize.

This script catches the "works only for the mocked/example sentence" failure
mode. It verifies both the production normalization path and deterministic
Portuguese translations that should not require loading the full MT model.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.context_resolver import ContextResolver
from app.ml.slang_detector import SlangDetector
from app.ml.slang_dictionary import SlangDictionary
from app.ml.translator import get_translator
from scripts.smoke_test_slang_pipeline import normalize_for_match, normalize_sentence


@dataclass(frozen=True)
class DynamicCase:
    input: str
    expected_normalized: str
    expected_pt: str


DYNAMIC_CASES = [
    DynamicCase(
        input="i ship gabriella and deryck",
        expected_normalized="i want gabriella and deryck to be a couple",
        expected_pt="Eu torço para gabriella e deryck ficarem juntos",
    ),
    DynamicCase(
        input="i ship maya and leo",
        expected_normalized="i want maya and leo to be a couple",
        expected_pt="Eu torço para maya e leo ficarem juntos",
    ),
    DynamicCase(
        input="i ship ana-maria and joao",
        expected_normalized="i want ana-maria and joao to be a couple",
        expected_pt="Eu torço para ana-maria e joao ficarem juntos",
    ),
]

LITERAL_NORMALIZATION_CASES = [
    ("the ship crossed the ocean", "the ship crossed the ocean"),
    ("they will ship the package tomorrow", "they will ship the package tomorrow"),
    ("the company ships internationally", "the company ships internationally"),
]


def main() -> None:
    dictionary = SlangDictionary()
    detector = SlangDetector()
    resolver = ContextResolver()
    translator = get_translator()
    failures = []

    for case in DYNAMIC_CASES:
        normalized = normalize_sentence(case.input, dictionary, detector, resolver)
        translated = translator.translate(normalized)
        norm_ok = normalize_for_match(normalized) == normalize_for_match(case.expected_normalized)
        pt_ok = normalize_for_match(translated) == normalize_for_match(case.expected_pt)
        status = "PASS" if norm_ok and pt_ok else "FAIL"
        print(
            f"{status} dynamic input={case.input!r} "
            f"normalized={normalized!r} translated={translated!r}"
        )
        if not norm_ok or not pt_ok:
            failures.append(case.input)

    for input_text, expected in LITERAL_NORMALIZATION_CASES:
        normalized = normalize_sentence(input_text, dictionary, detector, resolver)
        ok = normalize_for_match(normalized) == normalize_for_match(expected)
        status = "PASS" if ok else "FAIL"
        print(f"{status} literal input={input_text!r} normalized={normalized!r}")
        if not ok:
            failures.append(input_text)

    if failures:
        print(f"SUMMARY failed={len(failures)} cases={failures!r}")
        raise SystemExit(1)

    print(f"SUMMARY pass={len(DYNAMIC_CASES) + len(LITERAL_NORMALIZATION_CASES)}")


if __name__ == "__main__":
    main()
