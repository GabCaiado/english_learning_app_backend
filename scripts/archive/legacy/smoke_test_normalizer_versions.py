"""
Performs a smoke test of local slang normalizer model folders against curated production cases.

This is intentionally small and strict. Use it before pointing the app at a new normalizer version.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.normalizer import SlangNormalizer


DEFAULT_MODELS = [
    "models/slang_normalizer",
    "models/slang_normalizer_v2",
    "models/slang_normalizer_v3_1",
    "models/slang_normalizer_v3_2",
]


@dataclass(frozen=True)
class Case:
    text: str
    expected: str
    kind: str


CASES = [
    Case("the concert was lit", "the concert was exciting", "slang"),
    Case("the streets were lit by neon signs", "the streets were lit by neon signs", "literal"),
    Case("spill the tea", "share the gossip", "slang"),
    Case("i like green tea", "i like green tea", "literal"),
    Case("that song is my jam", "that song is my favorite", "slang"),
    Case("i put jam on my bread", "i put jam on my bread", "literal"),
    Case("can i crash at your place?", "can i sleep at your place?", "slang"),
    Case("the car crash blocked traffic", "the car crash blocked traffic", "literal"),
    Case("they have beef with the coworker", "they have a conflict with the coworker", "slang"),
    Case("she ordered beef tacos", "she ordered beef tacos", "literal"),
    Case("that guitar solo was nasty", "that guitar solo was amazing", "slang"),
    Case("the bathroom smells nasty", "the bathroom smells nasty", "literal"),
    Case("she ate that performance", "she did very well in that performance", "slang"),
    Case("she ate dinner", "she ate dinner", "literal"),
    Case("he dipped after the party", "he left after the party", "slang"),
    Case("dip the bread in the soup", "dip the bread in the soup", "literal"),
    Case("they're tripping about one missed practice", "they're overreacting about one missed practice", "slang"),
    Case("be careful not to trip on the step", "be careful not to trip on the step", "literal"),
]


BAD_FRAGMENTS = [
    "blackjack",
    "bookmark",
    "clinician",
    "dealers",
    "jerusalem",
    "marketers",
    "synchronous",
    "did extremely well after the party",
    "doing extremely well about",
]


def normalize_for_match(text: str) -> str:
    return " ".join((text or "").lower().strip(" .!?").split())


def is_exactish(predicted: str, expected: str) -> bool:
    return normalize_for_match(predicted) == normalize_for_match(expected)


def has_bad_fragment(predicted: str) -> bool:
    lower = predicted.lower()
    return any(fragment in lower for fragment in BAD_FRAGMENTS)


def run_model(model_path: str) -> tuple[int, int, int]:
    path = Path(model_path)
    if not path.exists():
        print(f"\nMODEL {model_path} [missing]")
        return 0, 0, 0

    print(f"\nMODEL {model_path}")
    normalizer = SlangNormalizer(model_path)
    passed = 0
    changed_slang = 0
    bad_outputs = 0

    for case in CASES:
        predicted = normalizer.normalize_sentence(case.text)
        ok = is_exactish(predicted, case.expected)
        changed = normalize_for_match(predicted) != normalize_for_match(case.text)
        bad = has_bad_fragment(predicted)
        passed += int(ok)
        changed_slang += int(case.kind == "slang" and changed)
        bad_outputs += int(bad)
        status = "PASS" if ok and not bad else "FAIL"
        print(
            f"{status:4} {case.kind:7} input={case.text!r} "
            f"expected={case.expected!r} predicted={predicted!r}"
        )

    print(
        f"SUMMARY pass={passed}/{len(CASES)} "
        f"changed_slang={changed_slang}/{sum(c.kind == 'slang' for c in CASES)} "
        f"bad_outputs={bad_outputs}"
    )
    return passed, changed_slang, bad_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test local normalizer model versions.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    args = parser.parse_args()

    for model_path in args.models:
        run_model(model_path)


if __name__ == "__main__":
    main()
