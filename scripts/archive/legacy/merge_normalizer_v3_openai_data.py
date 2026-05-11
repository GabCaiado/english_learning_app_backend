"""
Merge OpenAI-augmented normalizer V3.1 rows into the training split.

Run after scripts/expand_normalizer_v3_with_openai.py:
  python scripts/merge_normalizer_v3_openai_data.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


TRAIN_PATH = Path("data/slang_normalizer_v3_1_train.json")
AUGMENTED_PATH = Path("data/slang_normalizer_v3_1_openai_augmented.json")

BAD_WRAPPER_PREFIXES = (
    "honestly, ",
    "honestly ",
    "i think ",
    "everyone said ",
    "people online said ",
    "the comments agreed",
)

BAD_WRAPPER_FRAGMENTS = (
    "the comments agreed",
    "people online said",
    "everyone said",
)


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def clean(text: str) -> str:
    return " ".join((text or "").strip().split())


def has_bad_wrapper(text: str) -> bool:
    lower = clean(text).lower()
    return lower.startswith(BAD_WRAPPER_PREFIXES) or any(fragment in lower for fragment in BAD_WRAPPER_FRAGMENTS)


def validate_row(source: str, target: str, sense: str) -> str | None:
    if not source or not target:
        return "missing input or target"
    if has_bad_wrapper(source) or has_bad_wrapper(target):
        return "wrapper artifact"
    if sense == "literal" and source.lower() != target.lower():
        return "literal row must be identity"
    if len(target) > max(40, int(len(source) * 1.6)):
        return "target is too long"
    return None


def main() -> None:
    train_rows = read_rows(TRAIN_PATH)
    augmented_rows = read_rows(AUGMENTED_PATH)
    seen = {clean(row.get("input", "")).lower() for row in train_rows}
    merged = train_rows[:]
    added = 0

    for row in augmented_rows:
        source = clean(row.get("input", ""))
        target = clean(row.get("target", ""))
        sense = row.get("sense", "")
        if validate_row(source, target, sense):
            continue
        key = source.lower()
        if key in seen:
            continue
        seen.add(key)
        merged.append(
            {
                "input": source,
                "target": target,
                "term": row.get("term", ""),
                "sense": sense,
                "source": row.get("source", "openai_augmented"),
            }
        )
        added += 1

    with TRAIN_PATH.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Added {added} rows.")
    print(f"Training rows: {len(train_rows)} -> {len(merged)}")


if __name__ == "__main__":
    main()
