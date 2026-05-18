"""
Build a training split that intentionally oversamples reviewed feedback rows.

The canonical train file keeps unique examples. This derived file repeats
feedback-owned examples so a fine-tune sees the new corrections often enough
to learn them instead of drowning them in the older base distribution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_TRAIN = Path("data/slang_normalizer_v4_train.json")
DEFAULT_OUTPUT = Path("data/slang_normalizer_v4_feedback_oversampled_train.json")
FEEDBACK_SOURCES = {
    "feedback_approved",
    "feedback_augmented",
    "feedback_literal_guard",
}


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a JSON list")
    return [row for row in rows if isinstance(row, dict)]


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create feedback-oversampled normalizer train data.")
    parser.add_argument("--train-data", default=str(DEFAULT_TRAIN))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args()

    rows = read_rows(Path(args.train_data))
    feedback = [row for row in rows if row.get("source") in FEEDBACK_SOURCES]
    base = [row for row in rows if row.get("source") not in FEEDBACK_SOURCES]
    merged = base + feedback * max(args.repeat, 1)

    write_rows(Path(args.output), merged)
    print(f"Base rows: {len(base)}")
    print(f"Feedback rows: {len(feedback)}")
    print(f"Repeat: {max(args.repeat, 1)}")
    print(f"Oversampled rows: {len(merged)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
