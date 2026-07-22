"""
Export approved failed_translations rows directly into training-ready JSON.

Unlike build_datasets_from_failures.py (which spends GPT calls re-deriving
labels from failures), every row this script reads already carries a trusted
label — either a human-approved correction (expected_is_slang /
expected_normalized) or an auto-approved AI-generated row where the local
detector and the LLM already agreed. So this is a straight, free export.

Run from the backend root:
  python scripts/export_approved_to_training_data.py

Output:
  data/detector_train.json    [{ "text": ..., "is_slang": bool }, ...]
  data/normalizer_train.json  [{ "input": ..., "target": ... }, ...]

Then train with:
  python scripts/archive/legacy/train_detector_improved.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from sqlalchemy import select

from app.database import SessionLocal
from app.models.failed_translation import FailedTranslation

DATA_DIR = BACKEND_ROOT / "data"
DETECTOR_OUT = DATA_DIR / "detector_train.json"
NORMALIZER_OUT = DATA_DIR / "normalizer_train.json"


def clean(text: Any) -> str:
    return " ".join(str(text or "").strip().split())


def write_json(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> None:
    db = SessionLocal()
    try:
        stmt = select(FailedTranslation).where(FailedTranslation.status == "approved")
        rows = db.scalars(stmt).all()
    finally:
        db.close()

    print(f"Fetched {len(rows)} approved rows.")

    detector_rows: list[dict] = []
    normalizer_rows: list[dict] = []
    seen_detector: set[str] = set()
    seen_normalizer: set[str] = set()

    for row in rows:
        text = clean(row.input_text)
        if not text or row.expected_is_slang is None:
            continue

        key = text.lower()
        if key not in seen_detector:
            seen_detector.add(key)
            detector_rows.append({"text": text, "is_slang": bool(row.expected_is_slang)})

        normalized = clean(row.expected_normalized)
        if normalized and key not in seen_normalizer:
            seen_normalizer.add(key)
            normalizer_rows.append({"input": text, "target": normalized})

    write_json(DETECTOR_OUT, detector_rows)
    write_json(NORMALIZER_OUT, normalizer_rows)

    slang_count = sum(1 for r in detector_rows if r["is_slang"])
    print(f"\nWrote {len(detector_rows)} detector rows → {DETECTOR_OUT}")
    print(f"  slang: {slang_count}  |  neutral: {len(detector_rows) - slang_count}")
    print(f"Wrote {len(normalizer_rows)} normalizer rows → {NORMALIZER_OUT}")
    print("\nNext step:")
    print("  python scripts/archive/legacy/train_detector_improved.py")


if __name__ == "__main__":
    main()
