"""
Export admin-approved translation corrections into a runtime overrides file.

Unlike export_approved_to_training_data.py (which feeds the detector/normalizer
trainers), this exports expected_translation — the field nothing currently
consumes — into a lookup file that Translator loads directly at runtime. No
training involved: this is a direct "admin fixed it once, app uses that fix
from now on" patch, same mechanism as the hardcoded phrase_overrides dict in
app/ml/translator.py.

Only pulls from real user-report sources (word_book, word_detail_sentence,
word_detail_word) — the ai_generated/ai_amplified synthetic rows never had a
real translation attached and are excluded.

Run from the backend root:
  python scripts/export_translation_overrides.py

Output:
  data/translation_overrides.json   { "<normalized english>": "<pt translation>", ... }

Translator picks this up automatically on next process start (or set
TRANSLATION_OVERRIDES_PATH to point elsewhere).
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
OUT_PATH = DATA_DIR / "translation_overrides.json"

REAL_SOURCES = ("word_book", "word_detail_sentence", "word_detail_word")


def clean(text: Any) -> str:
    return " ".join(str(text or "").strip().split())


def override_key(text: str) -> str:
    return clean(text).lower().strip(" .!?")


def main() -> None:
    db = SessionLocal()
    try:
        stmt = select(FailedTranslation).where(
            FailedTranslation.status == "approved",
            FailedTranslation.source.in_(REAL_SOURCES),
            FailedTranslation.expected_translation.isnot(None),
        )
        rows = db.scalars(stmt).all()
    finally:
        db.close()

    print(f"Fetched {len(rows)} approved real-user corrections with an expected translation.")

    overrides: dict[str, str] = {}
    skipped_noop = 0
    for row in rows:
        key_text = row.expected_normalized or row.model_normalized or row.input_text
        key = override_key(key_text)
        translation = clean(row.expected_translation)
        if not key or not translation:
            continue
        # Guards against bad review data: an approved row whose "expected
        # translation" is just the English text left untranslated would make
        # the app stop translating that phrase entirely.
        if override_key(translation) == key:
            skipped_noop += 1
            continue
        overrides[key] = translation

    if skipped_noop:
        print(f"Skipped {skipped_noop} row(s) where expected_translation matched the English text (untranslated).")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(overrides, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote {len(overrides)} translation overrides -> {OUT_PATH}")


if __name__ == "__main__":
    main()
