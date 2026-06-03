"""
seed_slangs.py
--------------
Seeds the `slang_dictionary` table from the static in-memory cache defined in
app/ml/slang_dictionary.py.  Skips entries that already exist (idempotent).

Run after `alembic upgrade head` and `seed_admin.py`:
  python scripts/seed_slangs.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import select
from app.database import SessionLocal
from app.ml.slang_dictionary import SlangDictionary, SlangInfo
from app.models.vocabulary import SlangDictionaryEntry, WordExample


def seed_slangs() -> None:
    dictionary = SlangDictionary()

    db = SessionLocal()
    try:
        seeded = 0
        skipped = 0
        for word, info in dictionary._cache.items():
            existing = db.scalars(
                select(SlangDictionaryEntry).where(SlangDictionaryEntry.word == word)
            ).first()

            if existing:
                skipped += 1
                continue

            entry = SlangDictionaryEntry(
                word=word,
                normalized_form=info.normalized if info.normalized != word else None,
                translation_pt=info.meaning_pt or word,
                meaning_en=info.meaning_en or None,
                meaning_pt=info.meaning_pt or None,
                is_slang=True,
                formality_level=info.formality or "informal",
                category=info.category or None,
                region=info.region or "universal",
                is_verified=True,
            )
            db.add(entry)

            for ex_text in (info.examples or []):
                if ex_text:
                    entry.examples.append(
                        WordExample(example_en=ex_text, source="static_seed")
                    )

            seeded += 1

        db.commit()
        print(f"Seeded {seeded} slang entries, skipped {skipped} already existing.")
    except Exception as exc:
        db.rollback()
        print(f"ERROR: {exc}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    seed_slangs()
