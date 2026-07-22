"""
fix_nonslang_normalized.py
---------------------------
One-off repair for dataset_generator.py's pre-fix bug: rows generated with
is_slang=false sometimes had model_normalized/expected_normalized paraphrased
into a synonym instead of staying identical to the original sentence. This
forces normalized text back to the original input for every non-slang row.

Run:
  python scripts/fix_nonslang_normalized.py --dry-run
  python scripts/fix_nonslang_normalized.py
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import select, and_, or_
from app.database import SessionLocal
from app.models.failed_translation import FailedTranslation


def run_fix(dry_run: bool = False) -> None:
    db = SessionLocal()
    try:
        stmt = select(FailedTranslation).where(
            or_(
                and_(
                    FailedTranslation.model_is_slang.is_(False),
                    FailedTranslation.model_normalized.isnot(None),
                    FailedTranslation.model_normalized != FailedTranslation.input_text,
                ),
                and_(
                    FailedTranslation.expected_is_slang.is_(False),
                    FailedTranslation.expected_normalized.isnot(None),
                    FailedTranslation.expected_normalized != FailedTranslation.input_text,
                ),
            )
        )
        rows = list(db.scalars(stmt).all())
        print(f"Rows to fix: {len(rows)}")

        for row in rows:
            if row.model_is_slang is False and row.model_normalized != row.input_text:
                print(f"  [{row.id}] model_normalized: {row.model_normalized!r} -> {row.input_text!r}")
                if not dry_run:
                    row.model_normalized = row.input_text
            if row.expected_is_slang is False and row.expected_normalized != row.input_text:
                print(f"  [{row.id}] expected_normalized: {row.expected_normalized!r} -> {row.input_text!r}")
                if not dry_run:
                    row.expected_normalized = row.input_text

        if not dry_run:
            db.commit()
            print("Fix committed.")
        else:
            print("Dry-run mode — no changes committed.")

    except Exception as exc:
        db.rollback()
        print(f"ERROR during fix: {exc}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix paraphrased normalized text on non-slang rows")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without committing")
    args = parser.parse_args()
    run_fix(dry_run=args.dry_run)
