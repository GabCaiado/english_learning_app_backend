"""
Deletes needs_review failed_translations rows that came from the synthetic
dataset generator (source ai_generated / ai_amplified), leaving real
user-submitted feedback untouched.

Usage:
  python scripts/clear_ai_review_queue.py            # dry run, only counts
  python scripts/clear_ai_review_queue.py --confirm  # actually deletes
"""

from __future__ import annotations

import argparse

from app.database import SessionLocal
from app.models.failed_translation import FailedTranslation

AI_SOURCES = ("ai_generated", "ai_amplified")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm", action="store_true", help="Actually delete rows instead of just counting them.")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        query = db.query(FailedTranslation).filter(
            FailedTranslation.status == "needs_review",
            FailedTranslation.source.in_(AI_SOURCES),
        )
        count = query.count()
        print(f"Matching rows (status=needs_review, source in {AI_SOURCES}): {count}")

        if not args.confirm:
            print("Dry run only. Re-run with --confirm to delete.")
            return

        deleted = query.delete(synchronize_session=False)
        db.commit()
        print(f"Deleted {deleted} rows.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
