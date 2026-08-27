"""
Freezes a fixed, held-out evaluation sample from approved failed_translations
rows so fine-tuning runs have a stable set to score against — one that never
gets folded into training data, and never gets reshuffled on rerun.

Run from the backend root:
  python scripts/export_eval_holdout.py --size 75

Output:
  data/eval_holdout.json   [{ "text", "is_slang", "normalized", "translation" }, ...]

Rows frozen here are tagged in the DB (model_metadata.is_eval_holdout = true),
so export_approved_to_training_data.py excludes them from train/normalizer
exports going forward. Re-running with a larger --size only adds new rows on
top of the existing frozen set; it never removes or reshuffles what's already
frozen, so eval numbers stay comparable across checkpoints over time.
"""

from __future__ import annotations

import argparse
import json
import random
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
HOLDOUT_OUT = DATA_DIR / "eval_holdout.json"
RANDOM_SEED = 20260812  # fixed so previously-selected rows are never reshuffled


def clean(text: Any) -> str:
    return " ".join(str(text or "").strip().split())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, default=75, help="Target number of frozen eval rows (default: 75)")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        stmt = select(FailedTranslation).where(FailedTranslation.status == "approved")
        rows = list(db.scalars(stmt).all())

        already_frozen = [r for r in rows if (r.model_metadata or {}).get("is_eval_holdout")]
        candidates = [r for r in rows if not (r.model_metadata or {}).get("is_eval_holdout")]
        # Deterministic order before the seeded shuffle, so the chosen subset
        # doesn't depend on non-guaranteed DB result ordering.
        candidates.sort(key=lambda r: str(r.id))

        print(f"Approved rows: {len(rows)} total, {len(already_frozen)} already frozen, {len(candidates)} eligible.")

        needed = max(0, args.size - len(already_frozen))
        newly_frozen: list[FailedTranslation] = []
        if needed == 0:
            print(f"Already have {len(already_frozen)} frozen row(s) (>= requested {args.size}). Not adding more.")
        else:
            rng = random.Random(RANDOM_SEED)
            rng.shuffle(candidates)
            newly_frozen = candidates[:needed]
            for row in newly_frozen:
                metadata = dict(row.model_metadata or {})
                metadata["is_eval_holdout"] = True
                row.model_metadata = metadata
                db.add(row)
            db.commit()
            print(f"Froze {len(newly_frozen)} new row(s) into the eval holdout.")

        frozen_rows = already_frozen + newly_frozen
    finally:
        db.close()

    output: list[dict] = []
    for row in frozen_rows:
        text = clean(row.input_text)
        if not text or row.expected_is_slang is None:
            continue
        output.append({
            "text": text,
            "is_slang": bool(row.expected_is_slang),
            "normalized": clean(row.expected_normalized),
            "translation": clean(row.expected_translation) if row.expected_translation else None,
        })

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with HOLDOUT_OUT.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
        f.write("\n")

    slang_count = sum(1 for r in output if r["is_slang"])
    print(f"\nWrote {len(output)} eval rows -> {HOLDOUT_OUT}")
    print(f"  slang: {slang_count}  |  literal: {len(output) - slang_count}")
    print("\nThese rows are now excluded from export_approved_to_training_data.py.")


if __name__ == "__main__":
    main()
