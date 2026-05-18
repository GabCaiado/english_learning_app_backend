"""
Review failed translation reports and export approved corrections.

Typical flow:
  1. Users click "Wrong" in the app.
  2. Run this script with --review to approve/reject/correct reports.
  3. Run this script with --export-approved to create training/eval candidates.

The script intentionally does not retrain models. It prepares clean, reviewed
data that can safely feed future training and gold evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.database import get_supabase

DEFAULT_GOLD_EXPORT = BACKEND_ROOT / "data" / "feedback_gold_candidates.json"
DEFAULT_TRAIN_EXPORT = BACKEND_ROOT / "data" / "feedback_training_candidates.jsonl"


def fetch_rows(status: str, limit: int) -> list[dict[str, Any]]:
    supabase = get_supabase()
    result = (
        supabase.table("failed_translations")
        .select("*")
        .eq("status", status)
        .order("created_at", desc=False)
        .limit(limit)
        .execute()
    )
    return result.data or []


def print_row(row: dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print(f"id: {row['id']}")
    print(f"created_at: {row.get('created_at')}")
    print(f"source: {row.get('source')} | feedback: {row.get('user_feedback')}")
    print(f"\nINPUT:\n{row.get('input_text')}")
    print(f"\nMODEL NORMALIZED:\n{row.get('model_normalized') or ''}")
    print(f"\nMODEL TRANSLATION:\n{row.get('model_translation') or ''}")
    print(f"\nMODEL IS SLANG:\n{row.get('model_is_slang')}")


def prompt_default(label: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{label}{suffix}: ").strip()
    return value if value else (default or "")


def prompt_bool(label: str, default: bool | None = None) -> bool | None:
    default_label = "y" if default is True else "n" if default is False else ""
    raw = prompt_default(f"{label} (y/n/blank unknown)", default_label).lower()
    if raw in {"y", "yes", "true", "1"}:
        return True
    if raw in {"n", "no", "false", "0"}:
        return False
    return None


def review_rows(limit: int) -> None:
    rows = fetch_rows("needs_review", limit)
    if not rows:
        print("No failed translations need review.")
        return

    supabase = get_supabase()
    for row in rows:
        print_row(row)
        action = prompt_default("Action: approve / reject / skip", "skip").lower()

        if action in {"skip", "s", ""}:
            continue

        if action in {"reject", "r"}:
            supabase.table("failed_translations").update(
                {
                    "status": "rejected",
                    "reviewed_at": datetime.now(UTC).isoformat(),
                }
            ).eq("id", row["id"]).execute()
            print("Rejected.")
            continue

        if action not in {"approve", "a"}:
            print("Unknown action; skipped.")
            continue

        expected_normalized = prompt_default(
            "Expected normalized English",
            row.get("model_normalized") or row.get("input_text") or "",
        )
        expected_translation = prompt_default(
            "Expected Portuguese",
            row.get("model_translation") or "",
        )
        expected_is_slang = prompt_bool("Expected is slang", row.get("model_is_slang"))
        failure_type = prompt_default(
            "Failure type",
            "wrong_slang_sense",
        )

        supabase.table("failed_translations").update(
            {
                "expected_normalized": expected_normalized,
                "expected_translation": expected_translation,
                "expected_is_slang": expected_is_slang,
                "failure_type": failure_type,
                "status": "approved",
                "reviewed_at": datetime.now(UTC).isoformat(),
            }
        ).eq("id", row["id"]).execute()
        print("Approved.")


def gold_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "input": row["input_text"],
        "expected": row["expected_normalized"],
        "kind": "slang" if row.get("expected_is_slang") else "literal",
        "source_feedback_id": row["id"],
        "failure_type": row.get("failure_type"),
    }


def training_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "input": row["input_text"],
        "target": row["expected_normalized"],
        "translation_pt": row.get("expected_translation"),
        "sense": "slang" if row.get("expected_is_slang") else "literal",
        "failure_type": row.get("failure_type"),
        "source_feedback_id": row["id"],
    }


def export_approved(
    limit: int,
    gold_path: Path,
    train_path: Path,
    mark_exported: bool,
) -> None:
    rows = [
        row
        for row in fetch_rows("approved", limit)
        if row.get("expected_normalized")
    ]
    if not rows:
        print("No approved failed translations are ready to export.")
        return

    gold_rows = [gold_row(row) for row in rows]
    training_rows = [training_row(row) for row in rows]

    gold_path.parent.mkdir(parents=True, exist_ok=True)
    with gold_path.open("w", encoding="utf-8") as f:
        json.dump(gold_rows, f, ensure_ascii=False, indent=2)
        f.write("\n")

    train_path.parent.mkdir(parents=True, exist_ok=True)
    with train_path.open("w", encoding="utf-8") as f:
        for row in training_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if mark_exported:
        supabase = get_supabase()
        exported_at = datetime.now(UTC).isoformat()
        for row in rows:
            supabase.table("failed_translations").update(
                {
                    "status": "added_to_training",
                    "model_metadata": {
                        **(row.get("model_metadata") or {}),
                        "exported_at": exported_at,
                    },
                }
            ).eq("id", row["id"]).execute()

    print(f"Exported {len(rows)} approved rows.")
    print(f"Gold candidates: {gold_path}")
    print(f"Training candidates: {train_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Review/export failed translation feedback.")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--review", action="store_true", help="Interactively review needs_review rows.")
    parser.add_argument("--list", action="store_true", help="List needs_review rows without editing.")
    parser.add_argument("--export-approved", action="store_true", help="Export approved rows to data files.")
    parser.add_argument("--gold-path", default=str(DEFAULT_GOLD_EXPORT))
    parser.add_argument("--train-path", default=str(DEFAULT_TRAIN_EXPORT))
    parser.add_argument("--mark-exported", action="store_true")
    args = parser.parse_args()

    if args.list:
        rows = fetch_rows("needs_review", args.limit)
        for row in rows:
            print_row(row)
        print(f"\nListed {len(rows)} rows.")
        return

    if args.review:
        review_rows(args.limit)
        return

    if args.export_approved:
        export_approved(
            limit=args.limit,
            gold_path=Path(args.gold_path),
            train_path=Path(args.train_path),
            mark_exported=args.mark_exported,
        )
        return

    parser.print_help()


if __name__ == "__main__":
    main()
