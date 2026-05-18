"""
Build DistilBERT and Flan-T5 training datasets by combining two sources:
  1. failed_translations table from Supabase (known model failures)
  2. simulated_eval candidates from simulate_user_eval_with_openai.py (optional)

For each failed translation, GPT generates correctly labeled training pairs
for both models. The two sources are then merged and deduplicated.

Run from the backend root:
  python scripts/build_datasets_from_failures.py

Include simulated eval output (run simulate_user_eval_with_openai.py first):
  python scripts/build_datasets_from_failures.py \\
    --simulated-detector data/simulated_eval_detector_candidates.json \\
    --simulated-normalizer data/simulated_eval_normalizer_candidates.json

Output files (review before merging into training):
  data/combined_detector_candidates.json
      → review, then append approved rows into data/detector_train.json

  data/combined_normalizer_candidates.json
      → review, then merge via:
        python scripts/build_augmented_normalizer_train.py \\
          --openai-approved data/combined_normalizer_candidates.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from dotenv import load_dotenv

from app.database import get_supabase


DATA_DIR = BACKEND_ROOT / "data"
DEFAULT_SIMULATED_DETECTOR = DATA_DIR / "simulated_eval_detector_candidates.json"
DEFAULT_SIMULATED_NORMALIZER = DATA_DIR / "simulated_eval_normalizer_candidates.json"
DEFAULT_DETECTOR_OUT = DATA_DIR / "combined_detector_candidates.json"
DEFAULT_NORMALIZER_OUT = DATA_DIR / "combined_normalizer_candidates.json"
DEFAULT_REPORT_OUT = DATA_DIR / "combined_datasets_report.json"

MODEL_PRICES_PER_1M: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def clean(text: Any) -> str:
    return " ".join(str(text or "").strip().split())


def norm(text: Any) -> str:
    return clean(text).lower().strip(" .!?")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def read_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [row for row in data if isinstance(row, dict)] if isinstance(data, list) else []


def usage_value(usage: Any, key: str) -> int:
    if usage is None:
        return 0
    if isinstance(usage, dict):
        return int(usage.get(key) or 0)
    return int(getattr(usage, key, 0) or 0)


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    prices = MODEL_PRICES_PER_1M.get(model)
    if not prices:
        return None
    return input_tokens * prices["input"] / 1_000_000 + output_tokens * prices["output"] / 1_000_000


# ── Step 1: Fetch failed_translations from Supabase ───────────────────────────

def fetch_failed_translations(limit: int) -> list[dict[str, Any]]:
    supabase = get_supabase()
    result = (
        supabase.table("failed_translations")
        .select(
            "id, input_text, model_normalized, model_translation, model_is_slang, "
            "model_metadata, expected_normalized, expected_is_slang, failure_type, status"
        )
        .eq("status", "approved")
        .order("created_at", desc=False)
        .limit(limit)
        .execute()
    )
    rows = result.data or []
    # Keep only rows that have an input
    return [row for row in rows if clean(row.get("input_text", ""))]


# ── Step 2: GPT generates training pairs from failed_translations ─────────────

_GENERATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "word": {"type": "string"},
                    "correct_is_slang": {"type": "boolean"},
                    "correct_normalized": {"type": "string"},
                    "detector_candidates": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "text": {"type": "string"},
                                "is_slang": {"type": "boolean"},
                                "confidence": {"type": "number"},
                            },
                            "required": ["text", "is_slang", "confidence"],
                        },
                    },
                    "normalizer_candidates": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "input": {"type": "string"},
                                "target": {"type": "string"},
                                "term": {"type": "string"},
                                "sense": {"type": "string", "enum": ["slang", "literal"]},
                            },
                            "required": ["input", "target", "term", "sense"],
                        },
                    },
                },
                "required": [
                    "word",
                    "correct_is_slang",
                    "correct_normalized",
                    "detector_candidates",
                    "normalizer_candidates",
                ],
            },
        }
    },
    "required": ["results"],
}


def _build_generation_prompt(batch: list[dict[str, Any]]) -> str:
    items = []
    for row in batch:
        metadata = row.get("model_metadata") or {}
        items.append({
            "input_text": clean(row.get("input_text", "")),
            "model_is_slang": row.get("model_is_slang"),
            "model_normalized": clean(row.get("model_normalized", "")),
            "model_translation_pt": clean(row.get("model_translation", "")),
            "formality_level": clean(str(metadata.get("formality_level", "") or "")),
            "user_expected_normalized": clean(row.get("expected_normalized", "")),
            "user_expected_is_slang": row.get("expected_is_slang"),
            "failure_type": clean(row.get("failure_type", "")),
        })

    items_json = json.dumps(items, ensure_ascii=False, indent=2)

    return f"""You are fixing failures from an English slang detection and normalization pipeline used by Brazilian Portuguese learners.

Each entry shows what the model produced AND what the user said the correct answer should be. These are KNOWN failures — the model output was wrong. Use `user_expected_normalized` and `user_expected_is_slang` as ground truth when present.

For each entry:
1. Set correct_is_slang and correct_normalized based on the user-provided expected values (or your own judgment if missing).
2. Generate training data for two models:

detector_candidates — binary slang classifier:
  - Generate 5–8 varied sentences using the same word/phrase, all correctly labeled.
  - For slang: include different contexts where it IS slang.
  - For ambiguous words (fire, dead, sick, etc.): also include 2–3 literal contrast sentences labeled is_slang=false.
  - confidence: 0.90–0.95 for clear cases, 0.75–0.85 for ambiguous ones.

normalizer_candidates — seq2seq normalizer:
  - sense="slang": input is a slang sentence, target is the correct standard English rewrite.
    Only rewrite the slang term, keep the rest of the sentence unchanged.
  - sense="literal": target must be byte-for-byte identical to input. Never rewrite literal sentences.
  - Generate 4–6 pairs including at least one literal contrast pair (same word used literally).
  - Slang rows: target must differ from input.
  - Literal rows: target must equal input exactly.

correct_normalized: the correct standard English form.
  - If slang: rewrite to standard English (e.g. "ofc" → "of course", "thx" → "thanks").
  - If NOT slang: copy the input_text exactly as-is.

The `word` field in your JSON output should contain the input_text value.

Known failures to fix:
{items_json}

Return JSON only.""".strip()


def generate_from_failures(
    client: Any,
    model: str,
    batch: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, int]:
    response = client.responses.create(
        model=model,
        instructions="You fix slang pipeline failures and generate training data for DistilBERT and Flan-T5. Return valid JSON only.",
        input=_build_generation_prompt(batch),
        text={
            "format": {
                "type": "json_schema",
                "name": "failure_training_data",
                "strict": True,
                "schema": _GENERATION_SCHEMA,
            }
        },
    )
    data = json.loads(response.output_text)
    in_tok = usage_value(response.usage, "input_tokens")
    out_tok = usage_value(response.usage, "output_tokens")
    return data.get("results", []), in_tok, out_tok


# ── Validation ────────────────────────────────────────────────────────────────

def _valid_detector_row(row: dict[str, Any]) -> bool:
    return (
        bool(clean(row.get("text", "")))
        and isinstance(row.get("is_slang"), bool)
        and 0.0 <= float(row.get("confidence", -1)) <= 1.0
    )


def _valid_normalizer_row(row: dict[str, Any]) -> bool:
    inp = clean(row.get("input", ""))
    target = clean(row.get("target", ""))
    sense = row.get("sense", "")
    if not inp or not target or sense not in {"slang", "literal"}:
        return False
    if sense == "literal" and inp.lower() != target.lower():
        return False
    if sense == "slang" and inp.lower() == target.lower():
        return False
    if len(target) > max(80, len(inp) * 2):
        return False
    return True


# ── Merge helpers ─────────────────────────────────────────────────────────────

def collect_detector(
    rows: list[dict[str, Any]],
    seen: set[str],
    source_label: str,
) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        text = clean(row.get("text", ""))
        if not text:
            continue
        if not _valid_detector_row(row):
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "text": text,
            "is_slang": row["is_slang"],
            "confidence": round(float(row["confidence"]), 2),
            "source": source_label,
        })
    return out


def collect_normalizer(
    rows: list[dict[str, Any]],
    seen: set[str],
    source_label: str,
) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        inp = clean(row.get("input", ""))
        target = clean(row.get("target", ""))
        if not inp or not target:
            continue
        if not _valid_normalizer_row(row):
            continue
        key = inp.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "input": inp,
            "target": target,
            "term": clean(row.get("term", "")),
            "sense": row["sense"],
            "source": source_label,
        })
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Build combined DistilBERT + Flan-T5 datasets from failed_translations and simulated eval."
    )
    parser.add_argument("--limit", type=int, default=200, help="Max rows to fetch from failed_translations.")
    parser.add_argument("--batch-size", type=int, default=8, help="Rows per GPT call.")
    parser.add_argument("--model", default=os.getenv("OPENAI_DATA_MODEL", "gpt-4o-mini"))
    parser.add_argument("--simulated-detector", type=Path, default=DEFAULT_SIMULATED_DETECTOR)
    parser.add_argument("--simulated-normalizer", type=Path, default=DEFAULT_SIMULATED_NORMALIZER)
    parser.add_argument("--detector-output", type=Path, default=DEFAULT_DETECTOR_OUT)
    parser.add_argument("--normalizer-output", type=Path, default=DEFAULT_NORMALIZER_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_OUT)
    parser.add_argument("--sleep", type=float, default=0.5)
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Install the OpenAI SDK: pip install openai") from exc

    client = OpenAI()
    total_input_tokens = 0
    total_output_tokens = 0

    # ── Step 1: Fetch failed_translations ─────────────────────────────────────
    print(f"Step 1/3 — Fetching failed_translations from Supabase (limit={args.limit})...", flush=True)
    failures = fetch_failed_translations(args.limit)
    print(f"  Fetched {len(failures)} rows.", flush=True)

    # ── Step 2: GPT generates training pairs from failures ────────────────────
    batches = [failures[i : i + args.batch_size] for i in range(0, len(failures), args.batch_size)]
    print(
        f"Step 2/3 — Generating training pairs for {len(failures)} failures in {len(batches)} batches...",
        flush=True,
    )

    all_results: list[dict[str, Any]] = []
    for idx, batch in enumerate(batches, start=1):
        results, in_tok, out_tok = generate_from_failures(client, args.model, batch)
        total_input_tokens += in_tok
        total_output_tokens += out_tok
        all_results.extend(results)
        cost = estimate_cost(args.model, total_input_tokens, total_output_tokens)
        cost_str = f" est_cost=${cost:.4f}" if cost is not None else ""
        print(
            f"  Batch {idx}/{len(batches)} done — "
            f"tokens in:{total_input_tokens} out:{total_output_tokens}{cost_str}",
            flush=True,
        )
        if args.sleep and idx < len(batches):
            time.sleep(args.sleep)

    # ── Step 3: Merge all sources ─────────────────────────────────────────────
    print("Step 3/3 — Merging all sources...", flush=True)

    detector_candidates: list[dict[str, Any]] = []
    normalizer_candidates: list[dict[str, Any]] = []
    seen_detector: set[str] = set()
    seen_normalizer: set[str] = set()
    source_counts: Counter[str] = Counter()

    # Source A: GPT-generated from failed_translations
    for result in all_results:
        det_rows = collect_detector(result.get("detector_candidates", []), seen_detector, "failed_translations_gpt")
        nor_rows = collect_normalizer(result.get("normalizer_candidates", []), seen_normalizer, "failed_translations_gpt")
        detector_candidates.extend(det_rows)
        normalizer_candidates.extend(nor_rows)
        source_counts["failed_translations_gpt_detector"] += len(det_rows)
        source_counts["failed_translations_gpt_normalizer"] += len(nor_rows)

    # Source B: simulated eval detector candidates
    simulated_detector_rows = read_json_list(args.simulated_detector)
    if simulated_detector_rows:
        det_rows = collect_detector(simulated_detector_rows, seen_detector, "simulated_eval")
        detector_candidates.extend(det_rows)
        source_counts["simulated_eval_detector"] += len(det_rows)
        print(f"  Loaded {len(simulated_detector_rows)} simulated detector rows ({len(det_rows)} new).", flush=True)
    else:
        print(f"  No simulated detector file found at {args.simulated_detector} — skipping.", flush=True)

    # Source C: simulated eval normalizer candidates
    simulated_normalizer_rows = read_json_list(args.simulated_normalizer)
    if simulated_normalizer_rows:
        nor_rows = collect_normalizer(simulated_normalizer_rows, seen_normalizer, "simulated_eval")
        normalizer_candidates.extend(nor_rows)
        source_counts["simulated_eval_normalizer"] += len(nor_rows)
        print(f"  Loaded {len(simulated_normalizer_rows)} simulated normalizer rows ({len(nor_rows)} new).", flush=True)
    else:
        print(f"  No simulated normalizer file found at {args.simulated_normalizer} — skipping.", flush=True)

    # Write outputs
    write_json(args.detector_output, detector_candidates)
    write_json(args.normalizer_output, normalizer_candidates)

    total_cost = estimate_cost(args.model, total_input_tokens, total_output_tokens)
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "model": args.model,
        "failed_translations_fetched": len(failures),
        "source_counts": dict(source_counts),
        "detector_candidates_total": len(detector_candidates),
        "normalizer_candidates_total": len(normalizer_candidates),
        "token_usage": {"input": total_input_tokens, "output": total_output_tokens},
        "estimated_cost_usd": round(total_cost, 4) if total_cost is not None else None,
    }
    write_json(args.report, report)

    print(f"\n{'-' * 50}")
    print(f"Failed translations processed : {len(failures)}")
    print(f"Detector candidates total     : {len(detector_candidates)}")
    print(f"Normalizer candidates total   : {len(normalizer_candidates)}")
    print(f"Source breakdown              : {dict(source_counts)}")
    if total_cost is not None:
        print(f"Estimated cost                : ${total_cost:.4f}")
    print(f"\nOutput files:")
    print(f"  {args.detector_output}")
    print(f"  {args.normalizer_output}")
    print(f"\nNext steps:")
    print(f"  1. Review {args.detector_output}")
    print(f"     → append approved rows to data/detector_train.json")
    print(f"  2. Review {args.normalizer_output}")
    print(f"     → python scripts/build_augmented_normalizer_train.py \\")
    print(f"         --openai-approved {args.normalizer_output}")


if __name__ == "__main__":
    main()
