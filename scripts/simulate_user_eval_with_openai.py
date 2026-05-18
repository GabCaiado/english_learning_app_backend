"""
Simulate a user inputting diverse English sentences, run them through the
production slang pipeline, have GPT judge each output, and generate training
data for both DistilBERT (detector) and Flan-T5 (normalizer) from failures.

Run from the backend root:
  python scripts/simulate_user_eval_with_openai.py

More sentences per category:
  python scripts/simulate_user_eval_with_openai.py --sentences-per-category 20

Cheaper / faster run:
  python scripts/simulate_user_eval_with_openai.py --sentences-per-category 8 --model gpt-4.1-nano

Output files (review before merging):
  data/simulated_eval_detector_candidates.json   → merge into data/detector_train.json
  data/simulated_eval_normalizer_candidates.json → merge via scripts/build_augmented_normalizer_train.py
  data/simulated_eval_report.json                → summary
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

from app.ml.context_resolver import ContextResolver
from app.ml.normalizer import SlangNormalizer
from app.ml.slang_detector import SlangDetector
from app.ml.slang_dictionary import SlangDictionary
from scripts.smoke_test_slang_pipeline import normalize_sentence_with_trace


DATA_DIR = BACKEND_ROOT / "data"
DEFAULT_DETECTOR_OUT = DATA_DIR / "simulated_eval_detector_candidates.json"
DEFAULT_NORMALIZER_OUT = DATA_DIR / "simulated_eval_normalizer_candidates.json"
DEFAULT_REPORT_OUT = DATA_DIR / "simulated_eval_report.json"

MODEL_PRICES_PER_1M: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
}

CATEGORIES = [
    {
        "name": "ambiguous",
        "discovery_goal": (
            "English words that have BOTH a common literal meaning AND a slang/idiomatic meaning, "
            "making them hard for a slang detector to classify correctly from context alone. "
            "Think broadly: adjectives, verbs, nouns that Gen Z repurposed. "
            "Include the obvious ones AND rare or surprising ones the model is unlikely to handle well."
        ),
        "seed_terms": [
            "fire", "dead", "sick", "hard", "down", "cold", "dope", "goat", "ghost",
            "lit", "tea", "shade", "beat", "cooked", "salty", "cap", "mid", "washed",
            "cracked", "sharp", "tight", "slap", "ate", "clean", "wild", "pressed",
            "cooked", "flex", "drip", "bussin", "extra", "lowkey",
        ],
        "sentence_rules": (
            "For every term, generate ONE sentence where it is clearly slang and ONE where it is "
            "clearly literal. Both sentences must use the same term. "
            "Spread evenly — every term in your list must appear at least once."
        ),
    },
    {
        "name": "abbreviation",
        "discovery_goal": (
            "Internet and text abbreviations, acronyms, and informal shortenings used in English "
            "online communication that a pipeline trained on formal English will likely fail on. "
            "Include very common ones AND less obvious ones."
        ),
        "seed_terms": [
            "ofc", "thx", "tks", "stfu", "gotcha", "ima", "ngl", "imo", "lmao",
            "smh", "brb", "btw", "idk", "omg", "tbh", "rn", "gg", "gl", "wdym",
            "imo", "irl", "fwiw", "istg", "idc", "wyd", "hmu", "ion", "yk",
        ],
        "sentence_rules": (
            "Each sentence should use one abbreviation naturally, as a learner would type it. "
            "Some can be standalone (just the abbreviation), others in a sentence. "
            "Spread evenly across all terms."
        ),
    },
    {
        "name": "slang_phrase",
        "discovery_goal": (
            "Modern English slang words, idioms, and multi-word phrases that a Brazilian Portuguese "
            "learner would encounter online and look up. These need normalization to standard English. "
            "Include Gen Z slang, gaming slang, social media phrases, and informal idioms "
            "that a pipeline might miss or normalize incorrectly."
        ),
        "seed_terms": [
            "simping", "spill the tea", "bailed", "flaked", "nailed it", "i'm dead",
            "are you down", "hang out", "showing off", "he's a jerk", "ghosted",
            "i'm in", "rizz", "no cap", "hits different", "rent free", "slay",
            "understood the assignment", "it's giving", "main character", "touch grass",
            "caught in 4k", "living in my head", "based", "mid", "ratio", "sus",
            "let him cook", "ate and left no crumbs",
        ],
        "sentence_rules": (
            "Each sentence should use one slang phrase naturally in context. "
            "Short, as a learner would type it. "
            "Spread evenly across all phrases in your list."
        ),
    },
    {
        "name": "literal",
        "discovery_goal": (
            "English words that have a well-known slang meaning BUT also have a completely normal "
            "literal use — and the pipeline might wrongly flag or rewrite the literal version. "
            "Think of words where the literal meaning is the PRIMARY meaning in everyday use."
        ),
        "seed_terms": [
            "fire", "dead", "sick", "hard", "down", "cold", "beat", "ghost", "tea",
            "shade", "goat", "dope", "lit", "cooked", "wild", "salty", "clean",
            "bright", "heavy", "solid", "dark", "raw", "smooth", "bitter",
        ],
        "sentence_rules": (
            "Every sentence must be completely standard English where the word is used literally. "
            "The pipeline must NOT rewrite these. "
            "Spread across all terms — every term must appear at least once."
        ),
    },
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def clean(text: Any) -> str:
    return " ".join(str(text or "").strip().split())


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


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


# ── Step 1: Generate test sentences (discover terms → generate sentences) ─────

_DISCOVERY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "terms": {
            "type": "array",
            "items": {"type": "string"},
        }
    },
    "required": ["terms"],
}

_SENTENCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "sentences": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "text": {"type": "string"},
                    "category": {
                        "type": "string",
                        "enum": ["ambiguous", "abbreviation", "slang_phrase", "literal"],
                    },
                    "target_term": {"type": "string"},
                },
                "required": ["text", "category", "target_term"],
            },
        }
    },
    "required": ["sentences"],
}


def _build_discovery_prompt(category: dict[str, Any], target_count: int) -> str:
    seeds = ", ".join(category["seed_terms"])
    return f"""You are helping build a test suite for an English slang detection and normalization pipeline.

Task: find {target_count} {category['name']} terms to use as test targets.

Goal: {category['discovery_goal']}

Seed terms already known (include all of these AND add more of your own):
{seeds}

Rules:
- Include ALL seed terms above.
- Add enough new terms to reach {target_count} total.
- New terms must genuinely fit the goal — think about what would actually trip up an NLP model.
- No duplicates, no offensive slurs, no brand names.
- Return a flat list of strings.

Return JSON only.""".strip()


def _build_sentence_prompt(
    category: dict[str, Any], terms: list[str], count: int, already: list[str] | None = None
) -> str:
    name = category["name"]
    rules = category["sentence_rules"]
    terms_str = ", ".join(terms)
    already_block = ""
    if already:
        already_str = "\n".join(f"  - {t}" for t in already)
        already_block = f"\nDo NOT repeat any of these already-generated sentences:\n{already_str}\n"
    return f"""You are simulating a Brazilian Portuguese speaker learning English who uses an app to look up words and sentences they encounter online.

Generate exactly {count} NEW English sentences or words such a user might type into the app.

Category: {name}
Terms to cover: {terms_str}
{already_block}
Sentence rules:
- {rules}
- Short and natural, as a real learner would type (1–2 clauses max).
- Lowercase only, except proper nouns.
- No duplicate sentences. Be creative — vary structures, contexts, and terms used.
- target_term: the exact term from the list being tested.
- category: always "{name}".

Return JSON only.""".strip()


def _discover_terms(
    client: Any, model: str, category: dict[str, Any], target_count: int
) -> tuple[list[str], int, int]:
    response = client.responses.create(
        model=model,
        instructions="You select high-value test terms for an English slang pipeline evaluation. Return valid JSON only.",
        input=_build_discovery_prompt(category, target_count),
        text={
            "format": {
                "type": "json_schema",
                "name": "term_discovery",
                "strict": True,
                "schema": _DISCOVERY_SCHEMA,
            }
        },
    )
    data = json.loads(response.output_text)
    terms = [t.strip().lower() for t in data.get("terms", []) if t.strip()]
    # Always include seeds, deduped
    seen: set[str] = set()
    merged: list[str] = []
    for t in category["seed_terms"] + terms:
        if t not in seen:
            seen.add(t)
            merged.append(t)
    in_tok = usage_value(response.usage, "input_tokens")
    out_tok = usage_value(response.usage, "output_tokens")
    return merged, in_tok, out_tok


def generate_test_sentences(
    client: Any, model: str, sentences_per_category: int, sleep: float,
    batch_size: int = 35, max_rounds: int = 6,
) -> tuple[list[dict[str, Any]], int, int]:
    terms_per_category = max(len(CATEGORIES[0]["seed_terms"]), sentences_per_category // 3)
    print(
        f"Step 1/3 — Discovering terms then generating ~{sentences_per_category * len(CATEGORIES)} sentences "
        f"({batch_size}/call, up to {max_rounds} rounds per category)...",
        flush=True,
    )
    all_sentences: list[dict[str, Any]] = []
    total_in = 0
    total_out = 0
    seen_texts: set[str] = set()

    for cat in CATEGORIES:
        # Substep A: discover terms
        terms, in_tok, out_tok = _discover_terms(client, model, cat, terms_per_category)
        total_in += in_tok
        total_out += out_tok
        print(f"  [{cat['name']}] discovered {len(terms)} terms.", flush=True)
        if sleep:
            time.sleep(sleep)

        # Substep B: loop in small batches until we hit the target
        cat_sentences: list[dict[str, Any]] = []
        rounds_without_new = 0

        for round_idx in range(max_rounds):
            if len(cat_sentences) >= sentences_per_category:
                break
            if rounds_without_new >= 2:
                break

            needed = sentences_per_category - len(cat_sentences)
            ask_for = min(batch_size, needed)
            already = [s["text"] for s in cat_sentences[-20:]]  # show last 20 to avoid repeats

            prompt = _build_sentence_prompt(cat, terms, ask_for, already)
            response = client.responses.create(
                model=model,
                instructions="You generate compact, valid JSON for English slang pipeline evaluation. Follow the schema exactly.",
                input=prompt,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "simulated_eval_sentences",
                        "strict": True,
                        "schema": _SENTENCE_SCHEMA,
                    }
                },
            )
            total_in += usage_value(response.usage, "input_tokens")
            total_out += usage_value(response.usage, "output_tokens")

            rows = [s for s in json.loads(response.output_text).get("sentences", []) if clean(s.get("text", ""))]
            new_this_round = 0
            for row in rows:
                key = clean(row["text"]).lower()
                if key in seen_texts:
                    continue
                seen_texts.add(key)
                cat_sentences.append(row)
                all_sentences.append(row)
                new_this_round += 1

            rounds_without_new = 0 if new_this_round > 0 else rounds_without_new + 1
            print(
                f"  [{cat['name']}] round {round_idx + 1}: +{new_this_round} "
                f"(total {len(cat_sentences)}/{sentences_per_category})",
                flush=True,
            )
            if sleep:
                time.sleep(sleep)

    print(f"  Total generated: {len(all_sentences)} sentences.", flush=True)
    return all_sentences, total_in, total_out


# ── Step 2: Run pipeline ──────────────────────────────────────────────────────

def run_pipeline(
    sentences: list[dict[str, Any]],
    dictionary: SlangDictionary,
    detector: SlangDetector,
    resolver: ContextResolver,
    normalizer: SlangNormalizer,
) -> list[dict[str, Any]]:
    print(f"Step 2/3 — Running pipeline on {len(sentences)} sentences...", flush=True)
    results = []
    for item in sentences:
        text = clean(item["text"])
        trace = normalize_sentence_with_trace(text, dictionary, detector, resolver, normalizer)
        normalized = clean(trace["normalized"])
        slangs_found = trace["slangs_found"]
        pipeline_is_slang = bool(slangs_found) or normalized.lower() != text.lower()
        results.append(
            {
                "text": text,
                "category": item.get("category", ""),
                "target_term": item.get("target_term", ""),
                "pipeline_is_slang": pipeline_is_slang,
                "pipeline_normalized": normalized,
                "pipeline_detector_score": round(float(trace["detector_score"]), 4),
                "pipeline_slangs_found": [s.get("original", "") for s in slangs_found],
            }
        )
    print(f"  Pipeline done.", flush=True)
    return results


# ── Step 3: GPT judges outputs ────────────────────────────────────────────────

_JUDGMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "text": {"type": "string"},
                    "pipeline_is_correct": {"type": "boolean"},
                    "correct_is_slang": {"type": "boolean"},
                    "correct_normalized": {"type": "string"},
                    "failure_type": {
                        "type": "string",
                        "enum": ["ok", "missed_slang", "false_positive", "wrong_normalization"],
                    },
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
                    "text",
                    "pipeline_is_correct",
                    "correct_is_slang",
                    "correct_normalized",
                    "failure_type",
                    "detector_candidates",
                    "normalizer_candidates",
                ],
            },
        }
    },
    "required": ["results"],
}


def _build_judgment_prompt(batch: list[dict[str, Any]]) -> str:
    items_json = json.dumps(
        [
            {
                "text": item["text"],
                "pipeline_is_slang": item["pipeline_is_slang"],
                "pipeline_normalized": item["pipeline_normalized"],
                "detector_score": item["pipeline_detector_score"],
            }
            for item in batch
        ],
        ensure_ascii=False,
        indent=2,
    )
    return f"""You are evaluating an English slang detection and normalization pipeline used by Brazilian Portuguese learners.

For each input, the pipeline returned: is_slang (bool) and normalized (standard English rewrite, or unchanged if not slang).

Evaluate each result and classify the failure type:
- "ok": pipeline was correct.
- "missed_slang": pipeline said not-slang but it is slang/idiomatic.
- "false_positive": pipeline said slang but it is standard English.
- "wrong_normalization": detected slang correctly but rewrote it incorrectly.

For every failure (not "ok"), generate training data for two models:

1. detector_candidates — DistilBERT binary classifier:
   - Generate 4–6 varied phrasings of the same idea, all correctly labeled.
   - Include at least one contrast sentence (literal use for missed_slang; slang use for false_positive).
   - confidence: 0.90–0.95 for obvious cases, 0.75–0.85 for ambiguous ones.

2. normalizer_candidates — Flan-T5 seq2seq normalizer:
   - sense="slang": input is the slang sentence, target is the standard English rewrite. Only rewrite the slang term, keep the rest unchanged.
   - sense="literal": target must be byte-for-byte identical to input. Never rewrite literal sentences.
   - Generate 3–5 pairs including at least one contrast (literal identity pair for the same word).

correct_normalized: the correct standard English form. If not slang, copy the input exactly.

For "ok" results, leave detector_candidates and normalizer_candidates as empty arrays.

Pipeline outputs to evaluate:
{items_json}

Return JSON only.""".strip()


def judge_batch(
    client: Any, model: str, batch: list[dict[str, Any]], retries: int = 5
) -> tuple[list[dict[str, Any]], int, int]:
    delay = 5.0
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            response = client.responses.create(
                model=model,
                instructions="You evaluate slang pipeline outputs and generate training data. Return valid JSON only.",
                input=_build_judgment_prompt(batch),
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "simulated_eval_judgment",
                        "strict": True,
                        "schema": _JUDGMENT_SCHEMA,
                    }
                },
            )
            data = json.loads(response.output_text)
            in_tok = usage_value(response.usage, "input_tokens")
            out_tok = usage_value(response.usage, "output_tokens")
            return data.get("results", []), in_tok, out_tok
        except Exception as exc:
            last_exc = exc
            print(f"  [warn] batch attempt {attempt + 1}/{retries} failed: {exc!r} — retrying in {delay:.0f}s", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
    raise RuntimeError(f"judge_batch failed after {retries} attempts") from last_exc


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


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Simulate user eval and generate DistilBERT + Flan-T5 training data with OpenAI."
    )
    parser.add_argument("--sentences-per-category", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=10, help="Sentences per judgment API call.")
    parser.add_argument("--model", default=os.getenv("OPENAI_DATA_MODEL", "gpt-4o-mini"))
    parser.add_argument("--detector-output", type=Path, default=DEFAULT_DETECTOR_OUT)
    parser.add_argument("--normalizer-output", type=Path, default=DEFAULT_NORMALIZER_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_OUT)
    parser.add_argument("--sleep", type=float, default=0.5, help="Seconds between API calls.")
    parser.add_argument("--checkpoint", type=Path, default=DATA_DIR / "simulated_eval_checkpoint.json",
                        help="Checkpoint file — resume from here if it exists.")
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

    # Step 1: Generate test sentences
    generated_sentences, in_tok, out_tok = generate_test_sentences(
        client, args.model, args.sentences_per_category, args.sleep
    )
    total_input_tokens += in_tok
    total_output_tokens += out_tok

    # Step 2: Run pipeline
    print("Loading pipeline models (this may take a moment)...", flush=True)
    dictionary = SlangDictionary()
    detector = SlangDetector()
    resolver = ContextResolver()
    normalizer = SlangNormalizer()
    pipeline_results = run_pipeline(generated_sentences, dictionary, detector, resolver, normalizer)

    # Step 3: Judge in batches
    batches = [
        pipeline_results[i : i + args.batch_size]
        for i in range(0, len(pipeline_results), args.batch_size)
    ]
    print(
        f"Step 3/3 — Judging {len(pipeline_results)} outputs in {len(batches)} batches...",
        flush=True,
    )

    # Load checkpoint if present
    checkpoint_start = 0
    all_judgments: list[dict[str, Any]] = []
    if args.checkpoint.exists():
        try:
            ckpt = json.loads(args.checkpoint.read_text())
            all_judgments = ckpt.get("judgments", [])
            checkpoint_start = ckpt.get("next_batch", 0)
            total_input_tokens += ckpt.get("input_tokens", 0)
            total_output_tokens += ckpt.get("output_tokens", 0)
            print(f"  Resuming from checkpoint: {checkpoint_start} batches already done, {len(all_judgments)} judgments loaded.", flush=True)
        except Exception as exc:
            print(f"  [warn] Could not load checkpoint ({exc}), starting fresh.", flush=True)

    for idx, batch in enumerate(batches, start=1):
        if idx <= checkpoint_start:
            continue
        judgments, in_tok, out_tok = judge_batch(client, args.model, batch)
        total_input_tokens += in_tok
        total_output_tokens += out_tok
        all_judgments.extend(judgments)
        # Save checkpoint after every batch
        args.checkpoint.write_text(json.dumps({
            "next_batch": idx,
            "judgments": all_judgments,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
        }, ensure_ascii=False))
        cost = estimate_cost(args.model, total_input_tokens, total_output_tokens)
        cost_str = f" est_cost=${cost:.4f}" if cost is not None else ""
        print(
            f"  Batch {idx}/{len(batches)} done — "
            f"tokens in:{total_input_tokens} out:{total_output_tokens}{cost_str}",
            flush=True,
        )
        if args.sleep and idx < len(batches):
            time.sleep(args.sleep)

    # Collect and validate training candidates from failures
    detector_candidates: list[dict[str, Any]] = []
    normalizer_candidates: list[dict[str, Any]] = []
    failure_counts: Counter[str] = Counter()
    seen_detector: set[str] = set()
    seen_normalizer: set[str] = set()

    for judgment in all_judgments:
        failure_type = judgment.get("failure_type", "ok")
        failure_counts[failure_type] += 1

        for row in judgment.get("detector_candidates", []):
            row["text"] = clean(row.get("text", ""))
            if not _valid_detector_row(row):
                continue
            key = row["text"].lower()
            if key in seen_detector:
                continue
            seen_detector.add(key)
            detector_candidates.append(
                {
                    "text": row["text"],
                    "is_slang": row["is_slang"],
                    "confidence": round(float(row["confidence"]), 2),
                }
            )

        for row in judgment.get("normalizer_candidates", []):
            row["input"] = clean(row.get("input", ""))
            row["target"] = clean(row.get("target", ""))
            if not _valid_normalizer_row(row):
                continue
            key = row["input"].lower()
            if key in seen_normalizer:
                continue
            seen_normalizer.add(key)
            normalizer_candidates.append(
                {
                    "input": row["input"],
                    "target": row["target"],
                    "term": clean(row.get("term", "")),
                    "sense": row["sense"],
                    "source": "simulated_eval",
                }
            )

    # Write outputs
    write_json(args.detector_output, detector_candidates)
    write_json(args.normalizer_output, normalizer_candidates)

    total_cost = estimate_cost(args.model, total_input_tokens, total_output_tokens)
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "model": args.model,
        "sentences_generated": len(generated_sentences),
        "sentences_evaluated": len(all_judgments),
        "failure_counts": dict(failure_counts),
        "pipeline_accuracy": round(failure_counts["ok"] / max(len(all_judgments), 1), 3),
        "detector_candidates_written": len(detector_candidates),
        "normalizer_candidates_written": len(normalizer_candidates),
        "token_usage": {"input": total_input_tokens, "output": total_output_tokens},
        "estimated_cost_usd": round(total_cost, 4) if total_cost is not None else None,
    }
    write_json(args.report, report)

    # Clean up checkpoint now that the run completed successfully
    if args.checkpoint.exists():
        args.checkpoint.unlink()

    print(f"\n{'─' * 50}")
    print(f"Sentences evaluated  : {len(all_judgments)}")
    print(f"Pipeline accuracy    : {report['pipeline_accuracy']:.1%}")
    print(f"Failure breakdown    : {dict(failure_counts)}")
    print(f"Detector candidates  : {len(detector_candidates)} → {args.detector_output}")
    print(f"Normalizer candidates: {len(normalizer_candidates)} → {args.normalizer_output}")
    if total_cost is not None:
        print(f"Estimated cost       : ${total_cost:.4f}")
    print(f"\nNext steps:")
    print(f"  1. Review {args.detector_output}")
    print(f"     → merge approved rows into data/detector_train.json")
    print(f"  2. Review {args.normalizer_output}")
    print(f"     → merge via: python scripts/build_augmented_normalizer_train.py --openai-approved {args.normalizer_output}")


if __name__ == "__main__":
    main()
