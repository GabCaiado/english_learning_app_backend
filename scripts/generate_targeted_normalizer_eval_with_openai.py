"""
Generates targeted normalizer evaluation candidates with the OpenAI API.

The generated rows are candidates, not training data. Review them first, then
copy approved rows into scripts/build_normalizer_v3_dataset.py so they become
both train signal and locked regression tests.

Run:
  python scripts/generate_targeted_normalizer_eval_with_openai.py --examples-per-term 12

Let OpenAI choose a broader set of ambiguous terms first:
  python scripts/generate_targeted_normalizer_eval_with_openai.py --discover-terms 150 --examples-per-term 8 --output data/targeted_normalizer_eval_candidates_batch2.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


OUTPUT_PATH = Path("data/targeted_normalizer_eval_candidates.json")
DISCOVERED_TERMS_PATH = Path("data/targeted_normalizer_discovered_terms.json")

MODEL_PRICES_PER_1M = {
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-5-nano": {"input": 0.05, "output": 0.40},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}

TARGET_TERMS = [
    {
        "term": "jam",
        "notes": "literal fruit spread vs slang for favorite song/thing",
        "examples": [
            ("i put jam on my bread.", "i put jam on my bread.", "literal"),
            ("this song is my jam.", "this song is my favorite.", "slang"),
        ],
    },
    {
        "term": "chill",
        "notes": "literal cold/cool weather vs relaxed personality or calm instruction",
        "examples": [
            ("the weather is chill today.", "the weather is chill today.", "literal"),
            ("he is a chill guy.", "he is a relaxed guy.", "slang"),
        ],
    },
    {
        "term": "tight",
        "notes": "literal physically tight vs close relationship or excellent",
        "examples": [
            ("these shoes are tight.", "these shoes are tight.", "literal"),
            ("we're tight.", "we are close friends.", "slang"),
        ],
    },
    {
        "term": "nasty",
        "notes": "literal disgusting vs impressive in music/sports",
        "examples": [
            ("the bathroom smells nasty.", "the bathroom smells nasty.", "literal"),
            ("that guitar solo was nasty.", "that guitar solo was amazing.", "slang"),
        ],
    },
    {
        "term": "cracked",
        "notes": "literal broken/cracked vs very skilled in games",
        "examples": [
            ("the glass is cracked.", "the glass is cracked.", "literal"),
            ("he is cracked at fortnite.", "he is very good at fortnite.", "slang"),
        ],
    },
    {
        "term": "washed",
        "notes": "literal cleaned vs no longer good at something",
        "examples": [
            ("the shirt was washed.", "the shirt was washed.", "literal"),
            ("that player is washed.", "that player is no longer good.", "slang"),
        ],
    },
    {
        "term": "cooked",
        "notes": "literal food cooked vs in trouble/doomed/exhausted",
        "examples": [
            ("the chicken is cooked.", "the chicken is cooked.", "literal"),
            ("i'm cooked for the exam.", "i am in trouble for the exam.", "slang"),
        ],
    },
    {
        "term": "ate",
        "notes": "literal eating vs did extremely well",
        "examples": [
            ("she ate dinner early.", "she ate dinner early.", "literal"),
            ("she ate that performance.", "she did extremely well in that performance.", "slang"),
        ],
    },
    {
        "term": "sharp",
        "notes": "literal sharp edge vs stylish/smart-looking",
        "examples": [
            ("the knife is sharp.", "the knife is sharp.", "literal"),
            ("you look sharp today.", "you look stylish today.", "slang"),
        ],
    },
    {
        "term": "hooked",
        "notes": "literal caught on a hook vs addicted/interested",
        "examples": [
            ("the fish was hooked.", "the fish was hooked.", "literal"),
            ("i'm hooked on this show.", "i am very interested in this show.", "slang"),
        ],
    },
]


def clean(text: str) -> str:
    return " ".join((text or "").strip().split())


def normalize_example(example: Any) -> tuple[str, str, str] | None:
    if isinstance(example, dict):
        source = clean(example.get("input", ""))
        target = clean(example.get("target", ""))
        sense = clean(example.get("sense", ""))
    elif isinstance(example, (list, tuple)) and len(example) == 3:
        source = clean(str(example[0]))
        target = clean(str(example[1]))
        sense = clean(str(example[2]))
    else:
        return None
    if not source or not target or sense not in {"literal", "slang"}:
        return None
    if sense == "literal" and source.lower() != target.lower():
        return None
    return source, target, sense


def normalize_term_spec(spec: dict[str, Any]) -> dict[str, Any] | None:
    term = clean(spec.get("term", "")).lower()
    notes = clean(spec.get("notes", ""))
    raw_examples = spec.get("examples", [])
    if not term or not notes or not isinstance(raw_examples, list):
        return None

    examples = []
    senses = set()
    for raw_example in raw_examples:
        example = normalize_example(raw_example)
        if not example:
            continue
        examples.append(example)
        senses.add(example[2])

    if "slang" not in senses:
        return None

    return {"term": term, "notes": notes, "examples": examples[:6], "senses": sorted(senses)}


def load_term_specs(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        raw_terms = data.get("terms") or data.get("data") or []
    elif isinstance(data, list):
        raw_terms = data
    else:
        raw_terms = []

    terms = []
    for item in raw_terms:
        if not isinstance(item, dict):
            continue
        spec = normalize_term_spec(item)
        if spec:
            terms.append(spec)
    return dedupe_term_specs(terms)


def dedupe_term_specs(term_specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique = []
    for spec in term_specs:
        term = spec["term"].lower()
        if term in seen:
            continue
        seen.add(term)
        unique.append(spec)
    return unique


def build_discovery_prompt(term_count: int, existing_terms: list[str]) -> str:
    return f"""
Find {term_count} English words or short phrases that are high-risk evaluation targets for an English slang normalizer.

Goal:
- The normalizer must rewrite slang/idiomatic meanings into standard English.
- It must leave literal/non-slang meanings unchanged.
- We need ambiguous terms where both meanings are plausible in short learner sentences.

Avoid:
- Terms that only have slang meanings and no common literal meaning.
- Offensive slurs or explicit sexual content.
- Proper nouns, brand names, and rare niche memes.
- Duplicate morphological variants unless they create a genuinely different ambiguity.
- These already-covered terms unless they are essential:
{json.dumps(existing_terms, ensure_ascii=False)}

For each term, include:
- term: lowercase word or short phrase.
- notes: concise description of the literal vs slang/idiomatic ambiguity.
- examples: exactly two seed examples, one literal identity row and one slang normalization row.

Return only JSON matching this shape:
{{"terms":[{{"term":"...","notes":"literal ... vs slang ...","examples":[{{"input":"...","target":"...","sense":"literal"}},{{"input":"...","target":"...","sense":"slang"}}]}}]}}
""".strip()


def parse_discovered_terms(text: str) -> list[dict[str, Any]]:
    data = json.loads(text)
    if not isinstance(data, dict):
        return []
    raw_terms = data.get("terms", [])
    if not isinstance(raw_terms, list):
        return []

    terms = []
    for item in raw_terms:
        if not isinstance(item, dict):
            continue
        spec = normalize_term_spec(item)
        if spec:
            terms.append(spec)
    return dedupe_term_specs(terms)


def discover_term_specs(
    client: Any,
    model: str,
    term_count: int,
    existing_term_specs: list[dict[str, Any]],
    output_path: Path,
    batch_size: int,
) -> list[dict[str, Any]]:
    discovered: list[dict[str, Any]] = []
    rounds_without_new_terms = 0
    print(f"Discovering {term_count} ambiguous terms with {model}...", flush=True)

    while len(discovered) < term_count and rounds_without_new_terms < 3:
        request_count = min(batch_size, term_count - len(discovered))
        existing_terms = [spec["term"] for spec in dedupe_term_specs(existing_term_specs + discovered)]
        print(
            f"Discovery batch: requesting {request_count} more terms "
            f"({len(discovered)}/{term_count} collected)...",
            flush=True,
        )
        response = client.responses.create(
            model=model,
            instructions=(
                "You are selecting compact, high-value test terms for an English "
                "slang normalizer. Output valid JSON only."
            ),
            input=build_discovery_prompt(request_count, existing_terms),
            text={
                "format": {
                    "type": "json_schema",
                    "name": "targeted_normalizer_discovered_terms",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "terms": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "term": {"type": "string"},
                                        "notes": {"type": "string"},
                                        "examples": {
                                            "type": "array",
                                            "minItems": 2,
                                            "maxItems": 2,
                                            "items": {
                                                "type": "object",
                                                "additionalProperties": False,
                                                "properties": {
                                                    "input": {"type": "string"},
                                                    "target": {"type": "string"},
                                                    "sense": {"type": "string", "enum": ["literal", "slang"]},
                                                },
                                                "required": ["input", "target", "sense"],
                                            },
                                        },
                                    },
                                    "required": ["term", "notes", "examples"],
                                },
                            }
                        },
                        "required": ["terms"],
                    },
                }
            },
        )

        before = len(discovered)
        discovered = dedupe_term_specs(discovered + parse_discovered_terms(response.output_text))
        added = len(discovered) - before
        print(f"Discovery batch accepted {added} new terms.", flush=True)
        rounds_without_new_terms = rounds_without_new_terms + 1 if added == 0 else 0

    if not discovered:
        raise SystemExit("The model did not return any valid discovered terms.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(discovered, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Wrote {len(discovered)} discovered terms to {output_path}", flush=True)
    return discovered


def build_prompt(term_spec: dict[str, Any], examples_per_term: int) -> str:
    senses = set(term_spec.get("senses") or [])
    if senses == {"literal", "slang"}:
        balance_instruction = (
            f"Include a balanced mix of literal and slang rows. If {examples_per_term} is even, "
            "generate exactly half literal and half slang."
        )
    else:
        balance_instruction = (
            "Generate mostly slang normalization rows. Include literal identity rows only if the term "
            "has a common ordinary non-slang meaning; never invent fake literal uses for slang-only terms."
        )

    return f"""
Generate exactly {examples_per_term} high-quality evaluation rows for an English slang normalizer.

Normalizer task:
- Input is English.
- Output target is standard English.
- If the input is literal/non-slang, target must copy input exactly.
- If the input is slang/idiomatic, target must preserve meaning in standard English.
- Do not translate to Portuguese.

Term:
{term_spec["term"]}

Meaning notes:
{term_spec["notes"]}

Seed examples:
{json.dumps(term_spec["examples"], ensure_ascii=False)}

Requirements:
- Return only JSON matching this schema:
  {{"data":[{{"input":"...","target":"...","term":"{term_spec["term"]}","sense":"literal|slang","source":"openai_targeted_eval"}}]}}
- {balance_instruction}
- Seed targets may be meaning hints from a source dataset; generated targets must be full standard-English sentence rewrites.
- Literal rows are safety rows: target must be byte-for-byte identical to input after trimming spaces.
- Literal rows should use ordinary non-slang contexts, especially physical or dictionary meanings.
- Slang rows should rewrite only the slang phrase, preserving the rest of the sentence.
- Do not mark ordinary standard English as slang just because it can be paraphrased.
- Example: "they beat us by five points" is standard English and should be literal, not slang.
- Include short natural sentences a learner might save in the app.
- Avoid wrappers like "people said", "everyone online said", "honestly", "i think".
- Avoid duplicate inputs.
- Keep all text lowercase except proper nouns like Fortnite.
""".strip()


def parse_response(text: str) -> list[dict[str, Any]]:
    data = json.loads(text)
    if isinstance(data, dict):
        rows = data.get("data", [])
    elif isinstance(data, list):
        rows = data
    else:
        rows = []
    return [row for row in rows if isinstance(row, dict)]


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
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
    return (
        input_tokens * prices["input"] / 1_000_000
        + output_tokens * prices["output"] / 1_000_000
    )


def validate_row(row: dict[str, Any], expected_term: str) -> str | None:
    source = clean(row.get("input", ""))
    target = clean(row.get("target", ""))
    sense = row.get("sense", "")

    if not source or not target:
        return "missing input or target"
    if row.get("term") != expected_term:
        return "wrong term"
    if sense not in {"literal", "slang"}:
        return "invalid sense"
    if sense == "literal" and source.lower() != target.lower():
        return "literal target must match input"
    if len(target) > max(50, int(len(source) * 1.8)):
        return "target too long"
    return None


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Generate targeted normalizer eval candidates.")
    parser.add_argument("--model", default=os.getenv("OPENAI_DATA_MODEL", "gpt-5-mini"))
    parser.add_argument("--examples-per-term", type=int, default=12)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--terms-path", type=Path, help="Read term specs from a previous discovery JSON file.")
    parser.add_argument("--discover-terms", type=int, default=0, help="Ask GPT to choose this many ambiguous terms before generating rows.")
    parser.add_argument("--discover-batch-size", type=int, default=40, help="Number of terms to request in each discovery API call.")
    parser.add_argument("--discovered-terms-output", type=Path, default=DISCOVERED_TERMS_PATH)
    parser.add_argument("--include-default-terms", action="store_true", help="Append the built-in first-batch terms when using --terms-path or --discover-terms.")
    parser.add_argument("--discover-only", action="store_true", help="Only write discovered terms; do not generate eval rows.")
    parser.add_argument("--start-index", type=int, default=0, help="Skip this many term specs before generation.")
    parser.add_argument("--max-terms", type=int, help="Limit the number of term specs used for row generation.")
    parser.add_argument("--sleep", type=float, default=0.2)
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Install the OpenAI SDK first: pip install openai") from exc

    client = OpenAI()
    default_terms = dedupe_term_specs([spec for spec in (normalize_term_spec(item) for item in TARGET_TERMS) if spec])
    term_specs = default_terms

    if args.terms_path:
        loaded_terms = load_term_specs(args.terms_path)
        term_specs = dedupe_term_specs(default_terms + loaded_terms) if args.include_default_terms else loaded_terms

    if args.discover_terms:
        discovered_terms = discover_term_specs(
            client=client,
            model=args.model,
            term_count=args.discover_terms,
            existing_term_specs=dedupe_term_specs(default_terms + term_specs),
            output_path=args.discovered_terms_output,
            batch_size=args.discover_batch_size,
        )
        term_specs = discovered_terms
        if args.include_default_terms:
            term_specs = dedupe_term_specs(default_terms + term_specs)

    if args.discover_only:
        return

    if args.start_index:
        term_specs = term_specs[max(0, args.start_index) :]

    if args.max_terms is not None:
        term_specs = term_specs[: args.max_terms]

    if not term_specs:
        raise SystemExit("No term specs were available for generation.")

    generated: list[dict[str, Any]] = []
    seen_inputs: set[str] = set()
    total_input_tokens = 0
    total_output_tokens = 0

    for idx, term_spec in enumerate(term_specs, start=1):
        print(f"{idx}/{len(term_specs)} {term_spec['term']}: requesting examples...", flush=True)
        response = client.responses.create(
            model=args.model,
            instructions=(
                "You generate compact, valid JSON for English slang normalizer "
                "evaluation. Follow the requested schema exactly."
            ),
            input=build_prompt(term_spec, args.examples_per_term),
            text={
                "format": {
                    "type": "json_schema",
                    "name": "targeted_normalizer_eval_rows",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "data": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "input": {"type": "string"},
                                        "target": {"type": "string"},
                                        "term": {"type": "string"},
                                        "sense": {"type": "string", "enum": ["literal", "slang"]},
                                        "source": {"type": "string"},
                                    },
                                    "required": ["input", "target", "term", "sense", "source"],
                                },
                            }
                        },
                        "required": ["data"],
                    },
                }
            },
        )

        input_tokens = usage_value(response.usage, "input_tokens")
        output_tokens = usage_value(response.usage, "output_tokens")
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

        accepted = 0
        skipped = 0
        try:
            rows = parse_response(response.output_text)
        except Exception as exc:
            print(f"[skip] {term_spec['term']}: could not parse JSON: {exc}", flush=True)
            continue

        for row in rows:
            row["input"] = clean(row.get("input", ""))
            row["target"] = clean(row.get("target", ""))
            row["term"] = clean(row.get("term", term_spec["term"])).lower()
            row["source"] = "openai_targeted_eval"

            reason = validate_row(row, term_spec["term"])
            key = row["input"].lower()
            if reason or key in seen_inputs:
                skipped += 1
                continue
            seen_inputs.add(key)
            generated.append(row)
            accepted += 1

        cost = estimate_cost(args.model, total_input_tokens, total_output_tokens)
        usage_suffix = (
            f" tokens=in:{total_input_tokens} out:{total_output_tokens}"
            + (f" est_cost=${cost:.4f}" if cost is not None else "")
        )
        print(
            f"{idx}/{len(term_specs)} {term_spec['term']}: "
            f"accepted={accepted} skipped={skipped}{usage_suffix}",
            flush=True,
        )
        write_rows(args.output, generated)
        if args.sleep:
            time.sleep(args.sleep)

    write_rows(args.output, generated)

    print(f"Wrote {len(generated)} candidate rows to {args.output}", flush=True)
    total_cost = estimate_cost(args.model, total_input_tokens, total_output_tokens)
    if total_cost is not None:
        print(
            f"Token usage: input={total_input_tokens} output={total_output_tokens} "
            f"estimated_cost=${total_cost:.4f}",
            flush=True,
        )
    print("Review candidates before copying approved rows into scripts/build_normalizer_v3_dataset.py.", flush=True)


if __name__ == "__main__":
    main()
