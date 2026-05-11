"""
Expands normalizer V3.1 data using the OpenAI API.

Reads existing normalizer rows and uses a small OpenAI model to produce JSON
paraphrases that preserve the same input -> target transformation.

Run:
  python scripts/expand_normalizer_v3_with_openai.py --max-seeds 80 --variants 3
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any


INPUT_PATH = Path("data/slang_normalizer_v3_1_train.json")
OUTPUT_PATH = Path("data/slang_normalizer_v3_1_openai_augmented.json")
SEED = 42

BAD_WRAPPER_PREFIXES = (
    "honestly, ",
    "honestly ",
    "i think ",
    "everyone said ",
    "people online said ",
    "the comments agreed",
)

BAD_WRAPPER_FRAGMENTS = (
    "the comments agreed",
    "people online said",
    "everyone said",
)


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON list.")
    return data


def parse_json_list(text: str) -> list[dict[str, Any]]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("Model output must be a JSON list.")
    return [item for item in data if isinstance(item, dict)]


def clean(text: str) -> str:
    return " ".join((text or "").strip().split())


def has_bad_wrapper(text: str) -> bool:
    lower = clean(text).lower()
    return lower.startswith(BAD_WRAPPER_PREFIXES) or any(fragment in lower for fragment in BAD_WRAPPER_FRAGMENTS)


def validate_generated_row(item: dict[str, Any], seed_row: dict[str, Any]) -> str | None:
    source = clean(item.get("input", ""))
    target = clean(item.get("target", ""))
    if not source or not target:
        return "missing input or target"
    if has_bad_wrapper(source) or has_bad_wrapper(target):
        return "wrapper artifact"
    sense = item.get("sense") or seed_row.get("sense", "")
    if sense == "literal" and source.lower() != target.lower():
        return "literal row must be identity"
    if len(target) > max(40, int(len(source) * 1.6)):
        return "target is too long"
    return None


def build_prompt(seed_row: dict[str, Any], variants: int) -> str:
    return f"""
Generate {variants} high-quality training rows for an English slang normalizer.

Task:
- Rewrite slang or idiomatic English into standard English.
- For literal/non-slang uses, copy the input exactly as the target.
- Preserve English only. Do not translate to Portuguese.
- Return JSON only: a list of objects.
- Each object must have: input, target, term, sense, source.
- source must be "openai_augmented".

Seed row:
{json.dumps(seed_row, ensure_ascii=False)}

Good style:
- Natural short sentences.
- Direct sentences only.
- No wrappers like "honestly", "i think", "everyone said", "people online said", or "the comments agreed".
- Include realistic contrast when sense is literal.
- Keep target close to input except for the slang/idiom normalization.
""".strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand normalizer v3 dataset with OpenAI.")
    parser.add_argument("--model", default=os.getenv("OPENAI_DATA_MODEL", "gpt-5-mini"))
    parser.add_argument("--max-seeds", type=int, default=80)
    parser.add_argument("--variants", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0.2)
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Install the OpenAI SDK first: pip install openai") from exc

    rows = read_rows(INPUT_PATH)
    rng = random.Random(SEED)
    seeds = rows[:]
    rng.shuffle(seeds)
    seeds = seeds[: args.max_seeds]

    client = OpenAI()
    generated: list[dict[str, Any]] = []

    for idx, seed_row in enumerate(seeds, start=1):
        response = client.responses.create(
            model=args.model,
            instructions=(
                "You generate compact, valid JSON training data for an English "
                "slang normalizer. Output JSON only."
            ),
            input=build_prompt(seed_row, args.variants),
        )
        try:
            items = parse_json_list(response.output_text)
        except Exception as exc:
            print(f"[skip] seed {idx}: could not parse output: {exc}")
            continue

        skipped = 0
        for item in items:
            item.setdefault("term", seed_row.get("term", ""))
            item.setdefault("sense", seed_row.get("sense", ""))
            item["source"] = "openai_augmented"
            reason = validate_generated_row(item, seed_row)
            if reason:
                skipped += 1
                continue
            item["input"] = clean(item["input"])
            item["target"] = clean(item["target"])
            generated.append(item)

        print(f"{idx}/{len(seeds)} generated={len(generated)} skipped={skipped}")
        if args.sleep:
            time.sleep(args.sleep)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(generated, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Wrote {len(generated)} rows to {OUTPUT_PATH}")
    print("To merge with training data:")
    print("  python scripts/merge_normalizer_v3_openai_data.py")


if __name__ == "__main__":
    main()
