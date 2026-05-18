"""
Build OpenAI normalizer term specs from MLBtrio/genz-slang-dataset.

The Hugging Face dataset is vocabulary/definition/example data, not final
normalizer training data. This script converts it into prompt-ready specs that
can be passed to generate_targeted_normalizer_eval_with_openai.py.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


DATASET_ID = "MLBtrio/genz-slang-dataset"
DATA_DIR = Path("data")
DEFAULT_BASE_SPECS = DATA_DIR / "feedback_normalizer_openai_terms.json"
DEFAULT_OUTPUT = DATA_DIR / "genz_slang_openai_terms.json"
DEFAULT_REPORT = DATA_DIR / "genz_slang_openai_terms_report.json"

MAX_EXAMPLES_PER_TERM = 3
MAX_TERM_WORDS = 4
MAX_EXAMPLE_CHARS = 180

RISKY_TERM_FRAGMENTS = {
    "bussy",
    "gyatt",
    "horny",
    "nsfw",
    "porn",
    "sexy",
    "thot",
}

PRIORITY_TERMS = {
    "aight",
    "ate",
    "based",
    "bop",
    "boujee",
    "bussin",
    "cancel culture",
    "catch these hands",
    "cheugy",
    "clap back",
    "cringe",
    "delulu",
    "dank",
    "drip",
    "extra",
    "fam",
    "finna",
    "flex",
    "ghosting",
    "glow up",
    "goat",
    "hits different",
    "iykyk",
    "lit",
    "mid",
    "no cap",
    "periodt",
    "rent free",
    "rizz",
    "salty",
    "shade",
    "ship",
    "simp",
    "slaps",
    "slay",
    "snatched",
    "stan",
    "sus",
    "tea",
    "vibe check",
    "woke",
}

USEFUL_SHORT_TERMS = {
    "af",
    "dm",
    "gg",
    "gm",
    "irl",
    "jk",
    "l",
    "lol",
    "ngl",
    "omg",
    "tbh",
    "w",
}


def clean(text: Any) -> str:
    return " ".join(str(text or "").strip().split())


def norm(text: Any) -> str:
    return clean(text).lower().strip(" .!?")


def read_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    return [row for row in data if isinstance(row, dict)]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def slug_term(term: Any) -> str:
    term = clean(term).lower()
    term = re.sub(r"^[^\w#@]+|[^\w!?]+$", "", term)
    return term


def is_safe_term(term: str, include_risky: bool) -> bool:
    if not term:
        return False
    if len(term.split()) > MAX_TERM_WORDS:
        return False
    if len(term) > 40:
        return False
    if any(char.isdigit() for char in term) and term not in USEFUL_SHORT_TERMS:
        return False
    if not include_risky and any(fragment in term for fragment in RISKY_TERM_FRAGMENTS):
        return False
    return True


def get_ci(row: dict[str, Any], *names: str) -> str:
    lookup = {key.lower(): value for key, value in row.items()}
    for name in names:
        value = lookup.get(name.lower())
        if value is not None:
            return clean(value)
    return ""


def load_genz_rows(max_rows: int | None) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install the datasets package first: pip install datasets") from exc

    dataset = load_dataset(DATASET_ID, split="train")
    if max_rows:
        dataset = dataset.select(range(min(max_rows, len(dataset))))
    return [dict(row) for row in dataset]


def normalize_genz_row(row: dict[str, Any], include_risky: bool) -> dict[str, str] | None:
    term = slug_term(get_ci(row, "Slang", "slang", "term", "word"))
    description = get_ci(row, "Description", "description", "definition", "meaning")
    example = get_ci(row, "Example", "example", "sentence")
    context = get_ci(row, "Context", "context")

    if not is_safe_term(term, include_risky):
        return None
    if not description:
        return None
    if len(example) > MAX_EXAMPLE_CHARS:
        example = ""

    return {
        "term": term,
        "description": description,
        "example": example,
        "context": context,
    }


def merge_with_base_specs(
    base_specs: list[dict[str, Any]],
    genz_rows: list[dict[str, str]],
    max_terms: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    specs_by_term: dict[str, dict[str, Any]] = {}
    for spec in base_specs:
        term = slug_term(spec.get("term", ""))
        if not term:
            continue
            specs_by_term[term] = {
                "term": term,
                "notes": clean(spec.get("notes", "")),
                "examples": [
                example
                for example in spec.get("examples", [])
                if isinstance(example, dict)
                and clean(example.get("input", ""))
                and clean(example.get("target", ""))
                    and clean(example.get("sense", "")) in {"literal", "slang"}
                ],
                "_base": True,
                "_genz": False,
            }

    added_terms = 0
    added_slang_examples = 0
    skipped_existing_examples = 0

    for row in genz_rows:
        if max_terms is not None and added_terms >= max_terms and row["term"] not in specs_by_term:
            break

        term = row["term"]
        created = False
        if term not in specs_by_term:
            specs_by_term[term] = {
                "term": term,
                "notes": "",
                "examples": [],
                "_base": False,
                "_genz": True,
            }
            created = True
            added_terms += 1

        spec = specs_by_term[term]
        spec["_genz"] = True
        description = row["description"]
        context = row["context"]
        genz_note = (
            f"GenZ dataset meaning: {description}. "
            "Generate normalizer rows that preserve the intended meaning in standard English. "
            "If the term has a common literal/non-slang meaning, include literal identity rows."
        )
        if context:
            genz_note += f" Context: {context}."
        if "GenZ dataset meaning:" not in spec["notes"]:
            spec["notes"] = clean((spec["notes"] + " " + genz_note).strip())

        if not row["example"]:
            continue

        existing_slang = [
            example
            for example in spec["examples"]
            if clean(example.get("sense", "")) == "slang"
        ]
        if len(existing_slang) >= MAX_EXAMPLES_PER_TERM:
            skipped_existing_examples += 1
            continue

        input_text = row["example"]
        if any(norm(example.get("input", "")) == norm(input_text) for example in spec["examples"]):
            skipped_existing_examples += 1
            continue

        target_hint = f"[rewrite using meaning: {description}]"
        spec["examples"].append({
            "input": input_text,
            "target": target_hint,
            "sense": "slang",
        })
        added_slang_examples += 1

        if created:
            added_terms += 0

    specs = list(specs_by_term.values())
    specs.sort(key=spec_sort_key)
    for spec in specs:
        spec.pop("_base", None)
        spec.pop("_genz", None)

    report = {
        "dataset_id": DATASET_ID,
        "base_specs_read": len(base_specs),
        "genz_rows_read": len(genz_rows),
        "output_specs": len(specs),
        "new_terms_added": max(0, len(specs_by_term) - len({slug_term(spec.get("term", "")) for spec in base_specs})),
        "slang_seed_examples_added": added_slang_examples,
        "skipped_existing_examples": skipped_existing_examples,
        "top_terms": Counter(row["term"] for row in genz_rows).most_common(30),
    }
    return specs, report


def spec_sort_key(spec: dict[str, Any]) -> tuple[int, str]:
    term = spec["term"]
    score = 0
    if spec.get("_genz"):
        score += 1000
    if term in PRIORITY_TERMS:
        score += 500
    if " " in term:
        score += 120
    if re.fullmatch(r"[a-z][a-z' -]*", term):
        score += 80
    if len(term) >= 4:
        score += 30
    if len(term) <= 2 and term not in USEFUL_SHORT_TERMS:
        score -= 100
    if not any(char in "aeiou" for char in term) and term not in USEFUL_SHORT_TERMS:
        score -= 80
    if re.search(r"[^a-z' -]", term):
        score -= 50
    return (-score, term)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build OpenAI term specs from MLBtrio/genz-slang-dataset.")
    parser.add_argument("--base-specs", type=Path, default=DEFAULT_BASE_SPECS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-rows", type=int, help="Limit Hugging Face rows read, useful for tests.")
    parser.add_argument("--max-new-terms", type=int, default=120)
    parser.add_argument("--include-risky", action="store_true", help="Include terms filtered by the conservative risky-term list.")
    args = parser.parse_args()

    base_specs = read_json_list(args.base_specs)
    genz_raw_rows = load_genz_rows(args.max_rows)
    genz_rows = [
        row
        for row in (normalize_genz_row(row, args.include_risky) for row in genz_raw_rows)
        if row
    ]

    specs, report = merge_with_base_specs(base_specs, genz_rows, args.max_new_terms)
    write_json(args.output, specs)
    write_json(args.report, report)

    print(f"Wrote {len(specs)} specs to {args.output}")
    print(f"Report: {args.report}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
