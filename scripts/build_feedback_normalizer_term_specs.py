"""
Build OpenAI term specs from reviewed feedback and known weak normalizer terms.

The output is compatible with generate_targeted_normalizer_eval_with_openai.py
via --terms-path. It deliberately creates compact term specs, not final data:
OpenAI-generated rows still need review before they are merged into training.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DATA_DIR = Path("data")
DEFAULT_FEEDBACK_TRAIN = DATA_DIR / "feedback_training_candidates.jsonl"
DEFAULT_BASE_TRAIN = DATA_DIR / "slang_normalizer_v4_train.json"
DEFAULT_BASE_TEST = DATA_DIR / "slang_normalizer_v4_test.json"
DEFAULT_WEAK_TERMS = DATA_DIR / "targeted_normalizer_weak_terms.json"
DEFAULT_OUTPUT = DATA_DIR / "feedback_normalizer_openai_terms.json"

FALLBACK_LITERAL_SEEDS = {
    "beat": ("i love that beat", "i love that beat"),
    "burn": ("the burn on my hand is healing", "the burn on my hand is healing"),
    "burns": ("the burns on my arm are healing", "the burns on my arm are healing"),
    "cap": ("the bottle has a blue cap", "the bottle has a blue cap"),
    "chill": ("a chill wind came through", "a chill wind came through"),
    "clean": ("please clean the kitchen", "please clean the kitchen"),
    "cold": ("the soup is cold", "the soup is cold"),
    "cracked": ("the glass is cracked", "the glass is cracked"),
    "crush": ("please crush the cans", "please crush the cans"),
    "dead": ("the battery is dead", "the battery is dead"),
    "drip": ("the faucet has a drip", "the faucet has a drip"),
    "fire": ("the house is on fire", "the house is on fire"),
    "flex": ("flex your arm slowly", "flex your arm slowly"),
    "ghost": ("the ghost story was scary", "the ghost story was scary"),
    "hang": ("hang the coat by the door", "hang the coat by the door"),
    "kill": ("do not kill the plant", "do not kill the plant"),
    "killed": ("the frost killed the plant", "the frost killed the plant"),
    "legit": ("the ticket is legit", "the ticket is legit"),
    "lit": ("the candle is lit", "the candle is lit"),
    "mid": ("we met in mid july", "we met in mid july"),
    "salty": ("the soup is salty", "the soup is salty"),
    "savage": ("the savage storm damaged the coast", "the savage storm damaged the coast"),
    "sick": ("he felt sick after lunch", "he felt sick after lunch"),
    "slay": ("the knight will slay the dragon", "the knight will slay the dragon"),
    "tea": ("she spilled tea on her shirt", "she spilled tea on her shirt"),
}

FALLBACK_SLANG_SEEDS = {
    "burn": ("that comeback was a burn", "that comeback was an insult"),
    "burns": ("stop with the burns", "stop making insulting remarks"),
    "cap": ("that story is cap", "that story is a lie"),
    "chill": ("chill out, you are overreacting", "calm down, you are overreacting"),
    "clean": ("that shot was clean", "that shot was very smooth"),
    "cold": ("that reply was cold", "that reply was harsh"),
    "cracked": ("the hacker cracked the server", "the hacker broke into the server"),
    "crush": ("i have a crush on my neighbor", "i am romantically attracted to my neighbor"),
    "dead": ("that joke has me dead", "that joke made me laugh a lot"),
    "drip": ("his outfit has drip", "his outfit has style"),
    "fire": ("that track is fire", "that track is excellent"),
    "flex": ("he only posted that to flex", "he only posted that to show off"),
    "ghost": ("she ghosted me after our date", "she stopped replying to me after our date"),
    "hang": ("let's hang after work", "let's spend time together after work"),
    "kill": ("they kill our team in ranked", "they defeat our team thoroughly in ranked"),
    "killed": ("they killed our team in the match", "they defeated our team thoroughly in the match"),
    "legit": ("the setup looks legit", "the setup looks excellent"),
    "lit": ("the party was lit", "the party was exciting"),
    "salty": ("he is salty about losing", "he is bitter about losing"),
    "savage": ("that comeback was savage", "that comeback was brutally clever"),
    "sick": ("that trick was sick", "that trick was amazing"),
    "slay": ("you slayed that presentation", "you did extremely well in that presentation"),
    "tea": ("spill the tea about the meeting", "share the gossip about the meeting"),
}

TERM_ALIASES = {
    "burns": "burn",
    "killed": "kill",
    "ghosted": "ghost",
    "capping": "cap",
    "grindding": "grinding",
    "grindin": "grinding",
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no} must contain a JSON object")
            rows.append(row)
    return rows


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def canonical_term(term: str) -> str:
    term = clean(term).lower()
    return TERM_ALIASES.get(term, term)


def term_from_row(row: dict[str, Any]) -> str:
    term = clean(row.get("term") or row.get("target_term") or "").lower()
    if term:
        return canonical_term(term)

    text = norm(row.get("input", ""))
    candidates = sorted(
        set(FALLBACK_LITERAL_SEEDS) | set(FALLBACK_SLANG_SEEDS) | set(TERM_ALIASES),
        key=len,
        reverse=True,
    )
    for candidate in candidates:
        if re.search(rf"(?<![a-z]){re.escape(candidate)}(?![a-z])", text):
            return canonical_term(candidate)
    return ""


def normalize_row(row: dict[str, Any], source_label: str) -> dict[str, str] | None:
    source = clean(row.get("input", ""))
    target = clean(row.get("target") or row.get("expected") or "")
    sense = clean(row.get("sense") or row.get("kind") or "").lower()
    term = term_from_row(row)
    if not source or not target or sense not in {"literal", "slang"} or not term:
        return None
    if sense == "literal" and source.lower() != target.lower():
        return None
    if sense == "slang" and norm(source) == norm(target):
        return None
    return {
        "input": source,
        "target": target,
        "sense": sense,
        "term": term,
        "source": source_label,
    }


def add_seed(
    examples_by_term: dict[str, dict[str, list[dict[str, str]]]],
    row: dict[str, str],
    max_examples_per_sense: int,
) -> None:
    term = canonical_term(row["term"])
    sense = row["sense"]
    bucket = examples_by_term[term][sense]
    key = (norm(row["input"]), norm(row["target"]), sense)
    if any((norm(item["input"]), norm(item["target"]), item["sense"]) == key for item in bucket):
        return
    if len(bucket) < max_examples_per_sense:
        bucket.append({
            "input": row["input"],
            "target": row["target"],
            "sense": sense,
        })


def add_fallback_seeds(
    examples_by_term: dict[str, dict[str, list[dict[str, str]]]],
    wanted_terms: set[str],
    max_examples_per_sense: int,
) -> None:
    for term in sorted(wanted_terms):
        for seed_map, sense in [(FALLBACK_LITERAL_SEEDS, "literal"), (FALLBACK_SLANG_SEEDS, "slang")]:
            seed = seed_map.get(term)
            if not seed:
                continue
            add_seed(
                examples_by_term,
                {
                    "input": seed[0],
                    "target": seed[1],
                    "sense": sense,
                    "term": term,
                    "source": "fallback_seed",
                },
                max_examples_per_sense,
            )


def load_weak_terms(path: Path, top_terms: int) -> set[str]:
    rows = read_json_list(path)
    terms = []
    for row in rows:
        term = canonical_term(row.get("term", ""))
        if term:
            terms.append(term)
    return set(terms[:top_terms])


def make_notes(term: str, counts: Counter[str]) -> str:
    literal_count = counts.get("literal", 0)
    slang_count = counts.get("slang", 0)
    return (
        "Generate balanced literal and slang/idiomatic variants for this English normalizer term. "
        "Literal rows are safety rows: their target must copy the input exactly. "
        "Slang rows should rewrite only the slang meaning into standard English. "
        f"Use the seed meanings but create new natural learner sentences. "
        f"Current reviewed seeds: {slang_count} slang, {literal_count} literal."
    )


def build_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    feedback_rows = [
        normalize_row(row, "feedback")
        for row in read_jsonl(args.feedback_train)
    ]
    base_rows = [
        normalize_row(row, "base_train")
        for row in read_json_list(args.base_train) + read_json_list(args.base_test)
    ]
    normalized_rows = [row for row in feedback_rows + base_rows if row]

    wanted_terms = {row["term"] for row in normalized_rows if row["source"] == "feedback"}
    wanted_terms |= load_weak_terms(args.weak_terms, args.top_weak_terms)
    wanted_terms |= {canonical_term(term) for term in args.include_terms}
    wanted_terms = {term for term in wanted_terms if term}

    examples_by_term: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    counts_by_term: dict[str, Counter[str]] = defaultdict(Counter)

    feedback_first = sorted(normalized_rows, key=lambda row: row["source"] != "feedback")
    for row in feedback_first:
        term = row["term"]
        if term not in wanted_terms:
            continue
        counts_by_term[term][row["sense"]] += 1
        add_seed(examples_by_term, row, args.max_examples_per_sense)

    add_fallback_seeds(examples_by_term, wanted_terms, args.max_examples_per_sense)

    specs = []
    for term in sorted(wanted_terms):
        examples = examples_by_term[term]
        senses = {sense for sense, rows in examples.items() if rows}
        if senses != {"literal", "slang"}:
            continue
        seed_examples = examples["literal"][: args.max_examples_per_sense] + examples["slang"][: args.max_examples_per_sense]
        specs.append({
            "term": term,
            "notes": make_notes(term, counts_by_term[term]),
            "examples": seed_examples,
        })

    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build OpenAI term specs from normalizer feedback.")
    parser.add_argument("--feedback-train", type=Path, default=DEFAULT_FEEDBACK_TRAIN)
    parser.add_argument("--base-train", type=Path, default=DEFAULT_BASE_TRAIN)
    parser.add_argument("--base-test", type=Path, default=DEFAULT_BASE_TEST)
    parser.add_argument("--weak-terms", type=Path, default=DEFAULT_WEAK_TERMS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-weak-terms", type=int, default=40)
    parser.add_argument("--max-examples-per-sense", type=int, default=3)
    parser.add_argument(
        "--include-terms",
        nargs="*",
        default=[
            "legit",
            "burn",
            "kill",
            "cracked",
            "hang",
            "chill",
            "crush",
            "fire",
            "lit",
            "dead",
            "sick",
            "cold",
            "cap",
            "salty",
            "drip",
            "clean",
            "flex",
            "ghost",
            "slay",
            "tea",
            "savage",
        ],
        help="Extra terms to force into the spec file when seeds are available.",
    )
    args = parser.parse_args()

    specs = build_specs(args)
    write_json(args.output, specs)
    print(f"Wrote {len(specs)} term specs to {args.output}")
    print(", ".join(spec["term"] for spec in specs[:80]))


if __name__ == "__main__":
    main()
