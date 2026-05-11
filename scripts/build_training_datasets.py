"""
Build canonical training and evaluation datasets for the slang pipeline.

It merges the existing synthetic datasets, adds hard negatives for ambiguous slang
and writes stable master files consumed by the new training and evaluation scripts.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

DATA_DIR = Path("data")
SEED = 42

DETECTOR_TRAIN_FILES = [
    "detector_train.json",
    "advanced_sentences_train.json",
    "neutral_sentences_train.json",
    "gap_sentences_train.json",
]

DETECTOR_TEST_FILES = [
    "detector_test.json",
    "advanced_sentences_test.json",
    "neutral_sentences_test.json",
]

NORMALIZER_TRAIN_FILES = [
    "normalizer_train.json",
    "advanced_sentences_train.json",
    "sentences_train.json",
    "neutral_sentences_train.json",
    "gap_sentences_train.json",
]

NORMALIZER_TEST_FILES = [
    "normalizer_test.json",
    "advanced_sentences_test.json",
    "sentences_test.json",
    "neutral_sentences_test.json",
]

AMBIGUOUS_TERMS = [
    "fire",
    "sick",
    "lit",
    "goat",
    "cap",
    "tea",
    "beef",
    "ghost",
    "flex",
    "drop",
    "hard",
]

TERM_SENSES = {
    "fire": {
        "slang": "excellent or impressive",
        "literal": "flames or combustion",
        "normalization": "excellent",
        "slang_objects": ["beat", "track", "outfit", "verse", "design", "performance", "playlist", "edit"],
        "literal_objects": ["house", "pan", "building", "forest", "car", "kitchen", "warehouse", "candle"],
    },
    "sick": {
        "slang": "excellent or impressive",
        "literal": "ill or unwell",
        "normalization": "excellent",
        "slang_objects": ["trick", "move", "solo", "design", "goal", "shot", "transition", "routine"],
        "literal_objects": ["child", "patient", "teacher", "friend", "traveler", "student", "manager", "neighbor"],
    },
    "lit": {
        "slang": "exciting or excellent",
        "literal": "illuminated",
        "normalization": "exciting",
        "slang_objects": ["party", "show", "timeline", "festival", "night", "chat", "concert", "room"],
        "literal_objects": ["lamp", "hallway", "sign", "candle", "stage", "screen", "porch", "street"],
    },
    "goat": {
        "slang": "greatest of all time",
        "literal": "animal",
        "normalization": "the greatest of all time",
        "slang_objects": ["player", "singer", "teacher", "developer", "coach", "artist", "chef", "writer"],
        "literal_objects": ["farm", "barn", "field", "veterinarian", "mountain", "petting zoo", "fence", "pasture"],
    },
    "cap": {
        "slang": "lie or exaggeration",
        "literal": "cover or hat",
        "normalization": "lie",
        "slang_objects": ["story", "excuse", "claim", "rumor", "answer", "caption", "post", "message"],
        "literal_objects": ["bottle", "pen", "marker", "gas tank", "camera lens", "jar", "tube", "hat"],
    },
    "tea": {
        "slang": "gossip or private news",
        "literal": "drink",
        "normalization": "gossip",
        "slang_objects": ["group chat", "office", "timeline", "friend", "comment section", "story", "thread", "meeting"],
        "literal_objects": ["cup", "kettle", "mug", "breakfast", "teapot", "cafe", "shelf", "menu"],
    },
    "beef": {
        "slang": "conflict or argument",
        "literal": "meat",
        "normalization": "conflict",
        "slang_objects": ["team", "rapper", "neighbor", "classmate", "coworker", "creator", "friend", "rival"],
        "literal_objects": ["stew", "burger", "taco", "freezer", "restaurant", "recipe", "market", "sandwich"],
    },
    "ghost": {
        "slang": "suddenly ignore someone",
        "literal": "spirit or apparition",
        "normalization": "ignore",
        "slang_objects": ["date", "friend", "recruiter", "client", "classmate", "group", "partner", "seller"],
        "literal_objects": ["story", "movie", "costume", "legend", "museum", "haunted house", "painting", "tour"],
    },
    "flex": {
        "slang": "show off",
        "literal": "bend a muscle",
        "normalization": "show off",
        "slang_objects": ["watch", "car", "promotion", "setup", "vacation", "award", "fit", "score"],
        "literal_objects": ["arm", "knee", "muscle", "ankle", "shoulder", "wrist", "toe", "leg"],
    },
    "drop": {
        "slang": "release something new",
        "literal": "let something fall",
        "normalization": "release",
        "slang_objects": ["album", "song", "trailer", "collection", "update", "video", "episode", "single"],
        "literal_objects": ["glass", "phone", "keys", "box", "plate", "bag", "coin", "book"],
    },
    "hard": {
        "slang": "impressive or intense",
        "literal": "difficult or solid",
        "normalization": "impressive",
        "slang_objects": ["beat", "line", "fit", "poster", "intro", "verse", "edit", "photo"],
        "literal_objects": ["exam", "rock", "chair", "problem", "surface", "decision", "question", "wood"],
    },
}


def read_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def infer_target_term(text: str) -> str | None:
    lowered = text.lower()
    for term in AMBIGUOUS_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", lowered):
            return term
    return None


def detector_row(
    text: str,
    label: bool,
    source: str,
    target_term: str | None = None,
    sense: str | None = None,
    is_hard_negative: bool = False,
) -> dict[str, Any]:
    text = clean_text(text)
    return {
        "text": text,
        "label": int(label),
        "is_slang": bool(label),
        "target_term": target_term or infer_target_term(text),
        "sense": sense or ("slang" if label else "literal"),
        "source": source,
        "is_hard_negative": is_hard_negative,
    }


def normalizer_row(
    informal: str,
    formal: str,
    source: str,
    target_term: str | None = None,
    sense: str | None = None,
) -> dict[str, Any] | None:
    informal = clean_text(informal)
    formal = clean_text(formal)
    if not informal or not formal:
        return None
    return {
        "slang": informal,
        "formal": formal,
        "input": informal,
        "target": formal,
        "target_term": target_term or infer_target_term(informal),
        "sense": sense or ("neutral" if informal.lower() == formal.lower() else "slang"),
        "source": source,
    }


def convert_to_detector_rows(filename: str) -> list[dict[str, Any]]:
    rows = []
    for item in read_json_list(DATA_DIR / filename):
        if "text" in item:
            label = bool(item.get("is_slang", item.get("label", False)))
            rows.append(detector_row(item["text"], label, filename))
            continue

        informal = item.get("informal") or item.get("slang")
        formal = item.get("formal")
        if informal and formal:
            label = clean_text(informal).lower() != clean_text(formal).lower()
            rows.append(detector_row(informal, label, filename))
    return rows


def convert_to_normalizer_rows(filename: str) -> list[dict[str, Any]]:
    rows = []
    for item in read_json_list(DATA_DIR / filename):
        informal = item.get("informal") or item.get("slang") or item.get("text")
        formal = item.get("formal")
        if formal is None and "is_slang" in item:
            formal = informal
        row = normalizer_row(informal or "", formal or "", filename)
        if row:
            rows.append(row)
    return rows


def make_hard_examples(per_term: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    detector_rows: list[dict[str, Any]] = []
    normalizer_rows: list[dict[str, Any]] = []
    slang_count = per_term // 2
    literal_count = per_term - slang_count

    slang_templates = [
        "This {obj} is {term}.",
        "That {obj} was honestly {term}.",
        "Everyone said the {obj} is {term}.",
        "No lie, the {obj} is {term}.",
        "The {obj} from last night was {term}.",
        "My friends thought the {obj} was {term}.",
        "The new {obj} sounds {term}.",
        "People online called the {obj} {term}.",
        "I did not expect that {obj} to be so {term}.",
        "The whole room agreed the {obj} was {term}.",
        "Her latest {obj} is seriously {term}.",
        "Their final {obj} came out {term}.",
        "That viral {obj} looked {term}.",
    ]
    literal_templates = [
        "The {obj} is {term}.",
        "Please check whether the {obj} is {term}.",
        "They mentioned the {obj} was {term}.",
        "The report says the {obj} is {term}.",
        "I noticed the {obj} was {term}.",
        "The label explains why the {obj} is {term}.",
        "The technician confirmed the {obj} was {term}.",
        "We learned that the {obj} can be {term}.",
        "The instructions say the {obj} should not be {term}.",
        "Someone asked if the {obj} was {term}.",
        "The inspector wrote that the {obj} is {term}.",
        "The classroom example used a {obj} that was {term}.",
        "The note warned us the {obj} might be {term}.",
    ]
    slang_scenes = [
        "People kept replaying it.",
        "It got shared all morning.",
        "My friends brought it up later.",
        "The comments agreed immediately.",
        "It stood out from everything else.",
        "Everyone reacted at once.",
        "It became the highlight of the day.",
    ]
    literal_scenes = [
        "The situation was handled carefully.",
        "Someone wrote it in the report.",
        "The detail mattered in context.",
        "We checked it again later.",
        "The example was meant literally.",
        "No slang meaning was intended.",
        "The sentence describes a real object.",
    ]

    for term, config in TERM_SENSES.items():
        slang_objects = config["slang_objects"]
        literal_objects = config["literal_objects"]
        normalization = config["normalization"]

        for i in range(slang_count):
            obj = slang_objects[i % len(slang_objects)]
            template = slang_templates[i % len(slang_templates)]
            scene = slang_scenes[i % len(slang_scenes)]
            text = f"{template.format(obj=obj, term=term)} {scene}"
            formal = text.replace(f" {term}", f" {normalization}")
            detector_rows.append(detector_row(text, True, "hard_pairs", term, "slang", False))
            row = normalizer_row(text, formal, "hard_pairs", term, "slang")
            if row:
                normalizer_rows.append(row)

        for i in range(literal_count):
            obj = literal_objects[i % len(literal_objects)]
            template = literal_templates[i % len(literal_templates)]
            scene = literal_scenes[i % len(literal_scenes)]
            text = f"{template.format(obj=obj, term=term)} {scene}"
            detector_rows.append(detector_row(text, False, "hard_pairs", term, "literal", True))
            row = normalizer_row(text, text, "hard_pairs", term, "literal")
            if row:
                normalizer_rows.append(row)

    required = [
        ("This beat is fire.", True, "fire", "slang", "This beat is excellent."),
        ("The house is on fire.", False, "fire", "literal", "The house is on fire."),
        ("That comeback was sick.", True, "sick", "slang", "That comeback was excellent."),
        ("She felt sick after lunch.", False, "sick", "literal", "She felt sick after lunch."),
        ("No cap, this is true.", True, "cap", "slang", "No lie, this is true."),
        ("The bottle has a blue cap.", False, "cap", "literal", "The bottle has a blue cap."),
        ("Spill the tea.", True, "tea", "slang", "Share the gossip."),
        ("I made tea.", False, "tea", "literal", "I made tea."),
    ]
    for text, label, term, sense, formal in required:
        detector_rows.append(detector_row(text, label, "required_golden", term, sense, not label))
        row = normalizer_row(text, formal, "required_golden", term, sense)
        if row:
            normalizer_rows.append(row)

    return detector_rows, normalizer_rows


def dedupe(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique = []
    for row in rows:
        value = clean_text(str(row[key])).lower()
        if value in seen:
            continue
        seen.add(value)
        unique.append(row)
    return unique


def split_rows(
    rows: list[dict[str, Any]],
    test_size: float,
    seed: int,
    stratify_key: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    if not stratify_key:
        shuffled = rows[:]
        rng.shuffle(shuffled)
        cut = int(len(shuffled) * (1.0 - test_size))
        return shuffled[:cut], shuffled[cut:]

    train_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    groups: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row.get(stratify_key), []).append(row)

    for group_rows in groups.values():
        shuffled = group_rows[:]
        rng.shuffle(shuffled)
        test_count = max(1, int(round(len(shuffled) * test_size)))
        test_rows.extend(shuffled[:test_count])
        train_rows.extend(shuffled[test_count:])

    rng.shuffle(train_rows)
    rng.shuffle(test_rows)
    return train_rows, test_rows


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
        f.write("\n")


def build(per_term: int) -> None:
    random.seed(SEED)

    detector_train = []
    detector_test = []
    for filename in DETECTOR_TRAIN_FILES:
        detector_train.extend(convert_to_detector_rows(filename))
    for filename in DETECTOR_TEST_FILES:
        detector_test.extend(convert_to_detector_rows(filename))

    normalizer_train = []
    normalizer_test = []
    for filename in NORMALIZER_TRAIN_FILES:
        normalizer_train.extend(convert_to_normalizer_rows(filename))
    for filename in NORMALIZER_TEST_FILES:
        normalizer_test.extend(convert_to_normalizer_rows(filename))

    hard_detector, hard_normalizer = make_hard_examples(per_term)
    hard_train, hard_eval = split_rows(hard_detector, test_size=0.15, seed=SEED, stratify_key="label")
    hard_norm_train, hard_norm_eval = split_rows(hard_normalizer, test_size=0.15, seed=SEED)

    detector_train.extend(hard_train)
    detector_test.extend(hard_eval)
    normalizer_train.extend(hard_norm_train)
    normalizer_test.extend(hard_norm_eval)

    detector_train = dedupe(detector_train, "text")
    detector_test = dedupe(detector_test, "text")
    normalizer_train = dedupe(normalizer_train, "slang")
    normalizer_test = dedupe(normalizer_test, "slang")

    golden = []
    for row in detector_test:
        if row["source"] in {"required_golden", "hard_pairs"} and len(golden) < 300:
            golden.append(row)
    for text in [
        "This beat is fire.",
        "The house is on fire.",
        "That trick was sick.",
        "I feel sick today.",
        "Spill the tea.",
        "I made tea.",
    ]:
        if not any(item["text"].lower() == text.lower() for item in golden):
            label = text in {"This beat is fire.", "That trick was sick.", "Spill the tea."}
            golden.append(detector_row(text, label, "manual_golden", infer_target_term(text), "slang" if label else "literal", not label))

    write_json(DATA_DIR / "master_detector_train.json", detector_train)
    write_json(DATA_DIR / "master_detector_test.json", detector_test)
    write_json(DATA_DIR / "master_normalizer_train.json", normalizer_train)
    write_json(DATA_DIR / "master_normalizer_test.json", normalizer_test)
    write_json(DATA_DIR / "golden_eval.json", golden[:300])

    print("Canonical datasets written:")
    print(f"  detector train:   {len(detector_train)}")
    print(f"  detector test:    {len(detector_test)}")
    print(f"  normalizer train: {len(normalizer_train)}")
    print(f"  normalizer test:  {len(normalizer_test)}")
    print(f"  golden eval:      {len(golden[:300])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical slang training datasets.")
    parser.add_argument("--per-term", type=int, default=260, help="Hard-pair examples per ambiguous term.")
    args = parser.parse_args()
    build(args.per_term)


if __name__ == "__main__":
    main()
