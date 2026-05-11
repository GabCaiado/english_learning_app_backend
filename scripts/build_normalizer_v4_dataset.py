"""
Build the v4 slang normalizer dataset.

Principles:
- Never train on exact inputs from data/slang_normalizer_gold_eval_v4.json.
- Keep strong literal contrast pairs.
- Oversample hard ambiguous cases that current candidates fail.
- Produce a separate dev/test set for training feedback; the gold v4 set remains
  locked for final model selection only.
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


SEED = 44
DEFAULT_BASE_TRAIN = Path("data/slang_normalizer_v3_1_train.json")
DEFAULT_BASE_TEST = Path("data/slang_normalizer_v3_1_test.json")
DEFAULT_GOLD = Path("data/slang_normalizer_gold_eval_v4.json")
DEFAULT_TRAIN_OUT = Path("data/slang_normalizer_v4_train.json")
DEFAULT_TEST_OUT = Path("data/slang_normalizer_v4_test.json")


def clean(text: str) -> str:
    return " ".join((text or "").strip().split())


def norm_key(text: str) -> str:
    return clean(text).lower().strip(" .!?")


def row(source: str, target: str, term: str, sense: str, source_type: str) -> dict[str, str]:
    return {
        "input": clean(source),
        "target": clean(target),
        "term": term,
        "sense": sense,
        "source": source_type,
    }


V4_HARD_TRAINING = [
    ("last night's show was lit", "last night's show was exciting", "lit", "slang"),
    ("the festival got lit after sunset", "the festival got exciting after sunset", "lit", "slang"),
    ("the hallway was brightly lit", "the hallway was brightly lit", "lit", "literal"),
    ("the lamp lit the room", "the lamp lit the room", "lit", "literal"),

    ("what's the tea about the group chat", "what is the gossip about the group chat", "tea", "slang"),
    ("share the tea from the office", "share the gossip from the office", "tea", "slang"),
    ("she brewed tea before class", "she brewed tea before class", "tea", "literal"),
    ("the tea kettle is hot", "the tea kettle is hot", "tea", "literal"),

    ("this track is my jam", "this track is my favorite", "my jam", "slang"),
    ("that new album is my jam", "that new album is my favorite", "my jam", "slang"),
    ("the jar of jam is empty", "the jar of jam is empty", "jam", "literal"),
    ("strawberry jam tastes sweet", "strawberry jam tastes sweet", "jam", "literal"),

    ("we can crash on her couch", "we can sleep on her couch", "crash", "slang"),
    ("they crashed at the guest room", "they slept at the guest room", "crash", "slang"),
    ("the system crash erased the file", "the system crash erased the file", "crash", "literal"),
    ("the bike crashed into the fence", "the bike crashed into the fence", "crash", "literal"),

    ("his beef with the neighbor got worse", "his conflict with the neighbor got worse", "beef", "slang"),
    ("she has beef with her classmate", "she has a conflict with her classmate", "beef", "slang"),
    ("ground beef is in the fridge", "ground beef is in the fridge", "beef", "literal"),
    ("he cooked a beef burger", "he cooked a beef burger", "beef", "literal"),

    ("that drum fill was nasty", "that drum fill was amazing", "nasty", "slang"),
    ("his goal was nasty", "his goal was amazing", "nasty", "slang"),
    ("the trash smelled nasty", "the trash smelled nasty", "nasty", "literal"),
    ("the milk tasted nasty", "the milk tasted nasty", "nasty", "literal"),

    ("they ate that routine", "they did very well in that routine", "ate", "slang"),
    ("he ate the presentation", "he did very well in the presentation", "ate", "slang"),
    ("they ate soup for lunch", "they ate soup for lunch", "ate", "literal"),
    ("she ate a sandwich", "she ate a sandwich", "ate", "literal"),
    ("they ate and left no crumbs", "they did very well", "ate and left no crumbs", "slang"),
    ("he ate and left no crumbs with that speech", "he did very well with that speech", "ate and left no crumbs", "slang"),
    ("he ate and left no crumbs on the plate", "he ate and left no crumbs on the plate", "crumbs", "literal"),

    ("she dipped before the meeting ended", "she left before the meeting ended", "dipped", "slang"),
    ("we dipped out after dinner", "we left after dinner", "dipped", "slang"),
    ("he dipped fries in ketchup", "he dipped fries in ketchup", "dipped", "literal"),
    ("dip the vegetables in hummus", "dip the vegetables in hummus", "dip", "literal"),

    ("why are you tripping about the homework", "why are you overreacting about the homework", "tripping", "slang"),
    ("he is tripping over one small mistake", "he is overreacting over one small mistake", "tripping", "slang"),
    ("she was tripping over the loose cable", "she was tripping over the loose cable", "tripping", "literal"),
    ("this rug is a tripping hazard", "this rug is a tripping hazard", "tripping", "literal"),

    ("that skateboard trick was sick", "that skateboard trick was excellent", "sick", "slang"),
    ("her new design is sick", "her new design is excellent", "sick", "slang"),
    ("he felt sick after lunch", "he felt sick after lunch", "sick", "literal"),
    ("the patient is sick", "the patient is sick", "sick", "literal"),

    ("this verse is fire", "this verse is excellent", "fire", "slang"),
    ("the edit was straight fire", "the edit was excellent", "fire", "slang"),
    ("the oven caught fire", "the oven caught fire", "fire", "literal"),
    ("the fire truck arrived", "the fire truck arrived", "fire", "literal"),

    ("that promotion is a flex", "that promotion is showing off", "flex", "slang"),
    ("she keeps flexing her score", "she keeps showing off her score", "flex", "slang"),
    ("flex your ankle gently", "flex your ankle gently", "flex", "literal"),
    ("he flexed his muscle", "he flexed his muscle", "flex", "literal"),

    ("no cap, this is true", "seriously, this is true", "cap", "slang"),
    ("his excuse was cap", "his excuse was a lie", "cap", "slang"),
    ("the lens cap fell off", "the lens cap fell off", "cap", "literal"),
    ("the gas cap is loose", "the gas cap is loose", "cap", "literal"),

    ("he ghosted my messages", "he stopped responding to my messages", "ghost", "slang"),
    ("they might ghost the client", "they might stop responding to the client", "ghost", "slang"),
    ("the haunted house had a ghost", "the haunted house had a ghost", "ghost", "literal"),
    ("the ghost movie was old", "the ghost movie was old", "ghost", "literal"),

    ("she is still salty about the review", "she is still upset about the review", "salty", "slang"),
    ("that comment sounded salty", "that comment sounded upset", "salty", "slang"),
    ("the broth was too salty", "the broth was too salty", "salty", "literal"),
    ("salty water covered the rocks", "salty water covered the rocks", "salty", "literal"),

    ("that seller seems shady", "that seller seems suspicious", "shady", "slang"),
    ("the excuse felt shady", "the excuse felt suspicious", "shady", "slang"),
    ("the shady spot was cool", "the shady spot was cool", "shady", "literal"),
    ("we rested on the shady side", "we rested on the shady side", "shady", "literal"),

    ("her jacket adds drip", "her jacket adds style", "drip", "slang"),
    ("the outfit has drip", "the outfit has style", "drip", "slang"),
    ("paint started to drip", "paint started to drip", "drip", "literal"),
    ("the pipe had a slow drip", "the pipe had a slow drip", "drip", "literal"),

    ("post the fit check", "post the outfit check", "fit", "slang"),
    ("that fit goes hard", "that outfit is impressive", "fit", "slang"),
    ("the key will fit the lock", "the key will fit the lock", "fit", "literal"),
    ("these pants fit well", "these pants fit well", "fit", "literal"),

    ("that entrance was extra", "that entrance was excessive", "extra", "slang"),
    ("stop being so extra", "stop being so excessive", "extra", "slang"),
    ("pack an extra blanket", "pack an extra blanket", "extra", "literal"),
    ("there are extra tickets", "there are extra tickets", "extra", "literal"),

    ("the episode was mid", "the episode was mediocre", "mid", "slang"),
    ("this meal is mid", "this meal is mediocre", "mid", "slang"),
    ("that's facts", "that is true", "facts", "slang"),
    ("your point is facts", "your point is true", "facts", "slang"),
    ("she is cracked at chess", "she is very good at chess", "cracked", "slang"),
    ("that goalie is washed", "that goalie is no longer good", "washed", "slang"),
    ("we are tight now", "we are close friends now", "tight", "slang"),
    ("the lid is tight", "the lid is tight", "tight", "literal"),
    ("his outfit looks sharp", "his outfit looks stylish", "sharp", "slang"),
    ("the needle is sharp", "the needle is sharp", "sharp", "literal"),

    ("i am cooked for the final", "i am in trouble for the final", "cooked", "slang"),
    ("they are cooked for the interview", "they are in trouble for the interview", "cooked", "slang"),
    ("we're cooked if we miss the deadline", "we're in trouble if we miss the deadline", "cooked", "slang"),
    ("i'm cooked without my notes", "i am in trouble without my notes", "cooked", "slang"),
    ("the vegetables are cooked", "the vegetables are cooked", "cooked", "literal"),
    ("the pasta was cooked well", "the pasta was cooked well", "cooked", "literal"),
    ("breakfast was cooked early", "breakfast was cooked early", "cooked", "literal"),

    ("she is a chill manager", "she is a relaxed manager", "chill", "slang"),
    ("that neighbor seems chill", "that neighbor seems relaxed", "chill", "slang"),
    ("she's super chill", "she's super relaxed", "chill", "slang"),
    ("the room was chill in the morning", "the room was chill in the morning", "chill", "literal"),
    ("a chill wind came through", "a chill wind came through", "chill", "literal"),

    ("this look is serving confidence", "this look is projecting confidence", "serving", "slang"),
    ("her outfit is serving main character energy", "her outfit is projecting main character energy", "serving", "slang"),
    ("the waiter is serving dinner", "the waiter is serving dinner", "serving", "literal"),
    ("she is serving soup to guests", "she is serving soup to guests", "serving", "literal"),

    ("her outfit looks snatched", "her outfit looks stylish and flattering", "snatched", "slang"),
    ("that fit looked snatched", "that outfit looked stylish and flattering", "snatched", "slang"),
    ("the thief snatched her bag", "the thief snatched her bag", "snatched", "literal"),
    ("he snatched the phone from the table", "he snatched the phone from the table", "snatched", "literal"),

    ("she slayed that presentation", "she did very well in that presentation", "slayed", "slang"),
    ("he slayed the performance", "he did very well in the performance", "slayed", "slang"),
    ("the knight slayed the dragon", "the knight slayed the dragon", "slayed", "literal"),
    ("the hero slayed a monster", "the hero slayed a monster", "slayed", "literal"),

    ("that beat slaps", "that beat is excellent", "slaps", "slang"),
    ("her soup is bussin", "her soup is delicious", "bussin", "slang"),
    ("deadass, this is true", "seriously, this is true", "deadass", "slang"),
    ("that scene hits different", "that scene feels especially meaningful", "hits different", "slang"),
    ("they understood the assignment", "they did exactly what was needed", "understood the assignment", "slang"),
    ("people ship that couple", "people support that couple as a couple", "ship", "slang"),
    ("the cargo ship arrived", "the cargo ship arrived", "ship", "literal"),
    ("ship the order today", "ship the order today", "ship", "literal"),
]

V4_DEV_EXTRA = [
    ("the party got lit fast", "the party got exciting fast", "lit", "slang"),
    ("the candle lit the hallway", "the candle lit the hallway", "lit", "literal"),
    ("tell me the tea", "tell me the gossip", "tea", "slang"),
    ("the tea bag ripped", "the tea bag ripped", "tea", "literal"),
    ("that old track is my jam", "that old track is my favorite", "my jam", "slang"),
    ("raspberry jam is sweet", "raspberry jam is sweet", "jam", "literal"),
    ("can we crash at their apartment", "can we sleep at their apartment", "crash", "slang"),
    ("the train crash was reported", "the train crash was reported", "crash", "literal"),
    ("their beef with the rival is old", "their conflict with the rival is old", "beef", "slang"),
    ("beef tacos are on the menu", "beef tacos are on the menu", "beef", "literal"),
    ("the bass line was nasty", "the bass line was amazing", "nasty", "slang"),
    ("the garbage smelled nasty", "the garbage smelled nasty", "nasty", "literal"),
    ("she ate the speech", "she did very well in the speech", "ate", "slang"),
    ("she ate rice", "she ate rice", "ate", "literal"),
    ("he dipped from the event", "he left the event", "dipped", "slang"),
    ("he dipped bread in oil", "he dipped bread in oil", "dipped", "literal"),
    ("you are tripping about nothing", "you are overreacting about nothing", "tripping", "slang"),
    ("he is tripping on the stairs", "he is tripping on the stairs", "tripping", "literal"),
    ("this trick is sick", "this trick is excellent", "sick", "slang"),
    ("she called in sick", "she called in sick", "sick", "literal"),
    ("the playlist is fire", "the playlist is excellent", "fire", "slang"),
    ("fire smoke filled the room", "fire smoke filled the room", "fire", "literal"),
    ("that's a big flex", "that is a big way to show off", "flex", "slang"),
    ("flex the ankle slowly", "flex the ankle slowly", "flex", "literal"),
    ("we're cooked without a backup plan", "we're in trouble without a backup plan", "cooked", "slang"),
    ("the rice is cooked", "the rice is cooked", "cooked", "literal"),
    ("this look is serving attitude", "this look is projecting attitude", "serving", "slang"),
    ("the cafe is serving breakfast", "the cafe is serving breakfast", "serving", "literal"),
    ("her waist looks snatched", "her waist looks stylish and flattering", "snatched", "slang"),
    ("someone snatched my purse", "someone snatched my purse", "snatched", "literal"),
    ("they slayed the routine", "they did very well in the routine", "slayed", "slang"),
    ("the warrior slayed the beast", "the warrior slayed the beast", "slayed", "literal"),
]


def read_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list.")
    return data


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
        f.write("\n")


def valid_row(item: dict[str, Any]) -> bool:
    return bool(item.get("input")) and bool(item.get("target")) and item.get("sense") in {"slang", "literal"}


def remove_gold_leakage(rows: list[dict[str, Any]], gold_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    gold_inputs = {norm_key(item["input"]) for item in gold_rows}
    kept = [item for item in rows if norm_key(item.get("input", "")) not in gold_inputs]
    return kept, len(rows) - len(kept)


def add_without_conflicts(rows: list[dict[str, Any]], extra_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    targets_by_input = {norm_key(item["input"]): norm_key(item["target"]) for item in rows}
    combined = rows[:]
    skipped = 0
    for item in extra_rows:
        key = norm_key(item["input"])
        target = norm_key(item["target"])
        if key in targets_by_input and targets_by_input[key] != target:
            skipped += 1
            continue
        targets_by_input[key] = target
        combined.append(item)
    return combined, skipped


def validate_no_conflicts(rows: list[dict[str, Any]]) -> list[str]:
    errors = []
    targets_by_input: dict[str, str] = {}
    for idx, item in enumerate(rows):
        if not valid_row(item):
            errors.append(f"row {idx}: invalid row {item}")
            continue
        if item["sense"] == "literal" and norm_key(item["input"]) != norm_key(item["target"]):
            errors.append(f"row {idx}: literal row is not identity {item}")
        key = norm_key(item["input"])
        target = norm_key(item["target"])
        if key in targets_by_input and targets_by_input[key] != target:
            errors.append(f"row {idx}: conflicting target for input {item['input']!r}")
        targets_by_input[key] = target
    return errors


def split_base_rows(rows: list[dict[str, Any]], dev_size: float, rng: random.Random) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in rows:
        grouped.setdefault((item.get("term", ""), item.get("sense", "")), []).append(item)

    train_rows = []
    dev_rows = []
    for group_rows in grouped.values():
        shuffled = group_rows[:]
        rng.shuffle(shuffled)
        dev_count = max(1, int(round(len(shuffled) * dev_size))) if len(shuffled) >= 5 else 0
        dev_rows.extend(shuffled[:dev_count])
        train_rows.extend(shuffled[dev_count:])
    return train_rows, dev_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build normalizer v4 train/test data.")
    parser.add_argument("--base-train", default=str(DEFAULT_BASE_TRAIN))
    parser.add_argument("--base-test", default=str(DEFAULT_BASE_TEST))
    parser.add_argument("--gold", default=str(DEFAULT_GOLD))
    parser.add_argument("--train-out", default=str(DEFAULT_TRAIN_OUT))
    parser.add_argument("--test-out", default=str(DEFAULT_TEST_OUT))
    parser.add_argument("--hard-repeat", type=int, default=35)
    parser.add_argument("--base-dev-size", type=float, default=0.12)
    args = parser.parse_args()

    rng = random.Random(SEED)
    gold_rows = read_json(Path(args.gold))
    base_rows = [
        item
        for item in read_json(Path(args.base_train)) + read_json(Path(args.base_test))
        if valid_row(item)
    ]
    base_rows, leaked = remove_gold_leakage(base_rows, gold_rows)
    base_train_rows, base_dev_rows = split_base_rows(base_rows, args.base_dev_size, rng)

    hard_train = [
        row(source, target, term, sense, "v4_hard_contrast")
        for source, target, term, sense in V4_HARD_TRAINING
        for _ in range(args.hard_repeat)
    ]
    hard_dev = [
        row(source, target, term, sense, "v4_dev_contrast")
        for source, target, term, sense in V4_DEV_EXTRA
    ]
    hard_train, leaked_hard = remove_gold_leakage(hard_train, gold_rows)
    hard_dev, leaked_dev = remove_gold_leakage(hard_dev, gold_rows)

    train_rows, skipped_train_conflicts = add_without_conflicts(base_train_rows, hard_train)
    test_rows, skipped_test_conflicts = add_without_conflicts(base_dev_rows, hard_dev)

    rng.shuffle(train_rows)
    rng.shuffle(test_rows)

    errors = validate_no_conflicts(train_rows + test_rows)
    if errors:
        preview = "\n".join(errors[:20])
        raise SystemExit(f"Dataset validation failed with {len(errors)} errors:\n{preview}")

    write_json(Path(args.train_out), train_rows)
    write_json(Path(args.test_out), test_rows)

    print(f"Wrote {len(train_rows)} train rows to {args.train_out}")
    print(f"Wrote {len(test_rows)} test rows to {args.test_out}")
    print(f"Removed {leaked} base rows with exact gold inputs")
    print(f"Removed {leaked_hard} hard-train rows with exact gold inputs")
    print(f"Removed {leaked_dev} hard-dev rows with exact gold inputs")
    print(f"Skipped train conflicts: {skipped_train_conflicts}")
    print(f"Skipped test conflicts: {skipped_test_conflicts}")
    for label, rows_for_label in [("train", train_rows), ("test", test_rows)]:
        print(
            f"{label}: slang={sum(1 for item in rows_for_label if item['sense'] == 'slang')} "
            f"literal={sum(1 for item in rows_for_label if item['sense'] == 'literal')} "
            f"identity={sum(1 for item in rows_for_label if norm_key(item['input']) == norm_key(item['target']))}"
        )


if __name__ == "__main__":
    main()
