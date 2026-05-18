"""
Merge reviewed feedback candidates into gold eval and normalizer training data.

Rules:
- Gold eval receives only human-reviewed originals from feedback_gold_candidates.json.
- Training receives human-reviewed originals plus deterministic safe variants.
- Literal guards are training-only unless they were explicitly reviewed feedback.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

DEFAULT_FEEDBACK_GOLD = BACKEND_ROOT / "data" / "feedback_gold_candidates.json"
DEFAULT_FEEDBACK_TRAIN = BACKEND_ROOT / "data" / "feedback_training_candidates.jsonl"
DEFAULT_PIPELINE_GOLD = BACKEND_ROOT / "data" / "slang_pipeline_gold_cases.json"
DEFAULT_NORMALIZER_TRAIN = BACKEND_ROOT / "data" / "slang_normalizer_v4_train.json"

FEEDBACK_GOLD_SOURCE = "feedback_approved"
FEEDBACK_TRAINING_SOURCES = {
    "feedback_approved",
    "feedback_augmented",
    "feedback_literal_guard",
}


KNOWN_TERMS = (
    "grindding",
    "grinding",
    "grindin",
    "lowkey",
    "capping",
    "cap",
    "free",
    "sold",
    "served",
    "shade",
    "salty",
    "ghosted",
    "died",
    "down bad",
    "sent me",
    "send",
    "folded",
    "clutch",
    "wild",
    "cooked",
    "carried",
    "hardstuck",
    "tilted",
    "sweaty",
    "pressed",
    "beat",
    "sick",
    "tea",
    "slaps",
    "slap",
    "mid",
    "bet",
    "ship",
    "washed",
    "rizz",
    "clean",
    "read",
    "humbled",
    "flex",
    "af",
)


LITERAL_GUARDS = [
    ("i'm grinding coffee beans", "i am grinding coffee beans", "grinding"),
    ("the machine is grinding metal", "the machine is grinding metal", "grinding"),
    ("she cooked rice for dinner", "she cooked rice for dinner", "cooked"),
    ("the chef cooked the soup", "the chef cooked the soup", "cooked"),
    ("she carried the box upstairs", "she carried the box upstairs", "carried"),
    ("the lobby has three chairs", "the lobby has three chairs", "lobby"),
    ("the soup is salty", "the soup is salty", "salty"),
    ("the cap is on the bottle", "the cap is on the bottle", "cap"),
    ("send me the file", "send me the file", "send"),
    ("the phone died after five years", "the phone died after five years", "died"),
    ("he pressed the red button", "he pressed the red button", "pressed"),
    ("the wild horse ran across the field", "the wild horse ran across the field", "wild"),
    ("the picture is tilted", "the picture is tilted", "tilted"),
    ("tilt the screen slightly", "tilt the screen slightly", "tilted"),
    ("the table is tilted to one side", "the table is tilted to one side", "tilted"),
    ("the chair tilted backward", "the chair tilted backward", "tilted"),
    ("she cooked pasta for dinner", "she cooked pasta for dinner", "cooked"),
    ("he cooked rice yesterday", "he cooked rice yesterday", "cooked"),
    ("the vegetables are cooked", "the vegetables are cooked", "cooked"),
    ("the pasta was cooked well", "the pasta was cooked well", "cooked"),
    ("she sold her old laptop", "she sold her old laptop", "sold"),
    ("they sold tickets online", "they sold tickets online", "sold"),
    ("he sold his car yesterday", "he sold his car yesterday", "sold"),
    ("the beat is steady", "the beat is steady", "beat"),
    ("i love that beat", "i love that beat", "beat"),
    ("that is a great beat", "that is a great beat", "beat"),
    ("i love listening to this beat", "i love listening to this beat", "beat"),
    ("the song has a slow beat", "the song has a slow beat", "beat"),
    ("the drummer kept the beat", "the drummer kept the beat", "beat"),
    ("drop the beat", "start the beat", "beat"),
    ("the beat drops after the intro", "the beat drops after the intro", "beat"),
    ("the beat is too fast", "the beat is too fast", "beat"),
    ("the beat starts at the chorus", "the beat starts at the chorus", "beat"),
    ("the producer changed the beat", "the producer changed the beat", "beat"),
    ("this song has a heavy beat", "this song has a heavy beat", "beat"),
    ("she spilled tea on the table", "she spilled tea on the table", "tea"),
    ("i drank green tea", "i drank green tea", "tea"),
    ("the tea kettle is hot", "the tea kettle is hot", "tea"),
    ("she brewed tea before class", "she brewed tea before class", "tea"),
    ("the tea bag ripped", "the tea bag ripped", "tea"),
    ("he spilled hot tea on his shirt", "he spilled hot tea on his shirt", "tea"),
    ("we met in mid July", "we met in mid July", "mid"),
    ("the meeting is in mid March", "the meeting is in mid March", "mid"),
    ("we are planning it for mid December", "we are planning it for mid December", "mid"),
    ("the project starts in mid August", "the project starts in mid August", "mid"),
    ("she moved here in mid September", "she moved here in mid September", "mid"),
    ("the store opens in mid November", "the store opens in mid November", "mid"),
    ("they made a bet on the game", "they made a bet on the game", "bet"),
    ("the workers ship packages daily", "the workers ship packages daily", "ship"),
    ("the cargo ship arrived late", "the cargo ship arrived late", "ship"),
    ("ship the order today", "ship the order today", "ship"),
    ("the shirt was washed yesterday", "the shirt was washed yesterday", "washed"),
    ("the glass is cracked", "the glass is cracked", "cracked"),
    ("he slaps the table when he laughs", "he slaps the table when he laughs", "slaps"),
    ("press the clutch slowly", "press the clutch slowly", "clutch"),
    ("his shirt is sweaty after the run", "his shirt is sweaty after the run", "sweaty"),
    ("the app is free to download", "the app is free to download", "free"),
    ("the concert is free tonight", "the concert is free tonight", "free"),
    ("this sample is free", "this sample is free", "free"),
    ("the free version has ads", "the free version has ads", "free"),
    ("basic math is useful", "basic math is useful", "basic"),
    ("bring an extra towel", "bring an extra towel", "extra"),
    ("we sat in the shade", "we sat in the shade", "shade"),
]


NON_IDENTITY_LITERAL_FAILURE_TYPES = {
    "literal_intensifier_guard",
    "music_phrase",
}


def clean(text: str) -> str:
    return " ".join((text or "").strip().split())


def norm_key(text: str) -> str:
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


def write_json_list(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_compact_json_list(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("[\n")
        for index, row in enumerate(rows):
            suffix = "," if index < len(rows) - 1 else ""
            f.write(f"  {json.dumps(row, ensure_ascii=False)}{suffix}\n")
        f.write("]\n")


def contains_term(text: str, term: str) -> bool:
    lower = norm_key(text)
    if " " in term:
        return term in lower
    return re.search(rf"(?<![a-z]){re.escape(term)}(?![a-z])", lower) is not None


def infer_term(text: str, fallback: str = "") -> str:
    for term in KNOWN_TERMS:
        if contains_term(text, term):
            if term in {"grindding", "grindin"}:
                return "grinding"
            return term
    return fallback


def training_row(
    input_text: str,
    target: str,
    term: str,
    sense: str,
    source: str,
    source_feedback_id: str | None = None,
    failure_type: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "input": clean(input_text),
        "target": clean(target),
        "term": term,
        "sense": sense,
        "source": source,
    }
    if source_feedback_id:
        row["source_feedback_id"] = source_feedback_id
    if failure_type:
        row["failure_type"] = failure_type
    return row


def validate_training_row(row: dict[str, Any]) -> str | None:
    source = clean(row.get("input", ""))
    target = clean(row.get("target", ""))
    sense = row.get("sense", "")
    if not source or not target:
        return "missing input or target"
    if sense not in {"slang", "literal"}:
        return "invalid sense"
    failure_type = row.get("failure_type")
    is_allowed_literal_rewrite = failure_type in NON_IDENTITY_LITERAL_FAILURE_TYPES
    if sense == "literal" and norm_key(source) != norm_key(target) and not is_allowed_literal_rewrite:
        return "literal row must be identity"
    if len(target) > max(90, int(len(source) * 2.4)):
        return "target is unexpectedly long"
    return None


def gaming_grinding_variants(source_feedback_id: str, failure_type: str | None) -> list[dict[str, Any]]:
    pairs = [
        ("i'm grinding roblox", "i am playing roblox intensely"),
        ("i'm grindding valorant", "i am playing valorant intensely"),
        ("i'm grinding fortnite", "i am playing fortnite intensely"),
        ("i'm grindin ranked", "i am playing ranked mode intensely"),
        ("i'm grinding valorant ranked", "i am playing ranked Valorant intensely"),
    ]
    return [
        training_row(source, target, "grinding", "slang", "feedback_augmented", source_feedback_id, failure_type)
        for source, target in pairs
    ]


def generated_variants(row: dict[str, Any]) -> list[dict[str, Any]]:
    source = norm_key(row.get("input", ""))
    source_feedback_id = row.get("source_feedback_id")
    failure_type = row.get("failure_type")
    variants: list[dict[str, Any]] = []

    def add(input_text: str, target: str, term: str, sense: str = "slang") -> None:
        variants.append(training_row(input_text, target, term, sense, "feedback_augmented", source_feedback_id, failure_type))

    def add_many(term: str, pairs: list[tuple[str, str]], sense: str = "slang") -> None:
        for input_text, target in pairs:
            add(input_text, target, term, sense)

    if "grind" in source:
        variants.extend(gaming_grinding_variants(source_feedback_id, failure_type))

    if "lowkey" in source:
        add_many("lowkey", [
            ("that game is lowkey fun", "that game is somewhat fun"),
            ("i lowkey want to leave", "i kind of want to leave"),
            ("she lowkey carried the team", "she somewhat carried the team"),
            ("this song is lowkey fire", "this song is somewhat excellent"),
            ("i lowkey need help", "i kind of need help"),
            ("that answer is lowkey wrong", "that answer is somewhat wrong"),
        ])

    if "capping" in source or " cap" in f" {source}":
        add_many("cap", [
            ("stop capping about your rank", "stop lying about your rank"),
            ("don't cap about your score", "do not lie about your score"),
            ("he is capping about being diamond", "he is lying about being diamond"),
            ("stop the cap", "stop lying"),
            ("that's cap", "that is a lie"),
            ("no cap that was scary", "honestly that was scary"),
        ])

    if "match is free" in source or "lobby is free" in source:
        add_many("free", [
            ("this lobby is free", "this lobby is easy to win"),
            ("that ranked match is free", "that ranked match is easy to win"),
            ("this game is free for us", "this game is easy for us to win"),
            ("that ranked game was free", "that ranked game was easy to win"),
            ("this round is free", "this round is easy to win"),
            ("their defense is free", "their defense is easy to beat"),
        ])

    if "sold" in source:
        add_many("sold", [
            ("i sold that round", "i played badly and caused us to lose that round"),
            ("i sold the match", "i played badly and caused the match to go poorly"),
            ("he sold the final fight", "he played badly and caused the final fight to go poorly"),
            ("she sold the clutch", "she played badly and failed at the critical moment"),
            ("we sold the game in overtime", "we played badly and caused the game to go poorly in overtime"),
            ("don't sell this round", "do not play badly and ruin this round"),
            ("i sold the one versus one", "i played badly and lost the one versus one"),
            ("our tank sold the fight", "our tank played badly and caused the fight to go poorly"),
            ("he sold my ranked game", "he played badly and ruined my ranked game"),
            ("we almost sold that round", "we almost played badly and ruined that round"),
            ("she sold the last push", "she played badly and ruined the last push"),
            ("please don't sell the match", "please do not play badly and ruin the match"),
        ])

    if "served looks" in source:
        add("she served looks at the party", "she looked very stylish at the party", "served")
        add("that outfit served looks", "that outfit looked very stylish", "served")
        add("he served looks in that jacket", "he looked very stylish in that jacket", "served")

    if "shade" in source:
        add_many("shade", [
            ("that comment was pure shade", "that comment was a subtle insult"),
            ("she threw shade in the meeting", "she made a subtle insult in the meeting"),
            ("his reply had shade", "his reply had a subtle insult"),
            ("she threw shade at him", "she made a subtle insult toward him"),
            ("that caption was shade", "that caption was a subtle insult"),
            ("he keeps throwing shade", "he keeps making subtle insults"),
        ])

    if "salty" in source:
        add_many("salty", [
            ("he is salty after losing", "he is upset after losing"),
            ("why are you so salty", "why are you so upset"),
            ("that reply sounded salty", "that reply sounded upset"),
            ("my teammates got me all tilted. now i'm salty af and playing like shit.", "my teammates made me very frustrated. now i am very upset and playing very badly."),
            ("she got salty after the match", "she became upset after the match"),
            ("that salty comment was unnecessary", "that upset comment was unnecessary"),
        ])

    if "ghosted" in source:
        add("she ghosted me after the date", "she stopped responding to me after the date", "ghosted")
        add("he ghosted the group chat", "he stopped responding to the group chat", "ghosted")
        add("my friend ghosted me again", "my friend stopped responding to me again", "ghosted")

    if "phone died" in source:
        add("my phone died", "my phone stopped working because the battery ran out", "died")
        add("my laptop died during class", "my laptop stopped working because the battery ran out during class", "died")
        add("his controller died mid game", "his controller stopped working because the battery ran out during the game", "died")

    if "down bad" in source:
        add("i'm down bad for her", "i am extremely attracted to her", "down bad")
        add("he is down bad for his crush", "he is extremely attracted to his crush", "down bad")
        add("they are down bad after one date", "they are extremely attracted after one date", "down bad")

    if "sent me" in source:
        add("that joke sent me", "that joke made me laugh a lot", "sent me")
        add("that meme sent me", "that meme made me laugh a lot", "sent me")
        add("her reaction sent me", "her reaction made me laugh a lot", "sent me")

    if "folded" in source:
        add("he folded under pressure", "he gave up or failed under pressure", "folded")
        add("our team folded in overtime", "our team failed in overtime", "folded")
        add("she folded during the argument", "she gave up during the argument", "folded")

    if contains_term(source, "af"):
        add("this game is laggy af", "this game is very laggy", "af")
        add("that lobby was sweaty af", "that lobby was very competitive", "af")
        add("my internet is slow af", "my internet is very slow", "af")

    if "clutch" in source:
        add_many("clutch", [
            ("i need to clutch this round", "i need to succeed in this round at a critical moment"),
            ("she clutched the final round", "she succeeded in the final round at a critical moment"),
            ("that save was clutch", "that save was important at a critical moment"),
            ("that clutch was insane", "that last moment win was incredible"),
            ("he clutched the one versus three", "he won the one versus three at a critical moment"),
            ("we need a clutch play", "we need an important play at a critical moment"),
            ("he needs to clutch this", "he needs to succeed at this critical moment"),
            ("i clutched the last fight", "i succeeded in the last fight at a critical moment"),
            ("that clutch won us the game", "that critical moment win won us the game"),
            ("she hit a clutch shot", "she hit an important shot at a critical moment"),
            ("clutch this for us", "succeed at this critical moment for us"),
        ])

    if "wild" in source:
        add("that take is wild", "that opinion is shocking or extreme", "wild")
        add("his story was wild", "his story was shocking or extreme", "wild")
        add("this comment section is wild", "this comment section is shocking or extreme", "wild")

    if "cooked" in source:
        add_many("cooked", [
            ("we got cooked in ranked", "we lost badly in ranked mode"),
            ("that team cooked us", "that team beat us badly"),
            ("they cooked us in overtime", "they beat us badly in overtime"),
            ("we got cooked by that squad", "we were beaten badly by that squad"),
            ("their top laner cooked us", "their top laner beat us badly"),
            ("he cooked in the debate", "he performed extremely well in the debate"),
            ("she cooked during the presentation", "she performed extremely well during the presentation"),
            ("let him cook", "let him keep going"),
            ("we got cooked in overtime", "we lost badly in overtime"),
            ("our team got cooked", "our team lost badly"),
            ("the enemy cooked us in ranked", "the enemy beat us badly in ranked mode"),
            ("their squad cooked our lobby", "their squad beat our lobby badly"),
            ("she cooked everyone in the debate", "she performed extremely well against everyone in the debate"),
            ("he cooked with that answer", "he performed extremely well with that answer"),
            ("they cooked on stage", "they performed extremely well on stage"),
            ("let her cook", "let her keep going"),
            ("i'm cooked for the final", "i am in trouble for the final"),
            ("we're cooked without a healer", "we are in trouble without a healer"),
        ])

    if "carried" in source:
        add("bro carried the lobby", "he performed very well for the whole lobby", "carried")
        add("she carried our team", "she performed very well for our team", "carried")
        add("he carried the match", "he performed very well for the whole match", "carried")

    if "hardstuck" in source:
        add_many("hardstuck", [
            ("i'm hardstuck bronze", "i am unable to rank up from bronze"),
            ("she is hardstuck silver", "she is unable to rank up from silver"),
            ("we are hardstuck gold", "we are unable to rank up from gold"),
            ("he is hardstuck platinum", "he is unable to rank up from platinum"),
            ("i've been hardstuck for weeks", "i have been unable to rank up for weeks"),
            ("our duo is hardstuck diamond", "our duo is unable to rank up from diamond"),
        ])

    if "tilted" in source:
        add_many("tilted", [
            ("i'm tilted after that game", "i am frustrated after that game"),
            ("he got tilted after losing", "he became frustrated after losing"),
            ("don't queue while tilted", "do not queue while frustrated"),
            ("im tilted", "i am frustrated"),
            ("i'm getting tilted already", "i am already getting frustrated"),
            ("that camper is making me so tilted", "that camper is making me very frustrated"),
            ("my teammates tilted me", "my teammates frustrated me"),
            ("this match tilted me so hard", "this match frustrated me a lot"),
            ("i'm tilted rn", "i am frustrated right now"),
            ("stop playing while tilted", "stop playing while frustrated"),
            ("that throw made me tilted", "that mistake made me frustrated"),
            ("losing that fight tilted me", "losing that fight frustrated me"),
            ("he is too tilted to play", "he is too frustrated to play"),
            ("she gets tilted after every loss", "she gets frustrated after every loss"),
            ("the lag tilted our whole team", "the lag frustrated our whole team"),
            ("i was tilted for the rest of the match", "i was frustrated for the rest of the match"),
            ("being camped tilted me", "being camped frustrated me"),
            ("mute chat if you get tilted", "mute chat if you get frustrated"),
            ("that bad call tilted everyone", "that bad call frustrated everyone"),
            ("i'm so tilted right now", "i am very frustrated right now"),
        ])

    if "sweaty" in source:
        add_many("sweaty", [
            ("this lobby is sweaty", "this lobby is very competitive"),
            ("that lobby was sweaty af", "that lobby was very competitive"),
            ("we queued into sweats", "we matched against very competitive players"),
            ("their team is sweaty", "their team is very competitive"),
            ("casual mode is sweaty today", "casual mode is very competitive today"),
            ("these players are sweaty", "these players are very competitive"),
            ("ranked is sweaty tonight", "ranked mode is very competitive tonight"),
            ("every match feels sweaty", "every match feels very competitive"),
            ("that enemy team was sweaty", "that enemy team was very competitive"),
        ])

    if "beat" in source or "sick" in source:
        add_many("sick", [
            ("that beat is sick", "that beat is excellent"),
            ("this beat is sick", "this beat is excellent"),
            ("that drum pattern is sick", "that drum pattern is excellent"),
            ("that guitar solo was sick", "that guitar solo was excellent"),
            ("this edit is sick", "this edit is excellent"),
            ("that beat sounds sick", "that beat sounds excellent"),
            ("the beat drop was sick", "the beat drop was excellent"),
            ("that bass line is sick", "that bass line is excellent"),
            ("this rhythm is sick", "this rhythm is excellent"),
            ("that transition was sick", "that transition was excellent"),
            ("his flow on that beat is sick", "his flow on that beat is excellent"),
        ])

    if "tea" in source:
        add_many("tea", [
            ("spill the tea", "share the gossip"),
            ("tell me the tea", "tell me the gossip"),
            ("what's the tea about the group chat", "what is the gossip about the group chat"),
            ("share the tea from the office", "share the gossip from the office"),
            ("she has tea about the drama", "she has gossip about the drama"),
            ("give me the tea", "give me the gossip"),
            ("he spilled the tea in the group chat", "he shared the gossip in the group chat"),
            ("there is tea about the breakup", "there is gossip about the breakup"),
            ("what's the tea on that rumor", "what is the gossip about that rumor"),
            ("don't spill the tea yet", "do not share the gossip yet"),
        ])

    if "mid" in source:
        add_many("mid", [
            ("that movie was mid", "that movie was mediocre"),
            ("the food was kinda mid", "the food was somewhat mediocre"),
            ("that episode was mid", "that episode was mediocre"),
            ("this skin is mid", "this skin is mediocre"),
            ("the update feels mid", "the update feels mediocre"),
            ("that game was mid", "that game was mediocre"),
            ("the new map is mid", "the new map is mediocre"),
            ("his verse was mid", "his verse was mediocre"),
            ("that restaurant is mid", "that restaurant is mediocre"),
            ("the trailer looked mid", "the trailer looked mediocre"),
        ])

    if "ship" in source:
        add_many("ship", [
            ("people ship them together", "people support them as a couple"),
            ("i ship those two characters", "i support those two characters as a couple"),
            ("fans ship the main characters", "fans support the main characters as a couple"),
            ("do you ship them", "do you support them as a couple"),
            ("everyone ships that couple", "everyone supports that couple as a couple"),
            ("i ship them so hard", "i strongly support them as a couple"),
            ("the fandom ships those two", "the fandom supports those two as a couple"),
            ("nobody ships that pairing", "nobody supports that pairing as a couple"),
            ("she ships the rivals", "she supports the rivals as a couple"),
        ])

    if "washed" in source:
        add_many("washed", [
            ("he is washed now", "he is no longer good now"),
            ("that player is washed", "that player is no longer good"),
            ("the old champion is washed", "the old champion is no longer good"),
            ("people say he is washed", "people say he is no longer good"),
        ])

    if "flex" in source:
        add_many("flex", [
            ("that promotion is a flex", "that promotion is showing off"),
            ("weird flex but okay", "strange way to show off but okay"),
            ("she keeps flexing her score", "she keeps showing off her score"),
            ("he flexed his new car", "he showed off his new car"),
            ("that expensive watch is a flex", "that expensive watch is showing off"),
            ("stop flexing your rank", "stop showing off your rank"),
            ("his score was a flex", "his score was showing off"),
        ])

    if "rizz" in source:
        add_many("rizz", [
            ("he's got rizz", "he has charisma"),
            ("she has rizz", "she has charisma"),
            ("that guy has no rizz", "that guy has no charisma"),
            ("his rizz is unreal", "his charisma is unreal"),
        ])

    if "clean" in source:
        add_many("clean", [
            ("i'm not gonna lie that was clean", "honestly that was impressive"),
            ("that play was clean", "that play was impressive"),
            ("the edit was clean", "the edit was impressive"),
            ("that combo was clean", "that combo was impressive"),
        ])

    if "read" in source:
        add_many("read", [
            ("she left me on read", "she read my message and did not respond"),
            ("he left me on read again", "he read my message and did not respond again"),
            ("don't leave me on read", "do not read my message without responding"),
        ])

    if "humbled" in source:
        add_many("humbled", [
            ("that test humbled me", "that test made me realize I was not as prepared as I thought"),
            ("ranked humbled me today", "ranked made me realize I was not as good as I thought today"),
            ("that loss humbled us", "that loss made us realize we were not as good as we thought"),
        ])

    return variants


def merge_gold(existing_rows: list[dict[str, Any]], feedback_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    base_rows = [row for row in existing_rows if row.get("source") != FEEDBACK_GOLD_SOURCE]
    seen = {(norm_key(row.get("input", "")), norm_key(row.get("expected", "")), row.get("kind", "")) for row in base_rows}
    merged = base_rows[:]
    added = 0

    for row in feedback_rows:
        input_text = clean(row.get("input", ""))
        expected = clean(row.get("expected", ""))
        kind = row.get("kind", "")
        if not input_text or not expected or kind not in {"slang", "literal"}:
            continue
        key = (norm_key(input_text), norm_key(expected), kind)
        if key in seen:
            continue
        seen.add(key)
        merged.append(
            {
                "input": input_text,
                "expected": expected,
                "kind": kind,
                "source": FEEDBACK_GOLD_SOURCE,
                "source_feedback_id": row.get("source_feedback_id"),
                "failure_type": row.get("failure_type"),
            }
        )
        added += 1

    return merged, added


def merge_training(existing_rows: list[dict[str, Any]], feedback_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int, int, int]:
    base_rows = [row for row in existing_rows if row.get("source") not in FEEDBACK_TRAINING_SOURCES]
    seen = {norm_key(row.get("input", "")) for row in base_rows}
    merged = base_rows[:]
    added_originals = 0
    added_augmented = 0
    skipped = 0

    def add_row(row: dict[str, Any], is_augmented: bool) -> None:
        nonlocal added_originals, added_augmented, skipped
        if validate_training_row(row):
            skipped += 1
            return
        key = norm_key(row.get("input", ""))
        if key in seen:
            skipped += 1
            return
        seen.add(key)
        merged.append(row)
        if is_augmented:
            added_augmented += 1
        else:
            added_originals += 1

    for row in feedback_rows:
        original = training_row(
            input_text=row.get("input", ""),
            target=row.get("target", ""),
            term=infer_term(row.get("input", "")),
            sense=row.get("sense", ""),
            source="feedback_approved",
            source_feedback_id=row.get("source_feedback_id"),
            failure_type=row.get("failure_type"),
        )
        add_row(original, is_augmented=False)

        if row.get("sense") == "slang":
            for variant in generated_variants(row):
                add_row(variant, is_augmented=True)

    for source, target, term in LITERAL_GUARDS:
        add_row(training_row(source, target, term, "literal", "feedback_literal_guard"), is_augmented=True)

    return merged, added_originals, added_augmented, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge feedback candidates into eval/training datasets.")
    parser.add_argument("--feedback-gold", default=str(DEFAULT_FEEDBACK_GOLD))
    parser.add_argument("--feedback-train", default=str(DEFAULT_FEEDBACK_TRAIN))
    parser.add_argument("--pipeline-gold", default=str(DEFAULT_PIPELINE_GOLD))
    parser.add_argument("--normalizer-train", default=str(DEFAULT_NORMALIZER_TRAIN))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    feedback_gold = read_json_list(Path(args.feedback_gold))
    feedback_train = read_jsonl(Path(args.feedback_train))
    pipeline_gold = read_json_list(Path(args.pipeline_gold))
    normalizer_train = read_json_list(Path(args.normalizer_train))

    merged_gold, added_gold = merge_gold(pipeline_gold, feedback_gold)
    merged_train, added_originals, added_augmented, skipped_train = merge_training(normalizer_train, feedback_train)

    if not args.dry_run:
        write_compact_json_list(Path(args.pipeline_gold), merged_gold)
        write_json_list(Path(args.normalizer_train), merged_train)

    print(f"Feedback gold rows: {len(feedback_gold)}")
    print(f"Feedback training rows: {len(feedback_train)}")
    print(f"Gold eval rows: {len(pipeline_gold)} -> {len(merged_gold)} (+{added_gold})")
    print(
        "Normalizer training rows: "
        f"{len(normalizer_train)} -> {len(merged_train)} "
        f"(+{added_originals} originals, +{added_augmented} augmented/guards, {skipped_train} skipped)"
    )
    if args.dry_run:
        print("Dry run only; no files changed.")


if __name__ == "__main__":
    main()
