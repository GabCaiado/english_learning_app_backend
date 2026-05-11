"""
Build a sentence-level slang normalizer V3.1 dataset.

This dataset trains T5/FLAN to convert slang or idiomatic English into standard English.
Run:
  python scripts/build_normalizer_v3_dataset.py
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


DATA_DIR = Path("data")
TRAIN_PATH = DATA_DIR / "slang_normalizer_v3_1_train.json"
TEST_PATH = DATA_DIR / "slang_normalizer_v3_1_test.json"
TARGETED_APPROVED_PATH = DATA_DIR / "targeted_normalizer_eval_approved.json"
TARGETED_HARD_TRAINING_PATH = DATA_DIR / "targeted_normalizer_training_hard_cases.json"
SEED = 42
HARD_CASE_REPEAT_COUNT = 60

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


def row(
    source: str,
    target: str,
    term: str,
    sense: str,
    source_type: str = "template",
    split: str | None = None,
) -> dict[str, Any]:
    item = {
        "input": clean(source),
        "target": clean(target),
        "term": term,
        "sense": sense,
        "source": source_type,
    }
    if split:
        item["split"] = split
    return item


def clean(text: str) -> str:
    return " ".join(text.strip().split())


SLANG_PATTERNS: list[dict[str, Any]] = [
    {
        "term": "lit",
        "meaning": "exciting",
        "objects": ["concert", "party", "show", "festival", "night", "chat", "set", "performance"],
        "templates": [
            ("the {obj} was lit", "the {obj} was exciting"),
            ("that {obj} was lit", "that {obj} was exciting"),
            ("the {obj} got lit", "the {obj} got exciting"),
            ("last night's {obj} was lit", "last night's {obj} was exciting"),
        ],
    },
    {
        "term": "spill the tea",
        "objects": ["office drama", "group chat", "meeting", "party", "timeline", "comment thread"],
        "templates": [
            ("spill the tea", "share the gossip"),
            ("she spilled the tea", "she shared the gossip"),
            ("spill the tea about the {obj}", "share the gossip about the {obj}"),
            ("what's the tea about the {obj}", "what is the gossip about the {obj}"),
        ],
    },
    {
        "term": "throw shade",
        "objects": ["him", "her", "them", "the singer", "the team", "her coworker"],
        "templates": [
            ("she threw shade at {obj}", "she subtly insulted {obj}"),
            ("he threw shade at {obj}", "he subtly insulted {obj}"),
            ("they were throwing shade at {obj}", "they were subtly insulting {obj}"),
            ("stop throwing shade at {obj}", "stop subtly insulting {obj}"),
        ],
    },
    {
        "term": "crash",
        "objects": ["your place", "my friend's house", "his apartment", "her couch", "their guest room"],
        "templates": [
            ("can i crash at {obj}", "can i sleep at {obj}"),
            ("can i crash at {obj}?", "can i sleep at {obj}?"),
            ("i crashed at {obj}", "i slept at {obj}"),
            ("we crashed at {obj}", "we slept at {obj}"),
        ],
    },
    {
        "term": "jam",
        "objects": ["song", "track", "playlist", "album", "beat"],
        "templates": [
            ("this {obj} is my jam", "this {obj} is my favorite"),
            ("that {obj} is my jam", "that {obj} is my favorite"),
            ("this is my jam", "this is my favorite song"),
            ("that new {obj} is my jam", "that new {obj} is my favorite"),
        ],
    },
    {
        "term": "cooked",
        "objects": ["exam", "interview", "presentation", "final", "meeting"],
        "templates": [
            ("i'm cooked for the {obj}", "i am in trouble for the {obj}"),
            ("i am cooked for the {obj}", "i am in trouble for the {obj}"),
            ("we're cooked for the {obj}", "we are in trouble for the {obj}"),
            ("he is cooked for the {obj}", "he is in trouble for the {obj}"),
        ],
    },
    {
        "term": "chill",
        "objects": ["guy", "teacher", "manager", "neighbor", "friend", "person"],
        "templates": [
            ("he is a chill {obj}", "he is a relaxed {obj}"),
            ("she is a chill {obj}", "she is a relaxed {obj}"),
            ("my {obj} is chill", "my {obj} is relaxed"),
            ("that {obj} seems chill", "that {obj} seems relaxed"),
        ],
    },
    {
        "term": "nasty",
        "objects": ["beat drop", "solo", "dunk", "goal", "transition", "verse"],
        "templates": [
            ("that {obj} was nasty", "that {obj} was amazing"),
            ("the {obj} was nasty", "the {obj} was amazing"),
            ("his {obj} was nasty", "his {obj} was amazing"),
            ("her {obj} was nasty", "her {obj} was amazing"),
        ],
    },
    {
        "term": "ate and left no crumbs",
        "objects": ["performance", "presentation", "outfit", "routine", "speech", "scene"],
        "templates": [
            ("she ate and left no crumbs", "she did extremely well"),
            ("he ate and left no crumbs", "he did extremely well"),
            ("she ate and left no crumbs in the {obj}", "she did extremely well in the {obj}"),
            ("they ate and left no crumbs with that {obj}", "they did extremely well with that {obj}"),
        ],
    },
    {
        "term": "ship",
        "objects": ["those two characters", "them", "that couple", "those two", "the main characters"],
        "templates": [
            ("i ship {obj}", "i support {obj} as a couple"),
            ("we ship {obj}", "we support {obj} as a couple"),
            ("people ship {obj}", "people support {obj} as a couple"),
            ("do you ship {obj}?", "do you support {obj} as a couple?"),
        ],
    },
    {
        "term": "clutch",
        "objects": ["goal", "shot", "save", "play", "answer", "assist"],
        "templates": [
            ("that last-minute {obj} was clutch", "that last-minute {obj} was decisive"),
            ("the {obj} was clutch", "the {obj} was decisive"),
            ("his {obj} was clutch", "his {obj} was decisive"),
            ("her {obj} came in clutch", "her {obj} was very helpful at the right moment"),
        ],
    },
    {
        "term": "slay",
        "objects": ["performance", "presentation", "outfit", "interview", "routine", "audition"],
        "templates": [
            ("she slayed the {obj}", "she did extremely well in the {obj}"),
            ("he slayed the {obj}", "he did extremely well in the {obj}"),
            ("they slayed the {obj}", "they did extremely well in the {obj}"),
            ("she is slaying the {obj}", "she is doing extremely well in the {obj}"),
        ],
    },
]


SLANG_PATTERNS.extend([
    {
        "term": "fire",
        "objects": ["beat", "song", "outfit", "verse", "playlist", "performance", "design", "edit"],
        "templates": [
            ("this {obj} is fire", "this {obj} is excellent"),
            ("that {obj} was fire", "that {obj} was excellent"),
            ("her {obj} is straight fire", "her {obj} is excellent"),
            ("the new {obj} sounds fire", "the new {obj} sounds excellent"),
        ],
    },
    {
        "term": "sick",
        "objects": ["trick", "move", "goal", "solo", "transition", "routine", "design", "shot"],
        "templates": [
            ("that {obj} was sick", "that {obj} was excellent"),
            ("this {obj} is sick", "this {obj} is excellent"),
            ("her {obj} looked sick", "her {obj} looked excellent"),
            ("the new {obj} was sick", "the new {obj} was excellent"),
        ],
    },
    {
        "term": "hard",
        "objects": ["beat", "line", "verse", "fit", "intro", "edit", "poster", "photo"],
        "templates": [
            ("this {obj} goes hard", "this {obj} is impressive"),
            ("that {obj} hit hard", "that {obj} was intense"),
            ("the {obj} went hard", "the {obj} was impressive"),
            ("her {obj} hits hard", "her {obj} is impressive"),
        ],
    },
    {
        "term": "legit",
        "objects": ["setup", "feature", "song", "plan", "win", "idea", "meal", "deal"],
        "templates": [
            ("that's legit", "that is excellent"),
            ("that {obj} is legit", "that {obj} is excellent"),
            ("this {obj} was legit", "this {obj} was excellent"),
            ("the {obj} looks legit", "the {obj} looks excellent"),
        ],
    },
    {
        "term": "shady",
        "objects": ["link", "deal", "seller", "message", "excuse", "behavior", "story", "comment"],
        "templates": [
            ("that {obj} seems shady", "that {obj} seems suspicious"),
            ("the {obj} looked shady", "the {obj} looked suspicious"),
            ("his {obj} was shady", "his {obj} was suspicious"),
            ("that whole {obj} felt shady", "that whole {obj} felt suspicious"),
        ],
    },
    {
        "term": "salty",
        "objects": ["comment", "reply", "fan", "player", "friend", "review", "post", "loser"],
        "templates": [
            ("he got salty about the {obj}", "he got upset about the {obj}"),
            ("that {obj} sounded salty", "that {obj} sounded upset"),
            ("the {obj} was salty", "the {obj} was bitter"),
            ("she is still salty about the {obj}", "she is still upset about the {obj}"),
        ],
    },
    {
        "term": "cap",
        "objects": ["story", "excuse", "claim", "rumor", "answer", "post", "message", "caption"],
        "templates": [
            ("that's cap", "that is a lie"),
            ("no cap, the {obj} is true", "seriously, the {obj} is true"),
            ("stop capping about the {obj}", "stop lying about the {obj}"),
            ("his {obj} was cap", "his {obj} was a lie"),
        ],
    },
    {
        "term": "ghost",
        "objects": ["me", "my texts", "the group chat", "the client", "the recruiter", "his date", "her friend"],
        "templates": [
            ("don't ghost {obj}", "do not ignore {obj}"),
            ("she ghosted {obj}", "she stopped responding to {obj}"),
            ("he might ghost {obj}", "he might stop responding to {obj}"),
            ("they ghosted {obj}", "they stopped responding to {obj}"),
        ],
    },
    {
        "term": "beef",
        "objects": ["team", "rapper", "neighbor", "classmate", "coworker", "creator", "friend", "rival"],
        "templates": [
            ("they have beef with the {obj}", "they have a conflict with the {obj}"),
            ("his beef with the {obj} got worse", "his conflict with the {obj} got worse"),
            ("she has beef with her {obj}", "she has a conflict with her {obj}"),
            ("their beef became public", "their conflict became public"),
        ],
    },
    {
        "term": "goat",
        "objects": ["player", "singer", "teacher", "developer", "coach", "artist", "chef", "writer"],
        "templates": [
            ("she is the goat", "she is the greatest of all time"),
            ("that {obj} is the goat", "that {obj} is the greatest of all time"),
            ("the {obj} is goated", "the {obj} is excellent"),
            ("everyone called the {obj} the goat", "everyone called the {obj} the greatest of all time"),
        ],
    },
    {
        "term": "flex",
        "objects": ["watch", "car", "promotion", "setup", "vacation", "award", "fit", "score"],
        "templates": [
            ("that {obj} was a flex", "that {obj} was showing off"),
            ("he tried to flex his {obj}", "he tried to show off his {obj}"),
            ("she keeps flexing her {obj}", "she keeps showing off her {obj}"),
            ("the new {obj} is a big flex", "the new {obj} is a big way to show off"),
        ],
    },
    {
        "term": "drip",
        "objects": ["outfit", "jacket", "sneakers", "look", "fit", "chain", "style", "wardrobe"],
        "templates": [
            ("his drip is clean", "his style is clean"),
            ("she has serious drip", "she has serious style"),
            ("that {obj} adds drip", "that {obj} adds style"),
            ("the {obj} has drip", "the {obj} has style"),
        ],
    },
    {
        "term": "fit",
        "objects": ["jacket", "sneakers", "look", "mirror selfie", "streetwear", "colors", "photo", "party outfit"],
        "templates": [
            ("nice fit today", "nice outfit today"),
            ("her fit was clean", "her outfit was stylish"),
            ("post the fit check", "post the outfit check"),
            ("that fit goes hard", "that outfit is impressive"),
        ],
    },
    {
        "term": "extra",
        "objects": ["reaction", "outfit", "speech", "decorations", "comment", "plan", "entrance", "makeup"],
        "templates": [
            ("she is being extra", "she is being excessive"),
            ("that {obj} was extra", "that {obj} was excessive"),
            ("the {obj} felt extra", "the {obj} felt excessive"),
            ("you don't have to be so extra", "you do not have to be so excessive"),
        ],
    },
    {
        "term": "lowkey",
        "objects": ["like this song", "want to leave", "need help", "miss that place", "feel nervous", "love this"],
        "templates": [
            ("i lowkey {obj}", "i somewhat {obj}"),
            ("she lowkey {obj}", "she somewhat {obj}"),
            ("we lowkey {obj}", "we somewhat {obj}"),
            ("they lowkey {obj}", "they somewhat {obj}"),
        ],
    },
    {
        "term": "mid",
        "objects": ["movie", "song", "meal", "game", "episode", "album", "show", "performance"],
        "templates": [
            ("that {obj} was mid", "that {obj} was mediocre"),
            ("the {obj} is mid", "the {obj} is mediocre"),
            ("everyone called the {obj} mid", "everyone called the {obj} mediocre"),
            ("this {obj} feels mid", "this {obj} feels mediocre"),
        ],
    },
    {
        "term": "sus",
        "objects": ["message", "link", "seller", "excuse", "behavior", "story", "deal", "comment"],
        "templates": [
            ("that {obj} is sus", "that {obj} is suspicious"),
            ("the {obj} looked sus", "the {obj} looked suspicious"),
            ("his {obj} seems sus", "his {obj} seems suspicious"),
            ("this whole {obj} feels sus", "this whole {obj} feels suspicious"),
        ],
    },
    {
        "term": "bet",
        "objects": ["plan", "idea", "time", "place", "deal", "answer"],
        "templates": [
            ("bet, let's do it", "okay, let's do it"),
            ("bet, that works", "okay, that works"),
            ("bet, the {obj} works", "okay, the {obj} works"),
            ("bet, i can help", "okay, i can help"),
        ],
    },
    {
        "term": "facts",
        "objects": ["point", "comment", "answer", "take", "opinion", "statement"],
        "templates": [
            ("facts", "that is true"),
            ("that's facts", "that is true"),
            ("your {obj} is facts", "your {obj} is true"),
            ("he said facts", "he said something true"),
        ],
    },
    {
        "term": "deadass",
        "objects": ["serious", "tired", "ready", "confused", "hungry", "done"],
        "templates": [
            ("i'm deadass {obj}", "i am seriously {obj}"),
            ("she is deadass {obj}", "she is seriously {obj}"),
            ("deadass, this is true", "seriously, this is true"),
            ("are you deadass?", "are you serious?"),
        ],
    },
    {
        "term": "bussin",
        "objects": ["food", "sandwich", "pizza", "meal", "soup", "dessert"],
        "templates": [
            ("this {obj} is bussin", "this {obj} is delicious"),
            ("that {obj} was bussin", "that {obj} was delicious"),
            ("the {obj} tastes bussin", "the {obj} tastes delicious"),
            ("her {obj} is bussin", "her {obj} is delicious"),
        ],
    },
    {
        "term": "slaps",
        "objects": ["song", "beat", "album", "playlist", "track", "verse"],
        "templates": [
            ("this {obj} slaps", "this {obj} is excellent"),
            ("that {obj} slaps", "that {obj} is excellent"),
            ("the new {obj} slaps", "the new {obj} is excellent"),
            ("her {obj} really slaps", "her {obj} is really excellent"),
        ],
    },
    {
        "term": "hits different",
        "objects": ["song", "coffee", "scene", "memory", "meal", "episode"],
        "templates": [
            ("this {obj} hits different", "this {obj} feels especially meaningful"),
            ("that {obj} hits different", "that {obj} feels especially meaningful"),
            ("the {obj} hits different today", "the {obj} feels especially meaningful today"),
            ("her {obj} hits different", "her {obj} feels especially meaningful"),
        ],
    },
    {
        "term": "main character",
        "objects": ["energy", "moment", "walk", "outfit", "entrance", "vibe"],
        "templates": [
            ("she has main character {obj}", "she seems very confident and central with her {obj}"),
            ("that was a main character {obj}", "that was a very confident and central {obj}"),
            ("he had main character {obj}", "he seemed very confident and central with his {obj}"),
            ("this is main character {obj}", "this feels very confident and central"),
        ],
    },
    {
        "term": "understood the assignment",
        "objects": ["outfit", "presentation", "performance", "speech", "design", "project"],
        "templates": [
            ("she understood the assignment", "she did exactly what was needed"),
            ("he understood the assignment", "he did exactly what was needed"),
            ("that {obj} understood the assignment", "that {obj} did exactly what was needed"),
            ("they understood the assignment with that {obj}", "they did exactly what was needed with that {obj}"),
        ],
    },
])


LITERAL_CONTRASTS = [
    ("the candle was lit", "the candle was lit", "lit"),
    ("the hallway was well lit", "the hallway was well lit", "lit"),
    ("the room has low-key lighting", "the room has low-key lighting", "low-key"),
    ("the weather is chill today", "the weather is chill today", "chill"),
    ("the knight slayed the dragon", "the knight slayed the dragon", "slay"),
    ("the hero slayed the monster", "the hero slayed the monster", "slay"),
    ("the ship left the harbor", "the ship left the harbor", "ship"),
    ("i need to ship this package", "i need to ship this package", "ship"),
    ("the clutch in the car is broken", "the clutch in the car is broken", "clutch"),
    ("he spilled tea on the table", "he spilled tea on the table", "tea"),
    ("she ate and left no crumbs on the plate", "she ate and left no crumbs on the plate", "crumbs"),
    ("the food was nasty", "the food was nasty", "nasty"),
    ("the jam is sweet", "the jam is sweet", "jam"),
    ("the chicken is cooked", "the chicken is cooked", "cooked"),
    ("the room was chilly", "the room was chilly", "chill"),
    ("the low-key lighting made the room calm", "the low-key lighting made the room calm", "low-key"),
    ("i crashed my car", "i crashed my car", "crash"),
    ("the computer crashed", "the computer crashed", "crash"),
]

LITERAL_CONTRASTS.extend([
    ("the house is on fire", "the house is on fire", "fire"),
    ("the fire alarm went off", "the fire alarm went off", "fire"),
    ("i feel sick today", "i feel sick today", "sick"),
    ("the sick child stayed home", "the sick child stayed home", "sick"),
    ("the exam was hard", "the exam was hard", "hard"),
    ("the rock is hard", "the rock is hard", "hard"),
    ("this app is legit", "this app is legit", "legit"),
    ("the website is legit", "the website is legit", "legit"),
    ("we sat under a shady tree", "we sat under a shady tree", "shady"),
    ("the garden has a shady spot", "the garden has a shady spot", "shady"),
    ("the soup is salty", "the soup is salty", "salty"),
    ("the ocean water is salty", "the ocean water is salty", "salty"),
    ("the bottle has a blue cap", "the bottle has a blue cap", "cap"),
    ("put the cap on the pen", "put the cap on the pen", "cap"),
    ("the ghost story was scary", "the ghost story was scary", "ghost"),
    ("he wore a ghost costume", "he wore a ghost costume", "ghost"),
    ("we cooked beef stew", "we cooked beef stew", "beef"),
    ("she ordered beef tacos", "she ordered beef tacos", "beef"),
    ("the goat lives on the farm", "the goat lives on the farm", "goat"),
    ("the goat jumped over the fence", "the goat jumped over the fence", "goat"),
    ("flex your arm slowly", "flex your arm slowly", "flex"),
    ("the doctor asked him to flex his knee", "the doctor asked him to flex his knee", "flex"),
    ("the faucet has a drip", "the faucet has a drip", "drip"),
    ("water started to drip from the pipe", "water started to drip from the pipe", "drip"),
    ("the shoes fit well", "the shoes fit well", "fit"),
    ("the key does not fit", "the key does not fit", "fit"),
    ("we need an extra chair", "we need an extra chair", "extra"),
    ("make an extra copy", "make an extra copy", "extra"),
    ("he made a bet on the game", "he made a bet on the game", "bet"),
    ("the book has facts about history", "the book has facts about history", "facts"),
    ("the bus is coming", "the bus is coming", "bussin"),
    ("the slap was loud", "the slap was loud", "slaps"),
    ("the main character entered the room", "the main character entered the room", "main character"),
    ("the assignment is due tomorrow", "the assignment is due tomorrow", "understood the assignment"),
])


MANUAL_GOLDEN = [
    ("the concert was lit", "the concert was exciting", "lit", "slang"),
    ("she threw shade at him.", "she subtly insulted him.", "throw shade", "slang"),
    ("spill the tea.", "share the gossip.", "spill the tea", "slang"),
    ("the knight slayed the dragon.", "the knight slayed the dragon.", "slay", "literal"),
    ("can i crash at your place?", "can i sleep at your place?", "crash", "slang"),
    ("this song is my jam.", "this song is my favorite.", "jam", "slang"),
    ("i'm cooked for the exam.", "i am in trouble for the exam.", "cooked", "slang"),
    ("the room has low-key lighting.", "the room has low-key lighting.", "low-key", "literal"),
    ("he is a chill guy.", "he is a relaxed guy.", "chill", "slang"),
    ("the weather is chill today.", "the weather is chill today.", "chill", "literal"),
    ("i crashed at my friend's house.", "i slept at my friend's house.", "crash", "slang"),
    ("that beat drop was nasty.", "that beat drop was amazing.", "nasty", "slang"),
    ("she ate and left no crumbs.", "she did extremely well.", "ate and left no crumbs", "slang"),
    ("i ship those two characters.", "i support those two characters as a couple.", "ship", "slang"),
    ("that last-minute goal was clutch.", "that last-minute goal was decisive.", "clutch", "slang"),
]

MANUAL_GOLDEN.extend([
    ("this beat is fire.", "this beat is excellent.", "fire", "slang"),
    ("the house is on fire.", "the house is on fire.", "fire", "literal"),
    ("that trick was sick.", "that trick was excellent.", "sick", "slang"),
    ("i feel sick today.", "i feel sick today.", "sick", "literal"),
    ("this beat goes hard.", "this beat is impressive.", "hard", "slang"),
    ("the exam was hard.", "the exam was hard.", "hard", "literal"),
    ("that's legit.", "that is excellent.", "legit", "slang"),
    ("this app is legit.", "this app is legit.", "legit", "literal"),
    ("the link looked shady.", "the link looked suspicious.", "shady", "slang"),
    ("we sat under a shady tree.", "we sat under a shady tree.", "shady", "literal"),
    ("he got salty after losing.", "he got upset after losing.", "salty", "slang"),
    ("the soup is salty.", "the soup is salty.", "salty", "literal"),
    ("no cap, this is true.", "seriously, this is true.", "cap", "slang"),
    ("the bottle has a blue cap.", "the bottle has a blue cap.", "cap", "literal"),
    ("she ghosted me.", "she stopped responding to me.", "ghost", "slang"),
    ("the ghost story was scary.", "the ghost story was scary.", "ghost", "literal"),
    ("they have beef.", "they have a conflict.", "beef", "slang"),
    ("she ordered beef tacos.", "she ordered beef tacos.", "beef", "literal"),
    ("she is the goat.", "she is the greatest of all time.", "goat", "slang"),
    ("the goat is in the barn.", "the goat is in the barn.", "goat", "literal"),
    ("that car is a flex.", "that car is showing off.", "flex", "slang"),
    ("flex your arm slowly.", "flex your arm slowly.", "flex", "literal"),
    ("his drip is clean.", "his style is clean.", "drip", "slang"),
    ("the faucet has a drip.", "the faucet has a drip.", "drip", "literal"),
    ("nice fit today.", "nice outfit today.", "fit", "slang"),
    ("the shoes fit well.", "the shoes fit well.", "fit", "literal"),
    ("the decorations were extra.", "the decorations were excessive.", "extra", "slang"),
    ("we need an extra chair.", "we need an extra chair.", "extra", "literal"),
    ("this song slaps.", "this song is excellent.", "slaps", "slang"),
    ("this pizza is bussin.", "this pizza is delicious.", "bussin", "slang"),
    ("that's facts.", "that is true.", "facts", "slang"),
    ("are you deadass?", "are you serious?", "deadass", "slang"),
    ("this episode hits different.", "this episode feels especially meaningful.", "hits different", "slang"),
    ("she understood the assignment.", "she did exactly what was needed.", "understood the assignment", "slang"),
    ("i put jam on my bread.", "i put jam on my bread.", "jam", "literal"),
    ("this song is my jam.", "this song is my favorite.", "jam", "slang"),
    ("the fish was hooked.", "the fish was hooked.", "hooked", "literal"),
    ("he is cracked at fortnite.", "he is very good at fortnite.", "cracked", "slang"),
    ("that player is washed.", "that player is no longer good.", "washed", "slang"),
    ("the weather is chill today.", "the weather is chill today.", "chill", "literal"),
    ("he is a chill guy.", "he is a relaxed guy.", "chill", "slang"),
    ("these shoes are tight.", "these shoes are tight.", "tight", "literal"),
    ("we're tight.", "we are close friends.", "tight", "slang"),
    ("you look sharp today.", "you look stylish today.", "sharp", "slang"),
    ("that guitar solo was nasty.", "that guitar solo was amazing.", "nasty", "slang"),
    ("the bathroom smells nasty.", "the bathroom smells nasty.", "nasty", "literal"),
    ("i'm cooked for the exam.", "i am in trouble for the exam.", "cooked", "slang"),
    ("she ate that performance.", "she did extremely well in that performance.", "ate", "slang"),
    ("the bus is coming.", "the bus is coming.", "bus", "literal"),
])


DIRECT_VARIANTS = [
    ("{text}", "{target}"),
    ("{text}.", "{target}."),
]

HARD_CASE_TRAINING = [
    ("i feel sick today", "i feel sick today", "sick", "literal"),
    ("flex your arm slowly", "flex your arm slowly", "flex", "literal"),
    ("spill the tea", "share the gossip", "spill the tea", "slang"),
    ("spill the tea.", "share the gossip.", "spill the tea", "slang"),
    ("he spilled tea on the table", "he spilled tea on the table", "tea", "literal"),
    ("she spilled tea on her shirt", "she spilled tea on her shirt", "tea", "literal"),
    ("that trick was sick", "that trick was excellent", "sick", "slang"),
    ("the sick child stayed home", "the sick child stayed home", "sick", "literal"),
    ("that car is a flex", "that car is showing off", "flex", "slang"),
    ("the doctor asked him to flex his knee", "the doctor asked him to flex his knee", "flex", "literal"),
    ("the website is legit", "the website is legit", "legit", "literal"),
    ("the website is legit.", "the website is legit.", "legit", "literal"),
    ("this app is legit", "this app is legit", "legit", "literal"),
    ("the company is legit", "the company is legit", "legit", "literal"),
    ("the business is legit", "the business is legit", "legit", "literal"),
    ("the food was nasty", "the food was nasty", "nasty", "literal"),
    ("the food was nasty.", "the food was nasty.", "nasty", "literal"),
    ("she ate and left no crumbs on the plate", "she ate and left no crumbs on the plate", "crumbs", "literal"),
    ("she ate and left no crumbs on the plate.", "she ate and left no crumbs on the plate.", "crumbs", "literal"),
    ("he is a chill guy", "he is a relaxed guy", "chill", "slang"),
    ("he is a chill guy.", "he is a relaxed guy.", "chill", "slang"),
    ("she is a chill friend", "she is a relaxed friend", "chill", "slang"),
    ("my teacher is chill", "my teacher is relaxed", "chill", "slang"),
    ("that album was mid", "that album was mediocre", "mid", "slang"),
    ("that album was mid.", "that album was mediocre.", "mid", "slang"),
    ("that performance was mid", "that performance was mediocre", "mid", "slang"),
    ("that performance was mid.", "that performance was mediocre.", "mid", "slang"),
    ("facts", "that is true", "facts", "slang"),
    ("facts.", "that is true.", "facts", "slang"),
    ("that's facts", "that is true", "facts", "slang"),
    ("he said facts", "he said something true", "facts", "slang"),
    ("they have beef with the coworker", "they have a conflict with the coworker", "beef", "slang"),
    ("they have beef with the coworker.", "they have a conflict with the coworker.", "beef", "slang"),
    ("they have beef with the team", "they have a conflict with the team", "beef", "slang"),
    ("their beef became public", "their conflict became public", "beef", "slang"),
    ("i put jam on my bread", "i put jam on my bread", "jam", "literal"),
    ("i put jam on my bread.", "i put jam on my bread.", "jam", "literal"),
    ("this song is my jam", "this song is my favorite", "jam", "slang"),
    ("this song is my jam.", "this song is my favorite.", "jam", "slang"),
    ("the fish was hooked", "the fish was hooked", "hooked", "literal"),
    ("the fish was hooked.", "the fish was hooked.", "hooked", "literal"),
    ("he is cracked at fortnite", "he is very good at fortnite", "cracked", "slang"),
    ("he is cracked at fortnite.", "he is very good at fortnite.", "cracked", "slang"),
    ("that player is washed", "that player is no longer good", "washed", "slang"),
    ("that player is washed.", "that player is no longer good.", "washed", "slang"),
    ("the weather is chill today", "the weather is chill today", "chill", "literal"),
    ("the weather is chill today.", "the weather is chill today.", "chill", "literal"),
    ("he is a chill guy", "he is a relaxed guy", "chill", "slang"),
    ("he is a chill guy.", "he is a relaxed guy.", "chill", "slang"),
    ("these shoes are tight", "these shoes are tight", "tight", "literal"),
    ("these shoes are tight.", "these shoes are tight.", "tight", "literal"),
    ("we're tight", "we are close friends", "tight", "slang"),
    ("we're tight.", "we are close friends.", "tight", "slang"),
    ("you look sharp today", "you look stylish today", "sharp", "slang"),
    ("you look sharp today.", "you look stylish today.", "sharp", "slang"),
    ("that guitar solo was nasty", "that guitar solo was amazing", "nasty", "slang"),
    ("that guitar solo was nasty.", "that guitar solo was amazing.", "nasty", "slang"),
    ("the bathroom smells nasty", "the bathroom smells nasty", "nasty", "literal"),
    ("the bathroom smells nasty.", "the bathroom smells nasty.", "nasty", "literal"),
    ("i'm cooked for the exam", "i am in trouble for the exam", "cooked", "slang"),
    ("i'm cooked for the exam.", "i am in trouble for the exam.", "cooked", "slang"),
    ("she ate that performance", "she did extremely well in that performance", "ate", "slang"),
    ("she ate that performance.", "she did extremely well in that performance.", "ate", "slang"),
    ("the bus is coming", "the bus is coming", "bus", "literal"),
    ("the bus is coming.", "the bus is coming.", "bus", "literal"),
]


def has_bad_wrapper(text: str) -> bool:
    lower = clean(text).lower()
    return lower.startswith(BAD_WRAPPER_PREFIXES) or any(fragment in lower for fragment in BAD_WRAPPER_FRAGMENTS)


def validate_row(item: dict[str, Any]) -> str | None:
    source = item.get("input", "")
    target = item.get("target", "")
    if not source or not target:
        return "missing input or target"
    if has_bad_wrapper(source) or has_bad_wrapper(target):
        return "wrapper artifact"
    if item.get("sense") == "literal" and clean(source).lower() != clean(target).lower():
        return "literal row must be identity"
    return None


def build_template_rows(cycles: int, hard_case_repeat_count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pattern in SLANG_PATTERNS:
        term = pattern["term"]
        objects = pattern["objects"]
        templates = pattern["templates"]
        produced = 0
        for obj in objects:
            for source_tpl, target_tpl in templates:
                base_source = source_tpl.format(obj=obj)
                base_target = target_tpl.format(obj=obj)
                for wrapper_source, wrapper_target in DIRECT_VARIANTS:
                    source = wrapper_source.format(text=base_source)
                    target = wrapper_target.format(target=base_target)
                    rows.append(row(source, target, term, "slang"))
                    produced += 1
                    if produced >= cycles:
                        break
                if produced >= cycles:
                    break
            if produced >= cycles:
                break

    for source, target, term in LITERAL_CONTRASTS:
        for wrapper_source, wrapper_target in DIRECT_VARIANTS:
            rows.append(
                row(
                    wrapper_source.format(text=source),
                    wrapper_target.format(target=target),
                    term,
                    "literal",
                )
            )

    for source, target, term, sense in MANUAL_GOLDEN:
        rows.append(row(source, target, term, sense, "manual_golden", split="test"))

    for _ in range(hard_case_repeat_count):
        for source, target, term, sense in HARD_CASE_TRAINING:
            rows.append(row(source, target, term, sense, "hard_case_repeat", split="train"))

    return rows


def dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique = []
    for item in rows:
        if item.get("source") in {"hard_case_repeat", "targeted_hard_training"}:
            unique.append(item)
            continue
        key = f"{item['input'].lower()}\t{item['target'].lower()}\t{item.get('sense', '')}"
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def validate_rows(rows: list[dict[str, Any]]) -> list[str]:
    errors = []
    targets_by_input: dict[str, str] = {}
    for idx, item in enumerate(rows):
        reason = validate_row(item)
        if reason:
            errors.append(f"row {idx}: {reason}: {item}")
            continue
        key = item["input"].lower()
        target = item["target"].lower()
        existing_target = targets_by_input.get(key)
        if existing_target is not None and existing_target != target:
            errors.append(f"row {idx}: conflicting target for input '{item['input']}'")
        targets_by_input[key] = target
    return errors


def add_rows_skipping_conflicts(
    base_rows: list[dict[str, Any]],
    extra_rows: list[dict[str, Any]],
    label: str,
) -> tuple[list[dict[str, Any]], int]:
    targets_by_input = {
        item["input"].lower(): item["target"].lower()
        for item in base_rows
    }
    combined = base_rows[:]
    skipped = 0

    for item in extra_rows:
        key = item["input"].lower()
        target = item["target"].lower()
        existing_target = targets_by_input.get(key)
        if existing_target is not None and existing_target != target:
            skipped += 1
            continue
        targets_by_input[key] = target
        item["source"] = item.get("source", label)
        combined.append(item)

    return combined, skipped


def read_targeted_approved_rows() -> list[dict[str, Any]]:
    if not TARGETED_APPROVED_PATH.exists():
        return []
    with TARGETED_APPROVED_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{TARGETED_APPROVED_PATH} must contain a JSON list.")

    approved_rows = []
    for item in data:
        source = item.get("input", "")
        target = item.get("target", "")
        term = item.get("term", "")
        sense = item.get("sense", "")
        approved_rows.append(row(source, target, term, sense, "targeted_approved", split="test"))
    return approved_rows


def read_targeted_hard_training_rows(repeat_cap: int | None = None) -> list[dict[str, Any]]:
    if not TARGETED_HARD_TRAINING_PATH.exists():
        return []
    with TARGETED_HARD_TRAINING_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{TARGETED_HARD_TRAINING_PATH} must contain a JSON list.")

    training_rows = []
    for item in data:
        source = item.get("input", "")
        target = item.get("target", "")
        term = item.get("term", "")
        sense = item.get("sense", "")
        repeat = max(1, int(item.get("repeat", 1)))
        if repeat_cap is not None:
            repeat = min(repeat, repeat_cap)
        for _ in range(repeat):
            training_rows.append(row(source, target, term, sense, "targeted_hard_training", split="train"))
    return training_rows


def split_rows(rows: list[dict[str, Any]], test_size: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(SEED)
    train_rows = [item for item in rows if item.get("split") == "train"]
    test_rows = [item for item in rows if item.get("split") == "test"]

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in rows:
        if item.get("split"):
            continue
        grouped.setdefault((item["term"], item["sense"]), []).append(item)

    for group_rows in grouped.values():
        shuffled = group_rows[:]
        rng.shuffle(shuffled)
        count = max(1, int(round(len(shuffled) * test_size)))
        test_rows.extend(shuffled[:count])
        train_rows.extend(shuffled[count:])

    rng.shuffle(train_rows)
    rng.shuffle(test_rows)
    for item in train_rows + test_rows:
        item.pop("split", None)
    return train_rows, test_rows


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build normalizer v3 dataset.")
    parser.add_argument("--cycles", type=int, default=300, help="Template cycles per slang pattern.")
    parser.add_argument("--test-size", type=float, default=0.12)
    parser.add_argument("--train-path", default=str(TRAIN_PATH))
    parser.add_argument("--test-path", default=str(TEST_PATH))
    parser.add_argument(
        "--hard-case-repeat-count",
        type=int,
        default=HARD_CASE_REPEAT_COUNT,
        help="Repeat count for the small manual hard-case set.",
    )
    parser.add_argument(
        "--targeted-repeat-cap",
        type=int,
        default=None,
        help="Optional maximum repeat for generated targeted hard-training rows.",
    )
    args = parser.parse_args()

    rows = dedupe(build_template_rows(args.cycles, args.hard_case_repeat_count))
    rows, skipped_targeted_conflicts = add_rows_skipping_conflicts(
        rows,
        read_targeted_approved_rows(),
        "targeted_approved",
    )
    rows, skipped_hard_training_conflicts = add_rows_skipping_conflicts(
        rows,
        read_targeted_hard_training_rows(args.targeted_repeat_cap),
        "targeted_hard_training",
    )
    errors = validate_rows(rows)
    if errors:
        preview = "\n".join(errors[:10])
        raise SystemExit(f"Dataset validation failed with {len(errors)} errors:\n{preview}")

    train_rows, test_rows = split_rows(rows, args.test_size)
    train_path = Path(args.train_path)
    test_path = Path(args.test_path)
    write_json(train_path, train_rows)
    write_json(test_path, test_rows)

    print(f"Wrote {len(train_rows)} train rows to {train_path}")
    print(f"Wrote {len(test_rows)} test rows to {test_path}")
    for sense in ["slang", "literal"]:
        print(f"  {sense}: train={sum(1 for item in train_rows if item['sense'] == sense)} test={sum(1 for item in test_rows if item['sense'] == sense)}")
    print(f"  identity rows: train={sum(1 for item in train_rows if item['input'].lower() == item['target'].lower())} test={sum(1 for item in test_rows if item['input'].lower() == item['target'].lower())}")
    print(f"  targeted hard training rows: train={sum(1 for item in train_rows if item['source'] == 'targeted_hard_training')}")
    print(f"  skipped targeted approved conflicts: {skipped_targeted_conflicts}")
    print(f"  skipped targeted hard training conflicts: {skipped_hard_training_conflicts}")


if __name__ == "__main__":
    main()
