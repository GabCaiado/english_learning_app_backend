"""
Build contrastive sense-classification data for ambiguous slang terms.

The output rows follow this schema:
  {
    "term": "tea",
    "sentence": "what's the tea?",
    "slang_meaning": "gossip",
    "label": 1
  }

Run:
  python scripts/build_slang_sense_dataset.py
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


DATA_DIR = Path("data")
TRAIN_PATH = DATA_DIR / "slang_sense_train.json"
TEST_PATH = DATA_DIR / "slang_sense_test.json"
SEED = 42


TERM_CONFIGS: dict[str, dict[str, Any]] = {
    "tea": {
        "meaning": "gossip",
        "replacement": "gossip",
        "slang_objects": ["group chat", "office", "timeline", "friend", "comment section", "story", "thread", "meeting"],
        "literal_objects": ["cup", "kettle", "mug", "teapot", "cafe", "menu", "shelf", "thermos"],
        "slang_phrases": [
            "what's the tea?",
            "spill the tea",
            "she shared the tea after class",
            "the tea in that thread was wild",
            "give me the tea about the meeting",
        ],
        "literal_phrases": [
            "i made tea after dinner",
            "i wanna drink some tea",
            "green tea is healthy",
            "the tea was too hot",
            "she bought tea at the cafe",
        ],
    },
    "sick": {
        "meaning": "excellent or impressive",
        "replacement": "excellent",
        "slang_objects": ["trick", "move", "solo", "design", "goal", "shot", "transition", "routine"],
        "literal_objects": ["child", "patient", "teacher", "friend", "traveler", "student", "manager", "neighbor"],
        "slang_phrases": [
            "that trick was sick",
            "this edit is sick",
            "her goal was sick",
            "the guitar solo sounded sick",
            "those moves are sick",
        ],
        "literal_phrases": [
            "i feel sick today",
            "the sick child stayed home",
            "he called in sick",
            "the patient is sick",
            "she felt sick after lunch",
        ],
    },
    "fire": {
        "meaning": "excellent or impressive",
        "replacement": "excellent",
        "slang_objects": ["beat", "track", "outfit", "verse", "design", "performance", "playlist", "edit"],
        "literal_objects": ["house", "pan", "building", "forest", "car", "kitchen", "warehouse", "candle"],
        "slang_phrases": [
            "this beat is fire",
            "her outfit was fire",
            "that verse is fire",
            "the playlist is fire",
            "this design came out fire",
        ],
        "literal_phrases": [
            "the house is on fire",
            "the fire alarm went off",
            "smoke came from the fire",
            "the kitchen caught fire",
            "they called the fire department",
        ],
    },
    "lit": {
        "meaning": "exciting or excellent",
        "replacement": "exciting",
        "slang_objects": ["party", "show", "timeline", "festival", "night", "chat", "concert", "room"],
        "literal_objects": ["lamp", "hallway", "sign", "candle", "stage", "screen", "porch", "street"],
        "slang_phrases": [
            "the party was lit",
            "that show was lit",
            "the concert got lit",
            "last night was lit",
            "the chat was lit after the announcement",
        ],
        "literal_phrases": [
            "the hallway was well lit",
            "she lit the candle",
            "the lamp lit the room",
            "the porch is poorly lit",
            "the screen lit up",
        ],
    },
    "hard": {
        "meaning": "impressive or intense",
        "replacement": "impressive",
        "slang_objects": ["beat", "line", "fit", "poster", "intro", "verse", "edit", "photo"],
        "literal_objects": ["exam", "rock", "chair", "problem", "surface", "decision", "question", "wood"],
        "slang_phrases": [
            "this beat goes hard",
            "that line hit hard",
            "the intro goes hard",
            "his fit went hard",
            "the edit slaps hard",
        ],
        "literal_phrases": [
            "the exam was hard",
            "the rock is hard",
            "that decision was hard",
            "the surface feels hard",
            "this is hard work",
        ],
    },
    "legit": {
        "meaning": "excellent, real, or credible depending on context",
        "replacement": "excellent",
        "slang_objects": ["idea", "setup", "plan", "feature", "win", "song", "deal", "meal"],
        "literal_objects": ["app", "website", "company", "license", "source", "document", "seller", "business"],
        "slang_phrases": [
            "that's legit",
            "this setup is legit",
            "their plan was legit",
            "that feature looks legit",
            "the win felt legit",
        ],
        "literal_phrases": [
            "this app is legit",
            "the website is legit",
            "the company is legit",
            "the license looks legit",
            "we verified the source is legit",
        ],
    },
    "shady": {
        "meaning": "suspicious or dishonest",
        "replacement": "suspicious",
        "slang_objects": ["message", "deal", "seller", "link", "excuse", "behavior", "story", "comment"],
        "literal_objects": ["tree", "spot", "area", "garden", "porch", "trail", "bench", "side"],
        "slang_phrases": [
            "that deal seems shady",
            "he was acting shady",
            "the link looked shady",
            "that excuse was shady",
            "her behavior felt shady",
        ],
        "literal_phrases": [
            "we sat under a shady tree",
            "the garden has a shady area",
            "the bench is in a shady spot",
            "the trail was shady and cool",
            "plants grow on the shady side",
        ],
    },
    "salty": {
        "meaning": "bitter or upset",
        "replacement": "upset",
        "slang_objects": ["comment", "reply", "loser", "friend", "fan", "player", "review", "post"],
        "literal_objects": ["soup", "chips", "water", "snack", "broth", "sauce", "pretzel", "meal"],
        "slang_phrases": [
            "he got salty after losing",
            "that reply sounded salty",
            "the fans were salty online",
            "she is still salty about it",
            "his comment was salty",
        ],
        "literal_phrases": [
            "the soup is salty",
            "the chips tasted salty",
            "the ocean water is salty",
            "this sauce is too salty",
            "the pretzel was salty",
        ],
    },
    "cap": {
        "meaning": "lie or exaggeration",
        "replacement": "lie",
        "slang_objects": ["story", "excuse", "claim", "rumor", "answer", "caption", "post", "message"],
        "literal_objects": ["bottle", "pen", "marker", "gas tank", "camera lens", "jar", "tube", "hat"],
        "slang_phrases": [
            "that's cap",
            "no cap, this is true",
            "stop capping about the price",
            "his story was cap",
            "that rumor is cap",
        ],
        "literal_phrases": [
            "the bottle has a blue cap",
            "put the cap on the pen",
            "the gas cap was loose",
            "she wore a red cap",
            "the jar cap is missing",
        ],
    },
    "ghost": {
        "meaning": "suddenly ignore someone",
        "replacement": "ignore",
        "slang_objects": ["date", "friend", "recruiter", "client", "classmate", "group", "partner", "seller"],
        "literal_objects": ["story", "movie", "costume", "legend", "museum", "haunted house", "painting", "tour"],
        "slang_phrases": [
            "don't ghost me",
            "she ghosted him after dinner",
            "the recruiter ghosted my emails",
            "he might ghost the group chat",
            "my date ghosted me",
        ],
        "literal_phrases": [
            "the ghost story was scary",
            "he wore a ghost costume",
            "the movie had a ghost",
            "the haunted house has a ghost legend",
            "the painting showed a ghost",
        ],
    },
    "beef": {
        "meaning": "conflict or argument",
        "replacement": "conflict",
        "slang_objects": ["team", "rapper", "neighbor", "classmate", "coworker", "creator", "friend", "rival"],
        "literal_objects": ["stew", "burger", "tacos", "sandwich", "chili", "skewers", "meatballs", "soup"],
        "slang_phrases": [
            "they have beef now",
            "his beef with the team got worse",
            "the rappers started beef online",
            "she has beef with her coworker",
            "their beef became public",
        ],
        "literal_phrases": [
            "we cooked beef stew",
            "the beef burger was fresh",
            "she ordered beef tacos",
            "the freezer has beef",
            "this recipe uses beef",
        ],
    },
    "goat": {
        "meaning": "greatest of all time",
        "replacement": "greatest of all time",
        "slang_objects": ["player", "singer", "teacher", "developer", "coach", "artist", "chef", "writer"],
        "literal_objects": ["farm", "barn", "field", "veterinarian", "mountain", "petting zoo", "fence", "pasture"],
        "slang_phrases": [
            "she is the goat",
            "that player is the goat",
            "he's goated at this",
            "the coach is the goat",
            "everyone called the artist the goat",
        ],
        "literal_phrases": [
            "the goat lives on the farm",
            "the barn has a goat",
            "the goat jumped over the fence",
            "the veterinarian checked the goat",
            "the goat ate grass in the pasture",
        ],
    },
    "flex": {
        "meaning": "show off",
        "replacement": "show off",
        "slang_objects": ["watch", "car", "promotion", "setup", "vacation", "award", "fit", "score"],
        "literal_objects": ["arm", "knee", "muscle", "ankle", "shoulder", "wrist", "toe", "leg"],
        "slang_phrases": [
            "weird flex but okay",
            "he tried to flex his new watch",
            "that car was a big flex",
            "she keeps flexing her promotion",
            "the setup is a flex",
        ],
        "literal_phrases": [
            "flex your arm slowly",
            "he flexed his muscle",
            "the doctor asked her to flex her knee",
            "do not flex the ankle yet",
            "flex the wrist during the stretch",
        ],
    },
    "drip": {
        "meaning": "stylish clothing or appearance",
        "replacement": "style",
        "slang_objects": ["outfit", "jacket", "sneakers", "look", "fit", "wardrobe", "chain", "style"],
        "literal_objects": ["faucet", "pipe", "ceiling", "coffee maker", "paint can", "iv bag", "tap", "roof"],
        "slang_phrases": [
            "his drip is clean",
            "she showed up with serious drip",
            "that jacket adds drip",
            "the sneakers gave the fit drip",
            "everyone noticed his drip",
        ],
        "literal_phrases": [
            "the faucet has a drip",
            "water started to drip from the pipe",
            "the ceiling drip got worse",
            "coffee began to drip into the pot",
            "paint will drip if it is too thin",
        ],
    },
    "fit": {
        "meaning": "outfit",
        "replacement": "outfit",
        "slang_objects": ["jacket", "sneakers", "look", "mirror selfie", "party outfit", "streetwear", "colors", "photo"],
        "literal_objects": ["shoe", "box", "key", "schedule", "ring", "helmet", "seat", "door"],
        "slang_phrases": [
            "nice fit today",
            "her fit was clean",
            "post the fit check",
            "that fit goes hard",
            "the whole fit matched",
        ],
        "literal_phrases": [
            "the shoes fit well",
            "the key does not fit",
            "the box will fit in the car",
            "this helmet should fit",
            "the appointment can fit my schedule",
        ],
    },
    "extra": {
        "meaning": "over the top or excessive",
        "replacement": "excessive",
        "slang_objects": ["reaction", "outfit", "speech", "decor", "comment", "plan", "entrance", "makeup"],
        "literal_objects": ["ticket", "chair", "time", "money", "copy", "battery", "blanket", "credit"],
        "slang_phrases": [
            "she is being extra",
            "that reaction was extra",
            "his entrance felt extra",
            "the decorations were extra",
            "you don't have to be so extra",
        ],
        "literal_phrases": [
            "we need an extra chair",
            "bring an extra ticket",
            "i have extra time today",
            "the bag has an extra battery",
            "make an extra copy",
        ],
    },
    "chill": {
        "meaning": "relaxed or easygoing",
        "replacement": "relaxed",
        "slang_objects": ["manager", "teacher", "friend", "neighbor", "person", "coach", "roommate", "host"],
        "literal_objects": ["wind", "air", "weather", "room", "morning", "hallway", "water", "breeze"],
        "slang_phrases": [
            "she's super chill",
            "he is a chill guy",
            "my teacher is chill",
            "that neighbor seems chill",
            "they are really chill",
        ],
        "literal_phrases": [
            "the weather is chill today",
            "a chill wind came through",
            "the room was chill in the morning",
            "the air felt chill",
            "the breeze was chill near the water",
        ],
    },
    "cooked": {
        "meaning": "in serious trouble or likely to fail",
        "replacement": "in trouble",
        "slang_objects": ["exam", "deadline", "interview", "meeting", "final", "project", "presentation", "competition"],
        "literal_objects": ["pasta", "rice", "vegetables", "dinner", "lunch", "breakfast", "meal", "meat"],
        "slang_phrases": [
            "we're cooked if we miss the deadline",
            "i'm cooked for the final",
            "they are cooked for the interview",
            "you are cooked without a backup plan",
            "we are cooked before the meeting starts",
        ],
        "literal_phrases": [
            "the pasta was cooked well",
            "the vegetables are cooked",
            "breakfast was cooked early",
            "the rice is cooked",
            "dinner was cooked by noon",
        ],
    },
    "serving": {
        "meaning": "projecting or giving off a strong vibe",
        "replacement": "projecting",
        "slang_objects": ["look", "outfit", "makeup", "style", "pose", "photo", "entrance", "jacket"],
        "literal_objects": ["waiter", "server", "cafe", "restaurant", "host", "cafeteria", "kitchen", "bar"],
        "slang_phrases": [
            "this look is serving confidence",
            "her outfit is serving main character energy",
            "that makeup is serving attitude",
            "the jacket is serving looks",
            "his pose is serving confidence",
        ],
        "literal_phrases": [
            "the waiter is serving dinner",
            "the cafe is serving breakfast",
            "she is serving soup to guests",
            "the restaurant is serving lunch",
            "the host is serving drinks",
        ],
    },
    "snatched": {
        "meaning": "very stylish, flattering, or well put together",
        "replacement": "stylish and flattering",
        "slang_objects": ["outfit", "fit", "look", "waist", "makeup", "dress", "style", "jacket"],
        "literal_objects": ["bag", "phone", "purse", "wallet", "keys", "package", "ticket", "camera"],
        "slang_phrases": [
            "her outfit looks snatched",
            "that fit looked snatched",
            "her waist looks snatched",
            "the whole look is snatched",
            "that dress was snatched",
        ],
        "literal_phrases": [
            "the thief snatched her bag",
            "someone snatched my purse",
            "he snatched the phone from the table",
            "the package was snatched from the porch",
            "she snatched the keys quickly",
        ],
    },
    "slayed": {
        "meaning": "did very well",
        "replacement": "did very well",
        "slang_objects": ["presentation", "performance", "routine", "speech", "outfit", "scene", "audition", "interview"],
        "literal_objects": ["dragon", "monster", "beast", "creature", "enemy", "villain", "serpent", "giant"],
        "slang_phrases": [
            "she slayed that presentation",
            "he slayed the performance",
            "they slayed the routine",
            "her outfit slayed",
            "you slayed that interview",
        ],
        "literal_phrases": [
            "the knight slayed the dragon",
            "the hero slayed a monster",
            "the warrior slayed the beast",
            "the hunter slayed the creature",
            "the legend says she slayed the serpent",
        ],
    },
}


SLANG_TEMPLATES = [
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

LITERAL_TEMPLATES = [
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

SLANG_SCENES = [
    "People kept replaying it.",
    "It got shared all morning.",
    "My friends brought it up later.",
    "The comments agreed immediately.",
    "It stood out from everything else.",
    "Everyone reacted at once.",
    "It became the highlight of the day.",
]

LITERAL_SCENES = [
    "The sentence is meant literally.",
    "Someone wrote it in the report.",
    "The detail mattered in context.",
    "We checked it again later.",
    "No slang meaning was intended.",
    "The sentence describes a real object.",
    "The teacher used it as a literal example.",
]

NEAR_MISS_PREFIXES = [
    "Even though people online use slang,",
    "In this sentence,",
    "For the dictionary example,",
    "During the lesson,",
    "Without any figurative meaning,",
]


def clean(text: str) -> str:
    return " ".join(text.strip().split())


def make_row(term: str, sentence: str, label: int, example_type: str, source: str) -> dict[str, Any]:
    config = TERM_CONFIGS[term]
    return {
        "term": term,
        "sentence": clean(sentence),
        "slang_meaning": config["meaning"],
        "label": label,
        "example_type": example_type,
        "source": source,
    }


def slang_sentence(term: str, config: dict[str, Any], i: int) -> str:
    obj = config["slang_objects"][i % len(config["slang_objects"])]
    scene = SLANG_SCENES[i % len(SLANG_SCENES)]
    patterns = {
        "tea": [
            "What's the tea about the {obj}?",
            "Spill the tea from the {obj}.",
            "She shared the tea about her {obj}.",
            "The tea in the {obj} was wild.",
            "Give me the tea on that {obj}.",
        ],
        "sick": [
            "That {obj} was sick.",
            "This {obj} is sick.",
            "The new {obj} looks sick.",
            "Everyone called the {obj} sick.",
            "That {obj} came out sick.",
        ],
        "fire": [
            "This {obj} is fire.",
            "That {obj} was fire.",
            "The new {obj} sounds fire.",
            "Everyone called the {obj} fire.",
            "Her latest {obj} is straight fire.",
        ],
        "lit": [
            "The {obj} was lit.",
            "That {obj} got lit.",
            "The whole {obj} felt lit.",
            "Everyone said the {obj} was lit.",
            "Last night's {obj} was lit.",
        ],
        "hard": [
            "This {obj} goes hard.",
            "That {obj} hit hard.",
            "The new {obj} goes hard.",
            "Everyone said the {obj} went hard.",
            "Her latest {obj} hits hard.",
        ],
        "legit": [
            "That's legit.",
            "This {obj} is legit.",
            "That {obj} was legit.",
            "The new {obj} looks legit.",
            "Everyone agreed the {obj} was legit.",
        ],
        "shady": [
            "That {obj} seems shady.",
            "The {obj} looked shady.",
            "His {obj} was shady.",
            "Everyone thought the {obj} felt shady.",
            "That whole {obj} was shady.",
        ],
        "salty": [
            "That {obj} sounded salty.",
            "The {obj} got salty fast.",
            "His {obj} was salty.",
            "Everyone said the {obj} seemed salty.",
            "She was still salty about the {obj}.",
        ],
        "cap": [
            "That's cap.",
            "No cap, the {obj} was true.",
            "Stop capping about the {obj}.",
            "His {obj} was cap.",
            "Everyone knew the {obj} was cap.",
        ],
        "ghost": [
            "The {obj} ghosted me.",
            "Don't ghost your {obj}.",
            "She got ghosted by the {obj}.",
            "He might ghost the {obj}.",
            "The {obj} stopped replying and ghosted.",
        ],
        "beef": [
            "They have beef with the {obj}.",
            "His beef with the {obj} got worse.",
            "The {obj} started beef online.",
            "She has beef with her {obj}.",
            "Their beef became public after the {obj}.",
        ],
        "goat": [
            "That {obj} is the goat.",
            "Everyone calls the {obj} the goat.",
            "She is goated as a {obj}.",
            "The {obj} is goated at this.",
            "People said the {obj} is the goat.",
        ],
        "flex": [
            "That {obj} was a big flex.",
            "He tried to flex his {obj}.",
            "She keeps flexing her {obj}.",
            "The new {obj} is a flex.",
            "Posting the {obj} felt like a flex.",
        ],
        "drip": [
            "His {obj} has drip.",
            "She showed up with serious drip.",
            "That {obj} adds drip.",
            "Everyone noticed the drip in his {obj}.",
            "The {obj} gave the whole look drip.",
        ],
        "fit": [
            "Nice fit today.",
            "Her {obj} made the fit clean.",
            "Post the fit check.",
            "That fit goes hard.",
            "The whole fit matched the {obj}.",
        ],
        "extra": [
            "That {obj} was extra.",
            "She was being extra about the {obj}.",
            "His {obj} felt extra.",
            "The whole {obj} was too extra.",
            "You don't have to be so extra about the {obj}.",
        ],
        "chill": [
            "The {obj} is chill.",
            "That {obj} seems chill.",
            "Everyone said the {obj} was chill.",
            "The new {obj} is super chill.",
            "My {obj} is really chill.",
        ],
        "cooked": [
            "We are cooked for the {obj}.",
            "I am cooked before the {obj}.",
            "They are cooked without help on the {obj}.",
            "We are cooked if we miss the {obj}.",
            "You are cooked for that {obj}.",
        ],
        "serving": [
            "This {obj} is serving confidence.",
            "That {obj} is serving looks.",
            "Her {obj} is serving attitude.",
            "The {obj} is serving main character energy.",
            "His {obj} is serving confidence.",
        ],
        "snatched": [
            "Her {obj} looks snatched.",
            "That {obj} looked snatched.",
            "The whole {obj} is snatched.",
            "Everyone said the {obj} was snatched.",
            "This {obj} looks snatched.",
        ],
        "slayed": [
            "She slayed that {obj}.",
            "He slayed the {obj}.",
            "They slayed the {obj}.",
            "Everyone said her {obj} slayed.",
            "You slayed that {obj}.",
        ],
    }
    sentence = patterns[term][i % len(patterns[term])].format(obj=obj)
    return f"{sentence} {scene}"


def literal_sentence(term: str, config: dict[str, Any], i: int) -> str:
    obj = config["literal_objects"][i % len(config["literal_objects"])]
    scene = LITERAL_SCENES[i % len(LITERAL_SCENES)]
    patterns = {
        "tea": [
            "I drank tea from the {obj}.",
            "She made tea for breakfast.",
            "Green tea was placed near the {obj}.",
            "The tea in the {obj} was hot.",
            "They brewed tea in the {obj}.",
        ],
        "sick": [
            "The {obj} felt sick today.",
            "The sick {obj} stayed home.",
            "A doctor checked the sick {obj}.",
            "The {obj} called in sick.",
            "The {obj} became sick after lunch.",
        ],
        "fire": [
            "The {obj} is on fire.",
            "A fire started near the {obj}.",
            "The {obj} caught fire.",
            "Smoke came from the fire by the {obj}.",
            "The fire department checked the {obj}.",
        ],
        "lit": [
            "The {obj} was well lit.",
            "She lit the {obj}.",
            "The {obj} was poorly lit.",
            "A small light lit the {obj}.",
            "The {obj} lit up after dark.",
        ],
        "hard": [
            "The {obj} was hard.",
            "This {obj} feels hard.",
            "The hard {obj} was difficult to move.",
            "Someone said the {obj} was hard.",
            "The {obj} became hard overnight.",
        ],
        "legit": [
            "The {obj} is legit.",
            "We verified the {obj} is legit.",
            "The {obj} looks legit after inspection.",
            "A reviewer confirmed the {obj} was legit.",
            "The {obj} is a legit source.",
        ],
        "shady": [
            "We sat by a shady {obj}.",
            "The {obj} was shady and cool.",
            "Plants grew in the shady {obj}.",
            "The map marked a shady {obj}.",
            "They rested on the shady {obj}.",
        ],
        "salty": [
            "The {obj} tasted salty.",
            "This {obj} is too salty.",
            "The salty {obj} needed more water.",
            "Someone said the {obj} was salty.",
            "The {obj} became salty after seasoning.",
        ],
        "cap": [
            "The {obj} cap was loose.",
            "She put the cap on the {obj}.",
            "The cap for the {obj} was missing.",
            "He replaced the {obj} cap.",
            "The {obj} came with a blue cap.",
        ],
        "ghost": [
            "The ghost {obj} was scary.",
            "He wore a ghost {obj}.",
            "The {obj} had a ghost in it.",
            "They told a ghost {obj}.",
            "The museum displayed a ghost {obj}.",
        ],
        "beef": [
            "We cooked {obj} with beef.",
            "She ordered a beef {obj}.",
            "The {obj} included beef.",
            "This beef {obj} was fresh.",
            "The recipe uses beef for the {obj}.",
        ],
        "goat": [
            "The goat stayed near the {obj}.",
            "A goat walked through the {obj}.",
            "The {obj} had a goat inside.",
            "The veterinarian checked the goat at the {obj}.",
            "The goat ate grass by the {obj}.",
        ],
        "flex": [
            "Flex your {obj} slowly.",
            "The doctor asked him to flex his {obj}.",
            "Do not flex the {obj} yet.",
            "She flexed her {obj} during the stretch.",
            "The exercise helps you flex the {obj}.",
        ],
        "drip": [
            "The {obj} has a drip.",
            "Water started to drip from the {obj}.",
            "The drip near the {obj} got worse.",
            "Coffee began to drip from the {obj}.",
            "Paint will drip from the {obj}.",
        ],
        "fit": [
            "The {obj} should fit.",
            "The {obj} does not fit.",
            "This {obj} will fit in the car.",
            "The new {obj} fit perfectly.",
            "They checked whether the {obj} would fit.",
        ],
        "extra": [
            "We need an extra {obj}.",
            "Bring an extra {obj}.",
            "I have extra {obj} today.",
            "The bag has an extra {obj}.",
            "Make an extra {obj}.",
        ],
        "chill": [
            "The {obj} felt chill.",
            "A chill {obj} came through.",
            "The {obj} was chill in the morning.",
            "Someone said the {obj} was chill.",
            "The {obj} became chill overnight.",
        ],
        "cooked": [
            "The {obj} was cooked well.",
            "The {obj} is cooked.",
            "Someone cooked the {obj}.",
            "The cooked {obj} cooled on the counter.",
            "They checked whether the {obj} was cooked.",
        ],
        "serving": [
            "The {obj} is serving dinner.",
            "The {obj} started serving lunch.",
            "The {obj} is serving customers.",
            "Someone said the {obj} was serving soup.",
            "The {obj} kept serving guests.",
        ],
        "snatched": [
            "The thief snatched the {obj}.",
            "Someone snatched my {obj}.",
            "He snatched the {obj} from the table.",
            "The {obj} was snatched quickly.",
            "She snatched the {obj} before leaving.",
        ],
        "slayed": [
            "The knight slayed the {obj}.",
            "The hero slayed a {obj}.",
            "The warrior slayed the {obj}.",
            "The legend says she slayed the {obj}.",
            "The hunter slayed a dangerous {obj}.",
        ],
    }
    sentence = patterns[term][i % len(patterns[term])].format(obj=obj)
    return f"{sentence} {scene}"


def build_rows(per_label_per_term: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for term, config in TERM_CONFIGS.items():
        for phrase in config["slang_phrases"]:
            rows.append(make_row(term, phrase, 1, "slang", "seed_phrase"))
        for phrase in config["literal_phrases"]:
            rows.append(make_row(term, phrase, 0, "literal", "seed_phrase"))

        for i in range(per_label_per_term):
            rows.append(make_row(term, slang_sentence(term, config, i), 1, "slang", "template"))

        for i in range(per_label_per_term):
            obj = config["literal_objects"][i % len(config["literal_objects"])]
            scene = LITERAL_SCENES[i % len(LITERAL_SCENES)]
            rows.append(make_row(term, literal_sentence(term, config, i), 0, "literal", "template"))

            if i % 4 == 0:
                prefix = NEAR_MISS_PREFIXES[i % len(NEAR_MISS_PREFIXES)]
                rows.append(
                    make_row(
                        term,
                        f"{prefix} the word {term} refers to the {obj}. {scene}",
                        0,
                        "near_miss",
                        "template",
                    )
                )

    return rows


def fixed_test_rows() -> list[dict[str, Any]]:
    cases = [
        ("tea", "i wanna drink some tea", 0, "literal"),
        ("tea", "what's the tea", 1, "slang"),
        ("tea", "spill the tea", 1, "slang"),
        ("tea", "green tea is healthy", 0, "literal"),
        ("fire", "the house is on fire", 0, "literal"),
        ("fire", "this beat is fire", 1, "slang"),
        ("sick", "i feel sick today", 0, "literal"),
        ("sick", "that trick was sick", 1, "slang"),
        ("legit", "this app is legit", 0, "literal"),
        ("legit", "that's legit", 1, "slang"),
        ("hard", "the exam was hard", 0, "literal"),
        ("hard", "this beat goes hard", 1, "slang"),
        ("cap", "the bottle cap is blue", 0, "literal"),
        ("cap", "no cap, this is true", 1, "slang"),
        ("ghost", "the ghost story was scary", 0, "literal"),
        ("ghost", "she ghosted me", 1, "slang"),
        ("beef", "the beef stew is ready", 0, "literal"),
        ("beef", "they have beef", 1, "slang"),
        ("goat", "the goat is in the barn", 0, "literal"),
        ("goat", "she is the goat", 1, "slang"),
        ("fit", "the shoes fit well", 0, "literal"),
        ("fit", "nice fit today", 1, "slang"),
        ("drip", "the faucet has a drip", 0, "literal"),
        ("drip", "his drip is clean", 1, "slang"),
        ("extra", "we need an extra chair", 0, "literal"),
        ("extra", "she is being extra", 1, "slang"),
        ("salty", "the soup is salty", 0, "literal"),
        ("salty", "he got salty after losing", 1, "slang"),
        ("shady", "we sat under a shady tree", 0, "literal"),
        ("shady", "that deal seems shady", 1, "slang"),
        ("lit", "the hallway was well lit", 0, "literal"),
        ("lit", "the party was lit", 1, "slang"),
        ("flex", "flex your arm slowly", 0, "literal"),
        ("flex", "that car was a big flex", 1, "slang"),
        ("chill", "the weather is chill today", 0, "literal"),
        ("chill", "she's super chill", 1, "slang"),
        ("cooked", "the pasta was cooked well", 0, "literal"),
        ("cooked", "we're cooked if we miss the deadline", 1, "slang"),
        ("serving", "the waiter is serving dinner", 0, "literal"),
        ("serving", "this look is serving confidence", 1, "slang"),
        ("snatched", "the thief snatched her bag", 0, "literal"),
        ("snatched", "her outfit looks snatched", 1, "slang"),
        ("slayed", "the knight slayed the dragon", 0, "literal"),
        ("slayed", "she slayed that presentation", 1, "slang"),
    ]
    return [make_row(term, sentence, label, example_type, "fixed_test") for term, sentence, label, example_type in cases]


def split_rows(rows: list[dict[str, Any]], fixed_tests: list[dict[str, Any]], test_size: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(SEED)
    fixed_keys = {(row["term"], row["sentence"].lower()) for row in fixed_tests}
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}

    for row in rows:
        key = (row["term"], row["sentence"].lower())
        if key in fixed_keys:
            continue
        grouped.setdefault((row["term"], int(row["label"])), []).append(row)

    train_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = fixed_tests[:]

    for group_rows in grouped.values():
        rng.shuffle(group_rows)
        count = max(8, int(round(len(group_rows) * test_size)))
        test_rows.extend(group_rows[:count])
        train_rows.extend(group_rows[count:])

    rng.shuffle(train_rows)
    rng.shuffle(test_rows)
    return train_rows, test_rows


def dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    unique = []
    for row in rows:
        key = (row["term"], row["sentence"].lower())
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build slang sense train/test data.")
    parser.add_argument("--per-label-per-term", type=int, default=90)
    parser.add_argument("--test-size", type=float, default=0.15)
    args = parser.parse_args()

    rows = dedupe(build_rows(args.per_label_per_term))
    tests = fixed_test_rows()
    train_rows, test_rows = split_rows(rows, tests, args.test_size)

    write_json(TRAIN_PATH, train_rows)
    write_json(TEST_PATH, test_rows)

    print(f"Wrote {len(train_rows)} train rows to {TRAIN_PATH}")
    print(f"Wrote {len(test_rows)} test rows to {TEST_PATH}")
    for term in sorted(TERM_CONFIGS):
        train_pos = sum(1 for row in train_rows if row["term"] == term and row["label"] == 1)
        train_neg = sum(1 for row in train_rows if row["term"] == term and row["label"] == 0)
        print(f"  {term}: train positive={train_pos}, train negative={train_neg}")


if __name__ == "__main__":
    main()
