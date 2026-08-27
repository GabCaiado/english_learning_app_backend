"""
Smoke tests for the slang sense classifier integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.context_resolver import ContextResolver


CASES = [
    ("tea", "i wanna drink some tea", "gossip", False),
    ("tea", "what's the tea", "gossip", True),
    ("fire", "the house is on fire", "excellent or impressive", False),
    ("fire", "this beat is fire", "excellent or impressive", True),
    ("sick", "i feel sick today", "excellent or impressive", False),
    ("sick", "that trick was sick", "excellent or impressive", True),
    ("legit", "this app is legit", "excellent, real, or credible depending on context", False),
    ("legit", "that's legit", "excellent, real, or credible depending on context", True),
    ("ghost", "the ghost story was scary", "suddenly ignore someone", False),
    ("shady", "the link looked shady", "suspicious or dishonest", True),
    ("extra", "the decorations were extra", "over the top or excessive", True),
    ("drip", "his drip is clean", "stylish clothing or appearance", True),
    ("chill", "the weather is chill today", "relaxed or easygoing", False),
    ("chill", "she's super chill", "relaxed or easygoing", True),
    ("cooked", "the pasta was cooked well", "in serious trouble or likely to fail", False),
    ("cooked", "we're cooked if we miss the deadline", "in serious trouble or likely to fail", True),
    ("serving", "the waiter is serving dinner", "projecting or giving off a strong vibe", False),
    ("serving", "this look is serving confidence", "projecting or giving off a strong vibe", True),
    ("snatched", "the thief snatched her bag", "very stylish, flattering, or well put together", False),
    ("snatched", "her outfit looks snatched", "very stylish, flattering, or well put together", True),
    ("slayed", "the knight slayed the dragon", "did very well", False),
    ("slayed", "she slayed that presentation", "did very well", True),
    ("the bomb", "we learned about the bomb's history in our history class today", "excellent", False),
    ("the bomb", "the atomic bomb changed the course of world war two", "excellent", False),
    ("the bomb", "this new song is the bomb", "excellent", True),
    ("the bomb", "the bomb exploded near the old bridge", "excellent", False),
    ("flopped", "the fish flopped around on the deck of the boat", "failed badly", False),
    ("flopped", "the new movie flopped at the box office this weekend", "failed badly", True),
    ("go off", "the fire alarm went off in the middle of the night", "an encouragement to perform confidently", False),
    ("go off", "go off, queen, that speech was amazing", "an encouragement to perform confidently", True),
    ("let him cook", "the kitchen was short-staffed, so let him cook dinner for the guests", "let him keep going because his idea may work", False),
    ("let him cook", "his plan sounds weird, but let him cook", "let him keep going because his idea may work", True),
    ("spill the tea", "please spill the tea into the sink before it gets cold", "share the gossip", False),
    ("spill the tea", "okay girl, spill the tea about what happened last night", "share the gossip", True),
]


def main() -> None:
    resolver = ContextResolver()
    failures = []

    for term, sentence, meaning, expected in CASES:
        decision = resolver.resolve(
            term=term,
            sentence=sentence,
            detector_score=0.5,
            dictionary_has_entry=True,
            slang_meaning=meaning,
        )
        ok = decision.should_normalize is expected
        marker = "OK" if ok else "FAIL"
        print(
            f"{marker} | {sentence!r} | expected={expected} "
            f"got={decision.should_normalize} | {decision.reason}"
        )
        if not ok:
            failures.append((sentence, expected, decision))

    if failures:
        raise SystemExit(f"{len(failures)} sense classifier integration cases failed")


if __name__ == "__main__":
    main()
