"""
Dataset Generator Service
--------------------------
Calls OpenAI to generate labelled English sentences, then scores each one
with the local SlangDetector.

Auto-approve logic:
  - LLM says is_slang=True  AND detector score > 0.75  → both agree → approved silently
  - LLM says is_slang=False AND detector score < 0.25  → both agree → approved silently
  - Otherwise                                          → needs_review (appears in admin queue)
"""

import json
import random
from datetime import UTC, datetime
from typing import Optional

from sqlalchemy.orm import Session

from app.config import get_settings
from app.models.failed_translation import FailedTranslation
from app.models.user import User

# ---------------------------------------------------------------------------
# Prompt building config
# ---------------------------------------------------------------------------

CATEGORIES: dict[str, str] = {
    "gen_z": (
        "Gen Z / TikTok slang "
        "(slay, rizz, bussin, no cap, it's giving, lowkey, NPC, rent free, "
        "understood the assignment, ate that, main character, delulu)"
    ),
    "internet": (
        "Internet / social media slang "
        "(sus, vibe check, stan, based, cringe, ratio, hits different, send tweet, touch grass)"
    ),
    "dating": (
        "Dating / relationship slang "
        "(ghosting, situationship, talking stage, red flag, green flag, "
        "breadcrumbing, love bombing, rizz, dry texter, soft launch)"
    ),
    "sports": (
        "Sports / hype culture "
        "(clutch, GOAT, W, L, built different, beast mode, on fire, carry, mid, diff)"
    ),
    "work": (
        "Hustle culture / workplace slang "
        "(grind, hustle, crushing it, side hustle, bandwidth, circle back, synergy, pivot)"
    ),
    "general": (
        "General American casual speech and colloquialisms "
        "(chill, solid, vibe, hang, bounce, dip, crash, catch up, bet)"
    ),
}

CONTEXTS: dict[str, str] = {
    "text_message": "text messages between close friends",
    "social_media": "Instagram / TikTok captions and comments",
    "group_chat": "WhatsApp / Discord group chat",
    "casual_speech": "casual face-to-face conversation",
    "reaction": "short reaction texts or replies (punchy, 4-10 words)",
}

# ---------------------------------------------------------------------------
# Lazy-loaded detector singleton
# ---------------------------------------------------------------------------

_detector = None


def _get_detector():
    global _detector
    if _detector is None:
        from app.ml.slang_detector import SlangDetector
        _detector = SlangDetector()
    return _detector


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(
    count: int,
    categories: list[str],
    context_style: str,
    slang_mix: float,
) -> str:
    active_cats = {k: v for k, v in CATEGORIES.items() if k in categories}
    if not active_cats:
        active_cats = dict(random.sample(list(CATEGORIES.items()), k=min(3, len(CATEGORIES))))

    context_desc = CONTEXTS.get(context_style, CONTEXTS["text_message"])
    slang_count = round(count * slang_mix)
    neutral_count = count - slang_count
    cat_lines = "\n".join(f"  - {v}" for v in active_cats.values())

    return f"""You are a linguist creating training data for an English slang detection model used by Brazilian learners of English.

Generate exactly {count} English sentences in the style of {context_desc}.

Requirements:
- Exactly {slang_count} sentences MUST contain slang from these categories:
{cat_lines}
- Exactly {neutral_count} sentences must be completely standard English with NO slang whatsoever.
- Sentences should feel 100% natural and realistic — not forced or textbook-like.
- Vary sentence length between 4 and 20 words.
- For slang sentences: the "normalized" field must be grammatically correct, natural standard English conveying the same meaning.
- Do NOT repeat the same slang word more than 3 times across all sentences.
- Mix sentence types: statements, questions, exclamations.

Respond ONLY with a valid JSON object. No extra text outside the JSON.

Format:
{{
  "sentences": [
    {{
      "sentence": "That fit is absolutely bussin, no cap",
      "is_slang": true,
      "normalized": "That outfit looks absolutely amazing, honestly",
      "formality": "casual",
      "slang_words": ["bussin", "no cap"],
      "category": "gen_z"
    }},
    {{
      "sentence": "I need to finish this report by Friday.",
      "is_slang": false,
      "normalized": "I need to finish this report by Friday.",
      "formality": "neutral",
      "slang_words": [],
      "category": "general"
    }}
  ]
}}"""


# ---------------------------------------------------------------------------
# Auto-approve decision
# ---------------------------------------------------------------------------

def _should_auto_approve(llm_is_slang: bool, detector_score: float) -> bool:
    """Returns True when the LLM label and local model agree with high confidence."""
    if llm_is_slang and detector_score > 0.75:
        return True
    if not llm_is_slang and detector_score < 0.25:
        return True
    return False


# ---------------------------------------------------------------------------
# Main service function
# ---------------------------------------------------------------------------

def generate_batch(
    db: Session,
    admin_user: User,
    count: int = 25,
    categories: Optional[list[str]] = None,
    context_style: str = "text_message",
    slang_mix: float = 0.6,
) -> dict:
    """
    Generates a labelled batch of sentences via OpenAI, scores each with
    the local detector, and persists them to failed_translations.

    Returns:
        dict with keys: generated, auto_approved, queued_for_review
    """
    from openai import OpenAI

    settings = get_settings()
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is not configured in .env")

    client = OpenAI(api_key=settings.openai_api_key)
    detector = _get_detector()

    if categories is None:
        categories = ["gen_z", "internet"]

    prompt = _build_prompt(count, categories, context_style, slang_mix)

    # --- Call OpenAI ---
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a dataset generation assistant. Always respond with valid JSON only. No markdown fences.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.9,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        data = json.loads(raw)
        sentences: list[dict] = data.get("sentences", [])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"OpenAI returned invalid JSON: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"OpenAI API call failed: {exc}") from exc

    auto_approved = 0
    queued_for_review = 0
    now = datetime.now(UTC)

    for item in sentences:
        sentence: str = (item.get("sentence") or "").strip()
        if not sentence:
            continue

        llm_is_slang: bool = bool(item.get("is_slang", False))
        normalized: str = (item.get("normalized") or sentence).strip()
        formality: str = item.get("formality", "neutral")
        slang_words: list = item.get("slang_words") or []
        category: str = item.get("category", "general")

        # Score with local model (returns 0.0 if model not loaded — treated as uncertain)
        try:
            detector_score: float = detector.predict_score(sentence)
        except Exception:
            detector_score = 0.5  # neutral fallback — goes to review queue

        auto = _should_auto_approve(llm_is_slang, detector_score)

        # agreement_confidence: how much the two sources agree
        agreement_confidence = detector_score if llm_is_slang else (1.0 - detector_score)

        record = FailedTranslation(
            user_id=admin_user.id,
            input_text=sentence,
            model_normalized=normalized,
            model_is_slang=llm_is_slang,
            model_metadata={
                "detector_score": round(detector_score, 4),
                "agreement_confidence": round(agreement_confidence, 4),
                "category": category,
                "context_style": context_style,
                "formality": formality,
                "slang_words": slang_words,
                "auto_approved": auto,
            },
            user_feedback="ai_generated",
            source="ai_generated",
            status="approved" if auto else "needs_review",
        )

        if auto:
            record.expected_normalized = normalized
            record.expected_is_slang = llm_is_slang
            record.reviewed_at = now
            auto_approved += 1
        else:
            queued_for_review += 1

        db.add(record)

    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        raise RuntimeError(f"Failed to persist generated data: {exc}") from exc

    return {
        "generated": len(sentences),
        "auto_approved": auto_approved,
        "queued_for_review": queued_for_review,
    }
