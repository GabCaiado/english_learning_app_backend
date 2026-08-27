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
import re
from collections import Counter
from datetime import UTC, datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models.failed_translation import FailedTranslation
from app.models.user import User

# ---------------------------------------------------------------------------
# Prompt building config
# ---------------------------------------------------------------------------

CATEGORIES: dict[str, dict] = {
    "gen_z": {
        "desc": "Gen Z / TikTok slang",
        "examples": [
            "slay", "rizz", "bussin", "no cap", "it's giving", "lowkey", "NPC",
            "rent free", "understood the assignment", "ate that", "main character",
            "delulu",
        ],
    },
    "internet": {
        "desc": "Internet / social media slang",
        "examples": [
            "sus", "vibe check", "stan", "based", "cringe", "ratio", "hits different",
            "send tweet", "touch grass",
        ],
    },
    "dating": {
        "desc": "Dating / relationship slang",
        "examples": [
            "ghosting", "situationship", "talking stage", "red flag", "green flag",
            "breadcrumbing", "love bombing", "rizz", "dry texter", "soft launch",
        ],
    },
    "sports": {
        "desc": "Sports / hype culture",
        "examples": [
            "clutch", "GOAT", "W", "L", "built different", "beast mode", "on fire",
            "carry", "mid", "diff",
        ],
    },
    "work": {
        "desc": "Hustle culture / workplace slang",
        "examples": [
            "grind", "hustle", "crushing it", "side hustle", "bandwidth",
            "circle back", "synergy", "pivot",
        ],
    },
    "general": {
        "desc": "General American casual speech and colloquialisms",
        "examples": [
            "chill", "solid", "vibe", "hang", "bounce", "dip", "crash", "catch up", "bet",
        ],
    },
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
# Cross-batch dedup — seen_sentences used to be local to a single
# generate_batch() call, so GPT reusing the same sentence skeleton
# ("This new song is fire." / "This new album is fire.") across separate
# runs was invisible. This loads exact + near-duplicate state from the whole
# table once per run, so repeats across batches get caught too.
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "a", "an", "the", "is", "was", "are", "were", "this", "that", "these",
    "those", "in", "on", "at", "to", "of", "for", "with", "and", "or", "so",
    "i", "you", "he", "she", "it", "we", "they", "my", "your", "his", "her",
    "its", "our", "their", "be", "been", "being", "has", "have", "had",
}


def _content_words(sentence: str) -> frozenset:
    return frozenset(
        w for w in re.findall(r"[a-z']+", (sentence or "").lower())
        if w not in _STOPWORDS and len(w) > 2
    )


def _is_near_duplicate(candidate_words: frozenset, recent_word_sets: list, max_diff_words: int = 2) -> bool:
    """
    Flags a sentence as a template repeat if its content words differ from a
    prior sentence by at most `max_diff_words` (default: one word swapped —
    e.g. "This new song is fire." vs "This new album is fire."). A plain
    Jaccard ratio is too lenient here: these generated sentences are short,
    so a single noun swap already looks very different by ratio alone even
    though it's exactly the low-diversity template reuse this is meant to
    catch.
    """
    if len(candidate_words) < 2:
        return False
    for words in recent_word_sets:
        if len(words) < 2:
            continue
        if len(candidate_words ^ words) <= max_diff_words:
            return True
    return False


def _load_dedup_state(db: Session, history_limit: int = 800) -> tuple:
    """Returns (seen_exact_lowercased_texts, recent_content_word_sets)."""
    texts = [t for t in db.scalars(select(FailedTranslation.input_text)).all() if t]
    seen = {t.strip().lower() for t in texts}
    recent_word_sets = [_content_words(t) for t in texts[-history_limit:]]
    return seen, recent_word_sets


def _is_duplicate(sentence: str, seen: set, recent_word_sets: list) -> bool:
    key = sentence.strip().lower()
    if not key or key in seen:
        return True
    return _is_near_duplicate(_content_words(sentence), recent_word_sets)


def _register_sentence(sentence: str, seen: set, recent_word_sets: list) -> None:
    seen.add(sentence.strip().lower())
    recent_word_sets.append(_content_words(sentence))


# ---------------------------------------------------------------------------
# Portuguese translation sanity checks — cheap, dependency-free heuristics to
# flag likely-bad translations for the reviewer instead of relying purely on
# a human noticing by eye.
# ---------------------------------------------------------------------------

_PT_MARKERS = set("áàâãéêíîóôõúüçÁÀÂÃÉÊÍÎÓÔÕÚÜÇ")
_PT_STOPWORDS = {
    "que", "nao", "não", "para", "com", "uma", "um", "de", "da", "do", "das",
    "dos", "e", "ou", "muito", "mas", "como", "isso", "essa", "esse", "ela",
    "ele", "eles", "elas", "sua", "seu", "voce", "você", "porque", "quando",
    "onde", "tambem", "também", "mais", "menos", "esta", "está", "estao",
    "estão", "sao", "são",
}


def _norm_for_compare(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", "", (text or "").lower()).split())


def _looks_portuguese(text: str) -> bool:
    if any(ch in _PT_MARKERS for ch in text):
        return True
    words = set(re.findall(r"[a-zà-ÿ]+", text.lower()))
    return bool(words & _PT_STOPWORDS)


def _validate_pt_translation(english_text: str, translation: str) -> list:
    """Returns a list of warning codes; empty list means no issues found."""
    warnings: list = []
    en = (english_text or "").strip()
    pt = (translation or "").strip()

    if not pt:
        return ["empty_translation"]

    if en and _norm_for_compare(en) == _norm_for_compare(pt):
        return ["untranslated"]

    if not _looks_portuguese(pt):
        warnings.append("possibly_not_portuguese")

    if en:
        ratio = len(pt) / max(1, len(en))
        if ratio < 0.3:
            warnings.append("length_ratio_low")
        elif ratio > 3.0:
            warnings.append("length_ratio_high")

    return warnings


def _get_real_model_output(db: Session, sentence: str) -> dict:
    """
    Runs the sentence through the actual production pipeline (SlangDetector +
    SlangNormalizer + Translator) so model_normalized/model_translation/
    model_is_slang reflect what the app would really produce — not GPT's
    generation-time guess. GPT's guess is kept separately as a suggestion to
    seed the human reviewer's "Expected" draft.
    """
    from app.ml.pipeline import get_pipeline
    result = get_pipeline(db).translate_sentence(sentence)
    return {
        "is_slang": result["is_slang_detected"],
        "normalized": result["normalized_english"],
        "translation": result["translation_pt"],
    }


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(
    bucket_counts: dict[str, int],
    categories: list[str],
    context_style: str,
    target_word: Optional[str] = None,
    avoid_words: Optional[list[str]] = None,
) -> str:
    """
    bucket_counts keys depend on mode:
      - target_word set:      {"slang": N, "neutral": M}
      - target_word is None:  {"category_slang": N, "plain_neutral": N}
        category_slang = pure slang/style words from CATEGORIES (no literal sense)
        plain_neutral   = ordinary vocabulary, no slang words at all

    AMBIGUOUS_SLANG word coverage (dual-meaning words used both as slang and
    literally) is deliberately NOT requested here — see generate_batch's
    second phase, which uses _build_amplify_prompt's per-word "this word is
    mandatory" style instead. A self-reported "vocab_words" field turned out
    to be unreliable (gpt-4o-mini doesn't consistently populate a brand-new
    field it wasn't trained to expect), causing those buckets to starve and
    whole batches to stall out well short of the requested count. The
    mandatory-word style already used by target_word/amplify_vocabulary is
    proven to work — GPT reliably includes an explicitly named word, and the
    code verifies it by regex rather than trusting a self-tag.
    """
    active_cats = {k: v for k, v in CATEGORIES.items() if k in categories}
    if not active_cats:
        active_cats = dict(random.sample(list(CATEGORIES.items()), k=min(3, len(CATEGORIES))))

    context_desc = CONTEXTS.get(context_style, CONTEXTS["text_message"])
    count = sum(bucket_counts.values())

    avoid_set = {w.strip().lower() for w in (avoid_words or [])}
    cat_lines_parts = []
    for cat in active_cats.values():
        fresh_examples = [ex for ex in cat["examples"] if ex.lower() not in avoid_set]
        # Keep at least a couple examples even if most got filtered out, so the
        # category still reads as a concrete style rather than an empty label.
        pool = fresh_examples if fresh_examples else cat["examples"]
        # Random subset (not always the same first N) — a small model like
        # gpt-4o-mini anchors on whatever examples it's shown regardless of
        # "don't just copy these" instructions, so a fixed list would keep
        # converging on the same handful of words every single call. Rotating
        # which examples appear is what actually varies the output, not the
        # wording of the instruction.
        shown = random.sample(pool, k=min(3, len(pool)))
        cat_lines_parts.append(f"  - {cat['desc']} (e.g. {', '.join(shown)}, or other terms in this style)")
    cat_lines = "\n".join(cat_lines_parts)

    avoid_constraint = ""
    if avoid_words:
        avoid_list = ", ".join(f'"{w}"' for w in avoid_words)
        avoid_constraint = (
            f"- Do NOT use these words/slang terms — they already have plenty of "
            f"training examples in the dataset: {avoid_list}\n"
        )

    if target_word:
        slang_count = bucket_counts.get("slang", 0)
        neutral_count = bucket_counts.get("neutral", 0)
        requirements = (
            f'- *** MANDATORY: EVERY single sentence MUST contain the word "{target_word}" — '
            f"no exceptions. Any sentence without it will be discarded. ***\n"
            f'- ~{slang_count} sentences: use "{target_word}" as SLANG (is_slang=true).\n'
            f'- ~{neutral_count} sentences: use "{target_word}" with its LITERAL/STANDARD meaning (is_slang=false).\n'
            f"- Exactly {slang_count} sentences MUST contain slang from these categories:\n{cat_lines}\n"
            f"- Exactly {neutral_count} sentences must be completely standard English with NO slang whatsoever.\n"
        )
        examples = f"""    {{
      "sentence": "That movie was {target_word}, honestly.",
      "is_slang": true,
      "normalized": "That movie was excellent, honestly.",
      "translation": "Aquele filme foi excelente, sinceramente.",
      "formality": "casual",
      "slang_words": ["{target_word}"],
      "category": "gen_z"
    }},
    {{
      "sentence": "She has a {target_word} personality.",
      "is_slang": false,
      "normalized": "She has a {target_word} personality.",
      "translation": "Ela tem uma personalidade {target_word}.",
      "formality": "neutral",
      "slang_words": [],
      "category": "general"
    }}"""
    else:
        category_slang_count = bucket_counts.get("category_slang", 0)
        plain_neutral_count = bucket_counts.get("plain_neutral", 0)

        requirement_lines = []
        if category_slang_count:
            requirement_lines.append(
                f'- Exactly {category_slang_count} sentences MUST contain slang from these categories '
                f'("is_slang": true, put the word(s) used in "vocab_words"):\n{cat_lines}'
            )
        if plain_neutral_count:
            requirement_lines.append(
                f'- Exactly {plain_neutral_count} sentences must be completely standard English about ordinary '
                f'topics — NO slang whatsoever ("is_slang": false, "vocab_words": []).'
            )
        requirements = "\n".join(requirement_lines) + "\n"

        examples = """    {{
      "sentence": "That fit is absolutely iconic, no cap",
      "is_slang": true,
      "normalized": "That outfit looks absolutely amazing, honestly",
      "translation": "Essa roupa está incrível, sério",
      "formality": "casual",
      "slang_words": ["iconic", "no cap"],
      "vocab_words": ["iconic"],
      "category": "gen_z"
    }},
    {{
      "sentence": "I need to finish this report by Friday.",
      "is_slang": false,
      "normalized": "I need to finish this report by Friday.",
      "translation": "Eu preciso terminar esse relatório até sexta-feira.",
      "formality": "neutral",
      "slang_words": [],
      "vocab_words": [],
      "category": "general"
    }}"""

    return f"""You are a linguist creating training data for an English slang detection model used by Brazilian learners of English.

Generate exactly {count} English sentences in the style of {context_desc}.

Note: the JSON example below shows OUTPUT FORMAT ONLY — its wording is not a vocabulary suggestion. Ignore any specific words in it if they conflict with the requirements above.

Requirements:
{avoid_constraint}{requirements}- Sentences should feel 100% natural and realistic — not forced or textbook-like.
- Vary sentence length between 4 and 20 words.
- For slang sentences: the "normalized" field must be grammatically correct, natural standard English conveying the same meaning.
- "translation" must be a natural, idiomatic Brazilian Portuguese translation of the "normalized" field (never a literal word-for-word translation of the slang form).
- The example words shown are illustrative starting points only, NOT a checklist to copy from. Invent additional words/expressions in that same style rather than defaulting to the listed ones.
- Do NOT repeat the same slang word or dual-meaning word more than 2 times across all sentences in this batch.
- Mix sentence types: statements, questions, exclamations.
- "vocab_words" = the specific slang or dual-meaning word(s) this sentence is built around, lowercase, no punctuation. Use an empty list for plain standard-English sentences with no specific target word.

Respond ONLY with a valid JSON object. No extra text outside the JSON.

Format:
{{
  "sentences": [
{examples}
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


_context_resolver = None


def _get_context_resolver():
    global _context_resolver
    if _context_resolver is None:
        from app.ml.context_resolver import ContextResolver
        _context_resolver = ContextResolver()
    return _context_resolver


def _verify_ambiguous_slang_sense(word: str, sentence: str, detector_score: float) -> Optional[bool]:
    """
    Cross-checks a GPT is_slang=true label for an AMBIGUOUS_SLANG word against
    app/ml/context_resolver.py — the same disambiguation layer the live
    translation pipeline uses for polysemous words like "fire" or "cap".
    GPT's self-reported is_slang and the local SlangDetector's score both
    measure "informal register," not word SENSE, so neither catches a
    sentence like "he had one of his fits" being mislabeled as slang for
    "fit" (outfit) when it actually means a tantrum — a wrong-sense false
    positive, not a register disagreement.

    ContextResolver only has real disambiguation rules (slang_context_rules.
    json) for a subset of AMBIGUOUS_SLANG — words like "cool", "dope", or
    "later" have no entry, so resolve() would silently fall back to trusting
    detector_score/dictionary presence, i.e. exactly the blind spot this
    check exists to close. Returning None for those (skip, don't guess)
    instead of a false "confirmed" is deliberate per that gap.
    """
    resolver = _get_context_resolver()
    if word not in resolver.rule_set.ambiguous_terms:
        return None

    from app.ml.slang_dictionary import SlangDictionary
    info = SlangDictionary().lookup(word)
    slang_meaning = info.meaning_en if info else None
    if not slang_meaning:
        return None

    try:
        decision = resolver.resolve(
            term=word,
            sentence=sentence,
            detector_score=detector_score,
            dictionary_has_entry=True,
            slang_meaning=slang_meaning,
        )
    except Exception:
        return None

    return decision.should_normalize


_dictionary_gate_patterns: Optional[dict[str, "re.Pattern"]] = None


def _get_dictionary_gate_patterns() -> dict[str, "re.Pattern"]:
    """
    Lazily builds word-boundary regexes for hand-curated dictionary phrases
    that have no legitimate literal reading (i.e. NOT in AMBIGUOUS_SLANG,
    which is deliberately dual-meaning — "cap"/"broke"/"hard" are correctly
    is_slang=false when used literally, so they're excluded here). Phrases
    like "piece of cake" have no such excuse: they're figurative whenever
    they appear, so is_slang=false for one is a real error, not a judgment
    call. Found via an audit after GPT auto-approved "I really thought it
    would be a piece of cake." as non-slang, contradicting the app's own
    slang_dictionary.py entry for that exact phrase.
    """
    global _dictionary_gate_patterns
    if _dictionary_gate_patterns is None:
        from app.ml.slang_dictionary import SlangDictionary
        from app.ml.slang_detector import AMBIGUOUS_SLANG

        sd = SlangDictionary()
        phrases = [p for p in sd._cache.keys() if p not in AMBIGUOUS_SLANG]
        _dictionary_gate_patterns = {
            p: re.compile(r"\b" + re.escape(p) + r"\b", re.IGNORECASE) for p in phrases
        }
    return _dictionary_gate_patterns


def _contradicts_known_dictionary(sentence: str) -> bool:
    """
    True when `sentence` contains a non-ambiguous dictionary phrase (see
    _get_dictionary_gate_patterns) — used to veto auto-approval of an
    is_slang=false label GPT+detector agreed on, since agreement between two
    guessers isn't correctness when it contradicts already-confirmed ground
    truth. Only called for is_slang=false items; true instead just means the
    same phrase correctly got flagged as slang already.
    """
    sentence_lower = sentence.lower()
    return any(pattern.search(sentence_lower) for pattern in _get_dictionary_gate_patterns().values())


# ---------------------------------------------------------------------------
# Main service function
# ---------------------------------------------------------------------------

MAX_GENERATE_ATTEMPTS = 4

# When there's no target_word, the slang and neutral buckets each split
# further into "dual-meaning (ambiguous) word" vs "other" sentences. Weighted
# toward ambiguous by default — those are the hard-negative/hard-positive
# pairs that actually sharpen the classifier's decision boundary, per user
# preference (0.7 = ~70% of each bucket goes to ambiguous-word sentences).
AMBIGUOUS_WORD_WEIGHT = 0.7


def _bucket_targets(count: int, slang_mix: float, target_word: Optional[str]) -> dict[str, int]:
    slang_target = round(count * slang_mix)
    neutral_target = count - slang_target
    if target_word:
        return {"slang": slang_target, "neutral": neutral_target}

    ambiguous_slang_target = round(slang_target * AMBIGUOUS_WORD_WEIGHT)
    ambiguous_neutral_target = round(neutral_target * AMBIGUOUS_WORD_WEIGHT)
    return {
        "category_slang": slang_target - ambiguous_slang_target,
        "ambiguous_slang": ambiguous_slang_target,
        "ambiguous_neutral": ambiguous_neutral_target,
        "plain_neutral": neutral_target - ambiguous_neutral_target,
    }


def _distribute_evenly(total: int, n: int) -> list[int]:
    """Splits `total` into `n` near-equal non-negative integer parts."""
    if n <= 0:
        return []
    if total <= 0:
        return [0] * n
    base, extra = divmod(total, n)
    return [base + (1 if i < extra else 0) for i in range(n)]


def generate_batch(
    db: Session,
    admin_user: User,
    count: int = 25,
    categories: Optional[list[str]] = None,
    context_style: str = "text_message",
    slang_mix: float = 0.6,
    target_word: Optional[str] = None,
    avoid_overrepresented: bool = True,
) -> dict:
    """
    Generates a labelled batch of sentences via OpenAI, scores each with
    the local detector, and persists them to failed_translations.

    gpt-4o-mini frequently returns fewer usable sentences than requested
    (it doesn't reliably follow "generate exactly N", and the target_word
    filter can discard a lot of them), so a single API call can't be trusted
    to fill `count`. This retries with a growing request size, deduping
    sentences across attempts, until `count` is reached or attempts run out.

    When avoid_overrepresented is True, words that already have plenty of
    slang AND neutral examples (see get_overrepresented_words) are listed in
    the prompt as off-limits, so the category example lists don't keep
    steering the model back to the same handful of already-covered words.

    Without a target_word, each attempt does TWO kinds of requests (see
    _bucket_targets / AMBIGUOUS_WORD_WEIGHT for the four targets):

      - Phase 1 (_build_prompt): pure category slang + plain neutral filler,
        the original "exactly N of category X, exactly M plain" style.
      - Phase 2 (_build_amplify_prompt, same helper amplify_vocabulary uses):
        AMBIGUOUS_SLANG dual-meaning words, requested by NAME as a mandatory
        word per sentence — the same reliable pattern target_word already
        uses — rather than asking GPT to self-report which word a free-form
        sentence is "about" in a new "vocab_words" field. That self-tagging
        approach was tried first and dropped: gpt-4o-mini doesn't reliably
        populate a field it has no training exposure to, so the ambiguous
        buckets silently starved and whole batches stalled far short of the
        requested count. Naming the word and verifying it landed via regex
        (same check target_word already relies on) is what actually works.

    Every accepted item is still capped per-bucket in code (persisted_counts
    vs buckets) — even though phase 2 asks for an exact word/count, the
    per-word compliance still isn't perfect, so this is a backstop, not the
    primary mechanism.

    Returns:
        dict with keys: generated, auto_approved, queued_for_review, avoided_words
    """
    from openai import OpenAI
    from app.ml.slang_detector import AMBIGUOUS_SLANG

    settings = get_settings()
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is not configured in .env")

    client = OpenAI(api_key=settings.openai_api_key)
    detector = _get_detector()

    if categories is None:
        categories = ["gen_z", "internet"]

    avoided_words: list[str] = []
    if avoid_overrepresented:
        avoided_words = get_overrepresented_words(db)
        if target_word and target_word.strip().lower() in avoided_words:
            counts = count_examples_per_word_by_sense(db).get(target_word.strip().lower(), {})
            raise ValueError(
                f'"{target_word}" already has {counts.get("slang", 0)} slang and '
                f'{counts.get("neutral", 0)} neutral examples in the dataset — '
                "disable the 'avoid overrepresented words' toggle if you really want more."
            )
        # The target word is mandatory in this batch, so never list it as avoided.
        if target_word:
            avoided_words = [w for w in avoided_words if w != target_word.strip().lower()]

    avoided_set = {w.lower() for w in avoided_words}
    ambiguous_candidates = [w for w in sorted(AMBIGUOUS_SLANG) if w not in avoided_set]
    if not ambiguous_candidates:
        ambiguous_candidates = sorted(AMBIGUOUS_SLANG)

    buckets = _bucket_targets(count, slang_mix, target_word)
    persisted_counts: dict[str, int] = {k: 0 for k in buckets}
    phase1_keys = ("slang", "neutral") if target_word else ("category_slang", "plain_neutral")

    auto_approved = 0
    queued_for_review = 0
    persisted = 0
    now = datetime.now(UTC)
    seen_sentences, recent_word_sets = _load_dedup_state(db)

    def _accept(
        sentence: str,
        llm_is_slang: bool,
        normalized: str,
        formality: str,
        slang_words: list,
        vocab_words: list,
        category: str,
        bucket: str,
        translation: str = "",
        ambiguous_word: Optional[str] = None,
    ) -> None:
        nonlocal persisted, auto_approved, queued_for_review
        if persisted_counts[bucket] >= buckets[bucket]:
            return
        if not sentence or _is_duplicate(sentence, seen_sentences, recent_word_sets):
            return
        _register_sentence(sentence, seen_sentences, recent_word_sets)

        try:
            detector_score: float = detector.predict_score(sentence)
        except Exception:
            detector_score = 0.5  # neutral fallback — goes to review queue

        # Auto-approve is disabled: every generated row lands in needs_review,
        # with model_normalized/model_translation/model_is_slang pre-filled as
        # an editable draft (per user request — GPT proposes, a human always
        # confirms before anything becomes expected_* gold data). We still
        # compute the agreement signal below and store it as
        # "high_confidence" metadata, not to skip review, but so the review
        # UI/admin can spot which drafts are most likely already correct.
        high_confidence = _should_auto_approve(llm_is_slang, detector_score)
        if high_confidence and not llm_is_slang and _contradicts_known_dictionary(sentence):
            high_confidence = False

        # Only runs for the subset that would otherwise look high-confidence,
        # so it doesn't add ML cost to every sentence — just the ones GPT and
        # the detector already agree on. See _verify_ambiguous_slang_sense.
        sense_verified: Optional[bool] = None
        if high_confidence and llm_is_slang and ambiguous_word:
            sense_verified = _verify_ambiguous_slang_sense(ambiguous_word, sentence, detector_score)
            if sense_verified is False:
                high_confidence = False

        agreement_confidence = detector_score if llm_is_slang else (1.0 - detector_score)

        real_output = _get_real_model_output(db, sentence)
        translation_warnings = {
            "llm_suggested": _validate_pt_translation(sentence, translation),
            "model": _validate_pt_translation(real_output["normalized"], real_output["translation"]),
        }

        record = FailedTranslation(
            user_id=admin_user.id,
            input_text=sentence,
            model_normalized=real_output["normalized"],
            model_translation=real_output["translation"] or None,
            model_is_slang=real_output["is_slang"],
            model_metadata={
                "detector_score": round(detector_score, 4),
                "agreement_confidence": round(agreement_confidence, 4),
                "category": category,
                "context_style": context_style,
                "formality": formality,
                "slang_words": slang_words,
                "vocab_words": vocab_words,
                "auto_approved": False,
                "high_confidence": high_confidence,
                "sense_verified": sense_verified,
                "llm_suggested_is_slang": llm_is_slang,
                "llm_suggested_normalized": normalized,
                "llm_suggested_translation": translation or None,
                "translation_warnings": translation_warnings,
            },
            user_feedback="ai_generated",
            source="ai_generated",
            status="needs_review",
        )

        queued_for_review += 1

        db.add(record)
        persisted += 1
        persisted_counts[bucket] += 1

    def _checkpoint() -> None:
        """
        Commits whatever has been added so far. Called after each phase of
        each attempt (not just once at the end) — each attempt now makes up
        to two separate OpenAI calls, so if a later call fails (rate limit,
        timeout, bad JSON), earlier attempts' already-scored sentences are
        preserved instead of the whole multi-attempt batch being discarded
        when the exception propagates out of generate_batch.
        """
        try:
            db.commit()
        except Exception as exc:
            db.rollback()
            raise RuntimeError(f"Failed to persist generated data: {exc}") from exc

    attempts = 0
    while persisted < count and attempts < MAX_GENERATE_ATTEMPTS:
        attempts += 1

        # ---- Phase 1: category slang + plain neutral ----
        remaining_phase1 = {k: max(0, buckets[k] - persisted_counts[k]) for k in phase1_keys}
        remaining_total = sum(remaining_phase1.values())
        if remaining_total:
            base = remaining_total * 2 if target_word else remaining_total
            request_count = min(base + 20 * (attempts - 1), 100)

            keys = list(remaining_phase1.keys())
            if len(keys) == 1:
                request_bucket_counts = {keys[0]: request_count}
            else:
                n0 = round(request_count * remaining_phase1[keys[0]] / remaining_total)
                request_bucket_counts = {keys[0]: n0, keys[1]: max(0, request_count - n0)}

            prompt = _build_prompt(request_bucket_counts, categories, context_style, target_word, avoided_words)

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

            for item in sentences:
                if persisted >= count:
                    break
                sentence: str = (item.get("sentence") or "").strip()
                if not sentence:
                    continue

                if target_word and not re.search(
                    r"\b" + re.escape(target_word.lower()) + r"\b", sentence.lower()
                ):
                    continue

                llm_is_slang: bool = bool(item.get("is_slang", False))

                if target_word:
                    vocab_words = [target_word.strip().lower()]
                    bucket = "slang" if llm_is_slang else "neutral"
                else:
                    raw_vocab = item.get("vocab_words") or []
                    vocab_words = [w.strip().lower() for w in raw_vocab if isinstance(w, str) and w.strip()]
                    bucket = "category_slang" if llm_is_slang else "plain_neutral"

                normalized: str = sentence if not llm_is_slang else (item.get("normalized") or sentence).strip()
                translation: str = (item.get("translation") or "").strip()
                formality: str = item.get("formality", "neutral")
                slang_words: list = item.get("slang_words") or []
                category: str = item.get("category", "general")

                _accept(sentence, llm_is_slang, normalized, formality, slang_words, vocab_words, category, bucket, translation)

            _checkpoint()

        # ---- Phase 2: dual-meaning (AMBIGUOUS_SLANG) words, requested by name ----
        if not target_word:
            remaining_amb_slang = max(0, buckets["ambiguous_slang"] - persisted_counts["ambiguous_slang"])
            remaining_amb_neutral = max(0, buckets["ambiguous_neutral"] - persisted_counts["ambiguous_neutral"])
            if remaining_amb_slang or remaining_amb_neutral:
                n_words = min(8, len(ambiguous_candidates))
                words_this_round = random.sample(ambiguous_candidates, k=n_words) if n_words else []
                slang_alloc = _distribute_evenly(remaining_amb_slang, len(words_this_round))
                neutral_alloc = _distribute_evenly(remaining_amb_neutral, len(words_this_round))
                word_needs = {
                    w: {"slang": s_n, "neutral": n_n}
                    for w, s_n, n_n in zip(words_this_round, slang_alloc, neutral_alloc)
                    if s_n or n_n
                }

                if word_needs:
                    amb_prompt = _build_amplify_prompt(word_needs)
                    try:
                        response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {
                                    "role": "system",
                                    "content": "You are a dataset generation assistant. Always respond with valid JSON only. No markdown fences.",
                                },
                                {"role": "user", "content": amb_prompt},
                            ],
                            temperature=0.9,
                            response_format={"type": "json_object"},
                        )
                        raw = response.choices[0].message.content
                        data = json.loads(raw)
                        results: list[dict] = data.get("results", [])
                    except json.JSONDecodeError as exc:
                        raise RuntimeError(f"OpenAI returned invalid JSON: {exc}") from exc
                    except Exception as exc:
                        raise RuntimeError(f"OpenAI API call failed: {exc}") from exc

                    for entry in results:
                        if persisted >= count:
                            break
                        word = (entry.get("word") or "").strip().lower()
                        if not word:
                            continue
                        word_pattern = r"\b" + re.escape(word) + r"\b"
                        for s in entry.get("sentences") or []:
                            sentence = (s.get("sentence") or "").strip()
                            # Verify the mandated word actually landed — same regex
                            # check target_word already relies on — instead of
                            # trusting a self-reported field.
                            if not sentence or not re.search(word_pattern, sentence.lower()):
                                continue
                            llm_is_slang = bool(s.get("is_slang", True))
                            normalized = sentence if not llm_is_slang else (s.get("normalized") or sentence).strip()
                            translation = (s.get("translation") or "").strip()
                            bucket = "ambiguous_slang" if llm_is_slang else "ambiguous_neutral"
                            _accept(
                                sentence,
                                llm_is_slang,
                                normalized,
                                "casual" if llm_is_slang else "neutral",
                                [word] if llm_is_slang else [],
                                [word],
                                "ambiguous_vocabulary",
                                bucket,
                                translation,
                                ambiguous_word=word,
                            )

                    _checkpoint()

    _checkpoint()

    return {
        "generated": persisted,
        "auto_approved": auto_approved,
        "queued_for_review": queued_for_review,
        "avoided_words": avoided_words,
    }


# ---------------------------------------------------------------------------
# Vocabulary amplification — widen variation per already-known slang word
# ---------------------------------------------------------------------------

def count_examples_per_word(db: Session) -> dict[str, int]:
    """
    Returns how many failed_translations rows (any status) already reference
    each known slang/ambiguous word or phrase.

    Pulled from model_metadata.slang_words (populated for slang-labeled rows)
    UNIONED with model_metadata.vocab_words (populated for every row generated
    since the vocab_words field was added, slang or neutral) — the union
    matters because an AMBIGUOUS_SLANG word used only in its LITERAL sense
    never appears in slang_words, so relying on slang_words alone would make
    Amplify Vocabulary blind to ambiguous words Generate Batch only ever used
    literally. Older rows only have slang_words, which still works fine.
    """
    rows = db.scalars(select(FailedTranslation.model_metadata)).all()
    counts: Counter = Counter()
    for metadata in rows:
        if not metadata:
            continue
        words: set[str] = set()
        for w in (metadata.get("slang_words") or []) + (metadata.get("vocab_words") or []):
            cleaned = (w or "").strip().lower()
            if cleaned:
                words.add(cleaned)
        for w in words:
            counts[w] += 1
    return dict(counts)


def extract_known_vocabulary(db: Session) -> list[str]:
    """Returns every unique slang word/phrase already seen across all failed_translations rows."""
    return sorted(count_examples_per_word(db))


def count_examples_per_word_by_sense(db: Session) -> dict[str, dict[str, int]]:
    """
    Returns per-word example counts split by how the word was used, e.g.
    {"bussin": {"slang": 62, "neutral": 4}}.

    model_metadata.slang_words (used by count_examples_per_word) is only
    populated for slang usage, so it can't see how many NEUTRAL examples a
    word already has. This scans input_text directly for every known word/
    phrase instead, split by model_is_slang.
    """
    words = extract_known_vocabulary(db)
    if not words:
        return {}

    patterns = {w: re.compile(r"\b" + re.escape(w) + r"\b", re.IGNORECASE) for w in words}
    counts: dict[str, dict[str, int]] = {w: {"slang": 0, "neutral": 0} for w in words}

    rows = db.execute(
        select(FailedTranslation.input_text, FailedTranslation.model_is_slang)
    ).all()

    for input_text, is_slang in rows:
        if not input_text:
            continue
        sense = "slang" if is_slang else "neutral"
        for word, pattern in patterns.items():
            if pattern.search(input_text):
                counts[word][sense] += 1

    return counts


DEFAULT_MIN_SLANG_COVERAGE = 50
DEFAULT_MIN_NEUTRAL_COVERAGE = 30


def is_word_covered(
    word: str,
    counts: dict[str, int],
    min_slang: int = DEFAULT_MIN_SLANG_COVERAGE,
    min_neutral: int = DEFAULT_MIN_NEUTRAL_COVERAGE,
) -> bool:
    """
    A word has "enough" examples once its slang count clears min_slang.
    AMBIGUOUS_SLANG words (real literal + slang senses, e.g. "chill", "fire")
    additionally need min_neutral neutral examples. Everything else has no
    literal sense at all (e.g. "rizz", "bussin") — requiring a neutral count
    for those would mean they could never be considered covered.
    """
    from app.ml.slang_detector import AMBIGUOUS_SLANG

    if counts.get("slang", 0) < min_slang:
        return False
    if word in AMBIGUOUS_SLANG and counts.get("neutral", 0) < min_neutral:
        return False
    return True


def get_overrepresented_words(
    db: Session,
    min_slang: int = DEFAULT_MIN_SLANG_COVERAGE,
    min_neutral: int = DEFAULT_MIN_NEUTRAL_COVERAGE,
) -> list[str]:
    """
    Words that already have enough examples (see is_word_covered) — Generate
    Batch and Amplify Vocabulary should steer away from them instead of
    piling on more.
    """
    counts = count_examples_per_word_by_sense(db)
    return sorted(
        word for word, c in counts.items() if is_word_covered(word, c, min_slang, min_neutral)
    )


def _build_amplify_prompt(word_needs: dict[str, dict[str, int]]) -> str:
    """
    word_needs: {"chill": {"slang": 8, "neutral": 15}, "rizz": {"slang": 12, "neutral": 0}}

    A count of 0 on one sense just means that sense isn't needed right now
    (already covered, or this round's allocation gave it none) — it does NOT
    mean the word lacks that sense. Callers pass genuinely senseless
    combinations (e.g. a pure-slang neologism's neutral count) as 0, but a
    dual-meaning word like "chill" can just as easily land here with
    slang_n=0 while still needing neutral sentences (generate_batch's Phase 2
    splits a small slang target across up to 8 words, so most words get 0
    slang that round). Asserting "this word has no literal meaning" in that
    case is both false and — because it's paired with telling GPT not to
    generate the literal-meaning sentences either — silently drops the
    neutral request entirely, which is what broke the neutral/slang ratio.
    So the instruction only ever states what's needed and what to skip this
    round, never a claim about the word's actual senses.
    """
    lines = []
    for word, needs in word_needs.items():
        slang_n = needs.get("slang", 0)
        neutral_n = needs.get("neutral", 0)
        if slang_n and neutral_n:
            lines.append(
                f'- "{word}": exactly {slang_n} sentences using it AS SLANG (is_slang=true), '
                f"AND exactly {neutral_n} sentences using it with its LITERAL/STANDARD "
                f"meaning (is_slang=false)."
            )
        elif slang_n:
            lines.append(
                f'- "{word}": exactly {slang_n} sentences using it AS SLANG (is_slang=true) only. '
                f"Do NOT generate any is_slang=false sentences for this word right now."
            )
        else:
            lines.append(
                f'- "{word}": exactly {neutral_n} sentences using it with its LITERAL/STANDARD '
                f"meaning (is_slang=false) only. Do NOT generate any is_slang=true sentences "
                f"for this word right now."
            )

    word_lines = "\n".join(lines)

    return f"""You are a linguist creating training data for an English slang detection model used by Brazilian learners of English.

These words/phrases are CONFIRMED English slang. For EACH word below, generate exactly the requested number of NEW, distinct sentences per sense:

{word_lines}

Requirements:
- Vary sentence length (4-20 words), sentence type (statement/question/exclamation), and context (text message, social media, spoken conversation).
- Do not reuse the same sentence template across words.
- For is_slang=true sentences: "normalized" must be grammatically correct standard English with the same meaning, rewriting only the slang part.
- For is_slang=false sentences: "normalized" must be IDENTICAL to "sentence" — do not rewrite literal/standard usage.
- "translation" must be a natural, idiomatic Brazilian Portuguese translation of the "normalized" field (never a literal word-for-word translation of the slang form).

Respond ONLY with a valid JSON object. No extra text outside the JSON.

Format:
{{
  "results": [
    {{
      "word": "chill",
      "sentences": [
        {{"sentence": "This DJ set is so chill, I love it.", "is_slang": true, "normalized": "This DJ set is so relaxed, I love it.", "translation": "Esse set de DJ é tão relaxante, eu amei."}},
        {{"sentence": "The basement stays chill even in summer.", "is_slang": false, "normalized": "The basement stays chill even in summer.", "translation": "O porão continua fresco mesmo no verão."}}
      ]
    }}
  ]
}}"""


MAX_VARIATIONS_PER_REQUEST = 20


def amplify_vocabulary(
    db: Session,
    admin_user: User,
    words: Optional[list[str]] = None,
    batch_size: int = 8,
    min_slang: int = DEFAULT_MIN_SLANG_COVERAGE,
    min_neutral: int = DEFAULT_MIN_NEUTRAL_COVERAGE,
) -> dict:
    """
    Tops up each known word to the shared coverage thresholds (see
    is_word_covered) instead of a flat example count. AMBIGUOUS_SLANG words
    get topped up on BOTH slang and neutral senses; words with no literal
    meaning only get topped up on slang. Words already covered are skipped,
    so re-running this doesn't keep flooding already well-covered words with
    more duplicate variations — only genuinely underrepresented words (or
    senses) get new sentences.
    """
    from openai import OpenAI
    from app.ml.slang_detector import AMBIGUOUS_SLANG

    settings = get_settings()
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is not configured in .env")

    client = OpenAI(api_key=settings.openai_api_key)
    detector = _get_detector()
    seen_sentences, recent_word_sets = _load_dedup_state(db)

    counts = count_examples_per_word_by_sense(db)
    candidate_words = words if words is not None else sorted(counts)
    if not candidate_words:
        raise ValueError("No known vocabulary found to amplify. Generate a batch first.")

    needed_by_word: dict[str, dict[str, int]] = {}
    for word in candidate_words:
        key = word.strip().lower()
        c = counts.get(key, {"slang": 0, "neutral": 0})
        needed_slang = min(max(0, min_slang - c["slang"]), MAX_VARIATIONS_PER_REQUEST)
        needed_neutral = 0
        if key in AMBIGUOUS_SLANG:
            needed_neutral = min(max(0, min_neutral - c["neutral"]), MAX_VARIATIONS_PER_REQUEST)

        if needed_slang > 0 or needed_neutral > 0:
            needed_by_word[word] = {"slang": needed_slang, "neutral": needed_neutral}

    words_skipped = len(candidate_words) - len(needed_by_word)

    if not needed_by_word:
        return {
            "words_processed": 0,
            "words_skipped": words_skipped,
            "generated": 0,
            "auto_approved": 0,
            "queued_for_review": 0,
        }

    word_items = list(needed_by_word.items())
    word_batches = [dict(word_items[i : i + batch_size]) for i in range(0, len(word_items), batch_size)]

    generated = 0
    auto_approved = 0
    queued_for_review = 0
    now = datetime.now(UTC)

    for batch in word_batches:
        prompt = _build_amplify_prompt(batch)
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
            results: list[dict] = data.get("results", [])
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"OpenAI returned invalid JSON: {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"OpenAI API call failed: {exc}") from exc

        for entry in results:
            word = (entry.get("word") or "").strip().lower()
            word_pattern = re.compile(r"\b" + re.escape(word) + r"\b", re.IGNORECASE) if word else None
            for s in entry.get("sentences") or []:
                sentence = (s.get("sentence") or "").strip()
                if not sentence:
                    continue
                # Verify the mandated word actually landed in GPT's sentence —
                # same regex check generate_batch's Phase 2 already relies on.
                # Without this, GPT can report word="fit" while the sentence
                # only contains an unrelated inflection like "fits", and the
                # mismatch sails through untagged.
                if word_pattern and not word_pattern.search(sentence):
                    continue
                if _is_duplicate(sentence, seen_sentences, recent_word_sets):
                    continue
                _register_sentence(sentence, seen_sentences, recent_word_sets)
                llm_is_slang = bool(s.get("is_slang", True))
                normalized = sentence if not llm_is_slang else (s.get("normalized") or sentence).strip()
                translation = (s.get("translation") or "").strip()

                try:
                    detector_score: float = detector.predict_score(sentence)
                except Exception:
                    detector_score = 0.5

                high_confidence = _should_auto_approve(llm_is_slang, detector_score)
                if high_confidence and not llm_is_slang and _contradicts_known_dictionary(sentence):
                    high_confidence = False

                sense_verified: Optional[bool] = None
                if high_confidence and llm_is_slang and word:
                    sense_verified = _verify_ambiguous_slang_sense(word, sentence, detector_score)
                    if sense_verified is False:
                        high_confidence = False

                agreement_confidence = detector_score if llm_is_slang else (1.0 - detector_score)

                real_output = _get_real_model_output(db, sentence)
                translation_warnings = {
                    "llm_suggested": _validate_pt_translation(sentence, translation),
                    "model": _validate_pt_translation(real_output["normalized"], real_output["translation"]),
                }

                record = FailedTranslation(
                    user_id=admin_user.id,
                    input_text=sentence,
                    model_normalized=real_output["normalized"],
                    model_translation=real_output["translation"] or None,
                    model_is_slang=real_output["is_slang"],
                    model_metadata={
                        "detector_score": round(detector_score, 4),
                        "agreement_confidence": round(agreement_confidence, 4),
                        "category": "vocabulary_amplification",
                        "context_style": "mixed",
                        "formality": "casual" if llm_is_slang else "neutral",
                        "slang_words": [word] if word and llm_is_slang else [],
                        "vocab_words": [word] if word else [],
                        "auto_approved": False,
                        "high_confidence": high_confidence,
                        "sense_verified": sense_verified,
                        "llm_suggested_is_slang": llm_is_slang,
                        "llm_suggested_normalized": normalized,
                        "llm_suggested_translation": translation or None,
                        "translation_warnings": translation_warnings,
                    },
                    user_feedback="ai_generated",
                    source="ai_amplified",
                    status="needs_review",
                )

                queued_for_review += 1

                db.add(record)
                generated += 1

    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        raise RuntimeError(f"Failed to persist amplified data: {exc}") from exc

    return {
        "words_processed": len(needed_by_word),
        "words_skipped": words_skipped,
        "generated": generated,
        "auto_approved": auto_approved,
        "queued_for_review": queued_for_review,
    }
