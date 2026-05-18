from fastapi import APIRouter, Depends, HTTPException
from postgrest.exceptions import APIError
from typing import Optional
from datetime import date, timedelta

from app.database import get_supabase
from app.ml.pipeline import get_pipeline
from app.ml.slang_detector import AMBIGUOUS_SLANG
from app.schemas.word import WordCreate, WordResponse, ReviewRequest, ReviewResponse
from app.auth import get_current_user


# ---------------------------------------------------------------------------
# SM-2 spaced repetition algorithm
# ---------------------------------------------------------------------------

def sm2_update(ef: float, interval: int, reps: int, quality: int) -> tuple[float, int, int]:
    """
    Returns (new_easiness_factor, new_interval_days, new_repetitions).
    quality: 0-2 = failed, 3 = hard, 4 = good, 5 = perfect
    """
    new_ef = ef + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    new_ef = max(1.3, round(new_ef, 4))

    if quality < 3:
        # Failed — reset to beginning, review again tomorrow
        new_reps = 0
        new_interval = 1
    else:
        new_reps = reps + 1
        if new_reps == 1:
            new_interval = 1
        elif new_reps == 2:
            new_interval = 6
        else:
            new_interval = max(1, round(interval * new_ef))

    return new_ef, new_interval, new_reps


def get_mastery_level(reps: int) -> str:
    if reps == 0:
        return "new"
    if reps <= 3:
        return "learning"
    if reps <= 7:
        return "reviewing"
    return "mastered"

router = APIRouter(prefix="/words", tags=["words"])


def display_is_slang(row: dict) -> bool:
    word = (row.get("word") or "").strip().lower()
    has_context = bool((row.get("context_sentence") or "").strip())
    if word and " " not in word and word in AMBIGUOUS_SLANG and not has_context:
        return False
    return bool(row.get("is_slang", False))


def display_category(row: dict) -> Optional[str]:
    return row.get("category") if display_is_slang(row) else None


@router.post("/", response_model=WordResponse)
async def add_word(word_data: WordCreate, user_id: str = Depends(get_current_user)):
    """
    Adds a new word to the user's vocabulary.
    Automatically analyzes, translates, and generates embedding.
    """
    supabase = get_supabase()
    pipeline = get_pipeline(supabase)
    
    # 1- Analisa a palavra
    analysis = pipeline.analyze_word(word_data.word)
    
    # 2- Salva no banco
    insert_data = {
        "user_id": user_id,
        "word": word_data.word.lower(),
        "normalized_form": analysis.normalized,
        "translation": analysis.translation_pt,
        "is_slang": analysis.is_slang,
        "formality_level": analysis.formality,
        "category": analysis.category,
        "context_sentence": word_data.context_sentence,
        "source": word_data.source,
        "embedding": analysis.embedding,
        "next_review_date": date.today().isoformat(),
        "mastery_level": "new"
    }
    
    try:
        result = supabase.table("user_words").insert(insert_data).execute()
    except APIError as e:
        # Palavra ja existe para este usuario (constraint user_id + word)
        if e.code == "23505":
            existing = supabase.table("user_words")\
                .select("*, word_examples(example_en)")\
                .eq("user_id", user_id)\
                .eq("word", word_data.word.lower())\
                .single()\
                .execute()
            if existing.data:
                row = existing.data
                examples = [ex["example_en"] for ex in row.get("word_examples", [])]
                return WordResponse(
                    id=row["id"],
                    word=row["word"],
                    normalized_form=row.get("normalized_form"),
                    translation=row.get("translation"),
                    is_slang=display_is_slang(row),
                    formality_level=row.get("formality_level", "neutral"),
                    category=display_category(row),
                    meaning_en=analysis.meaning_en,
                    meaning_pt=analysis.meaning_pt,
                    examples=examples,
                    next_review_date=date.fromisoformat(row["next_review_date"]) if row.get("next_review_date") else date.today(),
                    mastery_level=row.get("mastery_level", "new"),
                    times_correct=row.get("times_correct", 0),
                    times_incorrect=row.get("times_incorrect", 0)
                )
        raise HTTPException(500, f"Erro ao salvar palavra: {e.message}")

    if not result.data:
        raise HTTPException(500, "Erro ao salvar palavra")
    
    saved = result.data[0]
    
    # 3- Salva exemplos
    if analysis.examples:
        examples_data = [
            {
                "user_word_id": saved["id"],
                "example_en": ex,
                "source": "dictionary"
            }
            for ex in analysis.examples[:3]  # Max 3 exemplos
        ]
        supabase.table("word_examples").insert(examples_data).execute()
    
    return WordResponse(
        id=saved["id"],
        word=saved["word"],
        normalized_form=saved["normalized_form"],
        translation=saved["translation"],
        is_slang=display_is_slang(saved),
        formality_level=saved["formality_level"],
        category=saved.get("category") if display_is_slang(saved) else None,
        meaning_en=analysis.meaning_en,
        meaning_pt=analysis.meaning_pt,
        examples=analysis.examples,
        next_review_date=date.fromisoformat(saved["next_review_date"]),
        mastery_level=saved["mastery_level"],
        times_correct=0,
        times_incorrect=0
    )


@router.get("/", response_model=list[WordResponse])
async def get_user_words(
    user_id: str = Depends(get_current_user),
    limit: int = 50,
    offset: int = 0
):
    """Returns the user's words"""
    supabase = get_supabase()
    
    result = supabase.table("user_words")\
        .select("*, word_examples(example_en)")\
        .eq("user_id", user_id)\
        .order("created_at", desc=True)\
        .range(offset, offset + limit - 1)\
        .execute()
    
    words = []
    for row in result.data:
        examples = [ex["example_en"] for ex in row.get("word_examples", [])]
        words.append(WordResponse(
            id=row["id"],
            word=row["word"],
            normalized_form=row.get("normalized_form"),
            translation=row.get("translation"),
            is_slang=display_is_slang(row),
            formality_level=row.get("formality_level", "neutral"),
            category=display_category(row),
            meaning_en=None,
            meaning_pt=None,
            examples=examples,
            next_review_date=date.fromisoformat(row["next_review_date"]) if row.get("next_review_date") else date.today(),
            mastery_level=row.get("mastery_level", "new"),
            times_correct=row.get("times_correct", 0),
            times_incorrect=row.get("times_incorrect", 0)
        ))
    
    return words


@router.get("/review")
async def get_words_for_review(user_id: str = Depends(get_current_user)):
    """Returns words due for review today, with their examples"""
    supabase = get_supabase()

    today = date.today().isoformat()

    result = supabase.table("user_words")\
        .select("*, word_examples(example_en)")\
        .eq("user_id", user_id)\
        .lte("next_review_date", today)\
        .eq("is_mastered", False)\
        .limit(20)\
        .execute()

    for row in result.data:
        row["is_slang"] = display_is_slang(row)
        row["category"] = display_category(row)

    return result.data


@router.post("/{word_id}/review", response_model=ReviewResponse)
async def review_word(
    word_id: str,
    data: ReviewRequest,
    user_id: str = Depends(get_current_user)
):
    """
    Records the user's review result and updates SM-2 fields.
    quality 0-2 = failed, 3 = hard, 4 = good, 5 = perfect
    """
    supabase = get_supabase()

    result = supabase.table("user_words")\
        .select("*")\
        .eq("id", word_id)\
        .eq("user_id", user_id)\
        .single()\
        .execute()

    if not result.data:
        raise HTTPException(404, "Word not found")

    row = result.data
    ef       = float(row.get("easiness_factor") or 2.5)
    interval = int(row.get("interval_days")     or 0)
    reps     = int(row.get("repetitions")       or 0)

    new_ef, new_interval, new_reps = sm2_update(ef, interval, reps, data.quality)
    new_mastery  = get_mastery_level(new_reps)
    is_mastered  = new_reps >= 8
    next_review  = date.today() + timedelta(days=new_interval)

    times_correct   = row.get("times_correct",   0) + (1 if data.quality >= 3 else 0)
    times_incorrect = row.get("times_incorrect", 0) + (1 if data.quality < 3  else 0)

    supabase.table("user_words").update({
        "easiness_factor":  new_ef,
        "interval_days":    new_interval,
        "repetitions":      new_reps,
        "next_review_date": next_review.isoformat(),
        "mastery_level":    new_mastery,
        "is_mastered":      is_mastered,
        "times_correct":    times_correct,
        "times_incorrect":  times_incorrect,
        "times_reviewed":   row.get("times_reviewed", 0) + 1,
    }).eq("id", word_id).execute()

    return ReviewResponse(
        id=word_id,
        word=row["word"],
        next_review_date=next_review,
        mastery_level=new_mastery,
        interval_days=new_interval,
        repetitions=new_reps,
        easiness_factor=new_ef,
        times_correct=times_correct,
        times_incorrect=times_incorrect,
    )


@router.delete("/{word_id}")
async def delete_word(word_id: str, user_id: str = Depends(get_current_user)):
    """Removes a word from the vocabulary"""
    supabase = get_supabase()
    
    result = supabase.table("user_words")\
        .delete()\
        .eq("id", word_id)\
        .eq("user_id", user_id)\
        .execute()
    
    return {"deleted": True}
