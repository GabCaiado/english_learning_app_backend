from fastapi import APIRouter, Depends, HTTPException
from postgrest.exceptions import APIError
from typing import Optional
from datetime import date

from app.database import get_supabase
from app.ml.pipeline import get_pipeline
from app.ml.slang_detector import AMBIGUOUS_SLANG
from app.schemas.word import WordCreate, WordResponse
from app.auth import get_current_user

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
    """Returns words that need to be reviewed today"""
    supabase = get_supabase()
    
    today = date.today().isoformat()
    
    result = supabase.table("user_words")\
        .select("*")\
        .eq("user_id", user_id)\
        .lte("next_review_date", today)\
        .eq("is_mastered", False)\
        .limit(20)\
        .execute()
    
    for row in result.data:
        row["is_slang"] = display_is_slang(row)
        row["category"] = display_category(row)

    return result.data


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
