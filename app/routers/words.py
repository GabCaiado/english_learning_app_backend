from fastapi import APIRouter, Depends, HTTPException
from typing import Optional
from datetime import date

from app.database import get_supabase
from app.ml.pipeline import get_pipeline
from app.schemas.word import WordCreate, WordResponse
from app.auth import get_current_user

router = APIRouter(prefix="/words", tags=["words"])


@router.post("/", response_model=WordResponse)
async def add_word(word_data: WordCreate, user_id: str = Depends(get_current_user)):
    """
    Adiciona uma nova palavra ao vocabulario do usuario.
    Automaticamente analisa, traduz, e gera embedding.
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
    
    result = supabase.table("user_words").insert(insert_data).execute()
    
    if not result.data:
        raise HTTPException(500, "Erro ao salvar palavra")
    
    saved = result.data[0]
    
    # 3- Salva exemplos
    if analysis.examples:
        examples_data = [
            {
                "word_id": saved["id"],
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
        is_slang=saved["is_slang"],
        formality_level=saved["formality_level"],
        category=analysis.category,
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
    """Retorna palavras do usuario"""
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
            is_slang=row.get("is_slang", False),
            formality_level=row.get("formality_level", "neutral"),
            category=row.get("category"),
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
    """Retorna palavras que precisam ser revisadas hoje"""
    supabase = get_supabase()
    
    today = date.today().isoformat()
    
    result = supabase.table("user_words")\
        .select("*")\
        .eq("user_id", user_id)\
        .lte("next_review_date", today)\
        .eq("is_mastered", False)\
        .limit(20)\
        .execute()
    
    return result.data


@router.delete("/{word_id}")
async def delete_word(word_id: str, user_id: str = Depends(get_current_user)):
    """Remove uma palavra do vocabulario"""
    supabase = get_supabase()
    
    result = supabase.table("user_words")\
        .delete()\
        .eq("id", word_id)\
        .eq("user_id", user_id)\
        .execute()
    
    return {"deleted": True}