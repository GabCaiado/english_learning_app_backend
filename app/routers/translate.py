from fastapi import APIRouter, Depends, HTTPException

from app.database import get_supabase
from app.ml.pipeline import get_pipeline
from app.schemas.word import WordAnalysisResponse, SentenceTranslationResponse
from app.auth import get_current_user

router = APIRouter(prefix="/translate", tags=["translate"])


@router.get("/word/{word}", response_model=WordAnalysisResponse)
async def analyze_word(word: str, user_id: str = Depends(get_current_user)):
    """
    Analisa uma palavra.
    Detecta se é giria, traduz, gera embedding.
    """
    supabase = get_supabase()
    pipeline = get_pipeline(supabase)
    
    result = pipeline.analyze_word(word)
    
    return WordAnalysisResponse(
        original=result.original,
        is_slang=result.is_slang,
        normalized=result.normalized,
        translation_pt=result.translation_pt,
        meaning_en=result.meaning_en,
        meaning_pt=result.meaning_pt,
        formality=result.formality,
        category=result.category,
        examples=result.examples
    )


@router.post("/sentence", response_model=SentenceTranslationResponse)
async def translate_sentence(sentence: str, user_id: str = Depends(get_current_user)):
    """
    Traduz uma frase completa.
    Detecta girias, normaliza, traduz.
    """
    if not sentence or len(sentence) > 1000:
        raise HTTPException(400, "Frase invalida ou muito longa")
    
    supabase = get_supabase()
    pipeline = get_pipeline(supabase)
    
    result = pipeline.translate_sentence(sentence)
    
    return SentenceTranslationResponse(**result)


@router.get("/health")
async def health_check():
    """Verifica se o servico esta ok (funcionando)"""
    return {"status": "ok", "service": "translate"}