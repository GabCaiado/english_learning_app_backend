from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session

from app.database import get_db
from app.services.translation_service import TranslationService
from app.schemas.word import WordAnalysisResponse, SentenceTranslationResponse
from app.auth import get_current_user
from app.models.user import User

router = APIRouter(prefix="/translate", tags=["translate"])


@router.get("/word/{word}", response_model=WordAnalysisResponse)
async def analyze_word(
    word: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Analyzes a word.
    Detects if it's slang, translates, generates embedding.
    """
    translation_service = TranslationService(db)
    result = translation_service.analyze_word(word)
    
    return WordAnalysisResponse(
        original=result.original,
        is_slang=result.is_slang,
        normalized=result.normalized,
        translation_pt=result.translation_pt,
        meaning_en=result.meaning_en,
        meaning_pt=result.meaning_pt,
        formality=result.formality,
        category=result.category,
        examples=result.examples,
        contextual_translations=result.contextual_translations
    )


@router.post("/sentence", response_model=SentenceTranslationResponse)
async def translate_sentence(
    sentence: str = Body(..., embed=True),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Translates a full sentence.
    Detects slangs, normalizes, translates.
    """
    if not sentence or len(sentence) > 1000:
        raise HTTPException(400, "Invalid or too long sentence")
    
    translation_service = TranslationService(db)
    result = translation_service.translate_sentence(sentence)
    
    return SentenceTranslationResponse(**result)


@router.get("/health")
async def health_check():
    """Checks if the service is ok (working)"""
    return {"status": "ok", "service": "translate"}