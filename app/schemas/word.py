from pydantic import BaseModel, Field
from typing import Optional
from datetime import date


class WordCreate(BaseModel):
    """Dados pra criar uma nova palavra"""
    word: str = Field(..., min_length=1, max_length=100)
    context_sentence: Optional[str] = None
    source: str = "manual"  # manual, youtube, import


class WordResponse(BaseModel):
    """Resposta com dados da palavra"""
    id: str
    word: str
    normalized_form: Optional[str]
    translation: Optional[str]
    is_slang: bool
    formality_level: str
    category: Optional[str]
    meaning_en: Optional[str]
    meaning_pt: Optional[str]
    examples: list[str]
    
    # Spaced Repetition
    next_review_date: date
    mastery_level: str
    times_correct: int
    times_incorrect: int


class WordAnalysisResponse(BaseModel):
    """Resposta da analise de palavra"""
    original: str
    is_slang: bool
    normalized: str
    translation_pt: str
    meaning_en: Optional[str]
    meaning_pt: Optional[str]
    formality: str
    category: str
    examples: list[str]


class SentenceTranslationResponse(BaseModel):
    """Resposta da traducao de frase"""
    original: str
    slangs_detected: list[dict]
    normalized_english: str
    translation_pt: str