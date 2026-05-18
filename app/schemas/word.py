from pydantic import BaseModel, Field
from typing import Optional
from datetime import date


class WordCreate(BaseModel):
    """Data to create a new word"""
    word: str = Field(..., min_length=1, max_length=100)
    context_sentence: Optional[str] = None
    source: str = "manual"  # manual, youtube, import


class WordResponse(BaseModel):
    """Response with word data"""
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
    """Response with word analysis"""
    original: str
    is_slang: bool
    normalized: str
    translation_pt: str
    meaning_en: Optional[str]
    meaning_pt: Optional[str]
    formality: str
    category: Optional[str]
    examples: list[str]
    contextual_translations: list[dict] = []


class ReviewRequest(BaseModel):
    """SM-2 quality score sent by the user after reviewing a card"""
    quality: int = Field(..., ge=0, le=5, description="0-5: 0-2 = failed, 3 = hard, 4 = good, 5 = perfect")


class ReviewResponse(BaseModel):
    """Updated spaced repetition state after a review"""
    id: str
    word: str
    next_review_date: date
    mastery_level: str
    interval_days: int
    repetitions: int
    easiness_factor: float
    times_correct: int
    times_incorrect: int


class SentenceTranslationResponse(BaseModel):
    """Response with sentence translation"""
    original: str
    slangs_detected: list[dict]
    normalized_english: str
    translation_pt: str
    contextual_translations: list[dict] = []