import uuid
from typing import Optional
from datetime import date
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.database import get_db
from app.auth import get_current_user
from app.models.user import User
from app.models.vocabulary import UserWord, WordExample
from app.ml.slang_detector import AMBIGUOUS_SLANG
from app.schemas.word import (
    WordCreate,
    WordResponse,
    WordListResponse,
    ReviewRequest,
    ReviewResponse,
    WordOfTheDayResponse,
    DailyWordCount,
)
from app.services.word_service import WordService
from app.services.translation_service import TranslationService
from app.repositories.word_repository import WordRepository

router = APIRouter(prefix="/words", tags=["words"])


def _mastery_level(user_word: UserWord) -> str:
    reps = user_word.repetitions
    if reps == 0:
        return "new"
    if reps <= 3:
        return "learning"
    if reps <= 7:
        return "reviewing"
    return "mastered"


def _display_is_slang(user_word: UserWord) -> bool:
    word = (user_word.word or "").strip().lower()
    has_context = bool((user_word.context_sentence or "").strip())
    if word and " " not in word and word in AMBIGUOUS_SLANG and not has_context:
        return False
    return bool(user_word.is_slang)


def _word_to_response(user_word: UserWord, meaning_en: Optional[str] = None, meaning_pt: Optional[str] = None) -> WordResponse:
    examples = [ex.example_en for ex in (user_word.word_examples or [])]
    is_slang = _display_is_slang(user_word)
    return WordResponse(
        id=str(user_word.id),
        word=user_word.word,
        normalized_form=user_word.normalized_form,
        translation=user_word.translation,
        is_slang=is_slang,
        formality_level=user_word.formality_level or "neutral",
        category=user_word.category if is_slang else None,
        meaning_en=meaning_en,
        meaning_pt=meaning_pt,
        examples=examples,
        next_review_date=user_word.next_review_date or date.today(),
        mastery_level=_mastery_level(user_word),
        times_correct=user_word.times_correct,
        times_incorrect=user_word.times_incorrect,
    )


@router.post("/", response_model=WordResponse)
async def add_word(
    word_data: WordCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Adds a new word to the user's vocabulary.
    Automatically analyzes, translates, and generates embedding.
    """
    translation_service = TranslationService(db)
    word_service = WordService(db)

    analysis = translation_service.analyze_word(word_data.word)

    try:
        user_word = word_service.add_user_word(
            user_id=current_user.id,
            word=word_data.word.lower(),
            translation=analysis.translation_pt,
            normalized_form=analysis.normalized,
            is_slang=analysis.is_slang,
            formality_level=analysis.formality,
            category=analysis.category,
            context_sentence=word_data.context_sentence,
            source=word_data.source,
            embedding=analysis.embedding,
        )
    except ValueError:
        from app.repositories.word_repository import WordRepository
        word_repo = WordRepository(db)
        user_word = word_repo.get_user_word_by_name(current_user.id, word_data.word)
        if not user_word:
            raise HTTPException(status_code=500, detail="Erro ao salvar palavra")

    # Persist examples (up to 3) if word was just created and has none
    if analysis.examples and not user_word.word_examples:
        for ex_text in analysis.examples[:3]:
            db.add(WordExample(user_word_id=user_word.id, example_en=ex_text, source="dictionary"))
        db.commit()
        db.refresh(user_word)

    return _word_to_response(user_word, meaning_en=analysis.meaning_en, meaning_pt=analysis.meaning_pt)


@router.get("/", response_model=WordListResponse)
async def get_user_words(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = 50,
    offset: int = 0,
):
    """Returns a page of the user's words"""
    word_repo = WordRepository(db)
    words, total = word_repo.get_user_words_page(current_user.id, limit=limit, offset=offset)
    return WordListResponse(items=[_word_to_response(w) for w in words], total=total)


@router.get("/review")
async def get_words_for_review(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Returns words due for review today"""
    word_service = WordService(db)
    due = word_service.get_due_reviews(current_user.id)
    return [_word_to_response(w) for w in due[:20]]


@router.get("/word-of-the-day", response_model=WordOfTheDayResponse)
async def get_word_of_the_day(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Returns a new word (not yet in the user's vocabulary) with translation and example"""
    word_service = WordService(db)
    translation_service = TranslationService(db)
    result = await word_service.get_word_of_the_day(current_user.id, translation_service)
    if not result:
        raise HTTPException(status_code=503, detail="Nao foi possivel buscar a palavra do dia")
    return WordOfTheDayResponse(**result)


@router.get("/progress/weekly", response_model=list[DailyWordCount])
async def get_weekly_progress(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Returns the count of words added per day for the last 7 days"""
    word_service = WordService(db)
    return word_service.get_weekly_progress(current_user.id)


@router.post("/{word_id}/review", response_model=ReviewResponse)
async def review_word(
    word_id: uuid.UUID,
    data: ReviewRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Records the user's review result and updates SM-2 SRS fields.
    quality 0-2 = failed, 3 = hard, 4 = good, 5 = perfect
    """
    word_service = WordService(db)
    try:
        _attempt, user_word = word_service.log_review_attempt(
            user_id=current_user.id,
            session_id=None,
            word_id=word_id,
            quality_rating=data.quality,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return ReviewResponse(
        id=str(user_word.id),
        word=user_word.word,
        next_review_date=user_word.next_review_date,
        mastery_level=_mastery_level(user_word),
        interval_days=user_word.interval_days,
        repetitions=user_word.repetitions,
        easiness_factor=round(user_word.easiness_factor, 4),
        times_correct=user_word.times_correct,
        times_incorrect=user_word.times_incorrect,
    )


@router.delete("/{word_id}")
async def delete_word(
    word_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Removes a word from the vocabulary"""
    word_service = WordService(db)
    try:
        word_service.delete_user_word(user_id=current_user.id, word_id=word_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"deleted": True}
