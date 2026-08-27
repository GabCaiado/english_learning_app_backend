import uuid
from datetime import UTC, datetime
from typing import Optional
from sqlalchemy import select, func
from sqlalchemy.orm import Session

from fastapi import APIRouter, Depends, HTTPException

from app.auth import get_current_admin, get_current_user
from app.database import get_db
from app.models.user import User
from app.models.failed_translation import FailedTranslation
from app.schemas.translation_feedback import (
    TranslationFeedbackApprove,
    TranslationFeedbackCreate,
    TranslationFeedbackListResponse,
    TranslationFeedbackReject,
    TranslationFeedbackResponse,
)

router = APIRouter(prefix="/translation-feedback", tags=["translation-feedback"])

AI_SOURCES = {"ai_generated", "ai_amplified"}


def _norm_text(value: Optional[str]) -> str:
    return " ".join((value or "").strip().lower().split())


def _seed_values(record: FailedTranslation) -> tuple[Optional[str], Optional[str], Optional[bool]]:
    """
    The value the reviewer's draft was pre-filled from — GPT's own guess for
    AI-generated/amplified rows (stored as a suggestion, not as model_*
    anymore), or the real model's own output for user-reported rows.
    """
    if record.source in AI_SOURCES:
        meta = record.model_metadata or {}
        return (
            meta.get("llm_suggested_normalized"),
            meta.get("llm_suggested_translation"),
            meta.get("llm_suggested_is_slang"),
        )
    return (record.model_normalized, record.model_translation, record.model_is_slang)


@router.post("/", response_model=TranslationFeedbackResponse)
async def create_translation_feedback(
    feedback: TranslationFeedbackCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Captures a bad translation/normalization result for later human review.

    This does not retrain anything live. It creates a reviewed-data queue item
    that can later become gold eval and training data.
    """
    user_word_id = None
    if feedback.user_word_id:
        try:
            user_word_id = uuid.UUID(feedback.user_word_id)
        except ValueError:
            pass

    record = FailedTranslation(
        user_id=current_user.id,
        input_text=feedback.input_text.strip(),
        model_normalized=feedback.model_normalized,
        model_translation=feedback.model_translation,
        model_is_slang=feedback.model_is_slang,
        model_metadata=feedback.model_metadata,
        user_feedback=feedback.user_feedback,
        source=feedback.source,
        user_word_id=user_word_id,
        status="needs_review",
    )
    db.add(record)
    try:
        db.commit()
        db.refresh(record)
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Erro ao salvar feedback: {str(exc)}")

    return record


@router.get("/pending", response_model=TranslationFeedbackListResponse)
async def list_pending_feedback(
    limit: int = 50,
    offset: int = 0,
    sources: str | None = None,
    current_admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    """
    Lists a page of pending feedback rows for admin review, oldest first.

    `sources`: optional comma-separated list of exact `source` values to
    filter to (e.g. "word_book,word_detail_sentence"). Without it, all
    sources are included — which mixes real user reports in with the much
    larger volume of ai_generated/ai_amplified synthetic seed rows from the
    dataset generator, so callers that care about real feedback specifically
    should pass this.
    """
    conditions = [FailedTranslation.status == "needs_review"]
    if sources:
        source_list = [s.strip() for s in sources.split(",") if s.strip()]
        if source_list:
            conditions.append(FailedTranslation.source.in_(source_list))

    total = db.scalar(select(func.count()).select_from(FailedTranslation).where(*conditions)) or 0
    stmt = (
        select(FailedTranslation)
        .where(*conditions)
        .order_by(FailedTranslation.created_at.asc())
        .limit(limit)
        .offset(offset)
    )
    items = list(db.scalars(stmt).all())
    return TranslationFeedbackListResponse(items=items, total=total)


@router.patch("/{feedback_id}/approve", response_model=TranslationFeedbackResponse)
async def approve_feedback(
    feedback_id: uuid.UUID,
    review: TranslationFeedbackApprove,
    current_admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    """Approves a feedback row as trusted data for future eval/training."""
    record = db.get(FailedTranslation, feedback_id)
    if not record:
        raise HTTPException(status_code=404, detail="Feedback not found")

    seed_normalized, seed_translation, seed_is_slang = _seed_values(record)
    was_edited = (
        _norm_text(review.expected_normalized) != _norm_text(seed_normalized)
        or _norm_text(review.expected_translation) != _norm_text(seed_translation)
        or review.expected_is_slang != seed_is_slang
    )

    # AI-generated/amplified rows must either be edited by the reviewer or
    # explicitly judged correct via failure_type — otherwise "approved" would
    # just mean "the button was clicked" and GPT's raw guess silently becomes
    # gold training data with no human signal behind it. failure_type is the
    # single source of truth here (not a separate checkbox) so a reviewer
    # can't claim a problem (e.g. "missed_slang") while leaving the text
    # exactly as GPT wrote it.
    if record.source in AI_SOURCES and not was_edited and review.failure_type != "correct":
        raise HTTPException(
            status_code=400,
            detail=(
                "This still matches GPT's untouched draft but the failure type "
                "claims a problem. Either edit the fields to fix it, or set "
                "failure type to 'Correct' if the draft is actually fine."
            ),
        )

    record.expected_normalized = review.expected_normalized.strip()
    record.expected_translation = review.expected_translation.strip() if review.expected_translation else None
    record.expected_is_slang = review.expected_is_slang
    record.failure_type = review.failure_type
    record.status = "approved"
    record.reviewed_at = datetime.now(UTC)

    metadata = dict(record.model_metadata or {})
    metadata["was_edited"] = was_edited
    metadata["reviewer_confirmed_unedited"] = bool(review.failure_type == "correct" and not was_edited)
    record.model_metadata = metadata

    db.add(record)
    db.commit()
    db.refresh(record)
    return record


@router.patch("/{feedback_id}/reject", response_model=TranslationFeedbackResponse)
async def reject_feedback(
    feedback_id: uuid.UUID,
    review: TranslationFeedbackReject = TranslationFeedbackReject(),
    current_admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    """Rejects a feedback row or marks it as a duplicate."""
    record = db.get(FailedTranslation, feedback_id)
    if not record:
        raise HTTPException(status_code=404, detail="Feedback not found")

    record.status = review.status
    record.failure_type = "duplicate" if review.status == "duplicate" else "rejected"
    record.reviewed_at = datetime.now(UTC)

    db.add(record)
    db.commit()
    db.refresh(record)
    return record
