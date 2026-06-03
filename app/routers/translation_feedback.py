import uuid
from datetime import UTC, datetime
from sqlalchemy import select
from sqlalchemy.orm import Session

from fastapi import APIRouter, Depends, HTTPException

from app.auth import get_current_admin, get_current_user
from app.database import get_db
from app.models.user import User
from app.models.failed_translation import FailedTranslation
from app.schemas.translation_feedback import (
    TranslationFeedbackApprove,
    TranslationFeedbackCreate,
    TranslationFeedbackReject,
    TranslationFeedbackResponse,
)

router = APIRouter(prefix="/translation-feedback", tags=["translation-feedback"])


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


@router.get("/pending", response_model=list[TranslationFeedbackResponse])
async def list_pending_feedback(
    limit: int = 50,
    current_admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    """Lists pending feedback rows for admin review."""
    stmt = (
        select(FailedTranslation)
        .where(FailedTranslation.status == "needs_review")
        .order_by(FailedTranslation.created_at.asc())
        .limit(limit)
    )
    return list(db.scalars(stmt).all())


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

    record.expected_normalized = review.expected_normalized.strip()
    record.expected_translation = review.expected_translation.strip() if review.expected_translation else None
    record.expected_is_slang = review.expected_is_slang
    record.failure_type = review.failure_type
    record.status = "approved"
    record.reviewed_at = datetime.now(UTC)

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
