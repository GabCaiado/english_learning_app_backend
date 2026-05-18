from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException
from postgrest.exceptions import APIError

from app.auth import get_current_admin, get_current_user
from app.database import get_supabase
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
    user_id: str = Depends(get_current_user),
):
    """
    Captures a bad translation/normalization result for later human review.

    This does not retrain anything live. It creates a reviewed-data queue item
    that can later become gold eval and training data.
    """
    supabase = get_supabase()
    insert_data = {
        "user_id": user_id,
        "input_text": feedback.input_text.strip(),
        "model_normalized": feedback.model_normalized,
        "model_translation": feedback.model_translation,
        "model_is_slang": feedback.model_is_slang,
        "model_metadata": feedback.model_metadata,
        "user_feedback": feedback.user_feedback,
        "source": feedback.source,
        "user_word_id": feedback.user_word_id,
        "status": "needs_review",
    }

    try:
        result = supabase.table("failed_translations").insert(insert_data).execute()
    except APIError as exc:
        raise HTTPException(status_code=500, detail=f"Erro ao salvar feedback: {exc.message}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erro ao salvar feedback: {str(exc)}")

    if not result.data:
        raise HTTPException(status_code=500, detail="Erro ao salvar feedback")

    return TranslationFeedbackResponse(**result.data[0])


@router.get("/pending", response_model=list[TranslationFeedbackResponse])
async def list_pending_feedback(
    limit: int = 50,
    admin_id: str = Depends(get_current_admin),
):
    """Lists pending feedback rows for admin review."""
    supabase = get_supabase()
    result = (
        supabase.table("failed_translations")
        .select("*")
        .eq("status", "needs_review")
        .order("created_at", desc=False)
        .limit(limit)
        .execute()
    )
    return [TranslationFeedbackResponse(**row) for row in (result.data or [])]


@router.patch("/{feedback_id}/approve", response_model=TranslationFeedbackResponse)
async def approve_feedback(
    feedback_id: str,
    review: TranslationFeedbackApprove,
    admin_id: str = Depends(get_current_admin),
):
    """Approves a feedback row as trusted data for future eval/training."""
    supabase = get_supabase()
    update_data = {
        "expected_normalized": review.expected_normalized.strip(),
        "expected_translation": review.expected_translation.strip() if review.expected_translation else None,
        "expected_is_slang": review.expected_is_slang,
        "failure_type": review.failure_type,
        "status": "approved",
        "reviewed_at": datetime.now(UTC).isoformat(),
    }
    result = (
        supabase.table("failed_translations")
        .update(update_data)
        .eq("id", feedback_id)
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=404, detail="Feedback not found")
    return TranslationFeedbackResponse(**result.data[0])


@router.patch("/{feedback_id}/reject", response_model=TranslationFeedbackResponse)
async def reject_feedback(
    feedback_id: str,
    review: TranslationFeedbackReject = TranslationFeedbackReject(),
    admin_id: str = Depends(get_current_admin),
):
    """Rejects a feedback row or marks it as a duplicate."""
    status = "rejected" if review.status == "duplicate" else review.status
    supabase = get_supabase()
    result = (
        supabase.table("failed_translations")
        .update(
            {
                "status": status,
                "failure_type": "duplicate" if review.status == "duplicate" else "rejected",
                "reviewed_at": datetime.now(UTC).isoformat(),
            }
        )
        .eq("id", feedback_id)
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=404, detail="Feedback not found")
    return TranslationFeedbackResponse(**result.data[0])
