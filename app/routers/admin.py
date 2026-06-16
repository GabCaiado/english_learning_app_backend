from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.auth import get_current_admin
from app.database import get_db
from app.models.failed_translation import FailedTranslation
from app.models.user import User
from app.schemas.admin import (
    DatasetGenerateRequest,
    DatasetGenerateResponse,
    DatasetStatsResponse,
    MilestoneInfo,
)
from app.services.dataset_generator import generate_batch

router = APIRouter(prefix="/admin", tags=["admin"])

MILESTONES = [
    {"count": 500,   "label": "First fine-tune", "emoji": "🟡"},
    {"count": 2000,  "label": "Good model",       "emoji": "🟠"},
    {"count": 5000,  "label": "Strong model",     "emoji": "🟢"},
    {"count": 10000, "label": "Excellent model",  "emoji": "🏆"},
]


@router.post("/dataset/generate", response_model=DatasetGenerateResponse)
async def generate_dataset(
    request: DatasetGenerateRequest,
    current_admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    """
    Generates a batch of labelled English sentences via OpenAI, scores each
    with the local slang detector, and stores them.

    - High-agreement items are auto-approved and never shown in the review queue.
    - Low-agreement items appear in the admin review queue for a quick click-approve.
    """
    try:
        result = generate_batch(
            db=db,
            admin_user=current_admin,
            count=request.count,
            categories=request.categories,
            context_style=request.context_style,
            slang_mix=request.slang_mix,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return DatasetGenerateResponse(
        generated=result["generated"],
        auto_approved=result["auto_approved"],
        queued_for_review=result["queued_for_review"],
        message=(
            f"Generated {result['generated']} sentences — "
            f"{result['auto_approved']} auto-approved, "
            f"{result['queued_for_review']} added to your review queue."
        ),
    )


@router.get("/dataset/stats", response_model=DatasetStatsResponse)
async def get_dataset_stats(
    current_admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    """Returns live dataset health stats for the admin dashboard."""

    total_approved: int = db.scalar(
        select(func.count(FailedTranslation.id)).where(
            FailedTranslation.status == "approved"
        )
    ) or 0

    slang_count: int = db.scalar(
        select(func.count(FailedTranslation.id)).where(
            FailedTranslation.status == "approved",
            FailedTranslation.expected_is_slang == True,  # noqa: E712
        )
    ) or 0

    neutral_count = total_approved - slang_count

    pending_review: int = db.scalar(
        select(func.count(FailedTranslation.id)).where(
            FailedTranslation.status == "needs_review"
        )
    ) or 0

    # Count approved items grouped by source
    source_rows = db.execute(
        select(FailedTranslation.source, func.count(FailedTranslation.id))
        .where(FailedTranslation.status == "approved")
        .group_by(FailedTranslation.source)
    ).all()
    by_source: dict[str, int] = {row[0]: row[1] for row in source_rows}

    # Build milestone objects
    milestone_objs: list[MilestoneInfo] = []
    next_milestone: MilestoneInfo | None = None

    for m in MILESTONES:
        reached = total_approved >= m["count"]
        obj = MilestoneInfo(
            count=m["count"],
            label=m["label"],
            emoji=m["emoji"],
            reached=reached,
        )
        milestone_objs.append(obj)
        if not reached and next_milestone is None:
            next_milestone = obj

    # Balance: ideally 35-65% slang
    if total_approved > 0:
        slang_pct = slang_count / total_approved
        balance_ok = 0.35 <= slang_pct <= 0.65
    else:
        balance_ok = True

    return DatasetStatsResponse(
        total_approved=total_approved,
        slang_count=slang_count,
        neutral_count=neutral_count,
        pending_review=pending_review,
        by_source=by_source,
        milestones=milestone_objs,
        next_milestone=next_milestone,
        balance_ok=balance_ok,
        distilbert_ready=total_approved >= 500,
        byt5_ready=total_approved >= 2000,
    )
