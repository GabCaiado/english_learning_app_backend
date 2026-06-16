from typing import Optional
from pydantic import BaseModel, Field


class DatasetGenerateRequest(BaseModel):
    count: int = Field(default=25, ge=10, le=100)
    categories: list[str] = Field(default_factory=lambda: ["gen_z", "internet"])
    context_style: str = Field(default="text_message")
    slang_mix: float = Field(default=0.6, ge=0.2, le=0.9)


class DatasetGenerateResponse(BaseModel):
    generated: int
    auto_approved: int
    queued_for_review: int
    message: str


class MilestoneInfo(BaseModel):
    count: int
    label: str
    emoji: str
    reached: bool


class DatasetStatsResponse(BaseModel):
    total_approved: int
    slang_count: int
    neutral_count: int
    pending_review: int
    by_source: dict[str, int]
    milestones: list[MilestoneInfo]
    next_milestone: Optional[MilestoneInfo]
    balance_ok: bool
    distilbert_ready: bool
    byt5_ready: bool
