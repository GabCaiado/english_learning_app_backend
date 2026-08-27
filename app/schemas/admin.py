from typing import Optional
from pydantic import BaseModel, Field


class DatasetGenerateRequest(BaseModel):
    count: int = Field(default=25, ge=10, le=100)
    categories: list[str] = Field(default_factory=lambda: ["gen_z", "internet"])
    context_style: str = Field(default="text_message")
    slang_mix: float = Field(default=0.6, ge=0.2, le=0.9)
    target_word: Optional[str] = None
    avoid_overrepresented: bool = Field(
        default=True,
        description="Steer the prompt away from words that already have plenty of slang AND neutral examples in the dataset.",
    )


class DatasetGenerateResponse(BaseModel):
    generated: int
    auto_approved: int
    queued_for_review: int
    avoided_words: list[str] = Field(default_factory=list)
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


class VocabularyInfoResponse(BaseModel):
    unique_words: int
    words: list[str]
    word_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Existing slang-usage example count per known word.",
    )
    overrepresented_words: list[str] = Field(
        default_factory=list,
        description="Words that already meet the shared coverage thresholds and would be skipped by Amplify Vocabulary.",
    )


class AmplifyVocabularyRequest(BaseModel):
    words: Optional[list[str]] = Field(
        default=None,
        description="Subset of words to amplify. Defaults to all known vocabulary.",
    )


class AmplifyVocabularyResponse(BaseModel):
    words_processed: int
    words_skipped: int
    generated: int
    auto_approved: int
    queued_for_review: int
    message: str
