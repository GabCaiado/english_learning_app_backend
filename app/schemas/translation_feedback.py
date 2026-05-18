from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class TranslationFeedbackCreate(BaseModel):
    input_text: str = Field(..., min_length=1, max_length=1000)
    model_normalized: Optional[str] = Field(default=None, max_length=1000)
    model_translation: Optional[str] = Field(default=None, max_length=1000)
    model_is_slang: Optional[bool] = None
    model_metadata: dict[str, Any] = Field(default_factory=dict)
    user_feedback: str = Field(default="wrong", max_length=80)
    source: str = Field(default="app", max_length=80)
    user_word_id: Optional[str] = None


class TranslationFeedbackResponse(BaseModel):
    id: str
    user_id: str
    input_text: str
    model_normalized: Optional[str]
    model_translation: Optional[str]
    model_is_slang: Optional[bool]
    model_metadata: dict[str, Any] = Field(default_factory=dict)
    user_feedback: str
    source: str
    expected_normalized: Optional[str] = None
    expected_translation: Optional[str] = None
    expected_is_slang: Optional[bool] = None
    failure_type: Optional[str] = None
    status: str
    created_at: Optional[datetime] = None
    reviewed_at: Optional[datetime] = None


class TranslationFeedbackApprove(BaseModel):
    expected_normalized: str = Field(..., min_length=1, max_length=1000)
    expected_translation: Optional[str] = Field(default=None, max_length=1000)
    expected_is_slang: Optional[bool] = None
    failure_type: str = Field(default="wrong_slang_sense", max_length=120)


class TranslationFeedbackReject(BaseModel):
    status: str = Field(default="rejected", pattern="^(rejected|duplicate)$")
