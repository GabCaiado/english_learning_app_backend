import uuid
from datetime import datetime, date
from typing import Optional
from pydantic import BaseModel, EmailStr, Field

class AuthSignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6)
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    full_name: Optional[str] = Field(None, max_length=100)
    avatar_url: Optional[str] = None

class AuthLoginRequest(BaseModel):
    email: EmailStr
    password: str
    device_name: Optional[str] = Field(None, max_length=100)

class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str = Field(..., min_length=6)

class VerifyEmailRequest(BaseModel):
    token: str

class ChangePasswordRequest(BaseModel):
    new_password: str = Field(..., min_length=6)

class UserResponse(BaseModel):
    id: uuid.UUID
    email: str
    username: Optional[str]
    full_name: Optional[str]
    avatar_url: Optional[str]
    native_language: str
    learning_level: str
    daily_goal: int
    total_xp: int
    current_streak: int
    longest_streak: int
    role: str
    status: str
    last_activity_date: Optional[date]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class AuthResponse(BaseModel):
    user: UserResponse

class ActiveSessionResponse(BaseModel):
    id: uuid.UUID
    device_name: Optional[str]
    ip_hash: Optional[str]
    user_agent: Optional[str]
    last_seen: datetime
    created_at: datetime

    class Config:
        from_attributes = True
