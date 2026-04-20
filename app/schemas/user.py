from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime, date

class UserProfile(BaseModel):
    """Dados do perfil completo do usuario"""
    id: str
    username: Optional[str] = None
    full_name: Optional[str] = None
    avatar_url: Optional[str] = None
    native_language: Optional[str] = None
    learning_level: Optional[str] = None
    daily_goal: int = 10
    total_xp: int = 0
    current_streak: int = 0
    longest_streak: int = 0
    last_activity_date: Optional[date] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class UserUpdate(BaseModel):
    """Campos que o usuario pode atualizar no perfil"""
    username: Optional[str] = None
    full_name: Optional[str] = None
    avatar_url: Optional[str] = None
    native_language: Optional[str] = None
    learning_level: Optional[str] = None
    daily_goal: Optional[int] = None
