import uuid
from typing import Optional
from datetime import datetime
from sqlalchemy import select, func
from sqlalchemy.orm import Session

from app.models.auth import PasswordResetToken

class PasswordResetRepository:
    def __init__(self, db: Session):
        self.db = db

    def store_reset_token(
        self,
        user_id: uuid.UUID,
        token_hash: str,
        expires_at: datetime
    ) -> PasswordResetToken:
        token = PasswordResetToken(
            user_id=user_id,
            token_hash=token_hash,
            expires_at=expires_at
        )
        self.db.add(token)
        self.db.commit()
        self.db.refresh(token)
        return token

    def get_active_reset_token_by_hash(self, token_hash: str) -> Optional[PasswordResetToken]:
        stmt = select(PasswordResetToken).where(
            PasswordResetToken.token_hash == token_hash,
            PasswordResetToken.used_at.is_(None),
            PasswordResetToken.expires_at > func.now()
        )
        return self.db.scalars(stmt).first()

    def mark_reset_token_used(self, token: PasswordResetToken) -> PasswordResetToken:
        token.used_at = func.now()
        self.db.add(token)
        self.db.commit()
        self.db.refresh(token)
        return token
