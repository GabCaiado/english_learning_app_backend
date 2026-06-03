import uuid
from typing import Optional
from datetime import datetime
from sqlalchemy import select, func
from sqlalchemy.orm import Session

from app.models.auth import EmailVerificationToken

class EmailVerificationRepository:
    def __init__(self, db: Session):
        self.db = db

    def store_verification_token(
        self,
        user_id: uuid.UUID,
        token_hash: str,
        expires_at: datetime
    ) -> EmailVerificationToken:
        token = EmailVerificationToken(
            user_id=user_id,
            token_hash=token_hash,
            expires_at=expires_at
        )
        self.db.add(token)
        self.db.commit()
        self.db.refresh(token)
        return token

    def get_active_verification_token_by_hash(self, token_hash: str) -> Optional[EmailVerificationToken]:
        stmt = select(EmailVerificationToken).where(
            EmailVerificationToken.token_hash == token_hash,
            EmailVerificationToken.used_at.is_(None),
            EmailVerificationToken.expires_at > func.now()
        )
        return self.db.scalars(stmt).first()

    def mark_verification_token_used(self, token: EmailVerificationToken) -> EmailVerificationToken:
        token.used_at = func.now()
        self.db.add(token)
        self.db.commit()
        self.db.refresh(token)
        return token
