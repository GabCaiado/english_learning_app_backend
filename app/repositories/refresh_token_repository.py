import uuid
from typing import Optional
from datetime import datetime
from sqlalchemy import select, func, update
from sqlalchemy.orm import Session

from app.models.auth import RefreshToken

class RefreshTokenRepository:
    def __init__(self, db: Session):
        self.db = db

    def store_refresh_token(
        self,
        user_id: uuid.UUID,
        token_hash: str,
        family_id: uuid.UUID,
        expires_at: datetime,
        ip_hash: Optional[str] = None,
        user_agent: Optional[str] = None,
        device_name: Optional[str] = None
    ) -> RefreshToken:
        token = RefreshToken(
            user_id=user_id,
            family_id=family_id,
            token_hash=token_hash,
            expires_at=expires_at,
            ip_hash=ip_hash,
            user_agent=user_agent,
            device_name=device_name
        )
        self.db.add(token)
        self.db.commit()
        self.db.refresh(token)
        return token

    def get_active_refresh_token_by_hash(self, token_hash: str) -> Optional[RefreshToken]:
        stmt = select(RefreshToken).where(
            RefreshToken.token_hash == token_hash,
            RefreshToken.revoked_at.is_(None),
            RefreshToken.expires_at > func.now()
        )
        return self.db.scalars(stmt).first()

    def get_refresh_token_by_hash_any_status(self, token_hash: str) -> Optional[RefreshToken]:
        stmt = select(RefreshToken).where(RefreshToken.token_hash == token_hash)
        return self.db.scalars(stmt).first()

    def revoke_refresh_token(self, token: RefreshToken) -> RefreshToken:
        token.revoked_at = func.now()
        self.db.add(token)
        self.db.commit()
        self.db.refresh(token)
        return token

    def revoke_entire_family(self, family_id: uuid.UUID) -> int:
        stmt = (
            update(RefreshToken)
            .where(RefreshToken.family_id == family_id, RefreshToken.revoked_at.is_(None))
            .values(revoked_at=func.now())
        )
        result = self.db.execute(stmt)
        self.db.commit()
        return result.rowcount

    def revoke_all_user_tokens(self, user_id: uuid.UUID) -> int:
        stmt = (
            update(RefreshToken)
            .where(RefreshToken.user_id == user_id, RefreshToken.revoked_at.is_(None))
            .values(revoked_at=func.now())
        )
        result = self.db.execute(stmt)
        self.db.commit()
        return result.rowcount
