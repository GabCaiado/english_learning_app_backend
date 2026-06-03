import uuid
from typing import List, Optional
from sqlalchemy import select, func, delete
from sqlalchemy.orm import Session

from app.models.auth import ActiveSession

class ActiveSessionRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_session(
        self,
        user_id: uuid.UUID,
        refresh_token_id: uuid.UUID,
        ip_hash: Optional[str] = None,
        user_agent: Optional[str] = None,
        device_name: Optional[str] = None
    ) -> ActiveSession:
        session = ActiveSession(
            user_id=user_id,
            refresh_token_id=refresh_token_id,
            ip_hash=ip_hash,
            user_agent=user_agent,
            device_name=device_name
        )
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)
        return session

    def get_user_sessions(self, user_id: uuid.UUID) -> List[ActiveSession]:
        stmt = select(ActiveSession).where(ActiveSession.user_id == user_id).order_by(ActiveSession.last_seen.desc())
        return list(self.db.scalars(stmt).all())

    def get_session_by_id(self, session_id: uuid.UUID) -> Optional[ActiveSession]:
        stmt = select(ActiveSession).where(ActiveSession.id == session_id)
        return self.db.scalars(stmt).first()

    def update_last_seen(self, session: ActiveSession) -> ActiveSession:
        session.last_seen = func.now()
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)
        return session

    def delete_session(self, session: ActiveSession) -> None:
        self.db.delete(session)
        self.db.commit()

    def delete_session_by_refresh_token_id(self, refresh_token_id: uuid.UUID) -> None:
        stmt = delete(ActiveSession).where(ActiveSession.refresh_token_id == refresh_token_id)
        self.db.execute(stmt)
        self.db.commit()
