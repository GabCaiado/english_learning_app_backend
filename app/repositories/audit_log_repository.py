import uuid
from typing import Optional
from sqlalchemy.orm import Session

from app.models.auth import AuditLog

class AuditLogRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_audit_log(
        self,
        user_id: Optional[uuid.UUID],
        event_type: str,
        request_id: Optional[str] = None,
        event_metadata: Optional[dict] = None,
        ip_hash: Optional[str] = None
    ) -> AuditLog:
        log = AuditLog(
            user_id=user_id,
            event_type=event_type,
            request_id=request_id,
            event_metadata=event_metadata,
            ip_hash=ip_hash
        )
        self.db.add(log)
        self.db.commit()
        self.db.refresh(log)
        return log
