import uuid
from typing import Optional
from pwdlib import PasswordHash

from app.models.user import User, UserStatus
from app.repositories.user_repository import UserRepository
from app.repositories.refresh_token_repository import RefreshTokenRepository
from app.repositories.active_session_repository import ActiveSessionRepository
from app.repositories.audit_log_repository import AuditLogRepository

class UserService:
    def __init__(self, db):
        self.db = db
        self.user_repo = UserRepository(db)
        self.refresh_repo = RefreshTokenRepository(db)
        self.session_repo = ActiveSessionRepository(db)
        self.audit_repo = AuditLogRepository(db)

    def get_user_profile(self, user_id: uuid.UUID) -> Optional[User]:
        return self.user_repo.get_by_id(user_id)

    def update_user_profile(
        self,
        user_id: uuid.UUID,
        update_data: dict,
        ip_address: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> User:
        user = self.user_repo.get_by_id(user_id)
        if not user:
            raise ValueError("Usuario nao encontrado")

        # If username is changing, ensure it's unique case-insensitively
        username = update_data.get("username")
        if username and username.strip().lower() != (user.username or "").lower():
            existing = self.user_repo.get_by_username_ci(username)
            if existing:
                raise ValueError("Nome de usuario ja em uso")
            update_data["username"] = username.strip()

        updated_user = self.user_repo.update_profile(user, update_data)

        # Audit event
        self.audit_repo.create_audit_log(
            user_id=user.id,
            event_type="PROFILE_UPDATED",
            request_id=request_id,
            ip_hash=hash_ip_helper(ip_address),
            event_metadata={"updated_fields": list(update_data.keys())}
        )

        return updated_user

    def change_password(
        self,
        user: User,
        new_password: str,
        ip_address: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> User:
        _hasher = PasswordHash.recommended()
        user.hashed_password = _hasher.hash(new_password)
        user.token_version += 1
        self.user_repo.update_profile(user, {})
        self.refresh_repo.revoke_all_user_tokens(user.id)
        self.audit_repo.create_audit_log(
            user_id=user.id,
            event_type="PASSWORD_CHANGED",
            request_id=request_id,
            ip_hash=hash_ip_helper(ip_address)
        )
        return user

    def delete_user_account(
        self,
        user_id: uuid.UUID,
        ip_address: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> User:
        user = self.user_repo.get_by_id(user_id)
        if not user:
            raise ValueError("Usuario nao encontrado")

        # Soft delete user record
        deleted_user = self.user_repo.soft_delete_user(user)

        # Revoke all refresh tokens
        self.refresh_repo.revoke_all_user_tokens(user_id)

        # Delete active sessions
        sessions = self.session_repo.get_user_sessions(user_id)
        for s in sessions:
            self.session_repo.delete_session(s)

        # Audit event
        self.audit_repo.create_audit_log(
            user_id=user_id,
            event_type="ACCOUNT_DELETED",
            request_id=request_id,
            ip_hash=hash_ip_helper(ip_address)
        )

        return deleted_user

def hash_ip_helper(ip_address: Optional[str]) -> Optional[str]:
    if not ip_address:
        return None
    import hashlib
    return hashlib.sha256(ip_address.encode()).hexdigest()
