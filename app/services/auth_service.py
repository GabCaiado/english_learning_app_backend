import os
import uuid
import hashlib
import jwt
import enum
from datetime import datetime, timedelta, date, timezone
from typing import Optional, List, Tuple
from cryptography.hazmat.primitives import serialization
from pwdlib import PasswordHash
import redis.asyncio as aioredis
from arq import create_pool
from arq.connections import RedisSettings

from app.config import get_settings
from app.models.user import User, UserStatus, UserRole
from app.models.auth import RefreshToken, ActiveSession
from app.repositories.user_repository import UserRepository
from app.repositories.refresh_token_repository import RefreshTokenRepository
from app.repositories.active_session_repository import ActiveSessionRepository
from app.repositories.password_reset_repository import PasswordResetRepository
from app.repositories.email_verification_repository import EmailVerificationRepository
from app.repositories.audit_log_repository import AuditLogRepository

settings = get_settings()
password_hash_context = PasswordHash.recommended()

redis_client = aioredis.from_url(settings.redis_url, decode_responses=True)

class RSAKeyManager:
    _private_key = None
    _public_key = None

    @classmethod
    def get_private_key(cls):
        if cls._private_key:
            return cls._private_key
        
        # 1. Production Secrets
        private_key_pem = os.getenv("JWT_PRIVATE_KEY")
        if private_key_pem:
            private_key_pem = private_key_pem.replace("\\n", "\n")
            cls._private_key = serialization.load_pem_private_key(
                private_key_pem.encode(), password=None
            )
            return cls._private_key

        # 2. Local File Fallback
        private_key_path = os.path.join("keys", "private.pem")
        if os.path.exists(private_key_path):
            with open(private_key_path, "rb") as f:
                cls._private_key = serialization.load_pem_private_key(
                    f.read(), password=None
                )
                return cls._private_key

        raise RuntimeError("JWT Private Key not found! Run generate_keys.py first.")

    @classmethod
    def get_public_key(cls):
        if cls._public_key:
            return cls._public_key

        # 1. Production Secrets
        public_key_pem = os.getenv("JWT_PUBLIC_KEY")
        if public_key_pem:
            public_key_pem = public_key_pem.replace("\\n", "\n")
            cls._public_key = serialization.load_pem_public_key(
                public_key_pem.encode()
            )
            return cls._public_key

        # 2. Local File Fallback
        public_key_path = os.path.join("keys", "public.pem")
        if os.path.exists(public_key_path):
            with open(public_key_path, "rb") as f:
                cls._public_key = serialization.load_pem_public_key(
                    f.read()
                )
                return cls._public_key

        raise RuntimeError("JWT Public Key not found! Run generate_keys.py first.")


class AuthService:
    def __init__(self, db):
        self.db = db
        self.user_repo = UserRepository(db)
        self.refresh_repo = RefreshTokenRepository(db)
        self.session_repo = ActiveSessionRepository(db)
        self.reset_repo = PasswordResetRepository(db)
        self.verify_repo = EmailVerificationRepository(db)
        self.audit_repo = AuditLogRepository(db)

    # Helper Cryptography Methods
    def hash_ip(self, ip_address: Optional[str]) -> Optional[str]:
        if not ip_address:
            return None
        return hashlib.sha256(ip_address.encode()).hexdigest()

    def hash_token(self, token: str) -> str:
        return hashlib.sha256((token + settings.token_pepper).encode()).hexdigest()

    def generate_random_token(self) -> str:
        return uuid.uuid4().hex + uuid.uuid4().hex

    # JWT Operations
    def create_access_token(self, user: User) -> Tuple[str, str]:
        """Returns (access_token_string, jti_uuid_string)"""
        jti = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        expire = now + timedelta(minutes=settings.jwt_expire_minutes)
        payload = {
            "sub": str(user.id),
            "role": user.role.value,
            "status": user.status.value,
            "token_version": user.token_version,
            "jti": jti,
            "iat": now,
            "exp": expire
        }
        private_key = RSAKeyManager.get_private_key()
        token = jwt.encode(payload, private_key, algorithm="RS256")
        return token, jti

    def decode_access_token(self, token: str) -> dict:
        public_key = RSAKeyManager.get_public_key()
        return jwt.decode(token, public_key, algorithms=["RS256"], options={"verify_aud": False})

    # arq Background Queue
    async def _enqueue_email(self, task_name: str, email: str, token: str):
        try:
            redis_settings = RedisSettings.from_dsn(settings.redis_url)
            redis = await create_pool(redis_settings)
            await redis.enqueue_job(task_name, email, token)
        except Exception as e:
            print(f"[AuthService] Falha ao enfileirar email ({task_name}): {e}")

    # Account Lockout (Redis)
    async def check_lockout(self, email: str) -> bool:
        key = f"login_failures:{email.lower().strip()}"
        failures = await redis_client.get(key)
        if failures and int(failures) >= 5:
            return True
        return False

    async def increment_failures(self, email: str):
        key = f"login_failures:{email.lower().strip()}"
        failures = await redis_client.incr(key)
        if failures == 1:
            await redis_client.expire(key, 900) # Lock duration: 15 minutes

    async def clear_failures(self, email: str):
        key = f"login_failures:{email.lower().strip()}"
        await redis_client.delete(key)

    # Main Service Flows
    async def signup(
        self,
        email: str,
        password: str,
        username: Optional[str] = None,
        full_name: Optional[str] = None,
        avatar_url: Optional[str] = None,
        ip_address: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> User:
        existing_user = self.user_repo.get_by_email(email)
        if existing_user:
            raise ValueError("Email ja cadastrado")

        if username:
            existing_username = self.user_repo.get_by_username_ci(username)
            if existing_username:
                raise ValueError("Nome de usuario ja cadastrado")

        # Hash password and create user
        hashed_password = password_hash_context.hash(password)
        user = self.user_repo.create_user(
            email=email,
            hashed_password=hashed_password,
            username=username,
            full_name=full_name,
            avatar_url=avatar_url,
            status=UserStatus.PENDING_VERIFICATION
        )

        # Generate email verification token
        raw_token = self.generate_random_token()
        token_hash = self.hash_token(raw_token)
        expires_at = datetime.now(timezone.utc) + timedelta(days=7)
        self.verify_repo.store_verification_token(user.id, token_hash, expires_at)

        await self._enqueue_email("send_verification_email", user.email, raw_token)

        # Audit event
        self.audit_repo.create_audit_log(
            user_id=user.id,
            event_type="USER_REGISTERED",
            request_id=request_id,
            ip_hash=self.hash_ip(ip_address),
            event_metadata={"email": user.email}
        )

        return user

    async def login(
        self,
        email: str,
        password: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        device_name: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> Tuple[str, str, str, User]:
        """Returns (access_token, refresh_token, csrf_token, user_object)"""
        ip_hash = self.hash_ip(ip_address)

        # Check Account Lockout
        if await self.check_lockout(email):
            self.audit_repo.create_audit_log(
                user_id=None,
                event_type="LOGIN_LOCKED",
                request_id=request_id,
                ip_hash=ip_hash,
                event_metadata={"email": email}
            )
            raise PermissionError("Conta temporariamente bloqueada por multiplas falhas de login. Tente novamente em 15 minutos.")

        user = self.user_repo.get_by_email(email)
        if not user or user.status == UserStatus.DELETED:
            await self.increment_failures(email)
            self.audit_repo.create_audit_log(
                user_id=None,
                event_type="LOGIN_FAILED",
                request_id=request_id,
                ip_hash=ip_hash,
                event_metadata={"email": email, "reason": "User not found or deleted"}
            )
            raise ValueError("Credenciais invalidas")

        # Verify status
        if user.status == UserStatus.PENDING_VERIFICATION:
            self.audit_repo.create_audit_log(
                user_id=user.id,
                event_type="LOGIN_BLOCKED_STATUS",
                request_id=request_id,
                ip_hash=ip_hash,
                event_metadata={"status": user.status.value}
            )
            raise PermissionError("EMAIL_NOT_VERIFIED")

        if user.status == UserStatus.DISABLED or user.status == UserStatus.BANNED:
            self.audit_repo.create_audit_log(
                user_id=user.id,
                event_type="LOGIN_BLOCKED_STATUS",
                request_id=request_id,
                ip_hash=ip_hash,
                event_metadata={"status": user.status.value}
            )
            raise PermissionError(f"Conta suspensa ou banida (status: {user.status.value})")

        # Verify password
        if not password_hash_context.verify(password, user.hashed_password):
            await self.increment_failures(email)
            self.audit_repo.create_audit_log(
                user_id=user.id,
                event_type="LOGIN_FAILED",
                request_id=request_id,
                ip_hash=ip_hash,
                event_metadata={"email": email, "reason": "Incorrect password"}
            )
            raise ValueError("Credenciais invalidas")

        # Clear login failures on success
        await self.clear_failures(email)

        # Create JWT Access Token and JTI
        access_token, jti = self.create_access_token(user)

        # Generate Refresh Token within family rotation
        raw_refresh = self.generate_random_token()
        refresh_hash = self.hash_token(raw_refresh)
        family_id = uuid.uuid4()
        refresh_expires = datetime.now(timezone.utc) + timedelta(days=settings.jwt_refresh_expire_days)

        # Store Refresh Token
        refresh_obj = self.refresh_repo.store_refresh_token(
            user_id=user.id,
            token_hash=refresh_hash,
            family_id=family_id,
            expires_at=refresh_expires,
            ip_hash=ip_hash,
            user_agent=user_agent,
            device_name=device_name
        )

        # Create Device Active Session
        self.session_repo.create_session(
            user_id=user.id,
            refresh_token_id=refresh_obj.id,
            ip_hash=ip_hash,
            user_agent=user_agent,
            device_name=device_name
        )

        # Generate CSRF Token
        csrf_token = self.generate_random_token()

        # Audit
        self.audit_repo.create_audit_log(
            user_id=user.id,
            event_type="LOGIN_SUCCESS",
            request_id=request_id,
            ip_hash=ip_hash,
            event_metadata={"family_id": str(family_id)}
        )

        return access_token, raw_refresh, csrf_token, user

    async def rotate_tokens(
        self,
        raw_refresh_token: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        device_name: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> Tuple[str, str, str]:
        """Rotates refresh token family. Returns (new_access_token, new_raw_refresh, new_csrf_token)"""
        ip_hash = self.hash_ip(ip_address)
        input_hash = self.hash_token(raw_refresh_token)

        # Check in DB for token status
        token_obj = self.refresh_repo.get_refresh_token_by_hash_any_status(input_hash)
        if not token_obj:
            raise ValueError("Refresh token invalido")

        user = self.user_repo.get_by_id(token_obj.user_id)
        if not user or user.status == UserStatus.DELETED:
            raise ValueError("Usuario nao encontrado ou excluido")

        if token_obj.revoked_at is not None:
            now = datetime.now(timezone.utc)
            revoked_at = token_obj.revoked_at
            if revoked_at.tzinfo is None:
                revoked_at = revoked_at.replace(tzinfo=timezone.utc)

            # 30-second grace period for concurrent requests during token rotation
            if (now - revoked_at).total_seconds() < 30:
                active_token = self.refresh_repo.get_active_token_in_family(token_obj.family_id)
                new_raw_refresh = self.generate_random_token()
                new_hash = self.hash_token(new_raw_refresh)
                refresh_expires = datetime.now(timezone.utc) + timedelta(days=settings.jwt_refresh_expire_days)

                if active_token:
                    self.refresh_repo.revoke_refresh_token(active_token)

                new_token_obj = self.refresh_repo.store_refresh_token(
                    user_id=user.id,
                    token_hash=new_hash,
                    family_id=token_obj.family_id,
                    expires_at=refresh_expires,
                    ip_hash=ip_hash,
                    user_agent=user_agent,
                    device_name=device_name
                )

                if active_token:
                    self.session_repo.delete_session_by_refresh_token_id(active_token.id)
                self.session_repo.create_session(
                    user_id=user.id,
                    refresh_token_id=new_token_obj.id,
                    ip_hash=ip_hash,
                    user_agent=user_agent,
                    device_name=device_name
                )

                new_access, jti = self.create_access_token(user)
                new_csrf = self.generate_random_token()

                return new_access, new_raw_refresh, new_csrf

            # Revoke entire token family to protect account
            self.refresh_repo.revoke_entire_family(token_obj.family_id)

            # Invalidate all user sessions (forces global logout)
            user.token_version += 1
            self.db.add(user)
            self.db.commit()

            # Audit event
            self.audit_repo.create_audit_log(
                user_id=user.id,
                event_type="REFRESH_TOKEN_REUSE_DETECTED",
                request_id=request_id,
                ip_hash=ip_hash,
                event_metadata={
                    "family_id": str(token_obj.family_id),
                    "token_id": str(token_obj.id),
                    "action": "revoked_entire_family_and_incremented_token_version"
                }
            )
            raise PermissionError("Reutilizacao de token de atualizacao detectada. Todas as sessoes ativas foram revogadas por seguranca.")

        if token_obj.expires_at < datetime.now(timezone.utc):
            raise ValueError("Refresh token expirado")

        # Token is valid: Revoke the old refresh token
        self.refresh_repo.revoke_refresh_token(token_obj)

        # Generate new refresh token in the same family
        new_raw_refresh = self.generate_random_token()
        new_hash = self.hash_token(new_raw_refresh)
        refresh_expires = datetime.now(timezone.utc) + timedelta(days=settings.jwt_refresh_expire_days)

        # Store new token
        new_token_obj = self.refresh_repo.store_refresh_token(
            user_id=user.id,
            token_hash=new_hash,
            family_id=token_obj.family_id,
            expires_at=refresh_expires,
            ip_hash=ip_hash,
            user_agent=user_agent,
            device_name=device_name
        )

        # Update Session linking
        self.session_repo.delete_session_by_refresh_token_id(token_obj.id)
        self.session_repo.create_session(
            user_id=user.id,
            refresh_token_id=new_token_obj.id,
            ip_hash=ip_hash,
            user_agent=user_agent,
            device_name=device_name
        )

        # Create new access token and CSRF token
        new_access, jti = self.create_access_token(user)
        new_csrf = self.generate_random_token()

        return new_access, new_raw_refresh, new_csrf

    async def logout(
        self,
        raw_refresh_token: str,
        request_id: Optional[str] = None
    ):
        input_hash = self.hash_token(raw_refresh_token)
        token_obj = self.refresh_repo.get_refresh_token_by_hash_any_status(input_hash)
        if token_obj:
            # Revoke refresh token
            self.refresh_repo.revoke_refresh_token(token_obj)

            # Delete ActiveSession
            self.session_repo.delete_session_by_refresh_token_id(token_obj.id)

            # Audit
            self.audit_repo.create_audit_log(
                user_id=token_obj.user_id,
                event_type="LOGOUT",
                request_id=request_id,
                event_metadata={"token_id": str(token_obj.id)}
            )

    async def forgot_password(
        self,
        email: str,
        ip_address: Optional[str] = None,
        request_id: Optional[str] = None
    ):
        ip_hash = self.hash_ip(ip_address)
        user = self.user_repo.get_by_email(email)

        if not user:
            print(f"[AuthService] Password reset requested for unregistered email: {email}")
            raise ValueError("E-mail não encontrado.")

        # Generate reset token
        raw_token = self.generate_random_token()
        token_hash = self.hash_token(raw_token)
        expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
        self.reset_repo.store_reset_token(user.id, token_hash, expires_at)

        # Enqueue email task
        await self._enqueue_email("send_password_reset_email", user.email, raw_token)

        # Audit
        self.audit_repo.create_audit_log(
            user_id=user.id,
            event_type="PASSWORD_RESET_REQUESTED",
            request_id=request_id,
            ip_hash=ip_hash
        )

    async def reset_password(
        self,
        raw_token: str,
        new_password: str,
        ip_address: Optional[str] = None,
        request_id: Optional[str] = None
    ):
        ip_hash = self.hash_ip(ip_address)
        token_hash = self.hash_token(raw_token)

        # Get active token
        reset_token = self.reset_repo.get_active_reset_token_by_hash(token_hash)
        if not reset_token:
            raise ValueError("Token de redefinicao invalido, usado ou expirado")

        user = self.user_repo.get_by_id(reset_token.user_id)
        if not user or user.status == UserStatus.DELETED:
            raise ValueError("Usuario nao encontrado")

        # Hash new password
        user.hashed_password = password_hash_context.hash(new_password)

        # Global Session Revocation: Increment token version + revoke all refresh tokens
        user.token_version += 1
        self.user_repo.update_profile(user, {})
        self.refresh_repo.revoke_all_user_tokens(user.id)

        # Mark token as used
        self.reset_repo.mark_reset_token_used(reset_token)

        # Audit
        self.audit_repo.create_audit_log(
            user_id=user.id,
            event_type="PASSWORD_RESET_SUCCESS",
            request_id=request_id,
            ip_hash=ip_hash
        )

    async def verify_email(
        self,
        raw_token: str,
        ip_address: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> User:
        ip_hash = self.hash_ip(ip_address)
        token_hash = self.hash_token(raw_token)

        verification_token = self.verify_repo.get_active_verification_token_by_hash(token_hash)
        if not verification_token:
            raise ValueError("Token de verificacao invalido, usado ou expirado")

        user = self.user_repo.get_by_id(verification_token.user_id)
        if not user or user.status == UserStatus.DELETED:
            raise ValueError("Usuario nao encontrado")

        # Update user status to active
        user.status = UserStatus.ACTIVE
        self.user_repo.update_profile(user, {})

        # Mark token as used
        self.verify_repo.mark_verification_token_used(verification_token)

        # Audit
        self.audit_repo.create_audit_log(
            user_id=user.id,
            event_type="EMAIL_VERIFIED",
            request_id=request_id,
            ip_hash=ip_hash
        )

        return user

    async def resend_verification(
        self,
        email: str,
        request_id: Optional[str] = None
    ):
        user = self.user_repo.get_by_email(email)
        if not user:
            raise ValueError("Usuario nao encontrado")

        if user.status == UserStatus.ACTIVE:
            raise ValueError("Email ja verificado")

        # Generate new token
        raw_token = self.generate_random_token()
        token_hash = self.hash_token(raw_token)
        expires_at = datetime.now(timezone.utc) + timedelta(days=7)
        self.verify_repo.store_verification_token(user.id, token_hash, expires_at)

        # Enqueue task
        await self._enqueue_email("send_verification_email", user.email, raw_token)

        # Audit
        self.audit_repo.create_audit_log(
            user_id=user.id,
            event_type="EMAIL_VERIFICATION_RESENT",
            request_id=request_id
        )
