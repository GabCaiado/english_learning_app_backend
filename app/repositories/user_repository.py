import uuid
from typing import Optional
from datetime import datetime
from sqlalchemy import select, func
from sqlalchemy.orm import Session

from app.models.user import User, UserStatus

class UserRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_id(self, user_id: uuid.UUID) -> Optional[User]:
        stmt = select(User).where(User.id == user_id, User.status != UserStatus.DELETED)
        return self.db.scalars(stmt).first()

    def get_by_email(self, email: str) -> Optional[User]:
        normalized_email = email.strip().lower()
        stmt = select(User).where(func.lower(User.email) == normalized_email, User.status != UserStatus.DELETED)
        return self.db.scalars(stmt).first()

    def get_by_username_ci(self, username: str) -> Optional[User]:
        stmt = select(User).where(func.lower(User.username) == username.strip().lower(), User.status != UserStatus.DELETED)
        return self.db.scalars(stmt).first()

    def create_user(
        self,
        email: str,
        hashed_password: str,
        username: Optional[str] = None,
        full_name: Optional[str] = None,
        avatar_url: Optional[str] = None,
        status: UserStatus = UserStatus.PENDING_VERIFICATION
    ) -> User:
        user = User(
            email=email.strip().lower(),
            hashed_password=hashed_password,
            username=username.strip() if username else None,
            full_name=full_name,
            avatar_url=avatar_url,
            status=status
        )
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user

    def update_profile(self, user: User, update_data: dict) -> User:
        for key, value in update_data.items():
            if hasattr(user, key):
                setattr(user, key, value)
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user

    def soft_delete_user(self, user: User) -> User:
        user.status = UserStatus.DELETED
        user.deleted_at = func.now()
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user
