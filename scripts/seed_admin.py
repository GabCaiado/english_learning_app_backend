"""
seed_admin.py
-------------
Creates (or idempotently updates) the admin user in the PostgreSQL database.
Reads ADMIN_EMAIL and ADMIN_PASSWORD from environment / .env file.

Run once after the first `alembic upgrade head`:
  python scripts/seed_admin.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.database import SessionLocal
from app.config import get_settings
from app.models.user import User, UserStatus, UserRole
from app.repositories.user_repository import UserRepository
from app.services.auth_service import password_hash_context

settings = get_settings()


def seed_admin() -> None:
    db = SessionLocal()
    try:
        user_repo = UserRepository(db)
        existing = user_repo.get_by_email(settings.admin_email)

        if existing:
            if existing.role != UserRole.ADMIN or existing.status != UserStatus.ACTIVE:
                existing.role = UserRole.ADMIN
                existing.status = UserStatus.ACTIVE
                db.add(existing)
                db.commit()
                print(f"Updated admin user: {settings.admin_email}")
            else:
                print(f"Admin user already exists and is active: {settings.admin_email}")
            return

        hashed_password = password_hash_context.hash(settings.admin_password)
        admin = User(
            email=settings.admin_email,
            username="admin",
            full_name="Admin",
            hashed_password=hashed_password,
            role=UserRole.ADMIN,
            status=UserStatus.ACTIVE,
        )
        db.add(admin)
        db.commit()
        db.refresh(admin)
        print(f"Admin user created: {admin.email} (id={admin.id})")
    except Exception as exc:
        db.rollback()
        print(f"ERROR: {exc}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    seed_admin()
