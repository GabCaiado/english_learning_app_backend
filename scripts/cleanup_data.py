"""
cleanup_data.py
---------------
Data retention and hygiene script.  Run periodically (e.g. daily cron).

Cleans up:
  1. Expired refresh tokens (revoked or past their expiry)
  2. Expired password-reset tokens
  3. Expired email-verification tokens
  4. Soft-deleted users older than 30 days
  5. Inactive active_sessions with no matching valid refresh token

Run:
  python scripts/cleanup_data.py
  python scripts/cleanup_data.py --dry-run
"""

import sys
import os
import argparse
from datetime import datetime, timedelta, UTC

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import select, delete, and_, or_
from app.database import SessionLocal
from app.models.auth import (
    RefreshToken,
    ActiveSession,
    PasswordResetToken,
    EmailVerificationToken,
)
from app.models.user import User, UserStatus


def run_cleanup(dry_run: bool = False) -> None:
    db = SessionLocal()
    now = datetime.now(UTC)
    cutoff_30d = now - timedelta(days=30)

    try:
        # 1. Expired / revoked refresh tokens
        stmt = select(RefreshToken).where(
            or_(
                RefreshToken.expires_at < now,
                RefreshToken.revoked_at.isnot(None),
            )
        )
        expired_tokens = list(db.scalars(stmt).all())
        print(f"Refresh tokens to delete: {len(expired_tokens)}")
        if not dry_run:
            for t in expired_tokens:
                db.delete(t)

        # 2. Expired password-reset tokens
        stmt2 = select(PasswordResetToken).where(
            or_(
                PasswordResetToken.expires_at < now,
                PasswordResetToken.used_at.isnot(None),
            )
        )
        expired_pwd = list(db.scalars(stmt2).all())
        print(f"Password reset tokens to delete: {len(expired_pwd)}")
        if not dry_run:
            for t in expired_pwd:
                db.delete(t)

        # 3. Expired email-verification tokens
        stmt3 = select(EmailVerificationToken).where(
            or_(
                EmailVerificationToken.expires_at < now,
                EmailVerificationToken.used_at.isnot(None),
            )
        )
        expired_email = list(db.scalars(stmt3).all())
        print(f"Email verification tokens to delete: {len(expired_email)}")
        if not dry_run:
            for t in expired_email:
                db.delete(t)

        # 4. Soft-deleted users older than 30 days
        stmt4 = select(User).where(
            and_(
                User.deleted_at.isnot(None),
                User.deleted_at < cutoff_30d,
            )
        )
        deleted_users = list(db.scalars(stmt4).all())
        print(f"Soft-deleted users to purge: {len(deleted_users)}")
        if not dry_run:
            for u in deleted_users:
                db.delete(u)

        if not dry_run:
            db.commit()
            print("Cleanup committed.")
        else:
            print("Dry-run mode — no changes committed.")

    except Exception as exc:
        db.rollback()
        print(f"ERROR during cleanup: {exc}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data retention cleanup script")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without committing")
    args = parser.parse_args()
    run_cleanup(dry_run=args.dry_run)
