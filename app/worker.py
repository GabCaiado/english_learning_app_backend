import asyncio
from arq.connections import RedisSettings
from app.config import get_settings

settings = get_settings()

async def send_verification_email(ctx, email: str, token: str):
    """Worker task to send email verification link"""
    # TODO - later i'll implement a real smtp or third-party email provider (e.g. Resend, SendGrid)
    print(f"[Worker] Envia email de verificacao para: {email} | Token: {token}")
    # Simulates SMTP network latency
    await asyncio.sleep(1)

async def send_password_reset_email(ctx, email: str, token: str):
    """Worker task to send password reset link"""
    print(f"[Worker] Envia email de reset de senha para: {email} | Token: {token}")
    await asyncio.sleep(1)

class WorkerSettings:
    redis_settings = RedisSettings.from_dsn(settings.redis_url)
    functions = [send_verification_email, send_password_reset_email]
