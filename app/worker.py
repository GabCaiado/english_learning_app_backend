import aiosmtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from arq.connections import RedisSettings
from app.config import get_settings

settings = get_settings()

_EMAIL_HTML = """
<!DOCTYPE html>
<html>
<body style="font-family:Arial,sans-serif;background:#f9f9f9;margin:0;padding:0">
  <div style="max-width:560px;margin:40px auto;background:#fff;border-radius:12px;padding:40px;box-shadow:0 2px 8px rgba(0,0,0,.08)">
    <h1 style="font-size:22px;color:#18181b;margin:0 0 8px">{title}</h1>
    <p style="color:#52525b;font-size:15px;line-height:1.6;margin:0 0 24px">{body}</p>
    <a href="{url}" style="display:inline-block;background:#7c3aed;color:#fff;text-decoration:none;padding:12px 28px;border-radius:8px;font-weight:600;font-size:15px">{cta}</a>
    <p style="color:#a1a1aa;font-size:13px;margin:24px 0 0">{footer}</p>
    <hr style="border:none;border-top:1px solid #f4f4f5;margin:24px 0">
    <p style="color:#a1a1aa;font-size:12px;margin:0">English Slang App &mdash; Seu dicionario inteligente de girias</p>
  </div>
</body>
</html>
"""


def _is_smtp_configured() -> bool:
    return bool(settings.smtp_host and settings.smtp_user and settings.smtp_pass)


async def _send_via_smtp(to: str, subject: str, html: str) -> bool:
    message = MIMEMultipart("alternative")
    message["From"] = settings.smtp_from
    message["To"] = to
    message["Subject"] = subject
    message.attach(MIMEText(html, "html", "utf-8"))

    try:
        await aiosmtplib.send(
            message,
            hostname=settings.smtp_host,
            port=settings.smtp_port,
            username=settings.smtp_user,
            password=settings.smtp_pass,
            use_tls=settings.smtp_secure,
            start_tls=not settings.smtp_secure,
        )
        return True
    except Exception as e:
        print(f"[Worker] Erro SMTP: {e}")
        return False


async def send_verification_email(ctx, email: str, token: str):
    url = f"{settings.frontend_url}/verify-email?token={token}"
    html = _EMAIL_HTML.format(
        title="Confirme seu e-mail",
        body="Obrigado por se cadastrar! Clique no botao abaixo para confirmar seu endereco de e-mail e ativar sua conta.",
        url=url,
        cta="Confirmar e-mail",
        footer="Este link expira em 7 dias. Se voce nao criou uma conta, ignore este e-mail.",
    )

    if not _is_smtp_configured():
        print(f"[Worker] SMTP nao configurado. Link de verificacao:\n{url}")
        return

    ok = await _send_via_smtp(email, "Confirme seu e-mail — English Slang App", html)
    if ok:
        print(f"[Worker] E-mail de verificacao enviado para {email}")


async def send_password_reset_email(ctx, email: str, token: str):
    url = f"{settings.frontend_url}/reset-password?token={token}"
    html = _EMAIL_HTML.format(
        title="Redefinicao de senha",
        body="Recebemos uma solicitacao para redefinir a senha da sua conta. Clique no botao abaixo para criar uma nova senha.",
        url=url,
        cta="Redefinir minha senha",
        footer="Este link expira em 1 hora. Se voce nao fez esta solicitacao, ignore este e-mail.",
    )

    if not _is_smtp_configured():
        print(f"[Worker] SMTP nao configurado. Link de reset:\n{url}")
        return

    ok = await _send_via_smtp(email, "Redefinicao de senha — English Slang App", html)
    if ok:
        print(f"[Worker] E-mail de reset enviado para {email}")


class WorkerSettings:
    redis_settings = RedisSettings.from_dsn(settings.redis_url)
    functions = [send_verification_email, send_password_reset_email]
