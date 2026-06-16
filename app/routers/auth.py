import uuid
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.orm import Session
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.database import get_db
from app.config import get_settings
from app.auth import get_current_user
from app.models.user import User
from app.models.auth import RefreshToken
from app.services.auth_service import AuthService
from app.schemas.auth import (
    AuthSignupRequest,
    AuthLoginRequest,
    ForgotPasswordRequest,
    ResetPasswordRequest,
    VerifyEmailRequest,
    UserResponse,
    AuthResponse,
    ActiveSessionResponse,
)

router = APIRouter(prefix="/auth", tags=["auth"])
settings = get_settings()

limiter = Limiter(key_func=get_remote_address)

def get_request_id(request: Request) -> str:
    req_id = request.headers.get("x-request-id")
    if not req_id:
        req_id = str(uuid.uuid4())
    return req_id

def set_auth_cookies(response: Response, access_token: str, refresh_token: str, csrf_token: str):
    is_prod = settings.environment == "production"
    domain = settings.cookie_domain if settings.cookie_domain else None

    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        secure=is_prod,
        samesite="lax",
        path="/",
        domain=domain,
        max_age=settings.jwt_expire_minutes * 60 # 5 minutes
    )

    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        secure=is_prod,
        samesite="lax",
        path="/",
        domain=domain,
        max_age=settings.jwt_refresh_expire_days * 24 * 3600 # 30 days
    )

    response.set_cookie(
        key="csrf_token",
        value=csrf_token,
        httponly=False,
        secure=is_prod,
        samesite="lax",
        path="/",
        domain=domain,
        max_age=settings.jwt_refresh_expire_days * 24 * 3600 # 30 days
    )

def clear_auth_cookies(response: Response):
    domain = settings.cookie_domain if settings.cookie_domain else None
    response.delete_cookie("access_token", path="/", domain=domain)
    response.delete_cookie("refresh_token", path="/", domain=domain)
    response.delete_cookie("csrf_token", path="/", domain=domain)


@router.post("/signup", response_model=UserResponse)
@limiter.limit("5/minute")
async def signup(
    request: Request,
    payload: AuthSignupRequest,
    db: Session = Depends(get_db)
):
    auth_service = AuthService(db)
    ip_address = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")
    request_id = get_request_id(request)

    try:
        user = await auth_service.signup(
            email=payload.email,
            password=payload.password,
            username=payload.username,
            full_name=payload.full_name,
            avatar_url=payload.avatar_url,
            ip_address=ip_address,
            request_id=request_id
        )
        return user
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/login", response_model=AuthResponse)
@limiter.limit("5/minute")
async def login(
    request: Request,
    response: Response,
    payload: AuthLoginRequest,
    db: Session = Depends(get_db)
):
    auth_service = AuthService(db)
    ip_address = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")
    request_id = get_request_id(request)

    try:
        access_token, refresh_token, csrf_token, user = await auth_service.login(
            email=payload.email,
            password=payload.password,
            ip_address=ip_address,
            user_agent=user_agent,
            device_name=payload.device_name,
            request_id=request_id
        )
        set_auth_cookies(response, access_token, refresh_token, csrf_token)
        return AuthResponse(user=UserResponse.from_orm(user))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))


@router.post("/refresh")
@limiter.limit("30/minute")
async def refresh(
    request: Request,
    response: Response,
    db: Session = Depends(get_db)
):
    auth_service = AuthService(db)
    ip_address = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")
    request_id = get_request_id(request)

    raw_refresh = request.cookies.get("refresh_token")
    if not raw_refresh:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token ausente")

    try:
        new_access, new_refresh, new_csrf = await auth_service.rotate_tokens(
            raw_refresh_token=raw_refresh,
            ip_address=ip_address,
            user_agent=user_agent,
            request_id=request_id
        )
        set_auth_cookies(response, new_access, new_refresh, new_csrf)
        return {"status": "success"}
    except (ValueError, PermissionError) as e:
        clear_auth_cookies(response)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))


@router.post("/logout")
async def logout(
    request: Request,
    response: Response,
    db: Session = Depends(get_db)
):
    auth_service = AuthService(db)
    request_id = get_request_id(request)
    raw_refresh = request.cookies.get("refresh_token")

    if raw_refresh:
        try:
            await auth_service.logout(raw_refresh, request_id=request_id)
        except Exception:
            pass

    clear_auth_cookies(response)
    return {"status": "success"}


@router.post("/forgot-password")
@limiter.limit("3/hour")
async def forgot_password(
    request: Request,
    payload: ForgotPasswordRequest,
    db: Session = Depends(get_db)
):
    auth_service = AuthService(db)
    ip_address = request.client.host if request.client else None
    request_id = get_request_id(request)

    try:
        await auth_service.forgot_password(
            email=payload.email,
            ip_address=ip_address,
            request_id=request_id
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return {"status": "success", "message": "Link de redefinição enviado com sucesso."}


@router.post("/reset-password")
@limiter.limit("3/hour")
async def reset_password(
    request: Request,
    payload: ResetPasswordRequest,
    db: Session = Depends(get_db)
):
    auth_service = AuthService(db)
    ip_address = request.client.host if request.client else None
    request_id = get_request_id(request)

    try:
        await auth_service.reset_password(
            raw_token=payload.token,
            new_password=payload.new_password,
            ip_address=ip_address,
            request_id=request_id
        )
        return {"status": "success", "message": "Senha redefinida com sucesso."}
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/verify-email", response_model=UserResponse)
async def verify_email(
    request: Request,
    payload: VerifyEmailRequest,
    db: Session = Depends(get_db)
):
    auth_service = AuthService(db)
    ip_address = request.client.host if request.client else None
    request_id = get_request_id(request)

    try:
        user = await auth_service.verify_email(
            raw_token=payload.token,
            ip_address=ip_address,
            request_id=request_id
        )
        return user
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/resend-verification")
@limiter.limit("3/hour")
async def resend_verification(
    request: Request,
    payload: ForgotPasswordRequest,
    db: Session = Depends(get_db)
):
    auth_service = AuthService(db)
    request_id = get_request_id(request)

    try:
        await auth_service.resend_verification(email=payload.email, request_id=request_id)
        return {"status": "success", "message": "Novo link de verificacao enviado se aplicavel."}
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/me", response_model=UserResponse)
async def get_me(
    current_user: User = Depends(get_current_user)
):
    return current_user


@router.get("/sessions", response_model=List[ActiveSessionResponse])
async def get_sessions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    auth_service = AuthService(db)
    sessions = auth_service.session_repo.get_user_sessions(current_user.id)
    return sessions


@router.post("/sessions/{session_id}/revoke")
async def revoke_session(
    session_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    auth_service = AuthService(db)
    session = auth_service.session_repo.get_session_by_id(session_id)
    if not session or session.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sessao nao encontrada")

    token_obj = auth_service.refresh_repo.db.get(RefreshToken, session.refresh_token_id)
    if token_obj:
        auth_service.refresh_repo.revoke_refresh_token(token_obj)
    
    auth_service.session_repo.delete_session(session)
    return {"status": "success"}
