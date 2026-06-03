import uuid
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.auth import get_current_user
from app.models.user import User
from app.schemas.user import UserProfile, UserUpdate
from app.schemas.auth import ChangePasswordRequest
from app.services.user_service import UserService

router = APIRouter(prefix="/users", tags=["users"])

def get_request_id(request: Request) -> str:
    req_id = request.headers.get("x-request-id")
    if not req_id:
        req_id = str(uuid.uuid4())
    return req_id

@router.get("/me", response_model=UserProfile)
async def get_me(current_user: User = Depends(get_current_user)):
    """
    Returns user profile.
    """
    return current_user

@router.patch("/me", response_model=UserProfile)
async def update_me(
    request: Request,
    data: UserUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Updates user profile.
    """
    user_service = UserService(db)
    
    update_data = data.model_dump(exclude_unset=True)
    if not update_data:
        raise HTTPException(status_code=400, detail="Nenhum dado para atualizar")
        
    ip_address = request.client.host if request.client else None
    request_id = get_request_id(request)
    
    try:
        updated_user = user_service.update_user_profile(
            user_id=current_user.id,
            update_data=update_data,
            ip_address=ip_address,
            request_id=request_id
        )
        return updated_user
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/me/change-password")
async def change_password(
    request: Request,
    data: ChangePasswordRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Changes user password and revokes all active sessions."""
    user_service = UserService(db)
    ip_address = request.client.host if request.client else None
    request_id = get_request_id(request)
    user_service.change_password(
        user=current_user,
        new_password=data.new_password,
        ip_address=ip_address,
        request_id=request_id
    )
    return {"status": "success", "message": "Senha alterada com sucesso"}


@router.delete("/me")
async def delete_me(
    request: Request,
    response: Response,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Deletes user account and revokes tokens/sessions.
    """
    user_service = UserService(db)
    ip_address = request.client.host if request.client else None
    request_id = get_request_id(request)
    
    user_service.delete_user_account(
        user_id=current_user.id,
        ip_address=ip_address,
        request_id=request_id
    )
    
    # Clear cookies
    from app.config import get_settings
    settings = get_settings()
    domain = settings.cookie_domain if settings.cookie_domain else None
    response.delete_cookie("access_token", path="/", domain=domain)
    response.delete_cookie("refresh_token", path="/", domain=domain)
    response.delete_cookie("csrf_token", path="/", domain=domain)
    
    return {"status": "success", "message": "Dados do usuario excluidos com sucesso"}

