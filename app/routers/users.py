from fastapi import APIRouter, Depends, HTTPException
from app.auth import get_current_user
from app.database import get_supabase
from app.schemas.user import UserProfile, UserUpdate

router = APIRouter(prefix="/users", tags=["users"])

@router.get("/me", response_model=UserProfile)
async def get_me(user_id: str = Depends(get_current_user)):
    """
    Retorna o perfil completo do usuário autenticado.
    """
    supabase = get_supabase()
    
    # Busca dados na tabela profiles
    res = supabase.table("profiles").select("*").eq("id", user_id).single().execute()
    
    if not res.data:
        # Se nao existir perfil ainda, retorna o basico (ou cria um)
        return UserProfile(id=user_id)
        
    return UserProfile(**res.data)

@router.patch("/me", response_model=UserProfile)
async def update_me(data: UserUpdate, user_id: str = Depends(get_current_user)):
    """
    Atualiza dados do perfil do usuário.
    """
    supabase = get_supabase()
    
    # Filtra apenas campos que foram enviados
    update_data = data.model_dump(exclude_unset=True)
    
    if not update_data:
        raise HTTPException(400, "Nenhum dado para atualizar")
        
    res = supabase.table("profiles").update(update_data).eq("id", user_id).execute()
    
    if not res.data:
        raise HTTPException(404, "Perfil nao encontrado")
        
    return UserProfile(**res.data[0])

@router.delete("/me")
async def delete_me(user_id: str = Depends(get_current_user)):
    """
    Exclui permanentemente o perfil do usuário e suas gírias.
    Nota: A exclusao do usuario no Auth requer permissao de admin.
    """
    supabase = get_supabase()
    
    # 1. Deletar gírias associadas ( cascades podem resolver isso se houver FK )
    # Aqui deletamos manualmente por garantia
    supabase.table("user_words").delete().eq("user_id", user_id).execute()
    
    # 2. Deletar perfil
    supabase.table("profiles").delete().eq("id", user_id).execute()
    
    return {"status": "success", "message": "Dados do usuario excluidos com sucesso"}
