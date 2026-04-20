import jwt
from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import get_settings
from app.database import get_supabase

security = HTTPBearer()
settings = get_settings()

def get_current_user(res: HTTPAuthorizationCredentials = Security(security)) -> str:
    """
    Valida o JWT emitido pelo Supabase e retorna o user_id (sub).
    """
    token = res.credentials
    
    import base64
    
    try:
        # Preferimos validar via Supabase Auth API.
        # Isso funciona tanto para JWTs HS256 quanto RS256.
        supabase = get_supabase()
        auth_res = supabase.auth.get_user(token)
        user = getattr(auth_res, "user", None)
        if user and getattr(user, "id", None):
            return user.id

    except Exception as e:
        # Fallback para validacao local de JWT (compatibilidade com setups antigos).
        print(f"[Auth] Fallback local JWT apos falha no get_user: {str(e)}")

    try:
        # Supabase JWT secrets are base64 encoded. 
        # Adding padding if necessary to avoid 'Incorrect padding' error.
        secret_str = settings.supabase_jwt_secret
        padding = len(secret_str) % 4
        if padding > 0:
            secret_str += "=" * (4 - padding)
            
        secret = base64.b64decode(secret_str)
        
        try:
            payload = jwt.decode(
                token, 
                secret, 
                algorithms=["HS256"],
                options={
                    "verify_aud": False,
                    "verify_iat": True,
                    "verify_exp": True
                }
            )
        except jwt.InvalidSignatureError:
            # Fallback for some configurations where the secret is literal
            payload = jwt.decode(
                token, 
                settings.supabase_jwt_secret, 
                algorithms=["HS256"],
                options={
                    "verify_aud": False,
                    "verify_iat": True,
                    "verify_exp": True
                }
            )
        
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Token invalido: 'sub' ausente")
            
        return user_id
        
    except jwt.ExpiredSignatureError:
        print("[Auth] Token expirado")
        raise HTTPException(status_code=401, detail="Token expirado")
    except jwt.InvalidTokenError as e:
        print(f"[Auth] Erro JWT: {str(e)}")
        raise HTTPException(status_code=401, detail=f"Token invalido: {str(e)}")
    except Exception as e:
        print(f"[Auth] Erro inesperado: {str(e)}")
        raise HTTPException(status_code=401, detail="Erro ao validar autenticacao")
