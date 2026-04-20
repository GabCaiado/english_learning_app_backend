from supabase import create_client, Client
from app.config import get_settings


def get_supabase() -> Client:
    """Cria cliente Supabase"""
    settings = get_settings()
    return create_client(
        settings.supabase_url,
        settings.supabase_service_key  # Usar service key no backend
    )


# Cliente global (para uso em services)
supabase: Client = get_supabase()