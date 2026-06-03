from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    database_url: str

    jwt_algorithm: str = "RS256"
    jwt_expire_minutes: int = 5
    jwt_refresh_expire_days: int = 30
    token_pepper: str
    cookie_domain: Optional[str] = None

    # RSA key files (development);
    jwt_private_key_path: str = "keys/private.pem"
    jwt_public_key_path: str = "keys/public.pem"
    # Production: inline PEM strings from secrets manager
    jwt_private_key: Optional[str] = None
    jwt_public_key: Optional[str] = None

    redis_url: str

    # App
    debug: bool = False
    environment: str = "development"
    allowed_origins: str

    # Admin seeding
    admin_email: str
    admin_password: str

    # Supabase (legacy)
    supabase_url: Optional[str] = None
    supabase_anon_key: Optional[str] = None
    supabase_service_key: Optional[str] = None
    supabase_jwt_secret: Optional[str] = None

    openai_api_key: Optional[str] = None

    # ML
    slang_normalizer_model_path: str = "models/slang_normalizer_v4_best"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


@lru_cache()
def get_settings() -> Settings:
    """Returns cached settings singleton"""
    return Settings()