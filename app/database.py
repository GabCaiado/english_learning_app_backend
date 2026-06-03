from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.config import get_settings

settings = get_settings()

engine = create_engine(settings.database_url, pool_pre_ping=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """FastAPI Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_supabase():
    try:
        from supabase import create_client
        if settings.supabase_url and settings.supabase_service_key:
            return create_client(settings.supabase_url, settings.supabase_service_key)
    except Exception:
        pass
    return None

supabase = get_supabase()