import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sqlalchemy import text

from app.config import get_settings
from app.database import engine, get_db
from app.routers import admin, auth, translate, translation_feedback, words, users

settings = get_settings()

# Rate limiter (slowapi)
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler – warm up models on startup."""
    print("Starting English Slang App API...")
    print("ML models will be loaded on first request (lazy loading).")
    yield
    print("Shutting down.")


app = FastAPI(
    title="English Slang App API",
    description="API para aprendizado de girias em ingles",
    version="2.0.0",
    debug=settings.debug,
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "X-CSRF-Token", "X-Request-ID"],
)


# Security Headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response: Response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    if settings.environment == "production":
        response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains; preload"
    return response


# Request-ID middleware
@app.middleware("http")
async def inject_request_id(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    response: Response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# CSRF Double-Submit Cookie protection
CSRF_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
CSRF_EXEMPT_PREFIXES = ("/auth/", "/health", "/docs", "/openapi", "/redoc")

@app.middleware("http")
async def csrf_protect(request: Request, call_next):
    if request.method in CSRF_SAFE_METHODS:
        return await call_next(request)
    if any(request.url.path.startswith(p) for p in CSRF_EXEMPT_PREFIXES):
        return await call_next(request)

    cookie_csrf = request.cookies.get("csrf_token")
    header_csrf = request.headers.get("x-csrf-token")

    if not cookie_csrf or not header_csrf or cookie_csrf != header_csrf:
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={"detail": "CSRF token invalido ou ausente"},
        )
    return await call_next(request)


# Routers
app.include_router(admin.router)
app.include_router(auth.router)
app.include_router(translate.router)
app.include_router(translation_feedback.router)
app.include_router(words.router)
app.include_router(users.router)


# Health endpoints
@app.get("/health/live", tags=["health"])
async def health_live():
    """Liveness probe – returns 200 when the process is running."""
    return {"status": "alive"}


@app.get("/health/ready", tags=["health"])
async def health_ready():
    """
    Readiness probe – checks database connectivity.
    Returns 200 only when the DB is reachable.
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "ready", "database": "ok"}
    except Exception as exc:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not_ready", "database": str(exc)},
        )


@app.get("/", tags=["root"])
async def root():
    return {
        "message": "English Slang App API",
        "version": "2.0.0",
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
