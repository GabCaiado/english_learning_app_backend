from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers import translate, words, users

settings = get_settings()

app = FastAPI(
    title="English Slang App API",
    description="API para aprendizado de girias em ingles",
    version="1.0.0",
    debug=settings.debug
)

# CORS para o frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rotas
app.include_router(translate.router)
app.include_router(words.router)
app.include_router(users.router)


@app.get("/")
async def root():
    return {
        "message": "English Slang App API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


# Evento de startup
@app.on_event("startup")
async def startup_event():
    """Carrega modelos ML no startup"""
    print("Iniciando servidor...")
    print("Os modelos ML serao carregados na primeira requisicao (lazy loading)")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)