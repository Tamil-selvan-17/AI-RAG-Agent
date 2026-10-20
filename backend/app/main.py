"""
Application entrypoint.

Wires together the FastAPI app, CORS, routers, exception handling, and
startup/shutdown lifecycle hooks (e.g. ensuring the Qdrant collection exists).
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api import auth_routes, chat_routes, health_routes, upload_routes
from app.core.auth import require_auth
from app.core.config import get_settings
from app.core.logging import logger
from app.services.container import build_services

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.app_name} ({settings.app_env})")
    services = build_services()
    app.state.services = services
    try:
        await services.qdrant.ensure_collection()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Qdrant not reachable at startup, will retry on first request: {exc}")
    try:
        await services.rag.recover_interrupted_ingestions()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Could not check for interrupted uploads at startup: {exc}")
    yield
    logger.info("Shutting down AI RAG Agent")
    await services.aclose()


app = FastAPI(
    title=settings.app_name,
    description="Production-ready local RAG AI agent powered by Ollama, Qdrant, and Redis.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    # Auth uses a Bearer header, not cookies, so credentials mode isn't needed
    # (and must not be combined with a wildcard origin).
    allow_credentials=settings.cors_origin_list != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    logger.warning(f"HTTPException on {request.method} {request.url.path}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code, content={"detail": exc.detail}, headers=exc.headers
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(f"Unhandled exception on {request.method} {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected internal error occurred."},
    )


app.include_router(health_routes.router)
app.include_router(auth_routes.router)
app.include_router(upload_routes.router, dependencies=[Depends(require_auth)])
app.include_router(chat_routes.router, dependencies=[Depends(require_auth)])

# Serve the plain HTML/CSS/JS frontend directly from the backend for convenience.
# Try common locations so this works both for local `uvicorn` runs (cwd=backend/)
# and inside the Docker container (where the frontend is mounted at /frontend).
_candidate_frontend_dirs = [
    Path(__file__).resolve().parent.parent.parent / "frontend",  # local: repo_root/frontend
    Path("/frontend"),  # docker: mounted volume
]
_frontend_dir = next((p for p in _candidate_frontend_dirs if p.is_dir()), None)

if _frontend_dir:
    app.mount("/app", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")
    logger.info(f"Serving frontend from {_frontend_dir}")
else:
    logger.warning("Frontend directory not found; static file serving disabled")


@app.get("/")
async def root() -> dict:
    return {
        "name": settings.app_name,
        "status": "running",
        "docs": "/docs",
        "frontend": "/app",
    }
