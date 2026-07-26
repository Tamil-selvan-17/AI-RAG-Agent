"""Health check endpoint reporting status of all backing services."""

from fastapi import APIRouter

from app.core.config import get_settings
from app.services.qdrant_service import QdrantService

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
async def health_check() -> dict:
    """Report liveness of the API and connectivity to the AI provider, Qdrant, and memory backend."""
    settings = get_settings()

    if settings.ai_provider == "gemini":
        from app.services.gemini_service import GeminiService

        provider_name = "gemini"
        provider_ok = await GeminiService().check_health()
    else:
        from app.services.ollama_service import OllamaService

        provider_name = "ollama"
        provider_ok = await OllamaService().check_health()

    qdrant_ok = await QdrantService().check_health()

    if settings.memory_backend == "memory":
        from app.services.memory_store_service import MemoryStoreService

        memory_name = "memory (in-process, non-persistent)"
        memory_ok = await MemoryStoreService().check_health()
    else:
        from app.services.redis_service import RedisService

        memory_name = "redis"
        memory_ok = await RedisService().check_health()

    overall = "healthy" if all([provider_ok, qdrant_ok, memory_ok]) else "degraded"

    return {
        "status": overall,
        "services": {
            "api": "healthy",
            provider_name: "healthy" if provider_ok else "unreachable",
            "qdrant": "healthy" if qdrant_ok else "unreachable",
            memory_name: "healthy" if memory_ok else "unreachable",
        },
    }
