"""Health check endpoint reporting status of all backing services."""

import time

from fastapi import APIRouter

from app.core.config import get_settings
from app.core.logging import logger
from app.services.qdrant_service import QdrantService

router = APIRouter(prefix="/health", tags=["health"])

# Render (and any similar host) pings /health every few seconds to confirm the
# service is alive. Without caching, that would silently burn through a
# rate-limited free-tier AI provider's quota just from routine health pings --
# nothing to do with actual user traffic. Qdrant/memory checks are free and
# instant, so only the AI provider check is cached.
_PROVIDER_HEALTH_CACHE_TTL_SECONDS = 60
_provider_health_cache: dict[str, tuple[float, bool]] = {}


async def _cached_provider_check(provider_name: str, check_fn) -> bool:
    now = time.monotonic()
    cached = _provider_health_cache.get(provider_name)
    if cached and (now - cached[0]) < _PROVIDER_HEALTH_CACHE_TTL_SECONDS:
        return cached[1]

    result = await check_fn()
    _provider_health_cache[provider_name] = (now, result)
    logger.debug(f"Refreshed cached health check for '{provider_name}': {result}")
    return result


@router.get("")
async def health_check() -> dict:
    """Report liveness of the API and connectivity to the AI provider, Qdrant, and memory backend."""
    settings = get_settings()

    if settings.ai_provider == "gemini":
        from app.services.gemini_service import GeminiService

        provider_name = "gemini"
        provider_ok = await _cached_provider_check("gemini", GeminiService().check_health)
    else:
        from app.services.ollama_service import OllamaService

        provider_name = "ollama"
        provider_ok = await _cached_provider_check("ollama", OllamaService().check_health)

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
