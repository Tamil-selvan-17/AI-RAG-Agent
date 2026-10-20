"""Health check endpoint reporting status of all backing services."""

import time

from fastapi import APIRouter, Depends

from app.core.config import get_settings
from app.core.logging import logger
from app.services.container import Services, get_services

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


@router.get("/live")
async def liveness() -> dict:
    """Instant liveness probe: no calls to any backing service.

    Point your host's health check (Render, Docker HEALTHCHECK, k8s liveness)
    here. Use GET /health for the full dependency check.
    """
    return {"status": "alive"}


@router.get("")
async def health_check(services: Services = Depends(get_services)) -> dict:
    """Report liveness of the API and connectivity to the AI provider, Qdrant, and memory backend."""
    settings = get_settings()

    provider_name = "gemini" if settings.ai_provider == "gemini" else "ollama"
    provider_ok = await _cached_provider_check(provider_name, services.llm.check_health)
    qdrant_ok = await services.qdrant.check_health()

    memory_name = (
        "memory (in-process, non-persistent)" if settings.memory_backend == "memory" else "redis"
    )
    memory_ok = await services.memory.check_health()

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
