"""Tests for the health check caching layer, which prevents frequent platform
health pings (e.g. Render hitting /health every few seconds) from burning
through a rate-limited AI provider's quota."""

import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.api import health_routes  # noqa: E402


@pytest.fixture(autouse=True)
def clear_health_cache():
    """Ensure each test starts with a clean cache."""
    health_routes._provider_health_cache.clear()
    yield
    health_routes._provider_health_cache.clear()


@pytest.mark.asyncio
async def test_repeated_calls_within_ttl_hit_cache_not_provider():
    check_fn = AsyncMock(return_value=True)

    result1 = await health_routes._cached_provider_check("gemini", check_fn)
    result2 = await health_routes._cached_provider_check("gemini", check_fn)
    result3 = await health_routes._cached_provider_check("gemini", check_fn)

    assert result1 is True
    assert result2 is True
    assert result3 is True
    check_fn.assert_awaited_once()  # only the first call actually hit the provider


@pytest.mark.asyncio
async def test_cache_expires_after_ttl():
    check_fn = AsyncMock(return_value=True)

    await health_routes._cached_provider_check("gemini", check_fn)
    check_fn.assert_awaited_once()

    # Simulate the TTL having elapsed by backdating the cache entry.
    expired_time = time.monotonic() - (health_routes._PROVIDER_HEALTH_CACHE_TTL_SECONDS + 1)
    health_routes._provider_health_cache["gemini"] = (expired_time, True)

    await health_routes._cached_provider_check("gemini", check_fn)
    assert check_fn.await_count == 2  # second real call happened after expiry


@pytest.mark.asyncio
async def test_different_providers_cached_independently():
    gemini_check = AsyncMock(return_value=True)
    ollama_check = AsyncMock(return_value=False)

    gemini_result = await health_routes._cached_provider_check("gemini", gemini_check)
    ollama_result = await health_routes._cached_provider_check("ollama", ollama_check)

    assert gemini_result is True
    assert ollama_result is False
    gemini_check.assert_awaited_once()
    ollama_check.assert_awaited_once()
