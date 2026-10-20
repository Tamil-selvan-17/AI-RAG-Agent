"""
In-memory sliding-window rate limiting for chat requests.

Each chat question costs AI provider quota (Gemini's free tier allows only a
handful of requests per minute), so a runaway script or a stuck browser tab
could otherwise burn through it for everyone. Limits are per client IP and
live in this process's memory -- enough for a single-instance deployment.
"""

import math
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request, status

from app.core.config import get_settings

_WINDOW_SECONDS = 60.0


class SlidingWindowLimiter:
    def __init__(self, window_seconds: float = _WINDOW_SECONDS) -> None:
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str, limit: int) -> float | None:
        """Record a hit for `key`. Return None if allowed, else seconds until retry."""
        now = time.monotonic()
        hits = self._hits[key]
        while hits and now - hits[0] >= self.window_seconds:
            hits.popleft()
        if len(hits) >= limit:
            return self.window_seconds - (now - hits[0])
        hits.append(now)
        return None

    def reset(self) -> None:
        self._hits.clear()


chat_limiter = SlidingWindowLimiter()


async def limit_chat_requests(request: Request) -> None:
    """FastAPI dependency: reject chat requests above CHAT_RATE_LIMIT_PER_MINUTE per client."""
    limit = get_settings().chat_rate_limit_per_minute
    if limit <= 0:
        return
    client = request.client.host if request.client else "unknown"
    retry_after = chat_limiter.check(client, limit)
    if retry_after is not None:
        seconds = max(1, math.ceil(retry_after))
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many questions in a short time (limit {limit}/minute). Try again in {seconds}s.",
            headers={"Retry-After": str(seconds)},
        )
