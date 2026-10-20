"""
Shared error handling and retry policy for AI provider clients (Ollama, Gemini).

Provider methods convert low-level httpx errors into HTTPExceptions so API
callers get a clean error message. Errors that are worth retrying (connection
failures, per-minute rate limits, 5xx responses) are raised as
TransientProviderError -- an HTTPException subclass -- so the retry decorator
below can recognize them. (Retrying on raw httpx.ConnectError never fired,
because it was always converted to a plain HTTPException before tenacity saw it.)
"""

import re

import httpx
from fastapi import HTTPException
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

_MAX_RETRY_AFTER_SECONDS = 60
_exponential_wait = wait_exponential(multiplier=1, min=1, max=10)


class TransientProviderError(HTTPException):
    """A provider failure that may succeed if retried shortly."""

    def __init__(self, status_code: int, detail: str, retry_after: float | None = None) -> None:
        super().__init__(status_code=status_code, detail=detail)
        self.retry_after = retry_after


def parse_retry_delay(value: str | None) -> float | None:
    """Parse Google's RetryInfo delay format (e.g. "23s" or "1.5s") into seconds."""
    if not value:
        return None
    match = re.match(r"^\s*([\d.]+)\s*s?\s*$", str(value))
    return float(match.group(1)) if match else None


def _wait(retry_state) -> float:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    retry_after = getattr(exc, "retry_after", None)
    if retry_after:
        return min(retry_after, _MAX_RETRY_AFTER_SECONDS)
    return _exponential_wait(retry_state)


def provider_retry(attempts: int = 4):
    """Retry decorator for non-streaming provider calls."""
    return retry(
        reraise=True,
        stop=stop_after_attempt(attempts),
        wait=_wait,
        retry=retry_if_exception_type(
            (TransientProviderError, httpx.ReadTimeout, httpx.RemoteProtocolError)
        ),
    )


async def safe_read_error_body(response: httpx.Response) -> str:
    """Read an error response's body, even if it came from a streaming request.

    A response obtained via `client.stream(...)` doesn't have its body buffered
    automatically -- calling `.text` directly on it raises httpx.ResponseNotRead.
    This reads it safely first, falling back to just the status code.
    """
    try:
        await response.aread()
        return response.text
    except Exception:  # noqa: BLE001
        return f"HTTP {response.status_code}"
