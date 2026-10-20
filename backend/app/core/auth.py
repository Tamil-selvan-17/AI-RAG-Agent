"""
Single-user authentication.

There is no user database: one admin account is configured through
ADMIN_USERNAME / ADMIN_PASSWORD. A successful login returns a signed,
expiring token (HMAC-SHA256 over "username|expiry"), which the frontend
sends back as `Authorization: Bearer <token>`. Tokens are stateless, so
"logout" simply means the browser discards its token.
"""

import base64
import hashlib
import hmac
import secrets
import time
from collections import defaultdict

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import get_settings
from app.core.logging import logger

_MAX_FAILED_ATTEMPTS = 5
_LOCKOUT_SECONDS = 300

_bearer_scheme = HTTPBearer(auto_error=False)
_failed_attempts: dict[str, list[float]] = defaultdict(list)
_generated_secret: bytes | None = None


def _secret_key() -> bytes:
    global _generated_secret
    configured = get_settings().auth_secret_key
    if configured:
        return configured.encode("utf-8")
    if _generated_secret is None:
        _generated_secret = secrets.token_bytes(32)
        logger.warning(
            "AUTH_SECRET_KEY is not set: using a random key for this process. "
            "Users will be logged out whenever the backend restarts."
        )
    return _generated_secret


def _sign(message: str) -> str:
    digest = hmac.new(_secret_key(), message.encode("utf-8"), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def create_token(username: str) -> tuple[str, int]:
    """Return a signed token for the user and its expiry (unix seconds)."""
    expires_at = int(time.time()) + get_settings().auth_session_hours * 3600
    payload = f"{username}|{expires_at}"
    encoded = base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii").rstrip("=")
    return f"{encoded}.{_sign(payload)}", expires_at


def verify_token(token: str) -> str | None:
    """Return the username if the token is authentic and unexpired, else None."""
    try:
        encoded, signature = token.split(".", 1)
        padded = encoded + "=" * (-len(encoded) % 4)
        payload = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
        username, expires_at = payload.rsplit("|", 1)
    except (ValueError, UnicodeDecodeError):
        return None

    if not hmac.compare_digest(signature, _sign(payload)):
        return None
    if int(expires_at) < time.time():
        return None
    if username != get_settings().admin_username:
        return None
    return username


def _client_key(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def check_login_allowed(request: Request) -> None:
    """Reject login attempts from a client that failed too many times recently."""
    key = _client_key(request)
    now = time.time()
    recent = [t for t in _failed_attempts[key] if now - t < _LOCKOUT_SECONDS]
    _failed_attempts[key] = recent
    if len(recent) >= _MAX_FAILED_ATTEMPTS:
        retry_in = int(_LOCKOUT_SECONDS - (now - recent[0])) + 1
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many failed login attempts. Try again in {retry_in} seconds.",
        )


def verify_credentials(request: Request, username: str, password: str) -> bool:
    settings = get_settings()
    valid = hmac.compare_digest(username.encode("utf-8"), settings.admin_username.encode("utf-8"))
    valid &= hmac.compare_digest(password.encode("utf-8"), settings.admin_password.encode("utf-8"))

    key = _client_key(request)
    if valid:
        _failed_attempts.pop(key, None)
    else:
        _failed_attempts[key].append(time.time())
        logger.warning(f"Failed login attempt for user '{username[:50]}' from {key}")
    return valid


async def require_auth(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> str:
    """FastAPI dependency: return the logged-in username or raise 401."""
    username = verify_token(credentials.credentials) if credentials else None
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not logged in or session expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return username
