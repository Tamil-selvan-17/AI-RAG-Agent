"""
Request ID middleware.

Gives every HTTP request an ID (reusing an incoming `X-Request-ID` header if
a proxy already set one), adds it to every log line written while handling
that request -- including streamed responses and background tasks, which run
inside the same ASGI call -- and returns it in the `X-Request-ID` response
header so a user-reported error can be matched to its log lines.
"""

import re
import uuid

from app.core.logging import logger

_HEADER = b"x-request-id"
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class RequestIdMiddleware:
    """Pure ASGI middleware (unlike BaseHTTPMiddleware it also covers streaming bodies)."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = dict(scope.get("headers") or []).get(_HEADER, b"").decode("latin-1")
        request_id = incoming if _SAFE_ID_RE.match(incoming) else uuid.uuid4().hex[:12]

        async def send_with_request_id(message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers") or [])
                headers.append((_HEADER, request_id.encode("latin-1")))
                message = {**message, "headers": headers}
            await send(message)

        with logger.contextualize(request_id=request_id):
            await self.app(scope, receive, send_with_request_id)
