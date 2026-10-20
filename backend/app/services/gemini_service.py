"""
Gemini service.

Free-tier cloud alternative to OllamaService, used when AI_PROVIDER=gemini.
Implements the exact same method signatures as OllamaService
(generate_embedding, generate_embeddings_batch, generate_chat_response,
check_health) so RagService and EmbeddingService can use either provider
interchangeably without any other code changes.

Uses Google's Generative Language API (https://ai.google.dev), free tier,
via a Google AI Studio API key.
"""

import base64
import json

import httpx
from fastapi import HTTPException, status

from app.core.config import get_settings
from app.core.logging import logger
from app.services.provider_errors import (
    TransientProviderError,
    parse_retry_delay,
    provider_retry,
    safe_read_error_body,
)

_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
# batchEmbedContents accepts at most 100 texts per request.
_EMBED_BATCH_SIZE = 100


def _describe_rate_limit(body_text: str) -> str:
    """Build a specific, actionable message from Google's 429 error body.

    Google's quota error responses include a `RetryInfo` detail with a
    `retryDelay` (e.g. "23s") for short-lived per-minute limits, and often
    name the specific quota metric that was exceeded (e.g. "PerDay" vs
    "PerMinute") in the error message text. Surfacing these specifics is much
    more useful than a generic "wait a minute" guess, since a daily quota
    won't actually clear for hours.
    """
    try:
        data = json.loads(body_text)
    except (json.JSONDecodeError, TypeError):
        data = {}

    message = ""
    retry_delay = None
    if isinstance(data, dict):
        error = data.get("error", {})
        message = error.get("message", "") or ""
        for detail in error.get("details", []):
            if isinstance(detail, dict) and "retryDelay" in detail:
                retry_delay = detail["retryDelay"]

    is_daily = "PerDay" in message or "per day" in message.lower()

    if is_daily:
        return (
            "Gemini's free-tier DAILY request limit has been used up for today. "
            "This resets at midnight Pacific Time -- it won't help to wait just a "
            "few minutes. Check your exact usage at https://aistudio.google.com/usage, "
            "or switch to a different free model in GEMINI_CHAT_MODEL in the meantime."
        )
    if retry_delay:
        return (
            f"Gemini's free-tier rate limit was hit. Google says to retry in "
            f"about {retry_delay}. Please wait and try again."
        )
    return (
        "Gemini's free-tier rate limit was hit (too many requests in a short "
        "time). Please wait a minute and try again, or check "
        "https://aistudio.google.com/usage to see exactly which limit you hit."
    )


def _http_error(exc: httpx.HTTPStatusError, body_text: str, what: str) -> HTTPException:
    """Map a Gemini HTTP error to the right exception (transient errors get retried)."""
    code = exc.response.status_code
    if code == 429:
        detail = _describe_rate_limit(body_text)
        if "DAILY" in detail:
            # Daily quota won't clear for hours -- retrying would only waste time.
            return HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=detail)
        retry_after = None
        try:
            for item in json.loads(body_text).get("error", {}).get("details", []):
                if isinstance(item, dict) and "retryDelay" in item:
                    retry_after = parse_retry_delay(item["retryDelay"])
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass
        return TransientProviderError(status.HTTP_429_TOO_MANY_REQUESTS, detail, retry_after)
    if code >= 500:
        return TransientProviderError(
            status.HTTP_502_BAD_GATEWAY, f"Gemini {what} failed (HTTP {code}): {body_text}"
        )
    return HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Gemini {what} failed: {body_text}"
    )


def _connect_error(exc: httpx.TransportError) -> TransientProviderError:
    logger.error(f"Cannot reach Gemini API: {exc}")
    return TransientProviderError(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "Cannot reach the Gemini API. Check your internet connection.",
    )


class GeminiService:
    """Async client for Google Gemini embeddings and chat generation (free tier)."""

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.gemini_api_key:
            logger.warning(
                "AI_PROVIDER=gemini but GEMINI_API_KEY is empty. "
                "Set it in .env or your hosting provider's environment variables."
            )
        self.api_key = settings.gemini_api_key
        self.chat_model = settings.gemini_chat_model
        self.embed_model = settings.gemini_embed_model
        self.embed_dimensions = settings.gemini_embed_dimensions
        self.timeout = settings.gemini_timeout_seconds

    def _client(self) -> httpx.AsyncClient:
        timeout = httpx.Timeout(connect=10.0, read=self.timeout, write=30.0, pool=10.0)
        # The key goes in a header rather than the URL so it never ends up in
        # request logs, proxy logs, or exception messages that include the URL.
        return httpx.AsyncClient(
            base_url=_API_BASE, timeout=timeout, headers={"x-goog-api-key": self.api_key}
        )

    def _require_api_key(self) -> None:
        if not self.api_key:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="GEMINI_API_KEY is not set. Add it to your environment variables.",
            )

    def _embed_request(self, text: str, task_type: str) -> dict:
        return {
            "model": f"models/{self.embed_model}",
            "content": {"parts": [{"text": text}]},
            "taskType": task_type,
            "outputDimensionality": self.embed_dimensions,
        }

    async def _post_json(self, url: str, payload: dict, what: str) -> dict:
        try:
            async with self._client() as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                return response.json()
        except httpx.ConnectError as exc:
            raise _connect_error(exc) from exc
        except httpx.HTTPStatusError as exc:
            body_text = await safe_read_error_body(exc.response)
            logger.error(f"Gemini {what} failed ({exc.response.status_code}): {body_text}")
            raise _http_error(exc, body_text, what) from exc
        except httpx.TransportError as exc:  # read errors, dropped connections, etc.
            raise _connect_error(exc) from exc

    @provider_retry()
    async def generate_embedding(self, text: str, task_type: str = "RETRIEVAL_QUERY") -> list[float]:
        """Embed a single text. Defaults to the query task type (used for user questions)."""
        if not text or not text.strip():
            raise ValueError("Cannot generate an embedding for empty text")
        self._require_api_key()

        data = await self._post_json(
            f"/models/{self.embed_model}:embedContent",
            self._embed_request(text, task_type),
            "embedding request",
        )
        values = data.get("embedding", {}).get("values")
        if not values:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Gemini returned an empty embedding response",
            )
        return values

    @provider_retry()
    async def _embed_batch_request(self, texts: list[str]) -> list[list[float]]:
        data = await self._post_json(
            f"/models/{self.embed_model}:batchEmbedContents",
            {"requests": [self._embed_request(t, "RETRIEVAL_DOCUMENT") for t in texts]},
            "batch embedding request",
        )
        embeddings = data.get("embeddings") or []
        vectors = [e.get("values") for e in embeddings]
        if len(vectors) != len(texts) or not all(vectors):
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Gemini returned an incomplete batch embedding response",
            )
        return vectors

    async def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed document chunks, up to 100 per request (document task type)."""
        self._require_api_key()
        vectors: list[list[float]] = []
        for i in range(0, len(texts), _EMBED_BATCH_SIZE):
            vectors.extend(await self._embed_batch_request(texts[i : i + _EMBED_BATCH_SIZE]))
        return vectors

    async def stream_chat_response(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
    ):
        """Yield answer tokens one at a time as Gemini generates them.

        See OllamaService.stream_chat_response for why this matters: forwarding
        each token to the browser as it arrives keeps the HTTP response "alive"
        the whole time, which is what actually prevents 504 Gateway Timeout
        errors from reverse proxies that give up on a silent request.

        Not retried here -- a generator that has already yielded partial output
        to a caller can't be safely restarted.
        """
        self._require_api_key()

        url = f"/models/{self.chat_model}:streamGenerateContent"
        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {"temperature": temperature},
        }

        try:
            async with self._client() as client:
                async with client.stream("POST", url, params={"alt": "sse"}, json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        raw = line[len("data: "):].strip()
                        if not raw or raw == "[DONE]":
                            continue
                        chunk = json.loads(raw)
                        candidates = chunk.get("candidates", [])
                        if not candidates:
                            continue
                        parts = candidates[0].get("content", {}).get("parts", [])
                        for part in parts:
                            text = part.get("text", "")
                            if text:
                                yield text
        except httpx.ConnectError as exc:
            raise _connect_error(exc) from exc
        except httpx.ReadTimeout as exc:
            logger.error(f"Gemini chat stream stalled: {exc}")
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Gemini did not respond in time. Please try again.",
            ) from exc
        except httpx.HTTPStatusError as exc:
            body_text = await safe_read_error_body(exc.response)
            logger.error(f"Gemini chat request failed ({exc.response.status_code}): {body_text}")
            raise _http_error(exc, body_text, "chat request") from exc
        except httpx.TransportError as exc:  # read errors, dropped connections, etc.
            raise _connect_error(exc) from exc

    @provider_retry(attempts=3)
    async def generate_chat_response(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
    ) -> str:
        """Generate a complete chat response by collecting the full token stream.

        Safe to retry (unlike stream_chat_response) because nothing has been
        returned to the caller until the whole answer is collected.
        """
        content_parts: list[str] = []
        async for token in self.stream_chat_response(system_prompt, user_prompt, temperature):
            content_parts.append(token)

        content = "".join(content_parts).strip()
        if not content:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Gemini returned an empty chat response",
            )
        return content

    @provider_retry(attempts=3)
    async def describe_image(self, image_bytes: bytes, mime_type: str) -> str:
        """Generate a rich text description of an image, including any visible text (OCR-like).

        This is what makes uploaded images searchable through the same RAG pipeline
        as text documents: the returned description is chunked and embedded exactly
        like extracted PDF/DOCX text, so a question like "what does the chart show"
        or "what does this screenshot say" can retrieve it normally.
        """
        self._require_api_key()

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": (
                                "Describe this image in detail for a search index. Include: "
                                "(1) a thorough description of what the image shows, "
                                "(2) a verbatim transcription of any visible text, numbers, "
                                "labels, or data (OCR), and (3) the type of image (photo, "
                                "screenshot, chart, diagram, document scan, etc.). Be "
                                "comprehensive and factual -- do not speculate about anything "
                                "not actually visible in the image."
                            )
                        },
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": base64.b64encode(image_bytes).decode("ascii"),
                            }
                        },
                    ],
                }
            ],
        }

        data = await self._post_json(
            f"/models/{self.chat_model}:generateContent", payload, "image description request"
        )

        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Gemini returned an unexpected response format for image description",
            ) from exc

        if not text or not text.strip():
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Gemini returned an empty image description",
            )
        return text.strip()

    async def check_health(self) -> bool:
        """Return True if the Gemini API is reachable and the API key is accepted."""
        if not self.api_key:
            return False
        try:
            async with self._client() as client:
                response = await client.get(f"/models/{self.embed_model}")
                return response.status_code == 200
        except httpx.HTTPError:
            return False
