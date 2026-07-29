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
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import get_settings
from app.core.logging import logger

_RETRYABLE_EXCEPTIONS = (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError)
_API_BASE = "https://generativelanguage.googleapis.com/v1beta"


async def _safe_read_error_body(response: httpx.Response) -> str:
    """Read an error response's body, even if it came from a streaming request.

    A response obtained via `client.stream(...)` doesn't have its body buffered
    automatically -- calling `.text` directly on it raises httpx.ResponseNotRead
    if the body hasn't been read yet, which would otherwise crash this exact
    error-handling code instead of reporting the real error. This reads it
    safely first, falling back to just the status code if reading fails too.
    """
    try:
        await response.aread()
        return response.text
    except Exception:  # noqa: BLE001
        return f"HTTP {response.status_code}"


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
        return httpx.AsyncClient(base_url=_API_BASE, timeout=timeout)

    def _require_api_key(self) -> None:
        if not self.api_key:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="GEMINI_API_KEY is not set. Add it to your environment variables.",
            )

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
    )
    async def generate_embedding(self, text: str) -> list[float]:
        """Generate an embedding vector for a single piece of text using Gemini."""
        if not text or not text.strip():
            raise ValueError("Cannot generate an embedding for empty text")
        self._require_api_key()

        url = f"/models/{self.embed_model}:embedContent"
        payload = {
            "model": f"models/{self.embed_model}",
            "content": {"parts": [{"text": text}]},
            "outputDimensionality": self.embed_dimensions,
        }

        try:
            async with self._client() as client:
                response = await client.post(
                    url, params={"key": self.api_key}, json=payload
                )
                response.raise_for_status()
                data = response.json()
        except httpx.ConnectError as exc:
            logger.error(f"Cannot reach Gemini API: {exc}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Cannot reach the Gemini API. Check your internet connection.",
            ) from exc
        except httpx.HTTPStatusError as exc:
            logger.error(f"Gemini embedding request failed: {exc.response.text}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Gemini embedding request failed: {exc.response.text}",
            ) from exc

        values = data.get("embedding", {}).get("values")
        if not values:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Gemini returned an empty embedding response",
            )
        return values

    async def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts sequentially."""
        vectors: list[list[float]] = []
        for text in texts:
            vectors.append(await self.generate_embedding(text))
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

        Not wrapped in @retry -- tenacity can't safely retry a generator that
        has already yielded partial output to a caller.
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
                async with client.stream(
                    "POST", url, params={"key": self.api_key, "alt": "sse"}, json=payload
                ) as response:
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
            logger.error(f"Cannot reach Gemini API: {exc}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Cannot reach the Gemini API. Check your internet connection.",
            ) from exc
        except httpx.ReadTimeout as exc:
            logger.error(f"Gemini chat stream stalled: {exc}")
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Gemini did not respond in time. Please try again.",
            ) from exc
        except httpx.HTTPStatusError as exc:
            # In streaming mode the response body isn't buffered automatically --
            # accessing exc.response.text directly here raises httpx.ResponseNotRead
            # and masks the real error. Read it safely first.
            body_text = await _safe_read_error_body(exc.response)
            logger.error(f"Gemini chat request failed ({exc.response.status_code}): {body_text}")

            if exc.response.status_code == 429:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=_describe_rate_limit(body_text),
                ) from exc
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Gemini chat request failed: {body_text}",
            ) from exc

    @retry(
        reraise=True,
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type((httpx.ConnectError, httpx.RemoteProtocolError)),
    )
    async def generate_chat_response(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
    ) -> str:
        """Generate a complete chat response by collecting the full token stream.

        Use this when you need the whole answer at once. For forwarding tokens
        to a browser as they're generated, use stream_chat_response instead.
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

    async def describe_image(self, image_bytes: bytes, mime_type: str) -> str:
        """Generate a rich text description of an image, including any visible text (OCR-like).

        This is what makes uploaded images searchable through the same RAG pipeline
        as text documents: the returned description is chunked and embedded exactly
        like extracted PDF/DOCX text, so a question like "what does the chart show"
        or "what does this screenshot say" can retrieve it normally.
        """
        self._require_api_key()

        url = f"/models/{self.chat_model}:generateContent"
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

        try:
            async with self._client() as client:
                response = await client.post(url, params={"key": self.api_key}, json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.ConnectError as exc:
            logger.error(f"Cannot reach Gemini API: {exc}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Cannot reach the Gemini API. Check your internet connection.",
            ) from exc
        except httpx.HTTPStatusError as exc:
            body_text = await _safe_read_error_body(exc.response)
            logger.error(f"Gemini image description request failed ({exc.response.status_code}): {body_text}")
            if exc.response.status_code == 429:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=_describe_rate_limit(body_text),
                ) from exc
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Gemini image description request failed: {body_text}",
            ) from exc

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
                response = await client.get(
                    f"/models/{self.embed_model}", params={"key": self.api_key}
                )
                return response.status_code == 200
        except httpx.HTTPError:
            return False
