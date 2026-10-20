"""
Ollama service.

Thin async client around the local Ollama REST API for two purposes:
- generating embeddings (bge-m3) via /api/embed
- generating chat completions (qwen2.5:7b) via /api/chat

All inference stays fully local; no external API calls are made.
"""

import base64
import json
from contextlib import asynccontextmanager

import httpx
from fastapi import HTTPException, status

from app.core.config import get_settings
from app.core.logging import logger
from app.services.provider_errors import (
    TransientProviderError,
    provider_retry,
    safe_read_error_body,
)

# Texts sent per /api/embed call (it accepts a list of inputs natively).
_EMBED_BATCH_SIZE = 32


class OllamaService:
    """Async client for local Ollama embedding and chat generation."""

    def __init__(self) -> None:
        settings = get_settings()
        self.base_url = settings.ollama_url.rstrip("/")
        self.embed_model = settings.ollama_embed_model
        self.chat_model = settings.ollama_chat_model
        self.vision_model = settings.ollama_vision_model or settings.ollama_chat_model
        self.timeout = settings.ollama_timeout_seconds
        self._http: httpx.AsyncClient | None = None

    @asynccontextmanager
    async def _client(self):
        """Yield this service's shared HTTP client (created on first use, reused after).

        Reusing one client keeps connections to Ollama alive between requests
        instead of opening a new TCP connection for every embedding/chat call.
        """
        if self._http is None or self._http.is_closed:
            # Read timeout applies per network read. For streaming responses this means
            # "seconds allowed between tokens", not total generation time -- so a slow
            # multi-minute answer no longer fails as long as tokens keep arriving.
            timeout = httpx.Timeout(connect=10.0, read=self.timeout, write=30.0, pool=10.0)
            self._http = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)
        yield self._http

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    def _unavailable(self, exc: httpx.TransportError) -> TransientProviderError:
        logger.error(f"Cannot connect to Ollama at {self.base_url}: {exc}")
        return TransientProviderError(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"Ollama service unavailable at {self.base_url}. Is Ollama running?",
        )

    @provider_retry(attempts=3)
    async def _embed(self, inputs: list[str]) -> list[list[float]]:
        payload = {"model": self.embed_model, "input": inputs}
        try:
            async with self._client() as client:
                response = await client.post("/api/embed", json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.ConnectError as exc:
            raise self._unavailable(exc) from exc
        except httpx.HTTPStatusError as exc:
            logger.error(f"Ollama embedding request failed: {exc.response.text}")
            detail = f"Ollama embedding request failed: {exc.response.text}"
            if exc.response.status_code >= 500:  # e.g. model still loading -- worth retrying
                raise TransientProviderError(status.HTTP_502_BAD_GATEWAY, detail) from exc
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=detail) from exc
        except httpx.TransportError as exc:  # read errors, dropped connections, etc.
            raise self._unavailable(exc) from exc

        embeddings = data.get("embeddings") or []
        if len(embeddings) != len(inputs) or not all(embeddings):
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Ollama returned an empty or incomplete embedding response",
            )
        return embeddings

    async def generate_embedding(self, text: str) -> list[float]:
        """Generate an embedding vector for a single piece of text using bge-m3."""
        if not text or not text.strip():
            raise ValueError("Cannot generate an embedding for empty text")
        return (await self._embed([text]))[0]

    async def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for many texts, several per request (order preserved)."""
        vectors: list[list[float]] = []
        for i in range(0, len(texts), _EMBED_BATCH_SIZE):
            vectors.extend(await self._embed(texts[i : i + _EMBED_BATCH_SIZE]))
        return vectors

    async def stream_chat_response(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        history: list[tuple[str, str]] | None = None,
    ):
        """Yield answer tokens one at a time as Ollama generates them.

        Unlike generate_chat_response (which waits and returns one final string),
        this lets the API route forward each token to the browser immediately as
        an HTTP chunk. That keeps bytes flowing to the client the whole time, which
        is what actually prevents 504 Gateway Timeout errors from reverse proxies
        (Render, nginx, etc.) that give up on a request producing no output for
        too long -- a real streamed response never looks "silent" to them.

        Note: not retried -- a generator that has already yielded partial output
        to a caller can't be safely restarted. Connection failures before any
        token is yielded still raise a clear HTTPException below.
        """
        payload = {
            "model": self.chat_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                *(
                    message
                    for question, answer in history or []
                    for message in (
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer},
                    )
                ),
                {"role": "user", "content": user_prompt},
            ],
            "stream": True,
            "options": {"temperature": temperature},
        }

        try:
            async with self._client() as client:
                async with client.stream("POST", "/api/chat", json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        chunk = json.loads(line)
                        if chunk.get("error"):
                            raise HTTPException(
                                status_code=status.HTTP_502_BAD_GATEWAY,
                                detail=f"Ollama returned an error: {chunk['error']}",
                            )
                        token = chunk.get("message", {}).get("content", "")
                        if token:
                            yield token
                        if chunk.get("done"):
                            break
        except httpx.ConnectError as exc:
            raise self._unavailable(exc) from exc
        except httpx.ReadTimeout as exc:
            logger.error(f"Ollama chat stream stalled (no token within {self.timeout}s): {exc}")
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=(
                    "Ollama stopped sending a response. The model may be overloaded "
                    "or too slow for this hardware. Try again, or increase "
                    "OLLAMA_TIMEOUT_SECONDS in .env."
                ),
            ) from exc
        except httpx.HTTPStatusError as exc:
            body_text = await safe_read_error_body(exc.response)
            logger.error(f"Ollama chat request failed ({exc.response.status_code}): {body_text}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Ollama chat request failed: {body_text}",
            ) from exc
        except httpx.TransportError as exc:  # read errors, dropped connections, etc.
            raise self._unavailable(exc) from exc

    @provider_retry(attempts=2)
    async def generate_chat_response(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        history: list[tuple[str, str]] | None = None,
    ) -> str:
        """Generate a complete chat response by collecting the full token stream.

        Use this when you need the whole answer at once (e.g. tests, or any
        non-streaming caller). For forwarding tokens to a browser as they're
        generated, use stream_chat_response instead.
        """
        content_parts: list[str] = []
        async for token in self.stream_chat_response(system_prompt, user_prompt, temperature, history):
            content_parts.append(token)

        content = "".join(content_parts).strip()
        if not content:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Ollama returned an empty chat response",
            )
        return content

    async def describe_image(self, image_bytes: bytes, mime_type: str) -> str:
        """Generate a text description of an image via Ollama's vision-capable models.

        Uses OLLAMA_VISION_MODEL (falling back to OLLAMA_CHAT_MODEL), which must be
        vision-capable (e.g. llava, moondream, qwen2.5vl) -- a text-only model like
        the default qwen2.5:7b ignores the image and returns an unrelated or empty
        response. Set it up with e.g. `ollama pull llava` and OLLAMA_VISION_MODEL=llava.
        """
        payload = {
            "model": self.vision_model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Describe this image in detail for a search index. Include a "
                        "thorough description of what it shows, a verbatim transcription "
                        "of any visible text/numbers/labels (OCR), and the type of image. "
                        "Be factual -- do not speculate about anything not actually visible."
                    ),
                    "images": [base64.b64encode(image_bytes).decode("ascii")],
                }
            ],
            "stream": False,
        }

        try:
            async with self._client() as client:
                response = await client.post("/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.ConnectError as exc:
            raise self._unavailable(exc) from exc
        except httpx.HTTPStatusError as exc:
            body_text = await safe_read_error_body(exc.response)
            logger.error(f"Ollama image description request failed ({exc.response.status_code}): {body_text}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=(
                    f"Ollama couldn't process the image -- is {self.vision_model} a "
                    f"vision-capable model? Set OLLAMA_VISION_MODEL (e.g. llava). Details: {body_text}"
                ),
            ) from exc
        except httpx.TransportError as exc:  # read errors, dropped connections, etc.
            raise self._unavailable(exc) from exc

        content = data.get("message", {}).get("content", "").strip()
        if not content:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=(
                    f"Ollama returned an empty image description. {self.vision_model} may "
                    "not support vision -- try `ollama pull llava` and set "
                    "OLLAMA_VISION_MODEL=llava, or switch AI_PROVIDER to gemini."
                ),
            )
        return content

    async def check_health(self) -> bool:
        """Return True if the Ollama server responds to a version check."""
        try:
            async with self._client() as client:
                response = await client.get("/api/version")
                return response.status_code == 200
        except httpx.HTTPError:
            return False
