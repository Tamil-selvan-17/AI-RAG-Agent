"""
Ollama service.

Thin async client around the local Ollama REST API for two purposes:
- generating embeddings (bge-m3) via /api/embed
- generating chat completions (qwen2.5:7b) via /api/chat

All inference stays fully local; no external API calls are made.
"""

import json

import httpx
from fastapi import HTTPException, status
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import get_settings
from app.core.logging import logger

_RETRYABLE_EXCEPTIONS = (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError)


class OllamaService:
    """Async client for local Ollama embedding and chat generation."""

    def __init__(self) -> None:
        settings = get_settings()
        self.base_url = settings.ollama_url.rstrip("/")
        self.embed_model = settings.ollama_embed_model
        self.chat_model = settings.ollama_chat_model
        self.timeout = settings.ollama_timeout_seconds

    def _client(self) -> httpx.AsyncClient:
        # Read timeout applies per network read. For streaming responses this means
        # "seconds allowed between tokens", not total generation time -- so a slow
        # multi-minute answer no longer fails as long as tokens keep arriving.
        timeout = httpx.Timeout(connect=10.0, read=self.timeout, write=30.0, pool=10.0)
        return httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
    )
    async def generate_embedding(self, text: str) -> list[float]:
        """Generate an embedding vector for a single piece of text using bge-m3."""
        if not text or not text.strip():
            raise ValueError("Cannot generate an embedding for empty text")

        payload = {"model": self.embed_model, "input": text}

        try:
            async with self._client() as client:
                response = await client.post("/api/embed", json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.ConnectError as exc:
            logger.error(f"Cannot connect to Ollama at {self.base_url}: {exc}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Ollama service unavailable at {self.base_url}. Is Ollama running?",
            ) from exc
        except httpx.HTTPStatusError as exc:
            logger.error(f"Ollama embedding request failed: {exc.response.text}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Ollama embedding request failed: {exc.response.text}",
            ) from exc

        embeddings = data.get("embeddings")
        if not embeddings or not embeddings[0]:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Ollama returned an empty embedding response",
            )
        return embeddings[0]

    async def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts sequentially (Ollama has no native batch embed)."""
        vectors: list[list[float]] = []
        for text in texts:
            vector = await self.generate_embedding(text)
            vectors.append(vector)
        return vectors

    async def stream_chat_response(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
    ):
        """Yield answer tokens one at a time as Ollama generates them.

        Unlike generate_chat_response (which waits and returns one final string),
        this lets the API route forward each token to the browser immediately as
        an HTTP chunk. That keeps bytes flowing to the client the whole time, which
        is what actually prevents 504 Gateway Timeout errors from reverse proxies
        (Render, nginx, etc.) that give up on a request producing no output for
        too long -- a real streamed response never looks "silent" to them.

        Note: not wrapped in @retry -- tenacity can't safely retry a generator that
        has already yielded partial output to a caller. Connection failures before
        any token is yielded still raise a clear HTTPException below.
        """
        payload = {
            "model": self.chat_model,
            "messages": [
                {"role": "system", "content": system_prompt},
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
            logger.error(f"Cannot connect to Ollama at {self.base_url}: {exc}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Ollama service unavailable at {self.base_url}. Is Ollama running?",
            ) from exc
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
            logger.error(f"Ollama chat request failed: {exc.response.text}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Ollama chat request failed: {exc.response.text}",
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

        Use this when you need the whole answer at once (e.g. tests, or any
        non-streaming caller). For forwarding tokens to a browser as they're
        generated, use stream_chat_response instead.
        """
        content_parts: list[str] = []
        async for token in self.stream_chat_response(system_prompt, user_prompt, temperature):
            content_parts.append(token)

        content = "".join(content_parts).strip()
        if not content:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Ollama returned an empty chat response",
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
