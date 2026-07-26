"""
Embedding service.

A thin, reusable wrapper dedicated to text -> vector conversion. Automatically
uses either the local Ollama model or the free Gemini API, based on the
AI_PROVIDER setting, so the rest of the app never needs to know which
provider is active.
"""

from typing import Protocol

from app.core.config import get_settings
from app.core.logging import logger


class _EmbeddingBackend(Protocol):
    async def generate_embedding(self, text: str) -> list[float]: ...
    async def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]: ...


def _default_backend() -> _EmbeddingBackend:
    settings = get_settings()
    if settings.ai_provider == "gemini":
        from app.services.gemini_service import GeminiService

        return GeminiService()
    from app.services.ollama_service import OllamaService

    return OllamaService()


class EmbeddingService:
    """Converts text into embedding vectors using the active AI provider."""

    def __init__(self, backend: _EmbeddingBackend | None = None) -> None:
        self._backend = backend or _default_backend()

    async def embed_text(self, text: str) -> list[float]:
        """Input: text. Output: embedding vector."""
        vector = await self._backend.generate_embedding(text)
        logger.debug(f"Generated embedding of dimension {len(vector)}")
        return vector

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Input: list of texts. Output: list of embedding vectors, order preserved."""
        return await self._backend.generate_embeddings_batch(texts)
