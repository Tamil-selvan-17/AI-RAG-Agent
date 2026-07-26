"""Unit tests for EmbeddingService with a mocked OllamaService."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.embedding_service import EmbeddingService  # noqa: E402


@pytest.mark.asyncio
async def test_embed_text_returns_vector():
    mock_backend = AsyncMock()
    mock_backend.generate_embedding.return_value = [0.1, 0.2, 0.3]

    service = EmbeddingService(backend=mock_backend)
    vector = await service.embed_text("hello world")

    assert vector == [0.1, 0.2, 0.3]
    mock_backend.generate_embedding.assert_awaited_once_with("hello world")


@pytest.mark.asyncio
async def test_embed_batch_preserves_order():
    mock_backend = AsyncMock()
    mock_backend.generate_embeddings_batch.return_value = [[0.1], [0.2], [0.3]]

    service = EmbeddingService(backend=mock_backend)
    vectors = await service.embed_batch(["a", "b", "c"])

    assert vectors == [[0.1], [0.2], [0.3]]
