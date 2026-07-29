"""Tests for ingesting image uploads via the LLM's vision/description capability."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.memory_store_service import MemoryStoreService  # noqa: E402
from app.services.rag_service import RagService  # noqa: E402


@pytest.mark.asyncio
async def test_image_upload_routes_through_describe_image(tmp_path):
    llm_mock = AsyncMock()
    llm_mock.describe_image.return_value = "A screenshot showing Total Revenue: $45,231, Q3 2026."
    embedding_mock = AsyncMock()
    embedding_mock.embed_batch.return_value = [[0.1] * 768]
    qdrant_mock = AsyncMock()

    memory = MemoryStoreService()
    rag = RagService(
        llm_service=llm_mock,
        embedding_service=embedding_mock,
        qdrant_service=qdrant_mock,
        redis_service=memory,
    )
    rag.settings.upload_dir = str(tmp_path)

    fake_image_bytes = b"\xff\xd8\xff\xe0FAKEJPEGDATA"
    document = await rag.ingest_document("chart.png", fake_image_bytes)

    assert document.status.value == "ready"
    assert document.chunk_count == 1
    llm_mock.describe_image.assert_awaited_once()
    call_args = llm_mock.describe_image.call_args
    assert call_args[0][0] == fake_image_bytes
    assert call_args[0][1] == "image/png"


@pytest.mark.asyncio
async def test_image_upload_fails_cleanly_on_empty_description(tmp_path):
    llm_mock = AsyncMock()
    llm_mock.describe_image.return_value = ""
    embedding_mock = AsyncMock()
    qdrant_mock = AsyncMock()

    memory = MemoryStoreService()
    rag = RagService(
        llm_service=llm_mock,
        embedding_service=embedding_mock,
        qdrant_service=qdrant_mock,
        redis_service=memory,
    )
    rag.settings.upload_dir = str(tmp_path)

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        await rag.ingest_document("blank.jpg", b"fakejpegbytes")

    assert exc_info.value.status_code == 422


@pytest.mark.asyncio
async def test_correct_mime_type_passed_for_each_image_extension(tmp_path):
    llm_mock = AsyncMock()
    llm_mock.describe_image.return_value = "A test image description."
    embedding_mock = AsyncMock()
    embedding_mock.embed_batch.return_value = [[0.1] * 768]
    qdrant_mock = AsyncMock()

    memory = MemoryStoreService()
    rag = RagService(
        llm_service=llm_mock,
        embedding_service=embedding_mock,
        qdrant_service=qdrant_mock,
        redis_service=memory,
    )
    rag.settings.upload_dir = str(tmp_path)

    cases = [
        ("photo.jpg", "image/jpeg"),
        ("photo.jpeg", "image/jpeg"),
        ("screenshot.png", "image/png"),
        ("image.webp", "image/webp"),
    ]

    for filename, expected_mime in cases:
        llm_mock.describe_image.reset_mock()
        await rag.ingest_document(filename, b"fakebytes")
        actual_mime = llm_mock.describe_image.call_args[0][1]
        assert actual_mime == expected_mime, f"{filename} expected {expected_mime}, got {actual_mime}"
