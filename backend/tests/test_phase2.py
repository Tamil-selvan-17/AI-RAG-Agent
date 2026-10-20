"""Tests for background ingestion, rate limiting, shared services, and store bounds."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core import rate_limit  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.models.document import DocumentStatus  # noqa: E402
from app.services import memory_store_service  # noqa: E402
from app.services.memory_store_service import MemoryStoreService  # noqa: E402
from app.services.rag_service import RagService  # noqa: E402


@pytest.fixture
def rag(tmp_path):
    llm = AsyncMock()
    embedding = AsyncMock()
    embedding.embed_batch.side_effect = lambda texts: [[0.1] * 4 for _ in texts]
    service = RagService(
        llm_service=llm,
        embedding_service=embedding,
        qdrant_service=AsyncMock(),
        redis_service=MemoryStoreService(),
    )
    service.settings.upload_dir = str(tmp_path)
    return service


# ---- Background ingestion ------------------------------------------------- #


@pytest.mark.asyncio
async def test_register_then_background_process_marks_ready(rag):
    document = await rag.register_upload("notes.txt", b"Quarterly revenue grew 12 percent.")
    assert (await rag.redis_service.get_document(document.document_id)).status == DocumentStatus.PROCESSING

    await rag.process_document_in_background(document, b"Quarterly revenue grew 12 percent.")

    stored = await rag.redis_service.get_document(document.document_id)
    assert stored.status == DocumentStatus.READY
    assert stored.chunk_count == 1


@pytest.mark.asyncio
async def test_background_failure_is_recorded_not_raised(rag):
    document = await rag.register_upload("empty.txt", b"   ")
    await rag.process_document_in_background(document, b"   ")  # must not raise

    stored = await rag.redis_service.get_document(document.document_id)
    assert stored.status == DocumentStatus.FAILED
    assert stored.error_message


@pytest.mark.asyncio
async def test_document_deleted_during_processing_is_not_resurrected(rag):
    document = await rag.register_upload("notes.txt", b"some text to index")

    async def delete_while_embedding(texts):
        await rag.redis_service.delete_document(document.document_id)
        return [[0.1] * 4 for _ in texts]

    rag.embedding_service.embed_batch.side_effect = delete_while_embedding
    await rag.process_document_in_background(document, b"some text to index")

    assert await rag.redis_service.get_document(document.document_id) is None
    rag.qdrant_service.delete_document.assert_awaited_once_with(document.document_id)


@pytest.mark.asyncio
async def test_interrupted_uploads_are_marked_failed_on_startup(rag):
    document = await rag.register_upload("big.pdf", b"%PDF-fake")  # left PROCESSING
    assert await rag.recover_interrupted_ingestions() == 1

    stored = await rag.redis_service.get_document(document.document_id)
    assert stored.status == DocumentStatus.FAILED
    assert "restart" in stored.error_message


# ---- Rate limiting --------------------------------------------------------- #


def test_chat_is_rate_limited_per_client(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "chat_rate_limit_per_minute", 3)
    statuses = [client.post("/api/chat", json={"question": "q"}).status_code for _ in range(4)]
    assert statuses == [200, 200, 200, 429]


def test_rate_limit_zero_means_unlimited(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "chat_rate_limit_per_minute", 0)
    assert all(client.post("/api/chat", json={"question": "q"}).status_code == 200 for _ in range(30))


def test_sliding_window_frees_slots_after_window(monkeypatch):
    limiter = rate_limit.SlidingWindowLimiter(window_seconds=60)
    now = [1000.0]
    monkeypatch.setattr(rate_limit.time, "monotonic", lambda: now[0])

    assert limiter.check("ip", 2) is None
    assert limiter.check("ip", 2) is None
    assert limiter.check("ip", 2) is not None  # third within the window is refused
    now[0] += 61
    assert limiter.check("ip", 2) is None  # window passed


# ---- Health / shared services ---------------------------------------------- #


def test_liveness_endpoint_is_public_and_instant(anon_client):
    response = anon_client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_routes_share_one_memory_store(monkeypatch):
    """With MEMORY_BACKEND=memory, every route must see the same store (previously
    each router built its own, so uploads couldn't clear the chat route's cache)."""
    from app.services import container

    monkeypatch.setattr(get_settings(), "memory_backend", "memory")
    services = container.build_services()
    assert services.rag.redis_service is services.memory
    assert services.rag.llm_service is services.llm
    assert services.rag.qdrant_service is services.qdrant


# ---- In-memory store bounds ------------------------------------------------- #


@pytest.mark.asyncio
async def test_memory_store_evicts_oldest_conversations(monkeypatch):
    from app.models.chat import ChatTurn

    monkeypatch.setattr(memory_store_service, "_MAX_CONVERSATIONS", 2)
    store = MemoryStoreService()
    for conv in ("a", "b", "c"):
        await store.append_chat_turn(ChatTurn(conversation_id=conv, question="q", answer="a"))

    assert await store.get_conversation_history("a") == []
    assert len(await store.get_conversation_history("c")) == 1


@pytest.mark.asyncio
async def test_memory_store_cache_expires(monkeypatch):
    store = MemoryStoreService()
    now = [500.0]
    monkeypatch.setattr(memory_store_service.time, "monotonic", lambda: now[0])

    await store.cache_answer("q", {"answer": "x", "sources": []})
    assert await store.get_cached_answer("q") is not None
    now[0] += memory_store_service._CACHE_TTL_SECONDS + 1
    assert await store.get_cached_answer("q") is None
