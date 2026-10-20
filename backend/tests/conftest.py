"""Shared pytest fixtures: a FastAPI TestClient with all external services mocked."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import main as main_module  # noqa: E402
from app.core.rate_limit import chat_limiter  # noqa: E402
from app.models.document import Document, DocumentStatus  # noqa: E402
from app.services.container import get_rag_service  # noqa: E402


@pytest.fixture(autouse=True)
def _hermetic_services(monkeypatch):
    """Start the app with mocked backends so tests never contact real Redis/Qdrant/AI."""
    from app.services.container import Services
    from app.services.memory_store_service import MemoryStoreService

    def fake_build_services() -> Services:
        llm = AsyncMock()
        qdrant = AsyncMock()
        return Services(llm=llm, memory=MemoryStoreService(), qdrant=qdrant, rag=AsyncMock())

    monkeypatch.setattr(main_module, "build_services", fake_build_services)


@pytest.fixture(autouse=True)
def _reset_rate_limits():
    chat_limiter.reset()
    yield
    chat_limiter.reset()


@pytest.fixture
def mock_services():
    """Replace the shared RagService the routes depend on with an async mock."""

    rag_mock = AsyncMock()

    async def fake_register(filename, contents):
        return Document(
            document_id="doc-123",
            filename=filename,
            file_extension=".txt",
            file_size_bytes=len(contents),
            status=DocumentStatus.PROCESSING,
        )

    async def fake_process(document, contents):
        return document.model_copy(update={"status": DocumentStatus.READY, "chunk_count": 3})

    rag_mock.register_upload.side_effect = fake_register
    rag_mock.process_document.side_effect = fake_process
    rag_mock.list_documents.return_value = [
        Document(
            document_id="doc-123",
            filename="sample.txt",
            file_extension=".txt",
            file_size_bytes=120,
            status=DocumentStatus.READY,
            chunk_count=3,
        )
    ]
    rag_mock.answer_question.return_value = ("conv-abc", "This is the answer.", [])
    rag_mock.get_history.return_value = []

    async def fake_delete(document_id):
        from fastapi import HTTPException

        if document_id != "doc-123":
            raise HTTPException(status_code=404, detail="No document found")
        return Document(
            document_id="doc-123",
            filename="sample.txt",
            file_extension=".txt",
            file_size_bytes=120,
            status=DocumentStatus.READY,
            chunk_count=3,
        )

    rag_mock.delete_document.side_effect = fake_delete
    rag_mock.delete_all_documents.return_value = 1

    async def fake_stream_answer(question, conversation_id, top_k=None, response_language=None):
        for word in ["This ", "is ", "a ", "streamed ", "answer."]:
            yield {"type": "token", "content": word}
        yield {"type": "done", "conversation_id": "conv-stream", "sources": []}

    rag_mock.stream_answer = fake_stream_answer

    main_module.app.dependency_overrides[get_rag_service] = lambda: rag_mock
    yield rag_mock
    main_module.app.dependency_overrides.pop(get_rag_service, None)


@pytest.fixture
def anon_client(mock_services):
    """A client that has not logged in."""
    with TestClient(main_module.app) as test_client:
        yield test_client


@pytest.fixture
def client(mock_services):
    """A client logged in as the configured admin user."""
    from app.core.config import get_settings

    settings = get_settings()
    with TestClient(main_module.app) as test_client:
        response = test_client.post(
            "/api/auth/login",
            json={"username": settings.admin_username, "password": settings.admin_password},
        )
        assert response.status_code == 200, response.text
        test_client.headers["Authorization"] = f"Bearer {response.json()['access_token']}"
        yield test_client
