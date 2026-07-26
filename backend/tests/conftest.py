"""Shared pytest fixtures: a FastAPI TestClient with all external services mocked."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import main as main_module  # noqa: E402
from app.api import chat_routes, upload_routes  # noqa: E402
from app.models.document import Document, DocumentStatus  # noqa: E402


@pytest.fixture
def mock_services(monkeypatch):
    """Patch the module-level service singletons used by the routers with async mocks."""

    rag_mock = AsyncMock()

    async def fake_ingest(filename, contents):
        return Document(
            document_id="doc-123",
            filename=filename,
            file_extension=".txt",
            file_size_bytes=len(contents),
            status=DocumentStatus.READY,
            chunk_count=3,
        )

    rag_mock.ingest_document.side_effect = fake_ingest
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

    monkeypatch.setattr(upload_routes, "_rag_service", rag_mock)
    monkeypatch.setattr(chat_routes, "_rag_service", rag_mock)

    return rag_mock


@pytest.fixture
def client(mock_services):
    with TestClient(main_module.app) as test_client:
        yield test_client
