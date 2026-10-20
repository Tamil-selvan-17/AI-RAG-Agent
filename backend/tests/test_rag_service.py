"""Tests for RagService ingestion, deletion, caching, and follow-up handling."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.memory_store_service import MemoryStoreService  # noqa: E402
from app.services.rag_service import RagService  # noqa: E402


def _chunk_hit(text: str, document_id: str = "doc-1") -> dict:
    return {
        "document_id": document_id,
        "filename": "resume.txt",
        "chunk_id": f"chunk-{abs(hash(text))}",
        "chunk_text": text,
        "score": 0.8,
    }


@pytest.fixture
def rag(tmp_path):
    llm = AsyncMock()
    llm.generate_chat_response.return_value = "Generated answer."
    embedding = AsyncMock()
    embedding.embed_text.return_value = [0.1] * 4
    embedding.embed_batch.side_effect = lambda texts: [[0.1] * 4 for _ in texts]
    qdrant = AsyncMock()
    qdrant.search.return_value = [_chunk_hit("Tamilselvan works at Katalyst since June 2022.")]

    service = RagService(
        llm_service=llm,
        embedding_service=embedding,
        qdrant_service=qdrant,
        redis_service=MemoryStoreService(),
    )
    service.settings.upload_dir = str(tmp_path)
    return service


@pytest.mark.asyncio
async def test_same_name_uploads_are_stored_and_deleted_separately(rag):
    first = await rag.ingest_document("resume.txt", b"first version of the resume")
    second = await rag.ingest_document("resume.txt", b"second version of the resume")

    first_path = rag.settings.upload_dir_path / first.stored_filename
    second_path = rag.settings.upload_dir_path / second.stored_filename
    assert first_path != second_path
    assert first.filename == second.filename == "resume.txt"

    await rag.delete_document(second.document_id)

    assert first_path.exists(), "deleting the second upload must not remove the first one's file"
    assert not second_path.exists()


@pytest.mark.asyncio
async def test_answer_cache_is_cleared_when_documents_change(rag):
    await rag.answer_question("Where does he work?", conversation_id=None)
    await rag.answer_question("Where does he work?", conversation_id=None)
    assert rag.llm_service.generate_chat_response.await_count == 1  # second one was cached

    document = await rag.ingest_document("notes.txt", b"some new notes")
    await rag.answer_question("Where does he work?", conversation_id=None)
    assert rag.llm_service.generate_chat_response.await_count == 2  # upload cleared the cache

    await rag.delete_document(document.document_id)
    await rag.answer_question("Where does he work?", conversation_id=None)
    assert rag.llm_service.generate_chat_response.await_count == 3  # delete cleared it too


@pytest.mark.asyncio
async def test_follow_up_question_is_rewritten_before_retrieval(rag):
    conversation_id, _, _ = await rag.answer_question("Where does Tamilselvan work?", None)

    rag.llm_service.generate_chat_response.side_effect = [
        "How many years has Tamilselvan worked at Katalyst?",  # the rewrite
        "About 4 years.",  # the answer
    ]
    _, answer, _ = await rag.answer_question("How many years has he worked there?", conversation_id)

    assert answer == "About 4 years."
    rag.embedding_service.embed_text.assert_awaited_with(
        "How many years has Tamilselvan worked at Katalyst?"
    )


@pytest.mark.asyncio
async def test_follow_up_answers_are_not_served_from_cache(rag):
    conv_a, _, _ = await rag.answer_question("Where does he work?", None)
    conv_b, _, _ = await rag.answer_question("Tell me about Priya", None)
    rag.settings.rag_query_rewrite = False
    try:
        await rag.answer_question("What about education?", conv_a)
        await rag.answer_question("What about education?", conv_b)
    finally:
        rag.settings.rag_query_rewrite = True

    # 2 first questions + 2 follow-ups, none of the follow-ups reused from cache.
    assert rag.llm_service.generate_chat_response.await_count == 4


@pytest.mark.asyncio
async def test_stream_answer_yields_tokens_then_done(rag):
    async def fake_stream(system_prompt, user_prompt):
        for token in ["Works ", "at ", "Katalyst."]:
            yield token

    rag.llm_service.stream_chat_response = fake_stream
    events = [e async for e in rag.stream_answer("Where does he work?", None)]

    assert [e["content"] for e in events if e["type"] == "token"] == ["Works ", "at ", "Katalyst."]
    assert events[-1]["type"] == "done"
    assert len(events[-1]["sources"]) == 1


@pytest.mark.asyncio
async def test_no_matching_chunks_returns_not_found_without_llm_call(rag):
    rag.qdrant_service.search.return_value = []
    _, answer, sources = await rag.answer_question("What is the capital of Mars?", None)
    assert "couldn't find" in answer
    assert sources == []
    rag.llm_service.generate_chat_response.assert_not_awaited()
