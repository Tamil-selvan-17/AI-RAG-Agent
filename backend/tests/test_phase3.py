"""Tests for Phase 3: chunk pages, BM25 rerank, thresholds, duplicates, OCR fallback, chat history."""

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import fitz
import httpx
import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import Settings, get_settings  # noqa: E402
from app.models.document import DocumentStatus  # noqa: E402
from app.services.chunking_service import ChunkingService  # noqa: E402
from app.services.memory_store_service import MemoryStoreService  # noqa: E402
from app.services.rag_service import RagService  # noqa: E402
from app.services.rerank_service import rerank  # noqa: E402


@pytest.fixture
def rag(tmp_path):
    llm = AsyncMock()
    llm.generate_chat_response.return_value = "Answer."
    embedding = AsyncMock()
    embedding.embed_text.return_value = [0.1] * 4
    embedding.embed_batch.side_effect = lambda texts: [[0.1] * 4 for _ in texts]
    qdrant = AsyncMock()
    qdrant.list_documents_from_vectors.return_value = []
    qdrant.search.return_value = []
    service = RagService(
        llm_service=llm, embedding_service=embedding, qdrant_service=qdrant, redis_service=MemoryStoreService()
    )
    service.settings.upload_dir = str(tmp_path)
    return service


def _hit(text, page_start=None, page_end=None):
    metadata = {"page_start": page_start, "page_end": page_end} if page_start else {}
    return {
        "document_id": "doc-1", "filename": "resume.pdf", "chunk_id": f"c-{abs(hash(text))}",
        "chunk_text": text, "score": 0.8, "metadata": metadata,
    }


# ---- Chunking ---------------------------------------------------------------- #


def test_chunks_record_page_ranges_and_never_contain_only_a_marker():
    text = (
        "[Page 1]\n" + "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu. " * 3
        + "\n[Page 2]\n" + "Upsilon phi chi psi omega one two three four five six seven. " * 3
    )
    chunks = ChunkingService(chunk_size_tokens=30, chunk_overlap_tokens=5).chunk_document(text, "d", "f.pdf")

    assert all(c.chunk_text.strip() not in ("[Page 1]", "[Page 2]") for c in chunks)
    assert chunks[0].metadata["page_start"] == 1
    assert chunks[-1].metadata["page_end"] == 2
    assert all(c.metadata["page_start"] <= c.metadata["page_end"] for c in chunks)


def test_text_without_page_markers_has_no_page_metadata():
    chunks = ChunkingService(chunk_size_tokens=100, chunk_overlap_tokens=10).chunk_document("Plain text.", "d", "f.txt")
    assert "page_start" not in chunks[0].metadata


def test_overlap_starts_on_a_word_boundary():
    words = " ".join(f"word{i}" for i in range(200))
    chunks = ChunkingService(chunk_size_tokens=40, chunk_overlap_tokens=7).split_text(words)
    for chunk in chunks[1:]:
        first_word = chunk.split()[0]
        assert first_word.startswith("word") and first_word[4:].isdigit(), first_word


def test_explicit_zero_overlap_is_respected():
    assert ChunkingService(chunk_size_tokens=50, chunk_overlap_tokens=0).chunk_overlap_chars == 0


# ---- Rerank / threshold ------------------------------------------------------ #


def test_bm25_promotes_chunk_with_rare_query_term():
    candidates = [
        {"chunk_id": "generic", "score": 0.60, "chunk_text": "He builds Angular apps. Angular apps are his focus."},
        {"chunk_id": "specific", "score": 0.55, "chunk_text": "He integrated TinyMCE for rich text editing."},
        {"chunk_id": "other", "score": 0.58, "chunk_text": "Angular dashboards with charts and Angular routing."},
    ]
    result = rerank("which editor library tinymce", candidates, top_k=3, mode="bm25")
    assert result[0]["chunk_id"] == "specific"


def test_rerank_none_keeps_vector_order():
    candidates = [
        {"chunk_id": "a", "score": 0.5, "chunk_text": "x"},
        {"chunk_id": "b", "score": 0.9, "chunk_text": "y"},
    ]
    assert [r["chunk_id"] for r in rerank("y", candidates, top_k=2, mode="none")] == ["b", "a"]


@pytest.mark.parametrize(
    "provider, explicit, expected",
    [("gemini", None, 0.5), ("ollama", None, 0.3), ("gemini", 0.42, 0.42)],
)
def test_effective_score_threshold(provider, explicit, expected):
    settings = Settings(_env_file=None, ai_provider=provider, rag_score_threshold=explicit)
    assert settings.effective_score_threshold == expected


# ---- Duplicate uploads ------------------------------------------------------- #


@pytest.mark.asyncio
async def test_duplicate_upload_is_rejected_until_original_is_deleted(rag):
    first = await rag.ingest_document("resume.txt", b"identical content")

    with pytest.raises(HTTPException) as exc_info:
        await rag.register_upload("resume-copy.txt", b"identical content")
    assert exc_info.value.status_code == 409
    assert "resume.txt" in exc_info.value.detail

    await rag.delete_document(first.document_id)
    again = await rag.register_upload("resume.txt", b"identical content")
    assert again.status == DocumentStatus.PROCESSING


@pytest.mark.asyncio
async def test_failed_upload_does_not_block_retry(rag):
    failed = await rag.register_upload("broken.txt", b"   ")
    await rag.process_document_in_background(failed, b"   ")
    assert (await rag.redis_service.get_document(failed.document_id)).status == DocumentStatus.FAILED

    retry = await rag.register_upload("broken.txt", b"   ")  # same bytes, allowed
    assert retry.document_id != failed.document_id


@pytest.mark.asyncio
async def test_duplicate_detection_survives_registry_reset(rag):
    """With MEMORY_BACKEND=memory the registry is rebuilt from Qdrant after a restart."""
    import hashlib

    content = b"resume bytes"
    rag.qdrant_service.list_documents_from_vectors.return_value = [
        {
            "document_id": "old-doc", "filename": "resume.pdf", "chunk_count": 3,
            "created_date": None, "content_hash": hashlib.sha256(content).hexdigest(),
        }
    ]
    with pytest.raises(HTTPException) as exc_info:
        await rag.register_upload("resume.pdf", content)
    assert exc_info.value.status_code == 409


# ---- Scanned PDF OCR fallback ------------------------------------------------ #


def _scanned_pdf_bytes(pages: int) -> bytes:
    """A PDF whose pages contain only drawings -- no text layer, like a scan."""
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page()
        page.draw_rect(fitz.Rect(50, 50, 300, 200), color=(0, 0, 0), fill=(0.8, 0.8, 0.8))
    data = doc.tobytes()
    doc.close()
    return data


@pytest.mark.asyncio
async def test_scanned_pdf_is_read_with_vision_model(rag):
    rag.llm_service.describe_image.side_effect = ["Invoice number 4471.", "Total due 120 EUR."]
    document = await rag.ingest_document("scan.pdf", _scanned_pdf_bytes(2))

    assert document.status == DocumentStatus.READY
    assert rag.llm_service.describe_image.await_count == 2
    assert rag.llm_service.describe_image.call_args[0][1] == "image/png"
    chunks = rag.qdrant_service.upsert_chunks.call_args[0][0]
    assert "Invoice number 4471" in chunks[0].chunk_text
    assert chunks[0].metadata["page_start"] == 1


@pytest.mark.asyncio
async def test_scanned_pdf_ocr_is_capped_by_max_ocr_pages(rag, monkeypatch):
    monkeypatch.setattr(rag.settings, "max_ocr_pages", 2)
    rag.llm_service.describe_image.return_value = "Some page text."
    await rag.ingest_document("long-scan.pdf", _scanned_pdf_bytes(5))
    assert rag.llm_service.describe_image.await_count == 2


# ---- Sources & chat history -------------------------------------------------- #


@pytest.mark.asyncio
async def test_sources_carry_page_numbers_into_answer_and_prompt(rag):
    rag.qdrant_service.search.return_value = [_hit("Education: Anna University 2021", 2, 2)]
    _, _, sources = await rag.answer_question("Where did he study?", None)

    assert sources[0].page_start == 2
    prompt = rag.llm_service.generate_chat_response.call_args.kwargs["user_prompt"]
    assert "resume.pdf, page 2]" in prompt


@pytest.mark.asyncio
async def test_follow_up_sends_history_as_chat_turns(rag):
    rag.settings.rag_query_rewrite = False
    rag.qdrant_service.search.return_value = [_hit("Works at Nova Techset since June 2022.")]
    try:
        conv, _, _ = await rag.answer_question("Where does he work?", None)
        rag.llm_service.generate_chat_response.return_value = "Since June 2022."
        await rag.answer_question("Since when?", conv)
    finally:
        rag.settings.rag_query_rewrite = True

    kwargs = rag.llm_service.generate_chat_response.call_args.kwargs
    assert kwargs["history"] == [("Where does he work?", "Answer.")]
    assert "Where does he work?" not in kwargs["user_prompt"]


@pytest.mark.asyncio
async def test_gemini_sends_history_as_user_and_model_turns(monkeypatch):
    from app.services.gemini_service import GeminiService

    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        body = 'data: {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}\n\n'
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    service = GeminiService()
    service.api_key = "k"
    monkeypatch.setattr(
        service, "_client",
        lambda: httpx.AsyncClient(base_url="https://x.test", transport=httpx.MockTransport(handler)),
    )
    await service.generate_chat_response("sys", "Context + question", history=[("q1", "a1")])

    assert [c["role"] for c in sent[0]["contents"]] == ["user", "model", "user"]
    assert sent[0]["contents"][1]["parts"][0]["text"] == "a1"


@pytest.mark.asyncio
async def test_ollama_uses_vision_model_for_images(monkeypatch):
    from app.services.ollama_service import OllamaService

    monkeypatch.setattr(get_settings(), "ollama_vision_model", "llava")
    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "A cat."}})

    service = OllamaService()
    monkeypatch.setattr(
        service, "_client",
        lambda: httpx.AsyncClient(base_url="http://x.test", transport=httpx.MockTransport(handler)),
    )
    assert await service.describe_image(b"img", "image/png") == "A cat."
    assert sent[0]["model"] == "llava"
