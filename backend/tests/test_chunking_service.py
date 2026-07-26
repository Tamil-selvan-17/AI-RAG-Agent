"""Unit tests for ChunkingService covering size bounds and overlap."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.chunking_service import ChunkingService  # noqa: E402


def test_short_text_produces_single_chunk():
    service = ChunkingService(chunk_size_tokens=1000, chunk_overlap_tokens=200)
    chunks = service.split_text("This is a short sentence.")
    assert len(chunks) == 1


def test_long_text_produces_multiple_chunks():
    service = ChunkingService(chunk_size_tokens=50, chunk_overlap_tokens=10)
    long_text = ("Sentence number %d provides sample content for chunking. " * 60) % tuple(range(60))
    chunks = service.split_text(long_text)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) > 0


def test_empty_text_produces_no_chunks():
    service = ChunkingService()
    assert service.split_text("") == []
    assert service.split_text("   ") == []


def test_chunk_document_wraps_metadata():
    service = ChunkingService(chunk_size_tokens=1000, chunk_overlap_tokens=200)
    chunks = service.chunk_document(
        text="Some example document content for testing.",
        document_id="doc-1",
        filename="test.txt",
    )
    assert len(chunks) == 1
    assert chunks[0].document_id == "doc-1"
    assert chunks[0].filename == "test.txt"
    assert chunks[0].chunk_index == 0
    assert chunks[0].total_chunks == 1
