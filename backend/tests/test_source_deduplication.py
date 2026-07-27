"""Unit tests for RagService._deduplicate_sources.

Duplicate identical chunks can appear in search results when the same document
gets uploaded more than once (e.g. across app restarts with a non-persistent
registry but a persistent vector store) -- this ensures the UI never shows the
same passage 2-3 times as if they were independent supporting sources.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models.chat import SourceReference  # noqa: E402
from app.services.rag_service import RagService  # noqa: E402


def test_removes_exact_duplicate_chunk_text():
    sources = [
        SourceReference(document_id="d1", filename="r.pdf", chunk_id="c1", chunk_text="same text", score=0.72),
        SourceReference(document_id="d2", filename="r.pdf", chunk_id="c2", chunk_text="same text", score=0.72),
        SourceReference(document_id="d3", filename="r.pdf", chunk_id="c3", chunk_text="same text", score=0.71),
    ]
    result = RagService._deduplicate_sources(sources)
    assert len(result) == 1
    assert result[0].chunk_id == "c1"  # first (highest-scoring) copy kept


def test_keeps_distinct_chunk_text():
    sources = [
        SourceReference(document_id="d1", filename="r.pdf", chunk_id="c1", chunk_text="text A", score=0.72),
        SourceReference(document_id="d1", filename="r.pdf", chunk_id="c2", chunk_text="text B", score=0.65),
    ]
    result = RagService._deduplicate_sources(sources)
    assert len(result) == 2


def test_empty_list_returns_empty_list():
    assert RagService._deduplicate_sources([]) == []


def test_preserves_original_order():
    sources = [
        SourceReference(document_id="d1", filename="r.pdf", chunk_id="c1", chunk_text="A", score=0.9),
        SourceReference(document_id="d1", filename="r.pdf", chunk_id="c2", chunk_text="B", score=0.8),
        SourceReference(document_id="d1", filename="r.pdf", chunk_id="c3", chunk_text="A", score=0.5),
    ]
    result = RagService._deduplicate_sources(sources)
    assert [s.chunk_id for s in result] == ["c1", "c2"]
