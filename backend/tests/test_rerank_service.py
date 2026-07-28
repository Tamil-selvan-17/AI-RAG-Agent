"""Unit tests for hybrid (vector + lexical) reranking."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.rerank_service import rerank  # noqa: E402


def test_empty_results_returns_empty_list():
    assert rerank("any question", [], top_k=5) == []


def test_respects_top_k_limit():
    candidates = [
        {"chunk_id": f"c{i}", "score": 0.5, "chunk_text": f"chunk number {i}"} for i in range(10)
    ]
    result = rerank("chunk", candidates, top_k=3)
    assert len(result) == 3


def test_lexically_relevant_chunk_promoted_over_purely_topical_one():
    candidates = [
        {
            "chunk_id": "generic",
            "score": 0.55,
            "chunk_text": "The company was founded in Chennai and focuses on enterprise delivery.",
        },
        {
            "chunk_id": "specific",
            "score": 0.45,
            "chunk_text": "Skills include AWS S3 secure file storage and cloud deployment pipelines.",
        },
    ]
    result = rerank("what aws skills does he have", candidates, top_k=2)
    assert result[0]["chunk_id"] == "specific"


def test_original_score_field_is_unchanged():
    candidates = [{"chunk_id": "c1", "score": 0.42, "chunk_text": "some text here"}]
    result = rerank("some question", candidates, top_k=1)
    assert result[0]["score"] == 0.42


def test_pure_vector_ranking_preserved_when_no_lexical_signal():
    # Neither chunk shares any significant words with the question, so
    # reranking should fall back to pure vector score ordering.
    candidates = [
        {"chunk_id": "low", "score": 0.3, "chunk_text": "zzz yyy xxx"},
        {"chunk_id": "high", "score": 0.8, "chunk_text": "qqq www vvv"},
    ]
    result = rerank("totally unrelated question terms", candidates, top_k=2)
    assert result[0]["chunk_id"] == "high"
