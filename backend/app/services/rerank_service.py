"""
Reranking service.

Implements hybrid reranking: after Qdrant returns candidates by vector
similarity alone, this rescoes them using a blend of the original vector
score and a lexical (keyword-overlap) score against the question, then
returns the top-k by that blended score.

Why this matters: pure vector similarity can favor chunks that are
*topically* related but don't actually contain the specific terms the
question asked about (e.g. a question naming a specific person/product may
retrieve a generically similar chunk over one that literally mentions that
name). Blending in a lexical signal corrects for this without needing a
separate paid reranking API/model -- a lightweight, dependency-free approach
suitable for a small self-hosted or free-tier deployment.
"""

import re

_STOPWORDS = {
    "the", "is", "are", "was", "were", "does", "do", "did", "have", "has", "had",
    "what", "how", "many", "much", "why", "who", "where", "when", "which",
    "tell", "me", "about", "in", "on", "at", "a", "an", "of", "to", "for",
    "and", "or", "but", "please", "can", "could", "you", "with", "this", "that",
    "he", "she", "they", "them", "it", "his", "her",
}

_WORD_RE = re.compile(r"[a-zA-Z0-9']+")


def _significant_words(text: str) -> set[str]:
    """Extract lowercased words from text, excluding common stopwords."""
    words = _WORD_RE.findall(text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 1}


def _lexical_overlap_score(question_words: set[str], chunk_text: str) -> float:
    """Fraction of the question's significant words that appear in the chunk (0-1)."""
    if not question_words:
        return 0.0
    chunk_words = _significant_words(chunk_text)
    if not chunk_words:
        return 0.0
    overlap = question_words & chunk_words
    return len(overlap) / len(question_words)


def rerank(
    question: str,
    results: list[dict],
    top_k: int,
    vector_weight: float = 0.7,
    lexical_weight: float = 0.3,
) -> list[dict]:
    """Rerank Qdrant search results using a blend of vector and lexical scores.

    Args:
        question: the original user question (used for lexical overlap scoring)
        results: candidate results from Qdrant, each a dict with at least
            "score" (vector similarity, 0-1) and "chunk_text"
        top_k: how many results to return after reranking
        vector_weight / lexical_weight: blend weights (should sum to ~1.0)

    Returns:
        Up to top_k results, reordered by blended score. Each result's original
        "score" field is left untouched (so the UI still shows the true vector
        similarity) -- reranking only affects *ordering and selection*, not the
        displayed match percentage.
    """
    if not results:
        return []

    question_words = _significant_words(question)

    scored = []
    for r in results:
        vector_score = r.get("score", 0.0)
        lexical_score = _lexical_overlap_score(question_words, r.get("chunk_text", ""))
        blended = (vector_weight * vector_score) + (lexical_weight * lexical_score)
        scored.append((blended, r))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [r for _, r in scored[:top_k]]
