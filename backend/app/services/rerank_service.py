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

import math
import re
from collections import Counter

_STOPWORDS = {
    "the", "is", "are", "was", "were", "does", "do", "did", "have", "has", "had",
    "what", "how", "many", "much", "why", "who", "where", "when", "which",
    "tell", "me", "about", "in", "on", "at", "a", "an", "of", "to", "for",
    "and", "or", "but", "please", "can", "could", "you", "with", "this", "that",
    "he", "she", "they", "them", "it", "his", "her",
}

# \w covers letters/digits in every script, but not Indic vowel signs and
# viramas (combining marks), which would otherwise split Tamil/Hindi/etc.
# words into fragments -- so the Indic Unicode blocks are included explicitly.
_WORD_RE = re.compile(r"[\wऀ-෿']+")


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


def _tokens(text: str) -> list[str]:
    return [w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS and len(w) > 1]


def _bm25_scores(question: str, texts: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """BM25 score of each text for the question, normalized to 0-1 (best = 1).

    IDF is computed over the candidate pool itself. That is an approximation of
    corpus-wide IDF, but it is exactly the set being reordered, and it needs no
    extra index: terms that appear in every candidate carry little weight, rare
    ones (names, product codes, numbers) carry a lot.
    """
    query_terms = set(_tokens(question))
    docs = [_tokens(t) for t in texts]
    if not query_terms or not docs:
        return [0.0] * len(texts)

    n = len(docs)
    avg_len = sum(len(d) for d in docs) / n or 1.0
    doc_freq = Counter(term for d in docs for term in set(d) if term in query_terms)

    scores = []
    for d in docs:
        tf = Counter(d)
        score = 0.0
        for term in query_terms:
            if not tf[term]:
                continue
            idf = math.log(1 + (n - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            norm = tf[term] * (k1 + 1) / (tf[term] + k1 * (1 - b + b * len(d) / avg_len))
            score += idf * norm
        scores.append(score)

    best = max(scores)
    return [s / best for s in scores] if best > 0 else scores


def rerank(
    question: str,
    results: list[dict],
    top_k: int,
    vector_weight: float = 0.7,
    lexical_weight: float = 0.3,
    mode: str = "overlap",
) -> list[dict]:
    """Rerank Qdrant search results using a blend of vector and lexical scores.

    Args:
        question: the original user question (used for lexical overlap scoring)
        results: candidate results from Qdrant, each a dict with at least
            "score" (vector similarity, 0-1) and "chunk_text"
        top_k: how many results to return after reranking
        vector_weight / lexical_weight: blend weights (should sum to ~1.0)
        mode: lexical signal -- "overlap" (fraction of question words present),
            "bm25" (term-frequency/rarity weighted), or "none" (vector order only)

    Returns:
        Up to top_k results, reordered by blended score. Each result's original
        "score" field is left untouched (so the UI still shows the true vector
        similarity) -- reranking only affects *ordering and selection*, not the
        displayed match percentage.
    """
    if not results:
        return []

    if mode == "none":
        return sorted(results, key=lambda r: r.get("score", 0.0), reverse=True)[:top_k]

    if mode == "bm25":
        lexical_scores = _bm25_scores(question, [r.get("chunk_text", "") for r in results])
    else:
        question_words = _significant_words(question)
        lexical_scores = [
            _lexical_overlap_score(question_words, r.get("chunk_text", "")) for r in results
        ]

    scored = []
    for r, lexical_score in zip(results, lexical_scores, strict=False):
        vector_score = r.get("score", 0.0)
        blended = (vector_weight * vector_score) + (lexical_weight * lexical_score)
        scored.append((blended, r))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [r for _, r in scored[:top_k]]
