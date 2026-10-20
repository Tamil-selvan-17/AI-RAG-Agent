"""
Retrieval evaluation: measures how often the right chunk is retrieved for a
fixed set of questions, so chunking/reranking changes are judged by numbers
instead of guesswork.

It uses the app's real ChunkingService, embedding provider (AI_PROVIDER from
.env) and rerank function, but runs the vector search in-process -- it never
writes to Qdrant, so it's safe to run against any environment.

Embeddings are cached in eval/.cache/, so re-running with the same corpus and
configs costs no API calls. Only embedding calls are made (no chat calls).

Usage (from backend/):
    python -m eval.retrieval_eval                        # default grid
    python -m eval.retrieval_eval --configs 1000:200 400:60 --modes overlap bm25
    python -m eval.retrieval_eval --verbose               # list misses for each config

Corpus: every .pdf/.docx/.txt/.md/.csv file in eval/corpus/ (git-ignored, since
it usually holds personal documents). Put copies of your real documents there.
"""

import argparse
import asyncio
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402
from app.core.tls import configure_tls  # noqa: E402
from app.services.chunking_service import ChunkingService  # noqa: E402
from app.services.document_service import DocumentService  # noqa: E402
from app.services.rerank_service import rerank  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
CACHE_FILE = EVAL_DIR / ".cache" / "embeddings.json"
DEFAULT_CONFIGS = ["1000:200", "600:100", "400:60", "250:40"]
DEFAULT_MODES = ["none", "overlap", "bm25"]


# --------------------------------------------------------------------------- #
# Embeddings (cached)
# --------------------------------------------------------------------------- #


class CachedEmbedder:
    def __init__(self) -> None:
        settings = get_settings()
        if settings.ai_provider == "gemini":
            from app.services.gemini_service import GeminiService

            self.backend = GeminiService()
            self.model_key = f"gemini:{settings.gemini_embed_model}:{settings.gemini_embed_dimensions}"
        else:
            from app.services.ollama_service import OllamaService

            self.backend = OllamaService()
            self.model_key = f"ollama:{settings.ollama_embed_model}"
        self.cache: dict[str, list[float]] = {}
        if CACHE_FILE.exists():
            self.cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        self.api_texts = 0

    def _key(self, text: str, task: str) -> str:
        return hashlib.sha256(f"{self.model_key}|{task}|{text}".encode()).hexdigest()

    async def embed(self, texts: list[str], task: str) -> list[list[float]]:
        missing = list(dict.fromkeys(t for t in texts if self._key(t, task) not in self.cache))
        if missing:
            self.api_texts += len(missing)
            if hasattr(self.backend, "embed_model") and type(self.backend).__name__ == "GeminiService":
                vectors = await self.backend.generate_embeddings_batch(missing, task_type=task)
            else:
                vectors = await self.backend.generate_embeddings_batch(missing)
            for text, vector in zip(missing, vectors, strict=False):
                self.cache[self._key(text, task)] = vector
            CACHE_FILE.parent.mkdir(exist_ok=True)
            CACHE_FILE.write_text(json.dumps(self.cache), encoding="utf-8")
        return [self.cache[self._key(t, task)] for t in texts]

    async def aclose(self) -> None:
        await self.backend.aclose()


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


# --------------------------------------------------------------------------- #
# Corpus / questions
# --------------------------------------------------------------------------- #


def load_corpus(corpus_dir: Path) -> dict[str, str]:
    documents: dict[str, str] = {}
    extractor = DocumentService()
    for path in sorted(corpus_dir.iterdir()):
        ext = path.suffix.lower()
        if ext in {".txt", ".md"}:
            documents[path.name] = path.read_text(encoding="utf-8")
        elif ext in {".pdf", ".docx", ".csv"}:
            documents[path.name] = extractor.extract_text(path, ext)
    return documents


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def contains_evidence(chunk_text: str, evidence: list[str]) -> bool:
    chunk = _normalize(chunk_text)
    return any(_normalize(e) in chunk for e in evidence)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


@dataclass
class ConfigResult:
    label: str
    chunk_count: int
    hit1: float
    hit3: float
    hit5: float
    mrr: float
    avg_context_chars: float
    unanswerable_matched: int
    unanswerable_total: int
    misses: list[str]


async def evaluate(
    embedder: CachedEmbedder,
    documents: dict[str, str],
    questions: list[dict],
    size: int,
    overlap: int,
    mode: str,
    top_k: int,
    threshold: float,
) -> ConfigResult:
    chunker = ChunkingService(chunk_size_tokens=size, chunk_overlap_tokens=overlap)
    chunks: list[dict] = []
    for name, text in documents.items():
        for i, piece in enumerate(chunker.split_text(text)):
            chunks.append({"filename": name, "chunk_id": f"{name}#{i}", "chunk_text": piece})

    chunk_vectors = await embedder.embed([c["chunk_text"] for c in chunks], "RETRIEVAL_DOCUMENT")
    question_vectors = await embedder.embed([q["question"] for q in questions], "RETRIEVAL_QUERY")

    ranks: list[int | None] = []
    context_chars: list[int] = []
    unanswerable_matched = 0
    unanswerable_total = 0
    misses: list[str] = []

    for q, q_vec in zip(questions, question_vectors, strict=False):
        scored = [
            {**c, "score": cosine(q_vec, v)} for c, v in zip(chunks, chunk_vectors, strict=False)
        ]
        scored = [c for c in scored if c["score"] >= threshold]
        scored.sort(key=lambda c: c["score"], reverse=True)
        candidates = scored[: min(top_k * 4, 40)]
        results = rerank(q["question"], candidates, top_k=top_k, mode=mode)

        if not q["evidence"]:
            unanswerable_total += 1
            if results:
                unanswerable_matched += 1
            continue

        context_chars.append(sum(len(r["chunk_text"]) for r in results))
        rank = next(
            (i for i, r in enumerate(results, start=1) if contains_evidence(r["chunk_text"], q["evidence"])),
            None,
        )
        ranks.append(rank)
        if rank != 1:
            top = results[0]["chunk_id"] if results else "nothing"
            misses.append(f"{q['id']}: rank={rank or '-'} (top: {top})")

    answerable = len(ranks) or 1
    return ConfigResult(
        label=f"{size}:{overlap} {mode}",
        chunk_count=len(chunks),
        hit1=sum(1 for r in ranks if r == 1) / answerable,
        hit3=sum(1 for r in ranks if r and r <= 3) / answerable,
        hit5=sum(1 for r in ranks if r and r <= 5) / answerable,
        mrr=sum(1 / r for r in ranks if r) / answerable,
        avg_context_chars=sum(context_chars) / (len(context_chars) or 1),
        unanswerable_matched=unanswerable_matched,
        unanswerable_total=unanswerable_total,
        misses=misses,
    )


def check_evidence_exists(documents: dict[str, str], questions: list[dict]) -> None:
    """Fail fast on a typo'd evidence string that can never match."""
    full = {name: _normalize(text) for name, text in documents.items()}
    for q in questions:
        for e in q["evidence"]:
            if not any(_normalize(e) in text for text in full.values()):
                raise SystemExit(f"Evidence for '{q['id']}' not found anywhere in the corpus: {e!r}")


async def main() -> None:
    configure_tls()
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus-dir", type=Path, default=EVAL_DIR / "corpus")
    parser.add_argument("--questions", type=Path, default=EVAL_DIR / "questions.json")
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS, help="chunk_size:overlap (tokens)")
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES, choices=DEFAULT_MODES)
    parser.add_argument("--top-k", type=int, default=settings.rag_top_k)
    parser.add_argument("--threshold", type=float, default=settings.effective_score_threshold)
    parser.add_argument("--verbose", action="store_true", help="print the questions each config missed at rank 1")
    args = parser.parse_args()

    documents = load_corpus(args.corpus_dir)
    if not documents:
        raise SystemExit(f"No documents found in {args.corpus_dir}")
    questions = json.loads(args.questions.read_text(encoding="utf-8"))["questions"]
    check_evidence_exists(documents, questions)

    embedder = CachedEmbedder()
    print(f"Corpus: {', '.join(documents)} | {len(questions)} questions | "
          f"top_k={args.top_k} threshold={args.threshold} | {embedder.model_key}\n")
    header = (
        f"{'config':<22}{'chunks':>7}{'hit@1':>8}{'hit@3':>8}{'hit@5':>8}"
        f"{'MRR':>7}{'ctx chars':>11}{'no-answer matched':>19}"
    )
    print(header)
    print("-" * len(header))

    try:
        for config in args.configs:
            size, overlap = (int(x) for x in config.split(":"))
            for mode in args.modes:
                r = await evaluate(embedder, documents, questions, size, overlap, mode, args.top_k, args.threshold)
                print(
                    f"{r.label:<22}{r.chunk_count:>7}{r.hit1:>8.0%}{r.hit3:>8.0%}{r.hit5:>8.0%}"
                    f"{r.mrr:>7.2f}{r.avg_context_chars:>11.0f}{r.unanswerable_matched:>12}/{r.unanswerable_total}"
                )
                if args.verbose and r.misses:
                    for miss in r.misses:
                        print(f"    miss  {miss}")
    finally:
        await embedder.aclose()

    print(f"\nEmbedding API calls this run: {embedder.api_texts} new texts (rest from cache)")


if __name__ == "__main__":
    asyncio.run(main())
