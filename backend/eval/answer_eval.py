"""
Answer-level evaluation: does agent mode actually answer better than the pipeline?

Indexes every document in eval/corpus/ into an in-memory Qdrant (never your real
collection), then asks each question in eval/answer_questions.json twice --
AGENT_MODE=off (one search, one answer) and AGENT_MODE=always (tool-using agent).
The "fallback" column is what the default mode would do: the agent answer for
questions the fallback rule sends to the agent, the pipeline answer otherwise.

Answers are graded by required facts (see answer_questions.json) and cached in
eval/.cache/answers.json, so a run cut short by the provider's daily quota can be
resumed later without re-asking finished questions. Use --fresh to ignore the cache.

This makes real chat calls (roughly 1 per question for the pipeline and 2-3 for
the agent). Usage (from backend/):
    python -m eval.answer_eval
    python -m eval.answer_eval --only compare-npms-nvms cgpa --show-answers
"""

import argparse
import asyncio
import json
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import HTTPException  # noqa: E402
from qdrant_client import AsyncQdrantClient  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.core.tls import configure_tls  # noqa: E402
from app.services.memory_store_service import MemoryStoreService  # noqa: E402
from app.services.qdrant_service import QdrantService  # noqa: E402
from app.services.rag_service import RagService  # noqa: E402
from eval.retrieval_eval import CachedEmbedder  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
ANSWER_CACHE = EVAL_DIR / ".cache" / "answers.json"
INGESTIBLE = {".txt", ".pdf", ".docx", ".csv"}


class CachedEmbeddingService:
    """EmbeddingService interface backed by the eval embedding cache."""

    def __init__(self, embedder: CachedEmbedder) -> None:
        self.embedder = embedder

    async def embed_text(self, text: str) -> list[float]:
        return (await self.embedder.embed([text], "RETRIEVAL_QUERY"))[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return await self.embedder.embed(texts, "RETRIEVAL_DOCUMENT")


class CountingLLM:
    """Wraps the real LLM service and counts chat/tool calls."""

    def __init__(self, llm) -> None:
        self._llm = llm
        self.calls = 0

    def __getattr__(self, name):
        attr = getattr(self._llm, name)
        if name in {"generate_chat_response", "generate_with_tools"}:
            async def counted(*args, **kwargs):
                self.calls += 1
                return await attr(*args, **kwargs)
            return counted
        return attr


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def grade(answer: str, require: list[list[str]]) -> tuple[bool, float, list[str]]:
    """(passed, fraction of required facts present, missing groups)."""
    text = _normalize(answer)
    missing = [" / ".join(group) for group in require if not any(_normalize(o) in text for o in group)]
    score = 1 - len(missing) / len(require) if require else 1.0
    return not missing, score, missing


async def build_rag(embedder: CachedEmbedder, upload_dir: str) -> tuple[RagService, CountingLLM]:
    settings = get_settings()
    settings.upload_dir = upload_dir
    if settings.ai_provider == "gemini":
        from app.services.gemini_service import GeminiService

        llm = CountingLLM(GeminiService())
    else:
        from app.services.ollama_service import OllamaService

        llm = CountingLLM(OllamaService())

    qdrant = QdrantService()
    await qdrant._client.close()
    qdrant._client = AsyncQdrantClient(location=":memory:")  # in-process, never the real cluster
    qdrant.collection_name = "answer_eval"

    rag = RagService(
        llm_service=llm,
        embedding_service=CachedEmbeddingService(embedder),
        qdrant_service=qdrant,
        redis_service=MemoryStoreService(),
    )
    return rag, llm


async def ingest_corpus(rag: RagService, corpus_dir: Path) -> None:
    for path in sorted(corpus_dir.iterdir()):
        ext = path.suffix.lower()
        if ext not in INGESTIBLE | {".md"}:
            continue
        name = path.name if ext in INGESTIBLE else f"{path.stem}.txt"  # .md -> uploaded as text
        document = await rag.ingest_document(name, path.read_bytes())
        print(f"  indexed {document.filename}: {document.chunk_count} chunks")


async def ask(rag: RagService, llm: CountingLLM, question: str, mode: str) -> dict:
    rag.settings.agent_mode = mode
    await rag.redis_service.clear_answer_cache()
    before = llm.calls
    _, answer, sources = await rag.answer_question(question, conversation_id=None)
    return {"answer": answer, "calls": llm.calls - before, "sources": len(sources)}


async def main() -> None:
    configure_tls()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus-dir", type=Path, default=EVAL_DIR / "corpus")
    parser.add_argument("--questions", type=Path, default=EVAL_DIR / "answer_questions.json")
    parser.add_argument("--only", nargs="+", help="question ids to run")
    parser.add_argument("--fresh", action="store_true", help="ignore cached answers")
    parser.add_argument("--show-answers", action="store_true")
    args = parser.parse_args()

    questions = json.loads(args.questions.read_text(encoding="utf-8"))["questions"]
    if args.only:
        questions = [q for q in questions if q["id"] in set(args.only)]
    cache: dict = {}
    if ANSWER_CACHE.exists() and not args.fresh:
        cache = json.loads(ANSWER_CACHE.read_text(encoding="utf-8"))

    def save() -> None:
        ANSWER_CACHE.parent.mkdir(exist_ok=True)
        ANSWER_CACHE.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")

    embedder = CachedEmbedder()
    with tempfile.TemporaryDirectory() as upload_dir:
        rag, llm = await build_rag(embedder, upload_dir)
        print("Indexing corpus into an in-memory Qdrant:")
        await ingest_corpus(rag, args.corpus_dir)

        quota_hit = None
        try:
            # Pipeline answers first (cheapest), then agent answers.
            for mode in ("off", "always"):
                for q in questions:
                    key = f"{mode}|{q['question']}"
                    if key in cache:
                        continue
                    try:
                        cache[key] = await ask(rag, llm, q["question"], mode)
                    except HTTPException as exc:
                        if exc.status_code == 429:
                            quota_hit = exc.detail
                            raise
                        cache[key] = {"answer": f"ERROR: {exc.detail}", "calls": 0, "sources": 0}
                    save()
                    print(f"  [{mode:6}] {q['id']}: {cache[key]['calls']} call(s)")
        except HTTPException:
            pass

        # Which questions would the default fallback mode send to the agent? (search only, no LLM)
        rag.settings.agent_mode = "fallback"
        triggers = {}
        for q in questions:
            await rag.redis_service.clear_answer_cache()  # a cached answer would skip the decision
            prepared = await rag._prepare_answer(q["question"], None, None, None)
            triggers[q["id"]] = prepared.agent_reason
        await rag.qdrant_service.aclose()
        await llm.aclose()
    await embedder.aclose()

    # ---- Report ------------------------------------------------------------------- #
    totals = {m: {"pass": 0, "score": 0.0, "calls": 0, "n": 0} for m in ("off", "always", "fallback")}
    print(f"\n{'question':<20}{'kind':<14}{'pipeline':>10}{'agent':>10}{'fallback':>10}  fallback uses agent?")
    print("-" * 92)
    for q in questions:
        row = {}
        for mode in ("off", "always"):
            entry = cache.get(f"{mode}|{q['question']}")
            row[mode] = entry and (*grade(entry["answer"], q["require"]), entry["calls"], entry["answer"])
        reason = triggers.get(q["id"], "")
        row["fallback"] = row["always"] if reason else row["off"]

        cells = []
        for mode in ("off", "always", "fallback"):
            result = row[mode]
            if not result:
                cells.append("—")
                continue
            passed, score, _, calls, _ = result
            totals[mode]["pass"] += passed
            totals[mode]["score"] += score
            totals[mode]["calls"] += calls
            totals[mode]["n"] += 1
            cells.append(f"{'PASS' if passed else f'{score:.0%}'} ({calls})")
        print(f"{q['id']:<20}{q['kind']:<14}{cells[0]:>10}{cells[1]:>10}{cells[2]:>10}  {reason or 'no'}")
        if args.show_answers:
            for mode in ("off", "always"):
                if row[mode]:
                    missing = row[mode][2]
                    note = f"  MISSING: {missing}" if missing else ""
                    print(f"      {mode:6} answer: {row[mode][4][:300]!r}{note}")

    print("-" * 92)
    for mode, label in (("off", "pipeline"), ("always", "agent"), ("fallback", "fallback")):
        t = totals[mode]
        if t["n"]:
            print(f"{label:<10} passed {t['pass']}/{t['n']}  facts {t['score'] / t['n']:.0%}  "
                  f"LLM calls {t['calls']} ({t['calls'] / t['n']:.1f}/question)")
    if quota_hit:
        print(f"\nStopped early -- provider quota reached: {quota_hit}\nRe-run later to continue (cached).")


if __name__ == "__main__":
    asyncio.run(main())
