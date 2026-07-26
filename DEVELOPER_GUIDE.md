# Developer Guide

This guide is for anyone extending or maintaining this codebase — it assumes you already know Python/FastAPI basics and focuses on how *this project specifically* is organized, its conventions, and where to hook in new work. (For a beginner-oriented explanation of what the project does and how to run it, see `README.md`.)

---

## 1. Architecture at a glance

```
Frontend (static HTML/CSS/JS)
        │  fetch() / SSE
        ▼
FastAPI app (app/main.py)
        │
   ┌────┴─────┬──────────┐
   ▼          ▼          ▼
upload_    chat_       health_
routes.py  routes.py   routes.py
   │          │
   └────┬─────┘
        ▼
   RagService  (app/services/rag_service.py)  ← orchestrator, no I/O of its own
        │
   ┌────┼────────┬───────────┬────────────┬─────────────┐
   ▼    ▼        ▼           ▼            ▼             ▼
Document Chunking Embedding  Qdrant       Memory        LLM
Service  Service   Service   Service   (Redis/RAM)    (Ollama/Gemini)
```

**Key principle:** `RagService` is the only place that knows the *order* of operations. Every other service is a stateless(ish) wrapper around one external dependency (a file format, Qdrant, Redis, an HTTP API). This makes each service trivially mockable in tests and means a bug is always traceable to exactly one file.

---

## 2. Provider abstraction pattern

Two dimensions are swappable at runtime via `.env`, and this project uses the same pattern for both — worth understanding before you add a third:

| Setting | Values | Swaps |
|---|---|---|
| `AI_PROVIDER` | `ollama` \| `gemini` | Embedding + chat generation backend |
| `MEMORY_BACKEND` | `redis` \| `memory` | Conversation/catalog/cache storage backend |

**The pattern**, e.g. in `rag_service.py`:
```python
def _default_llm_backend():
    settings = get_settings()
    if settings.ai_provider == "gemini":
        from app.services.gemini_service import GeminiService
        return GeminiService()
    from app.services.ollama_service import OllamaService
    return OllamaService()
```
Both backends implement the *exact same method signatures* (`generate_embedding`, `generate_embeddings_batch`, `generate_chat_response`, `stream_chat_response`, `check_health`). Nothing downstream needs an `if provider == ...` check anywhere else — `RagService`, `EmbeddingService`, and the health endpoint just call whatever `self.llm_service` / `self._backend` happens to be.

**If you add a third provider** (e.g. Anthropic, OpenAI-compatible, a local vLLM server): implement a new service class matching that exact interface, add a branch to the two `_default_*_backend()` factories (in `rag_service.py` and `embedding_service.py`), and add the corresponding `AI_PROVIDER=` value. Nothing else changes.

---

## 3. Streaming architecture

There are two parallel code paths for generation, and it's important to know when to touch which:

- **`generate_chat_response()`** — returns one complete string. Used by anything that wants a single final answer (tests, simple API consumers hitting `POST /api/chat`).
- **`stream_chat_response()`** — an async generator yielding tokens one at a time. Used by `POST /api/chat/stream` (Server-Sent Events) to keep bytes flowing to the browser continuously, which is what prevents reverse-proxy timeouts on slow answers.

`generate_chat_response()` is implemented as a thin wrapper that just consumes `stream_chat_response()` and joins the result — **don't duplicate the HTTP/parsing logic in both**; if you're touching the Ollama or Gemini request payload, you almost always only need to edit `stream_chat_response()`.

**Important gotcha:** `stream_chat_response()` is deliberately **not** wrapped in `@retry` (tenacity). Retrying a generator that has already yielded partial output to a caller doesn't make sense — you can't "retry" a response the browser has partially rendered. If you need retry behavior for a streaming call, it has to be at the layer above (e.g. don't start rendering until the first token succeeds), not inside the generator itself.

---

## 4. RAG pipeline internals (`rag_service.py`)

`answer_question()` (non-streaming) and `stream_answer()` (streaming) intentionally duplicate the retrieval logic rather than sharing a helper. This was a deliberate tradeoff for readability/safety over DRY-ness — the two methods diverge enough in control flow (one returns a tuple, one yields events) that a shared helper would need several callback parameters. **If you change the retrieval logic (chunk lookup, prompt building, caching), you must update both methods.** Search for `_build_context` and `_build_system_prompt` usage to find both call sites.

### The system prompt (`_build_system_prompt`)
This function is the single place answer *behavior* is controlled. It currently injects two dynamic pieces of ground truth the model can't know on its own:
1. **Today's real date** — LLMs don't know the actual wall-clock date, only their training cutoff, which breaks any "how many years since X" style question. Always keep this dynamic; never hardcode a date.
2. **The explicit answer language** (see below) — naming the language outperforms telling the model to "match the input language" and hoping it infers correctly.

If you need to change assistant behavior/tone/constraints, this is almost always the right function to edit — avoid scattering behavior instructions elsewhere.

### Language handling (`app/utils/language_utils.py`)
`detect_language_name()` wraps `langdetect` with a friendly name lookup table and returns `None` for text too short/ambiguous to classify (rather than guessing wrong). The resolution order in both `answer_question`/`stream_answer` is:
```python
language_name = response_language or detect_language_name(question)
```
i.e. an explicit `response_language` field on the request always wins over auto-detection. This same `language_name` is folded into the cache key (see below) so a forced-language request never returns a stale answer cached under a different language.

### The Q&A cache
`get_cached_answer` / `cache_answer` key on a hash of `question + language`, **not** just `question`. If you add any other parameter that changes the *content* of the answer (e.g. a future `top_k` override, a persona toggle, a document-scope filter), it needs to go into that cache key too, or you'll serve stale/wrong-context answers for parameter combinations that share a question string. Search for `_cache_key` in `redis_service.py` and `memory_store_service.py` — both must be updated together, since they implement the same interface.

---

## 5. Adding a new document type

1. Add the extension to `ALLOWED_EXTENSIONS` in `app/core/security.py`
2. Add an extraction method to `DocumentService` in `app/services/document_service.py`, matching the existing `_extract_pdf` / `_extract_docx` pattern (return plain text; let `clean_text()` in `file_utils.py` handle whitespace normalization)
3. Register it in the `extractors` dict inside `DocumentService.extract_text()`
4. No other file needs to change — chunking, embedding, and storage are format-agnostic once you have plain text

---

## 6. Adding a new API endpoint

Follow the existing three-router split by resource type (`upload_routes.py`, `chat_routes.py`, `health_routes.py`) rather than adding a fourth router unless the resource is genuinely new. Route handlers should stay thin — they parse the request, call exactly one `RagService` method, and shape the response via a Pydantic schema from `app/schemas/`. Business logic belongs in `RagService` or a lower service, never in the route function itself.

---

## 7. Testing conventions

- `tests/conftest.py` provides a `client` fixture: a `TestClient` with the module-level `_rag_service` singleton in each router replaced by an `AsyncMock`. This means route tests never touch real Ollama/Qdrant/Redis — they test request/response shape and status codes only.
- Service-level tests (e.g. `test_chunking_service.py`, `test_embedding_service.py`, `test_language_utils.py`) instantiate the real service class directly and inject mock dependencies via constructor parameters — every service is written to accept its dependencies as optional constructor args specifically to enable this.
- When mocking an async generator method (like `stream_answer` or `stream_chat_response`) for a test, **assign a plain async-generator function directly** (`mock.stream_answer = fake_generator_fn`) rather than using `.side_effect` on an `AsyncMock`. An `AsyncMock`'s default call behavior returns an awaitable coroutine, not an async-iterable generator, and `async for` will fail against it. See `conftest.py`'s `fake_stream_answer` for the working pattern.
- Run the full suite with `pytest -v` from `backend/`; `pytest.ini` sets `asyncio_mode = auto` so `async def test_...` functions don't need `@pytest.mark.asyncio` (though a few files still add it explicitly, harmlessly).

---

## 8. Configuration conventions

All runtime configuration lives in `app/core/config.py` as a single `Settings` (pydantic-settings) class, loaded once via `get_settings()` (`@lru_cache`, so it's a process-wide singleton). **Never read `os.environ` directly anywhere else in the codebase** — add a field to `Settings` instead, even for a one-off value, so every setting is discoverable in one place and typed.

Computed/derived values (e.g. `effective_vector_size`, which depends on which AI provider is active) are exposed as `@property` on `Settings` rather than computed ad-hoc where needed — check there first before recomputing something that looks derived.

---

## 9. Known intentional trade-offs (don't "fix" these without reading why)

- **`MemoryStoreService` is non-persistent by design** when `MEMORY_BACKEND=memory` — this is documented and intentional (a zero-dependency option for free hosting), not a bug.
- **`delete_document`/`delete_all_documents` treat Qdrant failures as non-fatal** — they log a warning and continue removing the registry entry/file regardless. This was a deliberate fix so a document stuck in a `failed` state (e.g. because Qdrant was down during upload) can still be removed from your list. Don't make Qdrant deletion failures raise again without re-solving that original problem.
- **`generate_chat_response` retries, `stream_chat_response` does not.** This is not an oversight — see section 3.
