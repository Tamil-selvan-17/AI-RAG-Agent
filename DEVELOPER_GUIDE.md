# Developer Guide

For anyone extending or maintaining this codebase. It assumes Python/FastAPI basics and covers how *this*
project is organized, its conventions, and where to hook in new work. For setup and usage see `README.md`;
for the history of what was changed and measured, see `IMPROVEMENT_PLAN.md`.

---

## 1. Architecture at a glance

```
Frontend (static: login.html, index.html, auth.js, app.js)
        │  fetch() + Server-Sent Events, Authorization: Bearer <token>
        ▼
FastAPI app (app/main.py)
  middleware: RequestIdMiddleware → CORS
  lifespan:   build_services() once → app.state.services; recover interrupted uploads; close on shutdown
        │
   ┌────┴──────────┬───────────────┬──────────────┐
   ▼               ▼               ▼              ▼
auth_routes    upload_routes   chat_routes    health_routes
(public)       (require_auth)  (require_auth  (public)
                               + rate limit)
                   │               │
                   └──────┬────────┘
                          ▼  Depends(get_rag_service)
                 RagService  (services/rag_service.py)  ← orchestrator
                          │
   ┌──────────┬───────────┼────────────┬─────────────┬──────────────┐
   ▼          ▼           ▼            ▼             ▼              ▼
Document  Chunking   Embedding     Qdrant        Memory          AgentService
Service   Service    Service ─┐    Service    (Redis | RAM)    (tool loop, fallback)
                              ▼                                     │
                      LLM: GeminiService | OllamaService ◄──────────┘
```

**Key principles**

- `RagService` is the only place that knows the *order* of operations. Every other service wraps one
  external dependency (a file format, Qdrant, Redis, an HTTP API) and is mockable on its own.
- Services are created **once per process** in `services/container.py` (`build_services()`, called from the
  app lifespan) and injected into routes with `Depends(get_rag_service)` / `Depends(get_services)`. Never
  instantiate `RagService()` or a provider inside a route: that creates extra connection pools, and with
  `MEMORY_BACKEND=memory` a *second in-memory store* that the rest of the app can't see.
- Routes stay thin: parse the request, call one `RagService` method, shape the response with a schema.

---

## 2. Request flow

### Upload (`POST /api/documents/upload`)
1. `validate_extension` + `validate_upload_size` (reads in 1 MB chunks, stops at the limit).
2. `RagService.register_upload` — sanitize name, SHA-256 **duplicate check** (409), save the file as
   `<document_id>_<name>` (worker thread), registry entry with status `processing`.
3. Return **202**; `process_document_in_background` runs as a FastAPI background task (or inline with `?wait=true`).
4. `process_document` (bounded by a semaphore, `MAX_CONCURRENT_INGESTIONS`): extract text in a worker
   thread → (no text layer in a PDF? render pages and `describe_image` them, `MAX_OCR_PAGES`) → chunk →
   embed → upsert → status `ready`, answer cache cleared. Failures set status `failed` + `error_message`.
   If the document was deleted meanwhile, its new vectors are removed instead of resurrecting it.

### Question (`POST /api/chat` and `/api/chat/stream`)
Both endpoints share one path, `RagService._prepare_answer`:

1. Small talk ("hi", "thanks") → canned reply, no retrieval.
2. Load the last 3 turns. **No history** → check the answer cache (standalone questions only).
3. **History** and `RAG_QUERY_REWRITE` → `_standalone_question` rewrites the follow-up (1 LLM call).
4. Embed → Qdrant search (`top_k*4` candidates ≥ `effective_score_threshold`) → `rerank` → top k → dedupe.
5. `_agent_reason` decides pipeline vs agent (section 5). No sources and no agent → "not found" (0 LLM calls).
6. Build the system prompt, context block (`[Source n - file, page p, section: S]`) and history turns.

Then `answer_question` (one JSON response) or `stream_answer` (events) generates the answer — via the agent
if `agent_reason` is set — and `_record_answer` saves the turn and caches it (standalone questions only).

**If you change retrieval or prompting, change `_prepare_answer`** — there is deliberately only one copy.

---

## 3. Provider abstraction

| Setting | Values | Swaps |
|---|---|---|
| `AI_PROVIDER` | `gemini` \| `ollama` | Chat, embeddings, vision |
| `MEMORY_BACKEND` | `redis` \| `memory` | History, document registry, answer cache |

`build_services()` picks the implementations; nothing downstream checks the provider. The interfaces:

**LLM** (`GeminiService`, `OllamaService`)
- `generate_embedding(text)` (Gemini: query task type) · `generate_embeddings_batch(texts)` (document task type;
  Gemini ≤100 per request, Ollama 32)
- `stream_chat_response(system_prompt, user_prompt, temperature, history)` — async generator of tokens
- `generate_chat_response(...)` — collects the stream; retried
- `describe_image(bytes, mime)` — vision/OCR (Ollama uses `OLLAMA_VISION_MODEL`)
- `generate_with_tools(system_prompt, contents, tools, allow_tools)` — **Gemini only**; its presence is what
  enables agent mode (`AgentService.available`)
- `check_health()`, `aclose()`

**Memory** (`RedisService`, `MemoryStoreService`) — conversation (`append_chat_turn`,
`get_conversation_history`), registry (`save_document`, `get_document`, `list_documents`,
`update_document_status`, `delete_document`), cache (`get_cached_answer`, `cache_answer`,
`clear_answer_cache`), `check_health`, `aclose`. Keep both in sync when adding a method.

**Adding a provider:** implement the LLM interface (tools optional), add a branch in `build_services()` and in
the fallback factories in `rag_service.py` / `embedding_service.py` (used by tests and the eval scripts), and
give `effective_score_threshold` / `effective_vector_size` sensible values for its embedding model.

### Errors and retries (`services/provider_errors.py`)
Providers convert httpx errors into `HTTPException`s. Retry-worthy ones — connection/read failures, per-minute
429s, 5xx — are raised as `TransientProviderError` (an `HTTPException` subclass), which `@provider_retry()`
retries, honouring Google's `retryDelay`. A Gemini **daily** quota 429 is a plain `HTTPException` and is not
retried. Catch `httpx.TransportError` (not just `ConnectError`) in new provider code — real failures include
`ReadError` and dropped connections.

**Do not wrap `stream_chat_response` in a retry**: a generator that already yielded tokens to the browser
can't be restarted. Retry at the non-streaming layer instead.

### Gemini specifics
- The API key goes in the `x-goog-api-key` header, never the URL (URLs end up in logs).
- Free tier: *each text in a batch* counts toward the embedding requests-per-minute limit; batching saves HTTP
  round trips, not quota. Ingesting a large document may pause on a 429 and resume automatically.
- Function calling: append the model's returned `content` to `contents` **unchanged** before sending tool
  results — newer models attach a `thoughtSignature` to `functionCall` parts that must be echoed back.
  Tool results go back as `{"role": "user", "parts": [{"functionResponse": {name, id, response}}]}`.

---

## 4. Chunking, retrieval and ranking

- **Chunks:** `ChunkingService` — recursive split (paragraph → line → sentence → word), 600/100 tokens
  (~4 chars/token), overlap starting on a word boundary. Per chunk metadata: `page_start`/`page_end` (from the
  extractor's `[Page N]` markers) and `section` (`_section_paths`).
- **Sections:** `_detect_heading` recognises markdown `#`, ALL-CAPS lines, spaced-letter headings ("S k i l l s")
  and numbered titles with a separator ("2. nVMS - …"); skips code fences, short acronyms ("XML") and lines that
  continue a wrapped previous line. Sections are tracked over the **non-overlapping** pieces — an overlapped chunk
  begins with the previous chunk's tail and would otherwise inherit the wrong section.
- **Contextual headers** (`contextual_text`, `RAG_CONTEXTUAL_HEADERS`) embed "Document/Section" + text. Measured
  as *not* helping on the current corpus, so off by default; sections are still given to the LLM and shown in the UI.
- **Threshold:** `Settings.effective_score_threshold` — `RAG_SCORE_THRESHOLD` if set, else 0.5 (Gemini) / 0.3
  (Ollama). Similarity scales differ per embedding model; re-measure when changing models.
- **Rerank** (`rerank_service.rerank`): blends vector score (0.7) with a lexical signal (0.3) — `overlap`
  (default) or `bm25`, or `none`. The displayed `score` stays the raw vector similarity. Tokenization is Unicode-
  aware (`[\wऀ-෿']+`) so Tamil/Hindi words are not split at vowel signs.
- **Vector size safety:** `QdrantService.ensure_collection` checks the existing collection's dimension once per
  process and refuses a mismatch with a clear message (changing provider/model needs a new collection).

---

## 5. Agent mode (`services/agent_service.py`)

`AGENT_MODE=off | fallback | always`. `RagService._agent_reason` decides per question:

- `always` → agent. `off`, or a provider without `generate_with_tools` → pipeline.
- `fallback` → agent if the question `looks_multi_part` (compare / both / each of / all of … / two "?"),
  or the best vector score is below `AGENT_FALLBACK_SCORE`. **No sources at all → pipeline's zero-call
  "not found"** (usually off-topic; the agent would only spend quota).

`AgentService.run_with_events` gives the model the first search's results plus history, and tools:
`search_documents(query)`, `list_documents()`, `calculate_duration(start, end)`. Each step is one LLM call; on
step `AGENT_MAX_STEPS` tools are disabled (`toolConfig` mode `NONE`), so total calls ≤ `AGENT_MAX_STEPS`.
It yields status strings (→ SSE `status` events) and finally an `AgentResult` (answer, merged/deduped sources,
steps, call count). In practice most agent answers take 1 call because the initial results suffice.

`RagService._run_agent` falls back to the normal pipeline answer if the agent fails — except on 429 (quota),
which is re-raised so we don't spend another call. Add a new tool by extending `TOOLS` (JSON-schema
declaration), `_describe` (status text) and `_execute`.

---

## 6. Prompts and language

`_build_system_prompt(language_name)` is the single place answer behaviour is controlled. It injects
**today's date** (models don't know it; required for "years of experience" questions) and an explicit
**answer language**. `AGENT_INSTRUCTIONS` is appended for the agent. Conversation history is sent as real
`user`/`model` turns (answers trimmed to 1500 chars), not pasted into the prompt.

`language_name = response_language or detect_language_name(question)`; `response_language` is validated
(`is_valid_language_name`) because it is inserted into the system prompt. The language is part of the answer
cache key. **If you add any request parameter that changes the answer, add it to `_cache_key` in both memory
backends.** The cache is cleared on every upload/delete and skipped for follow-ups.

---

## 7. Security model

- **Auth** (`core/auth.py`): one account from `ADMIN_USERNAME`/`ADMIN_PASSWORD` (constant-time compare).
  Tokens are stateless `base64(username|expiry).HMAC-SHA256` signed with `AUTH_SECRET_KEY` (random per process
  if unset). `require_auth` is applied per router in `main.py`; `/health*` and `/api/auth/login` are public.
  5 failed logins per IP → 5-minute lockout (in memory).
- **Rate limit** (`core/rate_limit.py`): sliding window per client IP on chat endpoints. In-memory — like the
  login lockout and restart recovery, it assumes **one server process**.
- **Uploads:** extension allow-list, size checked while reading, sanitized names, stored under the document ID.
- **Secrets:** only in `backend/.env` (git-ignored, in `.dockerignore`) or the host's env vars. Never commit
  `.env`; document new settings in `.env.example`.
- **HTTPS on corporate networks:** `USE_SYSTEM_CERTS=true` → `core/tls.py` injects `truststore` (OS certificate
  store) before any client is created. Verification is never disabled.

---

## 8. Observability

- Every request gets an ID (`core/request_id.py`, pure ASGI middleware so it also covers streamed bodies and
  background tasks), shown in every log line and returned as `X-Request-ID`.
- Loguru to stdout, plus `LOG_DIR/app.log` when `LOG_TO_FILE=true` (off on Render).
- `/health/live` = process is up (no I/O; use for host health checks). `/health` = dependency check; the AI
  provider check is cached for 60 s so platform pings don't spend quota.

---

## 9. Frontend

Plain HTML/CSS/JS, no build step; served by the backend at `/app/` (or any static server on port 5500, which
targets `http://localhost:8000`).

- `auth.js` — `API_BASE`, session in `localStorage`, `apiFetch()` (adds the Bearer token, redirects to
  `login.html` on 401). Use `apiFetch` for every authenticated call.
- `app.js` — library (polls while any document is `processing`), multi-file upload queue (XHR for progress),
  streaming chat (SSE parsing, `status` / `token` / `done` / `error`, markdown rendered per animation frame via
  marked + DOMPurify, Stop via `AbortController`), sources toggle, conversation restore from
  `/api/chat/history`, toasts, mobile drawer.
- Styling tokens are CSS variables in `:root` (`style.css`). Elements toggled with the `hidden` attribute rely
  on the global `[hidden] { display: none !important; }` rule.

---

## 10. Testing

- `pytest -q` from `backend/` (140 tests, offline, ~3 s); `ruff check .` must pass (config in `pyproject.toml`).
  CI runs both, plus a frontend syntax check and a Docker build/liveness check.
- `conftest.py`: the app starts with **mocked services** (`build_services` is patched) so no test touches real
  Redis/Qdrant/AI; `client` is logged in, `anon_client` is not; `mock_services` overrides `get_rag_service`
  via `app.dependency_overrides`; rate limits are reset per test.
- Service tests construct the real class with mocks (`RagService(llm_service=..., embedding_service=...,
  qdrant_service=..., redis_service=MemoryStoreService())`). Provider HTTP is tested with `httpx.MockTransport`
  (`test_provider_retry.py`, `test_phase3.py`).
- Mock async generators by assigning a plain `async def ...: yield` function (`mock.stream_answer = fake`), not
  `AsyncMock.side_effect` — an `AsyncMock` call returns a coroutine, which `async for` can't iterate.
- `AsyncMock` objects report *every* attribute as present, so `hasattr(llm, "generate_with_tools")` is true for a
  mocked LLM — tests that must not hit the agent set `agent_mode` or use a strong search score.

### Evaluation (`backend/eval/`)
Use these before changing chunking, ranking, thresholds, prompts or agent rules — and report the numbers:
- `retrieval_eval.py` + `questions.json` — hit@k / MRR / context size; embeddings only, cached.
- `answer_eval.py` + `answer_questions.json` — required-facts grading of full answers for `off` / `always` /
  `fallback`, in an in-memory Qdrant; real chat calls, cached and resumable.
- `eval/corpus/` and `eval/.cache/` are git-ignored (personal documents, embeddings).

---

## 11. Configuration conventions

All settings are fields on `Settings` in `core/config.py`, loaded once via `get_settings()` (`lru_cache`).
Never read `os.environ` elsewhere. Derived values are properties (`effective_vector_size`,
`effective_score_threshold`, `cors_origin_list`, `upload_dir_path`). Add every new setting to
`backend/.env.example` with a comment, and mention it in `FREE_CLOUD_DEPLOYMENT.md` if Render needs it.

---

## 12. Extending

- **New document type:** add the extension in `core/security.py`, an `_extract_*` method + `extractors` entry in
  `DocumentService`, the `accept` list in `index.html`. Return plain text (add `[Page N]` markers if the format
  has pages). Chunking, embedding and storage need no change.
- **New endpoint:** add it to the router for its resource; protect it via the router's `require_auth`
  dependency; get services with `Depends(get_rag_service)`; schema in `app/schemas/`; test in `tests/`.
- **New agent tool:** see section 5; add a unit test with a fake `generate_with_tools` (see `test_phase6.py`).

---

## 13. Intentional trade-offs (read before "fixing")

- **`MEMORY_BACKEND=memory` is non-persistent** by design (zero-dependency hosting). The document list is rebuilt
  from Qdrant after a restart; content hashes live in Qdrant payloads so duplicate detection survives too.
- **Deletes treat Qdrant failures as non-fatal**, so a document stuck in `failed` can always be removed.
- **Background ingestion and in-memory limits assume a single process.** With several workers, the rate limiter
  counts per worker and startup recovery could mark another worker's in-progress upload as failed.
- **`generate_chat_response` retries, `stream_chat_response` doesn't** (section 3).
- **Fallback mode never runs the agent when nothing matched** — cheaper on quota; set `AGENT_MODE=always` to let
  the agent try rephrasing such questions.
- **Contextual header embedding and BM25 exist but are off** — they were measured and didn't help this corpus;
  re-run the evals with a larger corpus before turning them on.
