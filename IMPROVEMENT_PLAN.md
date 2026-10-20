# Improvement Plan

Tracks the improvement work for the AI RAG Agent. `[x]` = done, `[ ]` = pending.

## Phase 1 — Login, sources UI, critical bugs, security (done)

### Login (single admin user, no database)
- [x] `ADMIN_USERNAME` / `ADMIN_PASSWORD` settings (default `admin` / `P@ss@123`), compared in constant time
- [x] Signed, expiring session token (HMAC-SHA256, no DB, no extra dependency)
- [x] `POST /api/auth/login`, `GET /api/auth/me`, `POST /api/auth/logout`
- [x] All `/api/documents` and `/api/chat` routes require login; `/health` stays public
- [x] Brute-force protection: 5 failed logins per IP -> 5 minute lockout
- [x] `login.html` page, redirect to login when not signed in / session expires, Logout button

### Sources UI
- [x] Sources hidden by default under each answer
- [x] Small "sources" icon button under the answer toggles them (shows count)
- [x] Double-click the answer also toggles sources + match %

### Critical bugs
- [x] Same-name uploads: file stored as `<document_id>_<name>`; delete removes the right file
- [x] Answer cache cleared on every upload / delete; skipped for follow-up questions
- [x] Follow-up questions rewritten into standalone questions before retrieval (`RAG_QUERY_REWRITE`)
- [x] Retries actually work now (connection errors, 429 per-minute limits, 5xx)
- [x] Clear error when the Qdrant collection's vector size doesn't match the active embedding model

### Security
- [x] `backend/.env` removed from git, `.env.example` added, no longer baked into the Docker image
- [x] Gemini API key sent in a header instead of the URL
- [x] Upload size enforced while reading (stops early instead of reading everything first)
- [x] `response_language` validated (no prompt injection through it)

### Performance / quality quick wins
- [x] PDF/DOCX parsing and file writes moved off the event loop (`asyncio.to_thread`)
- [x] Batched embeddings (Ollama list input, Gemini `batchEmbedContents`)
- [x] Gemini embedding task types (`RETRIEVAL_DOCUMENT` / `RETRIEVAL_QUERY`)
- [x] Reranker understands non-English words (Tamil, Hindi, ...)
- [x] `answer_question` / `stream_answer` share one retrieval path (no duplicated logic)

## Phase 2 — Scalability (done)
- [x] Background ingestion: upload returns 202, UI polls `processing -> ready/failed` (`?wait=true` for scripts)
- [x] At most `MAX_CONCURRENT_INGESTIONS` uploads processed at once; deleting during processing is safe
- [x] Uploads interrupted by a restart are marked failed at startup (no stuck "processing")
- [x] Shared `httpx.AsyncClient` per provider (connection reuse), closed on shutdown
- [x] Services created once at startup and injected with `Depends` (also fixes: with
      `MEMORY_BACKEND=memory` the upload and chat routes used to have separate stores)
- [x] Redis `MGET` for document list; size/TTL limits on the in-memory store
- [x] Rate limiting on `/api/chat` (`CHAT_RATE_LIMIT_PER_MINUTE`, per client IP)
- [x] `/health/live` endpoint for host pings (no backend calls)

## Phase 3 — Answer quality (done)
- [x] Retrieval eval: 36 questions (`backend/eval/questions.json`) + `python -m eval.retrieval_eval`
- [x] Chunks 600/100 tokens (was 1000/200): same hit@1 (88%) and hit@5 (100%), 34% less context per answer
- [x] Word-boundary overlap; no marker-only chunks
- [x] Score threshold per embedding model (Gemini 0.5, Ollama 0.3) -- 0.3 matched every off-topic question on Gemini
- [x] BM25 rerank mode evaluated: no gain over keyword overlap on this corpus, kept as `RAG_RERANK_MODE=bm25` option
- [x] Page numbers in chunk metadata, shown in source cards and given to the LLM
- [x] Duplicate-upload detection by content hash (409), survives registry resets via Qdrant
- [x] Scanned-PDF fallback to vision OCR (`MAX_OCR_PAGES`)
- [x] `OLLAMA_VISION_MODEL` setting for image uploads / scanned PDFs
- [x] Conversation history sent as real chat turns (user/model messages) instead of text in the prompt
- [ ] Re-evaluate chunking/BM25/threshold when more documents are uploaded (corpus is small today)

## Phase 4 — Engineering hygiene (done)
- [x] `RagService` unit tests with fake backends (99 tests total, ~3s, fully offline)
- [x] CI (GitHub Actions): ruff + pytest, frontend syntax check, Docker build + liveness check
- [x] ruff config (`backend/pyproject.toml`); codebase lint-clean
- [x] `requirements-dev.txt`; unused `numpy`, `python-dotenv` pins and test tools out of runtime deps
- [x] Dockerfile: non-root user, no `build-essential`, `HEALTHCHECK`; optional Qdrant + Redis via `docker-compose.infra.yml`
- [x] Compose no longer overrides model/vector settings from `.env`
- [x] `LOG_DIR` / `LOG_TO_FILE`; request ID on every log line and in the `X-Request-ID` response header
- [x] `USE_SYSTEM_CERTS` for corporate networks that inspect HTTPS (OS certificate store via `truststore`)

## Phase 5 — Frontend (done)
- [x] Multi-file upload (drop or pick several files; uploaded one after another)
- [x] Stop-generating button (keeps the partial answer, marked "Stopped")
- [x] Conversation restored after a page refresh (`/api/chat/history`)
- [x] Markdown rendered live while streaming; fixed extra blank lines inside answers
- [x] Toast messages instead of `alert()`, incl. "ready"/"failed" when background indexing finishes
- [x] Mobile layout: documents in a slide-in drawer, no horizontal overflow at 390px

## Phase 6 — Better match scores + agent mode (done)
### Step 1: Contextual chunk headers
- [x] Detect section headings while chunking (ALL-CAPS, markdown `#`, numbered titles, spaced-letter headings)
- [x] Section shown to the LLM and on source cards; header embedding built behind `RAG_CONTEXTUAL_HEADERS`
- [x] Measured: headers did not help (hit@1 88% -> 85%; avg correct-chunk score 0.635 both), so default off.
      3072-dim embeddings also tested: same 88% / MRR 0.92 -> not worth re-indexing
### Step 2: Agent mode (`AGENT_MODE=off | fallback | always`, default `fallback`)
- [x] Gemini function calling with tools: `search_documents`, `list_documents`, `calculate_duration`
- [x] Loop capped at `AGENT_MAX_STEPS` (default 3); last step must answer
- [x] `fallback`: agent only when the first search is weak (low top score / no match) or the question is multi-part
- [x] Streaming shows what the agent is doing ("Searching: ..."); Ollama falls back to the normal pipeline
### Step 3: Answer-level evaluation
- [x] `eval/answer_questions.json`: multi-step + simple questions, graded by required facts
- [x] `eval/answer_eval.py`: pipeline vs agent vs fallback, in-memory Qdrant (never touches production), answers cached so a run interrupted by the daily quota can resume
- [x] Results: pipeline 11/12 (1.0 calls/q), agent 12/12 (1.2), **fallback 12/12 (1.1)** -> fallback is the default
