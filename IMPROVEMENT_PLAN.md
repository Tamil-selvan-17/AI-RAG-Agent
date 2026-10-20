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

## Phase 4 — Engineering hygiene
- [ ] `RagService` unit tests with fake backends; CI (GitHub Actions) with pytest + ruff
- [ ] `requirements-dev.txt`; drop unused `numpy`, `python-dotenv`
- [ ] Dockerfile: non-root user, no `build-essential`, `HEALTHCHECK`; Qdrant + Redis in compose
- [ ] Configurable log path, request IDs in logs

## Phase 5 — Frontend
- [ ] Multi-file upload, Stop-generating button
- [ ] Restore conversation after refresh (`/api/chat/history`)
- [ ] Live markdown while streaming, toast messages instead of `alert()`, mobile layout
