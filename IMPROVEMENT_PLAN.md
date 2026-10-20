# Improvement Plan

Tracks the improvement work for the AI RAG Agent. `[x]` = done, `[ ]` = pending.

## Phase 1 — Login, sources UI, critical bugs, security (in progress)

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

## Phase 2 — Scalability
- [ ] Background ingestion (return 202, UI polls `processing -> ready`)
- [ ] Shared `httpx.AsyncClient` per provider (connection reuse)
- [ ] Services created once at startup and injected with `Depends`
- [ ] Redis `MGET` for document list; size/TTL limits on the in-memory store
- [ ] Rate limiting on `/api/chat`
- [ ] Separate `/health/live` endpoint for host pings

## Phase 3 — Answer quality
- [ ] Evaluation set (20-30 question -> expected chunk) + hit-rate script
- [ ] Smaller chunks (300-500 tokens) with word-boundary overlap, tuned using the eval set
- [ ] Page numbers in chunk metadata, shown in source cards
- [ ] Duplicate-upload detection by content hash
- [ ] Scanned-PDF fallback to vision OCR
- [ ] `OLLAMA_VISION_MODEL` setting for image uploads
- [ ] Qdrant hybrid search (dense + sparse/BM25) or cross-encoder reranker
- [ ] Send history as real chat messages instead of text in the prompt

## Phase 4 — Engineering hygiene
- [ ] `RagService` unit tests with fake backends; CI (GitHub Actions) with pytest + ruff
- [ ] `requirements-dev.txt`; drop unused `numpy`, `python-dotenv`
- [ ] Dockerfile: non-root user, no `build-essential`, `HEALTHCHECK`; Qdrant + Redis in compose
- [ ] Configurable log path, request IDs in logs

## Phase 5 — Frontend
- [ ] Multi-file upload, Stop-generating button
- [ ] Restore conversation after refresh (`/api/chat/history`)
- [ ] Live markdown while streaming, toast messages instead of `alert()`, mobile layout
