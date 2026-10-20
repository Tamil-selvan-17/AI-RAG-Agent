# Archive — RAG Document Assistant

Upload PDFs, Word documents, text files, CSVs or images, then ask questions about them. Every answer is
grounded in your documents, with the exact source passages (file, page, section, match %) one click away.

It runs in two ways:

| | **Local** | **Cloud (free tier)** |
|---|---|---|
| AI (chat + embeddings) | Ollama on your machine (`qwen2.5:7b`, `bge-m3`) — no data leaves your PC | Google Gemini API (`gemini-flash-lite-latest`, `gemini-embedding-001`) |
| Vector database | Qdrant in Docker | Qdrant Cloud |
| Memory / cache | Redis in Docker, or in-process RAM | In-process RAM (or Upstash Redis) |
| Hosting | `uvicorn` / Docker Compose | Render (auto-deploys from `main`) — see [FREE_CLOUD_DEPLOYMENT.md](FREE_CLOUD_DEPLOYMENT.md) |

The provider is chosen with `AI_PROVIDER` and `MEMORY_BACKEND` in `backend/.env`; nothing else changes.

---

## 1. Features

- **Login** — single admin account (no user database), signed expiring sessions, brute-force lockout.
- **Documents** — PDF, DOCX, TXT, CSV and images (JPG/PNG/WEBP/GIF/BMP, described/OCR'd by the vision model).
  Scanned PDFs without a text layer are read page by page with the vision model. Multi-file upload;
  indexing runs in the background; duplicate files are rejected.
- **Answers** — streamed token by token, rendered as markdown, in the language you pick (or the question's).
  Follow-up questions ("when did he join *there*?") are rewritten into standalone searches.
- **Sources** — hidden by default; click the small *sources* icon (or double-click the answer) to see each
  passage with file, page, section and match %.
- **Agent mode** — hard questions ("compare…", "all of his projects…") go to a tool-using agent that can search
  several times and do exact date maths (see [Agent mode](#agent-mode)).
- **Comfort** — Stop button, conversation restored after a refresh, toast notifications, mobile layout.

---

## 2. How it works

```
Browser (frontend/: login.html, index.html, app.js)
   │  REST + Server-Sent Events, Bearer token
   ▼
FastAPI backend (backend/app/main.py)
   ├─ /api/auth       login / me / logout
   ├─ /api/documents  upload (background indexing), list, delete
   ├─ /api/chat       ask (JSON or streamed), history
   └─ /health         full check  ·  /health/live  instant liveness
   │
   ▼
RagService ── DocumentService (PDF/DOCX/TXT/CSV text, scanned-PDF pages)
           ├─ ChunkingService (600-token chunks, pages + section headings)
           ├─ EmbeddingService ─┐
           ├─ LLM (Gemini | Ollama) ◄┘
           ├─ QdrantService (vectors)        ├─ AgentService (tools, fallback mode)
           └─ Memory (Redis | RAM: history, document list, answer cache)
```

**Ingestion:** upload → validate (type, size, duplicate) → save → extract text (or vision OCR) → chunk
(600 tokens, 100 overlap, page + section tracked) → embed → store in Qdrant.

**Question:** small-talk check → answer cache → (follow-up? rewrite it) → embed → Qdrant search (top 20
candidates above the score threshold) → rerank with keyword overlap → top 5 → LLM answer from those passages
only → save the turn. In `AGENT_MODE=fallback`, multi-part questions and weak searches go to the agent instead.

Retrieval settings were chosen with the evaluation scripts in `backend/eval/` (see [section 8](#8-measuring-quality)),
not by guesswork.

---

## 3. Quick start (Windows, Command Prompt)

Prerequisites: **Python 3.11+**, **Git**, and either **Ollama** (local mode) or a free
**Gemini API key** + **Qdrant Cloud** cluster (cloud mode). Docker Desktop is optional.

```bat
cd /d "<path>\AI-RAG-Agent\backend"
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements-dev.txt
copy .env.example .env
```

Edit `backend/.env` (see [section 4](#4-configuration)), then:

```bat
uvicorn app.main:app --reload --port 8000
```

Open **http://localhost:8000/app/** and sign in (default `admin` / `P@ss@123` — change it in `.env`).

> PowerShell users: activate with `.\venv\Scripts\Activate.ps1` (run
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once if scripts are blocked).
> macOS/Linux: `source venv/bin/activate` and `cp .env.example .env`.

Other URLs: API `http://localhost:8000`, Swagger docs `http://localhost:8000/docs`.

### Local services (local mode only)

```bash
ollama pull bge-m3
ollama pull qwen2.5:7b
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant
docker run -d --name redis -p 6379:6379 redis:7-alpine     # optional: or MEMORY_BACKEND=memory
```

For image uploads with Ollama, also pull a vision model (`ollama pull llava`) and set `OLLAMA_VISION_MODEL=llava`.

---

## 4. Configuration

All settings live in `backend/.env` (never committed; `backend/.env.example` is the documented template).

**Choose a setup**

```ini
# Cloud (Gemini + Qdrant Cloud, no Redis)
AI_PROVIDER=gemini
GEMINI_API_KEY=<your key>
QDRANT_URL=https://<cluster>.cloud.qdrant.io
QDRANT_API_KEY=<key>
QDRANT_VECTOR_SIZE=768
MEMORY_BACKEND=memory

# Local (Ollama + Docker Qdrant/Redis)
AI_PROVIDER=ollama
QDRANT_URL=
MEMORY_BACKEND=redis
```

**Always set**

| Setting | Why |
|---|---|
| `ADMIN_USERNAME`, `ADMIN_PASSWORD` | The only login. Change the default password before deploying. |
| `AUTH_SECRET_KEY` | Signs sessions; without it users are logged out on every restart. Generate: `python -c "import secrets; print(secrets.token_urlsafe(48))"` |

**Commonly changed** (defaults are the measured best values)

| Setting | Default | Meaning |
|---|---|---|
| `AGENT_MODE` | `fallback` | `off` / `fallback` / `always` — see [Agent mode](#agent-mode) |
| `CHAT_RATE_LIMIT_PER_MINUTE` | `20` | Questions per client IP per minute (`0` = unlimited) |
| `RAG_SCORE_THRESHOLD` | *(unset)* | Unset = 0.5 for Gemini, 0.3 for Ollama embeddings |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `600` / `100` | Tokens per chunk; affects new uploads only |
| `USE_SYSTEM_CERTS` | `false` | `true` on corporate networks that inspect HTTPS (`CERTIFICATE_VERIFY_FAILED`) |
| `LOG_TO_FILE` / `LOG_DIR` | `true` / `logs` | Set `LOG_TO_FILE=false` on Render |

See `backend/.env.example` for every setting with an explanation.

> **Changing the embedding model or `AI_PROVIDER`?** Vectors from different models are not compatible. The app
> refuses to use a collection with the wrong vector size and tells you so — use a new `QDRANT_COLLECTION_NAME`
> or *Clear all* and re-upload.

---

## 5. Using the app

- **Upload:** drop one or several files on the upload box (or click it). Files show *processing* while they
  are indexed in the background, then *ready* (or *failed* with the reason). Re-uploading an identical file is
  refused — delete the old copy first.
- **Ask:** type and press Enter. *Answer language* forces the reply language (or choose *Match question's language*).
  Press the orange ■ button to stop an answer.
- **Sources:** click the 📄 *N sources* icon under an answer, or double-click the answer.
- **Conversation:** kept after a page refresh; *Clear conversation* (*New chat* on phones) starts over.
- **Phones:** the document list opens from the *Documents* button.

### Agent mode

| `AGENT_MODE` | What happens | AI calls per question |
|---|---|---|
| `off` | One search, one answer | 1 (2 for follow-ups) |
| `fallback` *(default)* | Like `off`, but multi-part questions ("compare", "both", "all of…") and weak first searches go to the agent | ~1.1 on average |
| `always` | Agent for every question | ~1.2–3 |

The agent (Gemini only; Ollama always uses the normal pipeline) can call `search_documents`, `list_documents`
and `calculate_duration`, up to `AGENT_MAX_STEPS` (3) AI calls, and must answer on the last one. While it works
the chat shows *"Searching: …"*, and lists what it looked up under the answer. Questions with no matching
passage at all still cost zero AI calls. On the answer test set, `fallback` answered **12/12** questions
correctly vs **11/12** for `off`, at 1.1 calls per question.

---

## 6. Running with Docker Compose

```bash
docker compose up --build                     # app + nginx frontend; services from the host / cloud
docker compose -f docker-compose.yml -f docker-compose.infra.yml up --build   # also starts Qdrant + Redis
```

Settings come from `backend/.env`; compose only sets container hostnames and paths. Frontend:
`http://localhost:5500`, API: `http://localhost:8000`. The image runs as a non-root user with a
`HEALTHCHECK` on `/health/live`. Uploaded files persist in `./documents`.

---

## 7. Development

```bat
cd backend
venv\Scripts\activate.bat
ruff check .        # lint
pytest -q           # 140 tests, fully offline, ~3 seconds
```

GitHub Actions (`.github/workflows/ci.yml`) runs lint + tests, a frontend syntax check and a Docker
build/liveness check on every push and pull request to `main`. Render deploys `main` automatically.

See **[DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)** for architecture, conventions and how to extend the code.

---

## 8. Measuring quality

Two scripts in `backend/eval/` decide settings with numbers. Put copies of your documents in
`backend/eval/corpus/` (git-ignored — it usually contains personal files). Neither script writes to your
real Qdrant collection.

```bat
:: Retrieval: is the right passage found? (embedding calls only, cached)
python -m eval.retrieval_eval --verbose
python -m eval.retrieval_eval --configs 600:100 400:60 --modes overlap bm25 --headers off embed

:: Answers: does the final answer contain the required facts? pipeline vs agent vs fallback
python -m eval.answer_eval --show-answers
```

`answer_eval` makes real chat calls (~1 per question per mode) using an in-memory Qdrant; answers are cached
in `eval/.cache/`, so a run stopped by the free-tier daily quota resumes where it left off.

Current results (36 retrieval questions / 12 answer questions):

| Measurement | Result |
|---|---|
| Right passage ranked #1 / in top 5 | 88% / 100% (600-token chunks, keyword rerank) |
| Without the rerank step | 70% ranked #1 |
| Answers correct: pipeline / agent / fallback | 11/12 / 12/12 / **12/12** |

> The *match %* on source cards is semantic similarity, not "% correct". With Gemini embeddings even perfect
> matches typically show 55–75%. Tried and measured: section headers in embeddings and 3072-dim embeddings
> did not improve results, so they are off.

---

## 9. API reference

Base URL `http://localhost:8000`. Interactive docs at `/docs`.

**Auth:** `POST /api/auth/login` with `{"username": "...", "password": "..."}` returns `access_token`; send it as
`Authorization: Bearer <token>` on every `/api/documents` and `/api/chat` call. Five failed logins from one IP
lock login for 5 minutes. Every response carries an `X-Request-ID` header (quote it when reporting an error).

| Method & path | Purpose |
|---|---|
| `POST /api/auth/login` · `GET /api/auth/me` · `POST /api/auth/logout` | Session |
| `POST /api/documents/upload` | Upload one file (`multipart/form-data`, field `file`). Returns **202** `processing`; add `?wait=true` to block until indexed (201). 409 if it is a duplicate. |
| `GET /api/documents` | List documents with `status` (`processing` / `ready` / `failed`) and `error_message` |
| `DELETE /api/documents/{id}` · `DELETE /api/documents` | Delete one / all |
| `POST /api/chat` | Ask; returns `{conversation_id, answer, sources}` |
| `POST /api/chat/stream` | Ask; Server-Sent Events: `status` (agent progress), `token`, then `done` (with `sources`, `agent_steps`) or `error` |
| `GET /api/chat/history/{conversation_id}` | Stored turns (404 if none) |
| `GET /health/live` | Instant liveness (no backend calls) — use for host health checks |
| `GET /health` | AI provider, Qdrant and memory backend status |

Chat endpoints are rate limited (`CHAT_RATE_LIMIT_PER_MINUTE`) and return `429` with `Retry-After` when exceeded.

```bash
TOKEN=$(curl -s -X POST http://localhost:8000/api/auth/login -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"P@ss@123"}' | python -c "import sys,json;print(json.load(sys.stdin)['access_token'])")

curl -X POST "http://localhost:8000/api/documents/upload?wait=true" -H "Authorization: Bearer $TOKEN" -F "file=@./resume.pdf"

curl -X POST http://localhost:8000/api/chat -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"question": "What was his CGPA?"}'
```

A source in a chat response:

```json
{
  "document_id": "b3f1c2a4-...", "filename": "resume.pdf", "chunk_id": "e1d2c3b4-...",
  "chunk_text": "EDUCATION ... CGPA: 7.45 / 10", "score": 0.66,
  "page_start": 2, "page_end": 2, "section": "EDUCATION"
}
```

---

## 10. Project structure

```
AI-RAG-Agent/
├── backend/
│   ├── app/
│   │   ├── main.py            app, middleware (CORS, request IDs), routers, lifespan
│   │   ├── api/               auth, upload (documents), chat, health routes
│   │   ├── core/              config, auth, rate_limit, request_id, security, logging, tls
│   │   ├── services/          rag (orchestrator), agent, chunking, document, embedding,
│   │   │                      gemini, ollama, qdrant, redis, memory_store, rerank,
│   │   │                      provider_errors, container (shared instances)
│   │   ├── models/            Document, DocumentChunk, ChatTurn, SourceReference
│   │   ├── schemas/           API request/response models
│   │   └── utils/             file, language and small-talk helpers
│   ├── eval/                  retrieval_eval.py, answer_eval.py + question sets (corpus/ git-ignored)
│   ├── tests/                 pytest suite (all external services mocked)
│   ├── .env.example           settings template (copy to .env)
│   ├── requirements.txt       runtime deps   ·  requirements-dev.txt  + pytest, ruff
│   ├── pyproject.toml         ruff config    ·  pytest.ini
│   └── Dockerfile
├── frontend/                  login.html, index.html, auth.js, app.js, style.css
├── documents/                 uploaded files (git-ignored)
├── .github/workflows/ci.yml   CI
├── docker-compose.yml         ·  docker-compose.infra.yml (optional Qdrant + Redis)
├── IMPROVEMENT_PLAN.md        what was changed and measured, phase by phase
└── FREE_CLOUD_DEPLOYMENT.md   Gemini + Qdrant Cloud + Render setup
```

---

## 11. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `pip install -r requirements-dev.txt`: *No such file* | Not in the `backend` folder | `cd /d "<path>\AI-RAG-Agent\backend"` first |
| `pytest` / `uvicorn` *not recognized* | venv not activated (cmd needs `activate.bat`, not `.ps1`) | `venv\Scripts\activate.bat` — the prompt should start with `(venv)` |
| Redirected to the login page | Session expired or `AUTH_SECRET_KEY` changed / unset across a restart | Sign in again; set a fixed `AUTH_SECRET_KEY` |
| `CERTIFICATE_VERIFY_FAILED` calling Gemini (local only) | Corporate network inspects HTTPS | `USE_SYSTEM_CERTS=true` in `.env` |
| "Gemini's free-tier DAILY request limit has been used up" | Free quota exhausted | Wait for the reset (midnight Pacific Time), or use `AGENT_MODE=off` to save calls |
| `429 Too many questions` | `CHAT_RATE_LIMIT_PER_MINUTE` reached | Wait a minute or raise the limit |
| "Qdrant collection … stores N-dim vectors" | Embedding model / provider changed | New `QDRANT_COLLECTION_NAME`, or *Clear all* and re-upload |
| Upload `409` | Identical file already uploaded | Delete the existing document first |
| Upload `415` | Unsupported file type | Use PDF, DOCX, TXT, CSV or an image |
| Document *failed*: "No extractable text" | Empty file, or a scan the vision model couldn't read | Check the file; scanned PDFs are OCR'd up to `MAX_OCR_PAGES` pages |
| "I couldn't find any relevant information" | No passage above the score threshold | Rephrase, upload a document that covers it, or lower `RAG_SCORE_THRESHOLD` |
| `/health` shows `ollama: unreachable` | Ollama not running (local mode) | Start Ollama; `curl http://localhost:11434/api/version` |
| `/health` shows `qdrant`/`redis: unreachable` | Container stopped / wrong URL | `docker start qdrant` / `docker start redis`, or check `QDRANT_URL` / `REDIS_URL` |
