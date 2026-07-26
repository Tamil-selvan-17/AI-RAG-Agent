# Archive — Local Advanced RAG AI Agent

A production-style document assistant that runs **entirely on your local machine**. Upload PDFs, Word documents, text files, or CSVs; the system extracts, chunks, and embeds them with a local Ollama model, stores the vectors in Qdrant, and answers your questions using a local LLM — with every answer grounded in cited source passages. Conversation memory and a frequently-asked-question cache live in Redis.

No data ever leaves your machine. No OpenAI or other external AI API is used.

---

## 1. Architecture overview

```
                     ┌────────────────────┐
                     │   Frontend (HTML)  │
                     │  index.html/app.js │
                     └─────────┬──────────┘
                               │ REST (fetch/XHR)
                               ▼
                     ┌────────────────────┐
                     │   FastAPI backend   │
                     │     app/main.py     │
                     └─────────┬──────────┘
           ┌───────────────────┼───────────────────┐
           ▼                   ▼                   ▼
   upload_routes.py      chat_routes.py      health_routes.py
           │                   │
           ▼                   ▼
   ┌───────────────┐   ┌───────────────┐
   │  rag_service   │◄──┤  rag_service   │   (shared orchestrator)
   └───────┬───────┘   └───────┬───────┘
           │                   │
   ┌───────┴────────┬──────────┴─────────┬──────────────┐
   ▼                ▼                    ▼              ▼
document_service  chunking_service  embedding_service  ollama_service
(PyMuPDF/docx/csv) (recursive split)  (bge-m3 via Ollama) (qwen2.5:7b)
   │                                       │
   ▼                                       ▼
qdrant_service ◄───────────────── vectors stored/retrieved
   │
   ▼
redis_service (conversation memory, document registry, answer cache)
```

**Ingestion pipeline:** upload → validate → extract text → clean → recursively chunk (1000 tokens, 200 overlap) → embed each chunk with `bge-m3` → store in Qdrant with full metadata.

**Query pipeline:** question → embed with `bge-m3` → Qdrant cosine similarity search (top‑k=5) → build context from retrieved chunks + recent conversation history → generate answer with `qwen2.5:7b` (system prompt restricts it to answering only from the provided context) → persist the turn to Redis → return the answer with source references.

**Design choices worth knowing:**
- Chunking is a self-contained recursive character splitter (paragraph → sentence → word → character fallback) rather than a LangChain/LlamaIndex dependency, so the chunking behavior is fully transparent and has zero extra dependency-version risk.
- Redis doubles as the lightweight document registry (status, chunk counts) since the project intentionally has no separate relational database.
- Every service is written as an injectable class (constructor accepts its dependencies), so each one is independently unit-testable with mocks — see `backend/tests/`.

---

## 2. Prerequisites

Install on Windows (per your environment):

- **Docker Desktop** (with WSL2 backend enabled)
- **Python 3.12+**
- **Git**
- **Ollama** — [ollama.com](https://ollama.com)

---

## 3. Ollama setup

Pull the two models this project uses:

```bash
ollama pull bge-m3
ollama pull qwen2.5:7b
```

Confirm Ollama is serving the REST API:

```bash
curl http://localhost:11434/api/version
```

Ollama must remain running (`ollama serve`, or the Windows tray app) whenever you use the assistant.

---

## 4. Qdrant setup

Run Qdrant via Docker (this is the "external service already running" referenced by `docker-compose.yml`):

```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

Verify:

```bash
curl http://localhost:6333/collections
```

The backend creates its `documents` collection automatically on first run — no manual setup needed.

---

## 5. Redis setup

```bash
docker run -d --name redis -p 6379:6379 redis:7-alpine
```

Verify:

```bash
docker exec -it redis redis-cli ping
# -> PONG
```

---

## 6. Running the backend (local Python, no Docker)

```bash
cd AI-RAG-Agent/backend
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS/Linux

pip install -r requirements.txt

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The `.env` file in `backend/` already points at `localhost` for Ollama, Qdrant, and Redis — matching your local setup. Adjust it if any service runs elsewhere.

Once running:
- API: `http://localhost:8000`
- Interactive API docs (Swagger): `http://localhost:8000/docs`
- Frontend is also served at `http://localhost:8000/app`

---

## 7. Running the frontend

The backend already serves the frontend at `/app`, so no separate step is required. If you prefer to serve it independently (e.g. with VS Code's Live Server, or any static file server):

```bash
cd AI-RAG-Agent/frontend
python -m http.server 5500
```

Then open `http://localhost:5500`. The frontend automatically targets `http://localhost:8000` as the API base when served from a different port.

---

## 8. Running everything with Docker Compose

This starts the **backend API** and a small **nginx frontend container**. Qdrant, Redis, and Ollama are expected to already be running on the host (steps 3–5 above) — the containers reach them via `host.docker.internal`.

```bash
cd AI-RAG-Agent
docker compose up --build
```

- Frontend: `http://localhost:5500`
- Backend API: `http://localhost:8000`

Uploaded files are persisted to `./documents` on the host via a volume mount.

---

## 9. Running tests

```bash
cd AI-RAG-Agent/backend
pip install -r requirements.txt
pytest -v
```

Tests cover the upload API, chat API, chunking service, and embedding service, using mocked Ollama/Qdrant/Redis clients so they run without any external services.

---

## 10. API documentation

Base URL: `http://localhost:8000`

### `POST /api/documents/upload`
Upload a document (`multipart/form-data`, field name `file`). Accepts `.pdf`, `.docx`, `.txt`, `.csv`, up to 50MB.

```bash
curl -X POST http://localhost:8000/api/documents/upload \
  -F "file=@./contract.pdf"
```

Response `201`:
```json
{
  "document_id": "b3f1c2a4-...",
  "filename": "contract.pdf",
  "status": "ready",
  "chunk_count": 42,
  "message": "Document processed successfully into 42 chunks"
}
```

### `GET /api/documents`
List all uploaded documents and their status.

```bash
curl http://localhost:8000/api/documents
```

```json
{
  "total": 1,
  "documents": [
    {
      "document_id": "b3f1c2a4-...",
      "filename": "contract.pdf",
      "file_extension": ".pdf",
      "file_size_bytes": 184320,
      "status": "ready",
      "chunk_count": 42,
      "created_date": "2026-07-25T07:00:00Z",
      "error_message": null
    }
  ]
}
```

### `POST /api/chat`
Ask a question. Omit `conversation_id` to start a new conversation.

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the termination clause?", "conversation_id": null}'
```

```json
{
  "conversation_id": "9c8a7b6d-...",
  "question": "What is the termination clause?",
  "answer": "According to the contract, either party may terminate with 30 days written notice...",
  "sources": [
    {
      "document_id": "b3f1c2a4-...",
      "filename": "contract.pdf",
      "chunk_id": "e1d2c3b4-...",
      "chunk_text": "Either party may terminate this agreement by providing thirty (30) days...",
      "score": 0.84
    }
  ]
}
```

### `GET /api/chat/history/{conversation_id}`
Retrieve the stored turns for a conversation. Returns `404` if no history exists yet.

```bash
curl http://localhost:8000/api/chat/history/9c8a7b6d-...
```

### `GET /health`
Reports connectivity to Ollama, Qdrant, and Redis.

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "healthy",
  "services": {
    "api": "healthy",
    "ollama": "healthy",
    "qdrant": "healthy",
    "redis": "healthy"
  }
}
```

---

## 11. Project structure

```
AI-RAG-Agent/
├── backend/
│   ├── app/
│   │   ├── main.py                 FastAPI app, CORS, routers, lifespan
│   │   ├── api/                    upload_routes, chat_routes, health_routes
│   │   ├── core/                   config, logging, security
│   │   ├── services/                document/chunking/embedding/ollama/qdrant/redis/rag
│   │   ├── models/                  Document, DocumentChunk, ChatTurn, SourceReference
│   │   ├── schemas/                 Pydantic request/response schemas
│   │   └── utils/                   file_utils
│   ├── tests/                       pytest suite (mocked external services)
│   ├── requirements.txt
│   ├── .env
│   ├── pytest.ini
│   └── Dockerfile
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── documents/                       uploaded files land here
├── docker-compose.yml
├── .gitignore
└── README.md
```

---

## 12. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `/health` shows `ollama: unreachable` | Ollama isn't running | Start Ollama, confirm `curl http://localhost:11434/api/version` |
| `/health` shows `qdrant: unreachable` | Qdrant container not running | `docker start qdrant`, confirm port 6333 is free |
| `/health` shows `redis: unreachable` | Redis container not running | `docker start redis` |
| Upload returns `415` | File extension not in `.pdf/.docx/.txt/.csv` | Convert or re-save the file in a supported format |
| Upload returns `422` | Text extraction produced no content (e.g. scanned/image-only PDF) | Use a text-based PDF, or OCR it first |
| Chat answers "I couldn't find any relevant information" | No matching chunks above the similarity threshold | Upload a document that covers the topic, or lower `RAG_SCORE_THRESHOLD` in `.env` |
