"""
Application configuration.

Loads settings from environment variables / .env file using pydantic-settings.
A single cached Settings instance is shared across the application via get_settings().
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- AI Provider switch ----
    # "ollama" = local models on your own machine (default, original setup)
    # "gemini" = free Google Gemini API for both chat and embeddings (for cloud hosting)
    ai_provider: str = "ollama"

    # ---- Ollama (used when ai_provider = "ollama") ----
    ollama_url: str = "http://localhost:11434"
    ollama_chat_model: str = "qwen2.5:7b"
    ollama_embed_model: str = "bge-m3"
    # Vision-capable model for image uploads and scanned PDFs (e.g. "llava",
    # "qwen2.5vl"). Empty = use OLLAMA_CHAT_MODEL, which only works if it supports images.
    ollama_vision_model: str = ""
    ollama_timeout_seconds: int = 180

    # ---- Gemini (used when ai_provider = "gemini") ----
    gemini_api_key: str = ""
    gemini_chat_model: str = "gemini-flash-lite-latest"
    gemini_embed_model: str = "gemini-embedding-001"
    gemini_embed_dimensions: int = 768
    gemini_timeout_seconds: int = 60

    # ---- Qdrant ----
    # Local (Docker on your machine): leave qdrant_url empty, use host/port below.
    # Qdrant Cloud (free hosted cluster): set qdrant_url + qdrant_api_key instead.
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_grpc_port: int = 6334
    qdrant_collection_name: str = "documents"
    qdrant_vector_size: int = 1024

    # ---- Memory backend switch ----
    # "redis" = use Redis/Upstash for conversation memory, document catalog, and cache (default)
    # "memory" = keep everything in this process's RAM instead -- no Redis/Upstash needed at all,
    #            but history/catalog/cache reset whenever the process restarts
    memory_backend: str = "redis"

    # ---- Redis ----
    # Local (Docker on your machine): leave redis_url empty, use host/port below.
    # Upstash / Redis Cloud (free hosted instance): set redis_url instead
    # (looks like rediss://default:PASSWORD@your-host.upstash.io:6379).
    redis_url: str = ""
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_conversation_ttl_seconds: int = 604800

    # ---- Chunking ----
    # In tokens (~4 chars each). 600/100 matched 1000/200's retrieval accuracy on
    # the eval set (eval/retrieval_eval.py) while sending ~34% less text to the LLM.
    chunk_size: int = 600
    chunk_overlap: int = 100

    # ---- RAG ----
    rag_top_k: int = 5
    # Minimum similarity (0-1) for a chunk to count as a match. Scores are on a
    # different scale per embedding model, so leave this unset to use the
    # provider default (see effective_score_threshold), or tune it with the eval.
    rag_score_threshold: float | None = None
    # Lexical signal blended into vector scores when reranking: "overlap", "bm25" or "none".
    rag_rerank_mode: str = "overlap"

    # Rewrite follow-up questions ("what about his education?") into standalone
    # questions using the conversation history before searching Qdrant. Costs one
    # extra (short) LLM call per follow-up question; first questions are unaffected.
    rag_query_rewrite: bool = True
    # How many uploads may be parsed/embedded at the same time (others wait their turn).
    max_concurrent_ingestions: int = 2
    # Scanned PDFs (no text layer) are read page by page with the vision model.
    # Each page is one AI request, so this caps the cost of a single upload.
    max_ocr_pages: int = 10

    # ---- Rate limiting ----
    # Max chat questions per client IP per minute (0 = unlimited). Protects the
    # AI provider's quota (e.g. Gemini free tier) from runaway clients.
    chat_rate_limit_per_minute: int = 20

    # ---- Auth (single admin user, no database) ----
    admin_username: str = "admin"
    admin_password: str = "P@ss@123"
    # Secret used to sign login session tokens. If left empty, a random one is
    # generated at startup -- which works, but logs everyone out on every restart.
    auth_secret_key: str = ""
    auth_session_hours: int = 12

    # ---- App ----
    app_name: str = "AI RAG Agent"
    app_env: str = "development"
    log_level: str = "INFO"
    log_to_file: bool = True
    log_dir: str = "logs"
    # Verify outgoing HTTPS against the OS certificate store instead of Python's
    # bundled list -- needed on corporate networks that inspect HTTPS (see core/tls.py).
    use_system_certs: bool = False
    max_upload_size_mb: int = 50
    upload_dir: str = "../documents"
    cors_origins: str = "*"

    @property
    def cors_origin_list(self) -> list[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def effective_vector_size(self) -> int:
        """Embedding vector dimension actually in use, based on the active provider."""
        if self.ai_provider == "gemini":
            return self.gemini_embed_dimensions
        return self.qdrant_vector_size

    @property
    def effective_score_threshold(self) -> float:
        """RAG_SCORE_THRESHOLD if set, else a default measured for the active embedding model.

        gemini-embedding-001: correct chunks scored >= 0.55 on the eval set while
        off-topic questions topped out around 0.48-0.51. bge-m3 (Ollama) scores
        lower overall; 0.3 is what worked in earlier real testing.
        """
        if self.rag_score_threshold is not None:
            return self.rag_score_threshold
        return 0.5 if self.ai_provider == "gemini" else 0.3

    @property
    def upload_dir_path(self) -> Path:
        path = Path(self.upload_dir).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (loaded once per process)."""
    return Settings()
