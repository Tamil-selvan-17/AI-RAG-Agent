"""
Application configuration.

Loads settings from environment variables / .env file using pydantic-settings.
A single cached Settings instance is shared across the application via get_settings().
"""

from functools import lru_cache
from pathlib import Path
from typing import List

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
    ollama_timeout_seconds: int = 180

    # ---- Gemini (used when ai_provider = "gemini") ----
    gemini_api_key: str = ""
    gemini_chat_model: str = "gemini-2.0-flash"
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
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # ---- RAG ----
    rag_top_k: int = 5
    rag_score_threshold: float = 0.3

    # ---- App ----
    app_name: str = "AI RAG Agent"
    app_env: str = "development"
    log_level: str = "INFO"
    max_upload_size_mb: int = 50
    upload_dir: str = "../documents"
    cors_origins: str = "*"

    @property
    def cors_origin_list(self) -> List[str]:
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
    def upload_dir_path(self) -> Path:
        path = Path(self.upload_dir).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (loaded once per process)."""
    return Settings()
