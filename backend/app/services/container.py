"""
Service container.

Builds every backing service exactly once per process (at startup, in the app
lifespan) so all routes share the same instances: one HTTP connection pool per
AI provider, one Qdrant client, one Redis client -- and, with
MEMORY_BACKEND=memory, one in-memory store. (Previously each router built its
own RagService, so with the in-memory backend the upload and chat routes each
had a *separate* store: uploads couldn't clear the chat route's answer cache.)

Routes get services through the FastAPI dependencies below, which tests can
replace with `app.dependency_overrides`.
"""

from dataclasses import dataclass

from fastapi import Request

from app.core.config import get_settings
from app.core.logging import logger
from app.services.embedding_service import EmbeddingService
from app.services.qdrant_service import QdrantService
from app.services.rag_service import RagService


@dataclass
class Services:
    llm: object  # GeminiService | OllamaService
    memory: object  # RedisService | MemoryStoreService
    qdrant: QdrantService
    rag: RagService

    async def aclose(self) -> None:
        for service in (self.llm, self.memory, self.qdrant):
            close = getattr(service, "aclose", None)
            if close is None:
                continue
            try:
                await close()
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Error closing {type(service).__name__}: {exc}")


def build_services() -> Services:
    settings = get_settings()

    if settings.ai_provider == "gemini":
        from app.services.gemini_service import GeminiService

        llm = GeminiService()
    else:
        from app.services.ollama_service import OllamaService

        llm = OllamaService()

    if settings.memory_backend == "memory":
        from app.services.memory_store_service import MemoryStoreService

        memory = MemoryStoreService()
    else:
        from app.services.redis_service import RedisService

        memory = RedisService()

    qdrant = QdrantService()
    rag = RagService(
        embedding_service=EmbeddingService(backend=llm),
        qdrant_service=qdrant,
        redis_service=memory,
        llm_service=llm,
    )
    return Services(llm=llm, memory=memory, qdrant=qdrant, rag=rag)


def get_services(request: Request) -> Services:
    return request.app.state.services


def get_rag_service(request: Request) -> RagService:
    return request.app.state.services.rag
