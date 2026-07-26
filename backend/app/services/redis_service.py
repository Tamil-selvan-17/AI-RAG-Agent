"""
Redis service.

Responsibilities:
- Conversation memory: append/read chat turns per conversation_id
- Document registry: track uploaded document metadata/status (since there is
  no separate relational DB in this project, Redis doubles as the document
  catalog store)
- Query cache: cache answers for frequently asked, identical questions
"""

import hashlib
import json
import uuid

import redis.asyncio as redis
from fastapi import HTTPException, status

from app.core.config import get_settings
from app.core.logging import logger
from app.models.chat import ChatTurn
from app.models.document import Document, DocumentStatus

_CONVERSATION_KEY_PREFIX = "conversation:"
_DOCUMENT_KEY_PREFIX = "document:"
_DOCUMENT_INDEX_KEY = "documents:index"
_CACHE_KEY_PREFIX = "cache:qa:"
_CACHE_TTL_SECONDS = 3600


class RedisService:
    """Async Redis client wrapper for memory, document registry, and caching."""

    def __init__(self) -> None:
        settings = get_settings()
        self._ttl = settings.redis_conversation_ttl_seconds

        if settings.redis_url:
            # Hosted Redis (Upstash, Redis Cloud, etc.) via a full connection URL,
            # e.g. rediss://default:PASSWORD@your-host.upstash.io:6379
            self._client = redis.from_url(settings.redis_url, decode_responses=True)
        else:
            # Local Redis running in Docker on this machine.
            self._client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                decode_responses=True,
            )

    # ------------------------------------------------------------------ #
    # Conversation memory
    # ------------------------------------------------------------------ #

    @staticmethod
    def new_conversation_id() -> str:
        return str(uuid.uuid4())

    async def append_chat_turn(self, turn: ChatTurn) -> None:
        key = f"{_CONVERSATION_KEY_PREFIX}{turn.conversation_id}"
        try:
            await self._client.rpush(key, turn.model_dump_json())
            await self._client.expire(key, self._ttl)
        except redis.RedisError as exc:
            logger.error(f"Failed to append chat turn to Redis: {exc}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Failed to persist conversation memory: {exc}",
            ) from exc

    async def get_conversation_history(self, conversation_id: str, limit: int = 20) -> list[ChatTurn]:
        key = f"{_CONVERSATION_KEY_PREFIX}{conversation_id}"
        try:
            raw_turns = await self._client.lrange(key, -limit, -1)
        except redis.RedisError as exc:
            logger.error(f"Failed to read conversation history from Redis: {exc}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Failed to read conversation memory: {exc}",
            ) from exc
        return [ChatTurn.model_validate_json(raw) for raw in raw_turns]

    async def get_recent_context(self, conversation_id: str, max_turns: int = 3) -> str:
        """Return a compact string of recent Q/A turns for prompt context."""
        turns = await self.get_conversation_history(conversation_id, limit=max_turns)
        if not turns:
            return ""
        lines = [f"User: {t.question}\nAssistant: {t.answer}" for t in turns]
        return "\n\n".join(lines)

    # ------------------------------------------------------------------ #
    # Document registry
    # ------------------------------------------------------------------ #

    async def save_document(self, document: Document) -> None:
        key = f"{_DOCUMENT_KEY_PREFIX}{document.document_id}"
        try:
            await self._client.set(key, document.model_dump_json())
            await self._client.sadd(_DOCUMENT_INDEX_KEY, document.document_id)
        except redis.RedisError as exc:
            logger.error(f"Failed to save document metadata to Redis: {exc}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Failed to save document metadata: {exc}",
            ) from exc

    async def update_document_status(
        self,
        document_id: str,
        status_value: DocumentStatus,
        chunk_count: int | None = None,
        error_message: str | None = None,
    ) -> None:
        document = await self.get_document(document_id)
        if document is None:
            return
        document.status = status_value
        if chunk_count is not None:
            document.chunk_count = chunk_count
        if error_message is not None:
            document.error_message = error_message
        await self.save_document(document)

    async def get_document(self, document_id: str) -> Document | None:
        key = f"{_DOCUMENT_KEY_PREFIX}{document_id}"
        raw = await self._client.get(key)
        if raw is None:
            return None
        return Document.model_validate_json(raw)

    async def list_documents(self) -> list[Document]:
        try:
            document_ids = await self._client.smembers(_DOCUMENT_INDEX_KEY)
        except redis.RedisError as exc:
            logger.error(f"Failed to list documents from Redis: {exc}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Failed to list documents: {exc}",
            ) from exc

        documents: list[Document] = []
        for doc_id in document_ids:
            doc = await self.get_document(doc_id)
            if doc:
                documents.append(doc)
        documents.sort(key=lambda d: d.created_date, reverse=True)
        return documents

    async def delete_document(self, document_id: str) -> None:
        key = f"{_DOCUMENT_KEY_PREFIX}{document_id}"
        await self._client.delete(key)
        await self._client.srem(_DOCUMENT_INDEX_KEY, document_id)

    # ------------------------------------------------------------------ #
    # Frequently-asked-question cache
    # ------------------------------------------------------------------ #

    @staticmethod
    def _cache_key(question: str, language: str | None = None) -> str:
        normalized = f"{question.strip().lower()}|{(language or '').strip().lower()}"
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return f"{_CACHE_KEY_PREFIX}{digest}"

    async def get_cached_answer(self, question: str, language: str | None = None) -> dict | None:
        key = self._cache_key(question, language)
        raw = await self._client.get(key)
        if raw is None:
            return None
        logger.debug(f"Cache hit for question: {question[:50]}...")
        return json.loads(raw)

    async def cache_answer(self, question: str, answer_payload: dict, language: str | None = None) -> None:
        key = self._cache_key(question, language)
        await self._client.set(key, json.dumps(answer_payload), ex=_CACHE_TTL_SECONDS)

    # ------------------------------------------------------------------ #
    # Health
    # ------------------------------------------------------------------ #

    async def check_health(self) -> bool:
        try:
            return bool(await self._client.ping())
        except redis.RedisError:
            return False
