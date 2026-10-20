"""
In-memory store service.

A drop-in replacement for RedisService with the exact same method signatures,
used when MEMORY_BACKEND=memory so the project can run with no Redis/Upstash
dependency at all.

Trade-off: everything lives only in this process's RAM. Conversation history,
the document catalog, and the answer cache are all lost whenever the backend
process restarts (e.g. a Render free-tier instance waking from sleep, or a
local `--reload` restart). Document *knowledge* itself is unaffected, since
the actual chunk vectors remain safely in Qdrant.
"""

import hashlib
import json
import time
import uuid
from collections import OrderedDict

from app.core.logging import logger
from app.models.chat import ChatTurn
from app.models.document import Document, DocumentStatus

# Bounds so a long-running process can't grow without limit. The least recently
# used conversations / cache entries are dropped first. Cache TTL matches Redis.
_MAX_CONVERSATIONS = 1000
_MAX_TURNS_PER_CONVERSATION = 100
_MAX_CACHE_ENTRIES = 500
_CACHE_TTL_SECONDS = 3600


class MemoryStoreService:
    """In-process, non-persistent stand-in for RedisService."""

    def __init__(self) -> None:
        logger.warning(
            "MEMORY_BACKEND=memory is active: conversation history, the document "
            "catalog, and the answer cache all live only in RAM and will reset "
            "whenever this process restarts. Set MEMORY_BACKEND=redis with a "
            "REDIS_URL/host for persistence instead."
        )
        self._conversations: OrderedDict[str, list[ChatTurn]] = OrderedDict()
        self._documents: dict[str, Document] = {}
        self._cache: OrderedDict[str, tuple[float, dict]] = OrderedDict()

    # ------------------------------------------------------------------ #
    # Conversation memory
    # ------------------------------------------------------------------ #

    @staticmethod
    def new_conversation_id() -> str:
        return str(uuid.uuid4())

    async def append_chat_turn(self, turn: ChatTurn) -> None:
        turns = self._conversations.setdefault(turn.conversation_id, [])
        turns.append(turn)
        del turns[:-_MAX_TURNS_PER_CONVERSATION]
        self._conversations.move_to_end(turn.conversation_id)
        while len(self._conversations) > _MAX_CONVERSATIONS:
            self._conversations.popitem(last=False)

    async def get_conversation_history(self, conversation_id: str, limit: int = 20) -> list[ChatTurn]:
        return list(self._conversations.get(conversation_id, [])[-limit:])

    # ------------------------------------------------------------------ #
    # Document registry
    # ------------------------------------------------------------------ #

    async def save_document(self, document: Document) -> None:
        self._documents[document.document_id] = document

    async def update_document_status(
        self,
        document_id: str,
        status_value: DocumentStatus,
        chunk_count: int | None = None,
        error_message: str | None = None,
    ) -> None:
        document = self._documents.get(document_id)
        if document is None:
            return
        document.status = status_value
        if chunk_count is not None:
            document.chunk_count = chunk_count
        if error_message is not None:
            document.error_message = error_message
        self._documents[document_id] = document

    async def get_document(self, document_id: str) -> Document | None:
        return self._documents.get(document_id)

    async def list_documents(self) -> list[Document]:
        return sorted(self._documents.values(), key=lambda d: d.created_date, reverse=True)

    async def delete_document(self, document_id: str) -> None:
        self._documents.pop(document_id, None)

    # ------------------------------------------------------------------ #
    # Frequently-asked-question cache
    # ------------------------------------------------------------------ #

    @staticmethod
    def _cache_key(question: str, language: str | None = None) -> str:
        normalized = f"{question.strip().lower()}|{(language or '').strip().lower()}"
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    async def get_cached_answer(self, question: str, language: str | None = None) -> dict | None:
        key = self._cache_key(question, language)
        entry = self._cache.get(key)
        if entry is None:
            return None
        stored_at, payload = entry
        if time.monotonic() - stored_at > _CACHE_TTL_SECONDS:
            del self._cache[key]
            return None
        self._cache.move_to_end(key)
        logger.debug(f"In-memory cache hit for question: {question[:50]}...")
        return json.loads(json.dumps(payload))  # deep-ish copy to mirror Redis round-trip

    async def cache_answer(self, question: str, answer_payload: dict, language: str | None = None) -> None:
        key = self._cache_key(question, language)
        self._cache[key] = (time.monotonic(), answer_payload)
        self._cache.move_to_end(key)
        while len(self._cache) > _MAX_CACHE_ENTRIES:
            self._cache.popitem(last=False)

    async def clear_answer_cache(self) -> None:
        self._cache.clear()

    # ------------------------------------------------------------------ #
    # Health
    # ------------------------------------------------------------------ #

    async def check_health(self) -> bool:
        return True
