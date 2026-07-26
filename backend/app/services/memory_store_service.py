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
import uuid
from collections import defaultdict

from app.core.logging import logger
from app.models.chat import ChatTurn
from app.models.document import Document, DocumentStatus


class MemoryStoreService:
    """In-process, non-persistent stand-in for RedisService."""

    def __init__(self) -> None:
        logger.warning(
            "MEMORY_BACKEND=memory is active: conversation history, the document "
            "catalog, and the answer cache all live only in RAM and will reset "
            "whenever this process restarts. Set MEMORY_BACKEND=redis with a "
            "REDIS_URL/host for persistence instead."
        )
        self._conversations: dict[str, list[ChatTurn]] = defaultdict(list)
        self._documents: dict[str, Document] = {}
        self._cache: dict[str, dict] = {}

    # ------------------------------------------------------------------ #
    # Conversation memory
    # ------------------------------------------------------------------ #

    @staticmethod
    def new_conversation_id() -> str:
        return str(uuid.uuid4())

    async def append_chat_turn(self, turn: ChatTurn) -> None:
        self._conversations[turn.conversation_id].append(turn)

    async def get_conversation_history(self, conversation_id: str, limit: int = 20) -> list[ChatTurn]:
        return self._conversations.get(conversation_id, [])[-limit:]

    async def get_recent_context(self, conversation_id: str, max_turns: int = 3) -> str:
        turns = await self.get_conversation_history(conversation_id, limit=max_turns)
        if not turns:
            return ""
        lines = [f"User: {t.question}\nAssistant: {t.answer}" for t in turns]
        return "\n\n".join(lines)

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
        cached = self._cache.get(self._cache_key(question, language))
        if cached is None:
            return None
        logger.debug(f"In-memory cache hit for question: {question[:50]}...")
        return json.loads(json.dumps(cached))  # deep-ish copy to mirror Redis round-trip

    async def cache_answer(self, question: str, answer_payload: dict, language: str | None = None) -> None:
        self._cache[self._cache_key(question, language)] = answer_payload

    # ------------------------------------------------------------------ #
    # Health
    # ------------------------------------------------------------------ #

    async def check_health(self) -> bool:
        return True
