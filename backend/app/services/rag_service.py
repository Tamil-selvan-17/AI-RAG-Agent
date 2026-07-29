"""
RAG service.

Orchestrates the full pipeline end to end:

Ingestion:  file -> extract text -> clean -> chunk -> embed -> store in Qdrant
Retrieval:  query -> embed -> Qdrant similarity search -> top-k chunks
Generation: system prompt + retrieved context + question -> Ollama LLM answer
Memory:     persist each turn to Redis; use recent turns as extra context
"""

from datetime import datetime
from pathlib import Path

from fastapi import HTTPException, status

from app.core.config import get_settings
from app.core.logging import logger
from app.core.security import (
    generate_document_id,
    image_mime_type,
    is_image_extension,
    sanitize_filename,
    validate_extension,
)
from app.models.chat import ChatTurn, SourceReference
from app.models.document import Document, DocumentStatus
from app.services.chunking_service import ChunkingService
from app.services.document_service import DocumentService
from app.services.embedding_service import EmbeddingService
from app.services.qdrant_service import QdrantService
from app.services.redis_service import RedisService
from app.services.rerank_service import rerank
from app.utils.chitchat_utils import detect_smalltalk_reply
from app.utils.file_utils import clean_text, save_file_to_disk
from app.utils.language_utils import detect_language_name


def _default_llm_backend():
    """Return the chat-generation backend for the active AI_PROVIDER setting."""
    settings = get_settings()
    if settings.ai_provider == "gemini":
        from app.services.gemini_service import GeminiService

        return GeminiService()
    from app.services.ollama_service import OllamaService

    return OllamaService()


def _default_memory_backend():
    """Return the memory/catalog/cache backend for the active MEMORY_BACKEND setting."""
    settings = get_settings()
    if settings.memory_backend == "memory":
        from app.services.memory_store_service import MemoryStoreService

        return MemoryStoreService()
    return RedisService()


def _build_system_prompt(language_name: str | None = None) -> str:
    """Build the system prompt with today's real date and an explicit answer-language rule.

    Naming the language explicitly (e.g. "respond entirely in Tamil") is much more
    reliable than telling the model to "match the question's language" and hoping
    it infers correctly -- multilingual models like qwen2.5 can otherwise drift into
    a different language mid-answer, especially when the source documents are in a
    different language than the question.
    """
    today = datetime.now().strftime("%B %d, %Y")

    if language_name:
        language_rule = (
            f"The user's question is written in {language_name}. You must write your "
            f"entire answer in {language_name} from start to finish, even if the source "
            "documents are in a different language. Never switch languages partway "
            "through a response, and never mix two languages in the same answer unless "
            "the user explicitly asked for a translation."
        )
    else:
        language_rule = (
            "Always answer in the same language the user's question was written in, even if "
            "the source document is in a different language. Write your entire answer in that "
            "one language from start to finish — never switch languages partway through a "
            "response, and never mix two languages in the same answer unless the user "
            "explicitly asked for a translation."
        )

    return (
        "You are an AI assistant. Answer only from provided context. "
        f"Today's date is {today}. When the context contains dates (e.g. employment "
        "start dates, 'Present', durations), ALWAYS compute relative time spans (such as "
        "years of experience) yourself using today's date above — do not rely on your own "
        "assumption of the current date, and do not simply quote a summary phrase from the "
        "document (e.g. a resume saying '3+ years of experience') as your final answer. "
        "Documents are often written at an earlier point in time and such self-described "
        "durations go stale; a literal start date (e.g. 'June 2022 - Present') is always "
        "more reliable than a prose summary claim, so calculate the actual duration from "
        "that start date to today's real date and use that calculated figure as the answer, "
        "even if it differs from a phrase written in the document. "
        f"{language_rule} "
        "If the context does not contain enough information to answer the question, "
        "say so clearly instead of guessing. Cite specific details from the context "
        "where relevant, but do not fabricate information that is not present in it."
    )


class RagService:
    """Composes document processing, retrieval, and generation services."""

    def __init__(
        self,
        document_service: DocumentService | None = None,
        chunking_service: ChunkingService | None = None,
        embedding_service: EmbeddingService | None = None,
        qdrant_service: QdrantService | None = None,
        redis_service: RedisService | None = None,
        llm_service=None,
    ) -> None:
        settings = get_settings()
        self.settings = settings
        self.document_service = document_service or DocumentService()
        self.chunking_service = chunking_service or ChunkingService()
        self.embedding_service = embedding_service or EmbeddingService()
        self.qdrant_service = qdrant_service or QdrantService()
        self.redis_service = redis_service or _default_memory_backend()
        self.llm_service = llm_service or _default_llm_backend()


    # ------------------------------------------------------------------ #
    # Ingestion pipeline
    # ------------------------------------------------------------------ #

    async def ingest_document(self, original_filename: str, contents: bytes) -> Document:
        """Full ingestion pipeline: validate -> save -> extract -> chunk -> embed -> store."""
        safe_name = sanitize_filename(original_filename)
        extension = validate_extension(safe_name)
        document_id = generate_document_id()

        document = Document(
            document_id=document_id,
            filename=safe_name,
            file_extension=extension,
            file_size_bytes=len(contents),
            status=DocumentStatus.PROCESSING,
        )
        await self.redis_service.save_document(document)

        try:
            file_path = save_file_to_disk(self.settings.upload_dir_path, safe_name, contents)

            if is_image_extension(extension):
                # Images have no local text to extract -- describe/OCR them via the
                # active AI provider's vision capability instead, then treat that
                # description as the document's text for chunking/embedding, so an
                # uploaded image becomes searchable through the exact same RAG
                # pipeline as any PDF or DOCX.
                text = await self.llm_service.describe_image(
                    contents, image_mime_type(extension)
                )
                text = clean_text(text)
                if not text:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail="No description could be generated for this image",
                    )
            else:
                text = self.document_service.extract_text(file_path, extension)

            chunks = self.chunking_service.chunk_document(
                text=text,
                document_id=document_id,
                filename=safe_name,
                extra_metadata={"source_path": str(file_path), "file_extension": extension},
            )
            if not chunks:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Document produced no usable chunks",
                )

            await self.qdrant_service.ensure_collection()

            vectors = await self.embedding_service.embed_batch([c.chunk_text for c in chunks])

            await self.qdrant_service.upsert_chunks(chunks, vectors)

            document.status = DocumentStatus.READY
            document.chunk_count = len(chunks)
            await self.redis_service.save_document(document)

            logger.info(f"Ingested document '{safe_name}' ({document_id}) with {len(chunks)} chunks")
            return document

        except HTTPException as exc:
            await self.redis_service.update_document_status(
                document_id, DocumentStatus.FAILED, error_message=exc.detail
            )
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Unexpected ingestion failure for '{safe_name}': {exc}")
            await self.redis_service.update_document_status(
                document_id, DocumentStatus.FAILED, error_message=str(exc)
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to process document: {exc}",
            ) from exc

    async def list_documents(self) -> list[Document]:
        documents = await self.redis_service.list_documents()
        if documents:
            return documents

        # Registry is empty (e.g. after a restart with MEMORY_BACKEND=memory), but
        # Qdrant may still hold real, previously-uploaded vectors. Rebuild the
        # registry from Qdrant directly so those documents don't just disappear
        # from the UI, and persist the rebuilt entries for future fast lookups.
        rebuilt = await self.qdrant_service.list_documents_from_vectors()
        if not rebuilt:
            return []

        logger.info(f"Rebuilt document registry from Qdrant: {len(rebuilt)} document(s) found")
        restored: list[Document] = []
        for entry in rebuilt:
            filename = entry["filename"]
            file_extension = Path(filename).suffix or ""
            created_date_str = entry.get("created_date")
            try:
                created_date = datetime.fromisoformat(created_date_str) if created_date_str else datetime.now()
            except ValueError:
                created_date = datetime.now()

            document = Document(
                document_id=entry["document_id"],
                filename=filename,
                file_extension=file_extension,
                file_size_bytes=0,  # unknown after a registry reset -- not stored in Qdrant payload
                status=DocumentStatus.READY,
                chunk_count=entry["chunk_count"],
                created_date=created_date,
            )
            await self.redis_service.save_document(document)
            restored.append(document)

        restored.sort(key=lambda d: d.created_date, reverse=True)
        return restored

    async def delete_document(self, document_id: str) -> Document:
        """Delete a single document: its vectors in Qdrant, its registry entry, and its file on disk.

        Best-effort on the Qdrant step: if the vector store is unreachable (e.g. this
        document already failed to embed, or Qdrant is temporarily down), we still remove
        the registry entry and file so the document disappears from your list instead of
        being stuck forever.
        """
        document = await self.redis_service.get_document(document_id)
        if document is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No document found with id '{document_id}'",
            )

        try:
            await self.qdrant_service.delete_document(document_id)
        except HTTPException as exc:
            logger.warning(
                f"Could not delete vectors for '{document.filename}' from Qdrant "
                f"(continuing with registry/file cleanup): {exc.detail}"
            )

        await self.redis_service.delete_document(document_id)

        file_path = self.settings.upload_dir_path / document.filename
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Deleted file from disk: {file_path}")

        logger.info(f"Deleted document '{document.filename}' ({document_id})")
        return document

    async def delete_all_documents(self) -> int:
        """Delete every document: all vectors, all registry entries, and all files on disk.

        Wipes the entire Qdrant collection unconditionally rather than only deleting
        vectors for documents the current registry knows about. This matters with
        MEMORY_BACKEND=memory: the registry resets on every restart, but a persistent
        Qdrant instance doesn't, so repeated uploads across restarts can leave orphaned
        duplicate vectors that a registry-driven delete would never find or remove.
        """
        documents = await self.redis_service.list_documents()

        try:
            await self.qdrant_service.wipe_collection()
        except HTTPException as exc:
            logger.warning(f"Could not fully wipe Qdrant collection: {exc.detail}")

        for document in documents:
            await self.redis_service.delete_document(document.document_id)
            file_path = self.settings.upload_dir_path / document.filename
            if file_path.exists():
                file_path.unlink()

        logger.info(f"Deleted all {len(documents)} documents")
        return len(documents)

    # ------------------------------------------------------------------ #
    # Retrieval + generation pipeline
    # ------------------------------------------------------------------ #

    async def answer_question(
        self,
        question: str,
        conversation_id: str | None,
        top_k: int | None = None,
        response_language: str | None = None,
    ) -> tuple[str, str, list[SourceReference]]:
        """Run retrieval + generation for a question, persist memory, return results."""
        conversation_id = conversation_id or self.redis_service.new_conversation_id()
        k = top_k or self.settings.rag_top_k
        language_name = response_language or detect_language_name(question)

        smalltalk_reply = detect_smalltalk_reply(question)
        if smalltalk_reply:
            turn = ChatTurn(
                conversation_id=conversation_id,
                question=question,
                answer=smalltalk_reply,
                sources=[],
            )
            await self.redis_service.append_chat_turn(turn)
            return conversation_id, smalltalk_reply, []

        cached = await self.redis_service.get_cached_answer(question, language_name)
        if cached:
            sources = [SourceReference(**s) for s in cached["sources"]]
            turn = ChatTurn(
                conversation_id=conversation_id,
                question=question,
                answer=cached["answer"],
                sources=sources,
            )
            await self.redis_service.append_chat_turn(turn)
            return conversation_id, cached["answer"], sources

        query_vector = await self.embedding_service.embed_text(question)

        await self.qdrant_service.ensure_collection()
        # Fetch a wider candidate pool than we'll actually use, then rerank down
        # to top_k. Vector similarity alone can miss chunks that literally
        # contain the question's specific terms (names, numbers) in favor of
        # ones that are merely topically similar -- reranking with a lexical
        # signal corrects for that.
        candidate_pool_size = min(k * 4, 40)
        candidates = await self.qdrant_service.search(
            query_vector=query_vector,
            top_k=candidate_pool_size,
            score_threshold=self.settings.rag_score_threshold,
        )
        results = rerank(question, candidates, top_k=k)

        sources = [
            SourceReference(
                document_id=r["document_id"],
                filename=r["filename"],
                chunk_id=r["chunk_id"],
                chunk_text=r["chunk_text"],
                score=r["score"],
            )
            for r in results
        ]
        sources = self._deduplicate_sources(sources)

        context = self._build_context(sources)
        recent_history = await self.redis_service.get_recent_context(conversation_id, max_turns=3)

        user_prompt = self._build_user_prompt(question, context, recent_history)

        if not sources:
            answer = (
                "I couldn't find any relevant information in the uploaded documents "
                "to answer that question. Please try rephrasing, or upload a document "
                "that covers this topic."
            )
        else:
            answer = await self.llm_service.generate_chat_response(
                system_prompt=_build_system_prompt(language_name),
                user_prompt=user_prompt,
            )

        turn = ChatTurn(
            conversation_id=conversation_id,
            question=question,
            answer=answer,
            sources=sources,
        )
        await self.redis_service.append_chat_turn(turn)

        if sources:
            await self.redis_service.cache_answer(
                question,
                {"answer": answer, "sources": [s.model_dump() for s in sources]},
                language_name,
            )

        return conversation_id, answer, sources

    async def stream_answer(
        self,
        question: str,
        conversation_id: str | None,
        top_k: int | None = None,
        response_language: str | None = None,
    ):
        """Run retrieval + generation for a question, yielding events as the answer streams in.

        Yields dicts of one of these shapes, in order:
          {"type": "token", "content": "..."}          -- zero or more, as the answer is generated
          {"type": "done", "conversation_id": "...", "sources": [...]}  -- always last on success
          {"type": "error", "detail": "..."}             -- instead of "done", if generation failed

        The caller (the chat_routes streaming endpoint) is responsible for turning these
        into actual HTTP chunks (e.g. as Server-Sent Events) so bytes reach the browser
        as soon as each token is ready, rather than all at once at the very end.
        """
        conversation_id = conversation_id or self.redis_service.new_conversation_id()
        k = top_k or self.settings.rag_top_k
        language_name = response_language or detect_language_name(question)

        smalltalk_reply = detect_smalltalk_reply(question)
        if smalltalk_reply:
            yield {"type": "token", "content": smalltalk_reply}
            turn = ChatTurn(
                conversation_id=conversation_id,
                question=question,
                answer=smalltalk_reply,
                sources=[],
            )
            await self.redis_service.append_chat_turn(turn)
            yield {"type": "done", "conversation_id": conversation_id, "sources": []}
            return

        cached = await self.redis_service.get_cached_answer(question, language_name)
        if cached:
            sources = [SourceReference(**s) for s in cached["sources"]]
            yield {"type": "token", "content": cached["answer"]}
            turn = ChatTurn(
                conversation_id=conversation_id,
                question=question,
                answer=cached["answer"],
                sources=sources,
            )
            await self.redis_service.append_chat_turn(turn)
            yield {
                "type": "done",
                "conversation_id": conversation_id,
                "sources": [s.model_dump() for s in sources],
            }
            return

        try:
            query_vector = await self.embedding_service.embed_text(question)
            await self.qdrant_service.ensure_collection()
            candidate_pool_size = min(k * 4, 40)
            candidates = await self.qdrant_service.search(
                query_vector=query_vector,
                top_k=candidate_pool_size,
                score_threshold=self.settings.rag_score_threshold,
            )
            results = rerank(question, candidates, top_k=k)
        except HTTPException as exc:
            yield {"type": "error", "detail": exc.detail}
            return

        sources = [
            SourceReference(
                document_id=r["document_id"],
                filename=r["filename"],
                chunk_id=r["chunk_id"],
                chunk_text=r["chunk_text"],
                score=r["score"],
            )
            for r in results
        ]
        sources = self._deduplicate_sources(sources)

        if not sources:
            answer = (
                "I couldn't find any relevant information in the uploaded documents "
                "to answer that question. Please try rephrasing, or upload a document "
                "that covers this topic."
            )
            yield {"type": "token", "content": answer}
        else:
            context = self._build_context(sources)
            recent_history = await self.redis_service.get_recent_context(conversation_id, max_turns=3)
            user_prompt = self._build_user_prompt(question, context, recent_history)

            answer_parts: list[str] = []
            try:
                async for token in self.llm_service.stream_chat_response(
                    system_prompt=_build_system_prompt(language_name),
                    user_prompt=user_prompt,
                ):
                    answer_parts.append(token)
                    yield {"type": "token", "content": token}
            except HTTPException as exc:
                yield {"type": "error", "detail": exc.detail}
                return

            answer = "".join(answer_parts).strip()
            if not answer:
                yield {"type": "error", "detail": "The AI provider returned an empty response."}
                return

        turn = ChatTurn(
            conversation_id=conversation_id,
            question=question,
            answer=answer,
            sources=sources,
        )
        await self.redis_service.append_chat_turn(turn)

        if sources:
            await self.redis_service.cache_answer(
                question,
                {"answer": answer, "sources": [s.model_dump() for s in sources]},
                language_name,
            )

        yield {
            "type": "done",
            "conversation_id": conversation_id,
            "sources": [s.model_dump() for s in sources],
        }

    async def get_history(self, conversation_id: str) -> list[ChatTurn]:
        return await self.redis_service.get_conversation_history(conversation_id, limit=50)

    @staticmethod
    def _deduplicate_sources(sources: list[SourceReference]) -> list[SourceReference]:
        """Drop sources with identical chunk text, keeping the first (highest-scoring) one.

        Qdrant results are already sorted by score descending, so the first occurrence
        of a given chunk_text is always the best-scoring copy. Duplicates can happen if
        the same document gets uploaded more than once (e.g. across app restarts with a
        non-persistent registry but a persistent vector store), which would otherwise
        show the same passage 2-3 times as if they were independent supporting sources.
        """
        seen_text: set[str] = set()
        unique: list[SourceReference] = []
        for source in sources:
            if source.chunk_text in seen_text:
                continue
            seen_text.add(source.chunk_text)
            unique.append(source)
        return unique

    @staticmethod
    def _build_context(sources: list[SourceReference]) -> str:
        blocks = []
        for i, s in enumerate(sources, start=1):
            blocks.append(f"[Source {i} - {s.filename}]\n{s.chunk_text}")
        return "\n\n".join(blocks)

    @staticmethod
    def _build_user_prompt(question: str, context: str, recent_history: str) -> str:
        parts = []
        if recent_history:
            parts.append(f"Previous conversation:\n{recent_history}")
        parts.append(f"Context:\n{context}")
        parts.append(f"Question: {question}")
        return "\n\n".join(parts)
