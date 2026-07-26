"""
Advanced chunking service.

Implements a recursive character-based text splitter with configurable
overlap, similar in spirit to LangChain's RecursiveCharacterTextSplitter,
but self-contained (no external dependency) for full control and stability.

Sizes are measured in approximate tokens using a simple heuristic
(1 token ~= 4 characters), which is accurate enough for chunk sizing
purposes without requiring a tokenizer model.
"""

import uuid
from datetime import datetime, timezone

from app.core.config import get_settings
from app.core.logging import logger
from app.models.document import DocumentChunk

# Separators tried in order, from largest semantic unit to smallest.
_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]

_CHARS_PER_TOKEN = 4


class ChunkingService:
    """Splits cleaned document text into overlapping, semantically-aware chunks."""

    def __init__(self, chunk_size_tokens: int | None = None, chunk_overlap_tokens: int | None = None):
        settings = get_settings()
        self.chunk_size_tokens = chunk_size_tokens or settings.chunk_size
        self.chunk_overlap_tokens = chunk_overlap_tokens or settings.chunk_overlap
        self.chunk_size_chars = self.chunk_size_tokens * _CHARS_PER_TOKEN
        self.chunk_overlap_chars = self.chunk_overlap_tokens * _CHARS_PER_TOKEN

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using the first separator that yields small-enough pieces."""
        if len(text) <= self.chunk_size_chars:
            return [text] if text.strip() else []

        if not separators:
            # Base case: hard split by character length.
            return [
                text[i : i + self.chunk_size_chars]
                for i in range(0, len(text), self.chunk_size_chars)
            ]

        sep, remaining_separators = separators[0], separators[1:]
        if sep == "":
            parts = list(text)
        else:
            parts = text.split(sep)

        # Reassemble parts into chunks close to chunk_size_chars.
        chunks: list[str] = []
        current = ""
        for part in parts:
            candidate = current + (sep if current else "") + part if sep else current + part
            if len(candidate) <= self.chunk_size_chars:
                current = candidate
            else:
                if current.strip():
                    chunks.append(current)
                if len(part) > self.chunk_size_chars:
                    # Part itself too large; recurse with remaining separators.
                    chunks.extend(self._split_recursive(part, remaining_separators))
                    current = ""
                else:
                    current = part
        if current.strip():
            chunks.append(current)

        return chunks

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        """Prepend a trailing slice of the previous chunk to each chunk for context continuity."""
        if self.chunk_overlap_chars <= 0 or len(chunks) <= 1:
            return chunks

        overlapped: list[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-self.chunk_overlap_chars :]
            overlapped.append(f"{prev_tail} {chunks[i]}".strip())
        return overlapped

    def split_text(self, text: str) -> list[str]:
        """Split raw text into overlapping chunks bounded by chunk_size_tokens."""
        if not text or not text.strip():
            return []

        raw_chunks = self._split_recursive(text, _SEPARATORS)
        raw_chunks = [c.strip() for c in raw_chunks if c.strip()]
        return self._apply_overlap(raw_chunks)

    def chunk_document(
        self,
        text: str,
        document_id: str,
        filename: str,
        extra_metadata: dict | None = None,
    ) -> list[DocumentChunk]:
        """Split document text and wrap each piece as a DocumentChunk with metadata."""
        pieces = self.split_text(text)
        total = len(pieces)
        now = datetime.now(timezone.utc)

        chunks = [
            DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                document_id=document_id,
                filename=filename,
                chunk_text=piece,
                chunk_index=idx,
                total_chunks=total,
                metadata={
                    **(extra_metadata or {}),
                    "char_count": len(piece),
                },
                created_date=now,
            )
            for idx, piece in enumerate(pieces)
        ]

        logger.info(f"Chunked '{filename}' into {total} chunks (document_id={document_id})")
        return chunks
