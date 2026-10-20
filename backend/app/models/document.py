"""Domain models representing documents and their chunks."""

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class DocumentChunk(BaseModel):
    """A single chunk of a document, ready to be embedded and stored."""

    chunk_id: str
    document_id: str
    filename: str
    chunk_text: str
    chunk_index: int
    total_chunks: int
    metadata: dict = Field(default_factory=dict)
    created_date: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Document(BaseModel):
    """Represents an uploaded document and its processing state."""

    document_id: str
    filename: str
    file_extension: str
    file_size_bytes: int
    # Name of the file on disk (unique per document). None for documents saved
    # before this field existed, which were stored under their plain filename.
    stored_filename: str | None = None
    # SHA-256 of the uploaded bytes, used to reject duplicate uploads.
    content_hash: str | None = None
    status: DocumentStatus = DocumentStatus.PROCESSING
    chunk_count: int = 0
    error_message: str | None = None
    created_date: datetime = Field(default_factory=lambda: datetime.now(UTC))
