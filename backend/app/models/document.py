"""Domain models representing documents and their chunks."""

from datetime import datetime, timezone
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
    created_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Document(BaseModel):
    """Represents an uploaded document and its processing state."""

    document_id: str
    filename: str
    file_extension: str
    file_size_bytes: int
    status: DocumentStatus = DocumentStatus.PROCESSING
    chunk_count: int = 0
    error_message: str | None = None
    created_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
