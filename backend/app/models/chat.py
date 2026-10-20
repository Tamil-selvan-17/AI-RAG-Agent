"""Domain models representing chat turns and source references."""

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class SourceReference(BaseModel):
    """A retrieved chunk cited as a source for an answer."""

    document_id: str
    filename: str
    chunk_id: str
    chunk_text: str
    score: float
    # PDF page range the chunk came from (None for non-PDF / older uploads).
    page_start: int | None = None
    page_end: int | None = None
    # Section heading the chunk is under (e.g. "KEY PROJECTS > 2. nVMS"), if detected.
    section: str | None = None


class ChatTurn(BaseModel):
    """A single question/answer pair stored in conversation memory."""

    conversation_id: str
    question: str
    answer: str
    sources: list[SourceReference] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
