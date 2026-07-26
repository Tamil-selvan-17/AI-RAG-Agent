"""Domain models representing chat turns and source references."""

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class SourceReference(BaseModel):
    """A retrieved chunk cited as a source for an answer."""

    document_id: str
    filename: str
    chunk_id: str
    chunk_text: str
    score: float


class ChatTurn(BaseModel):
    """A single question/answer pair stored in conversation memory."""

    conversation_id: str
    question: str
    answer: str
    sources: list[SourceReference] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
