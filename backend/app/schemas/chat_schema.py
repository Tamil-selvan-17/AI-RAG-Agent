"""API request/response schemas for the chat endpoints."""

from datetime import datetime

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    conversation_id: str | None = Field(
        default=None, description="Existing conversation id; a new one is created if omitted."
    )
    top_k: int | None = Field(default=None, ge=1, le=20)
    response_language: str | None = Field(
        default=None,
        max_length=50,
        description=(
            "Optional override to force the answer into a specific language "
            "(e.g. 'English', 'Tamil'), regardless of what language the question "
            "was asked in. If omitted, the answer language is auto-detected from "
            "the question."
        ),
    )


class SourceReferenceResponse(BaseModel):
    document_id: str
    filename: str
    chunk_id: str
    chunk_text: str
    score: float


class ChatResponse(BaseModel):
    conversation_id: str
    question: str
    answer: str
    sources: list[SourceReferenceResponse]


class ChatHistoryTurn(BaseModel):
    question: str
    answer: str
    sources: list[SourceReferenceResponse]
    timestamp: datetime


class ChatHistoryResponse(BaseModel):
    conversation_id: str
    turns: list[ChatHistoryTurn]
