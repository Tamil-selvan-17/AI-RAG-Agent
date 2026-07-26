"""Chat and conversation history endpoints."""

import json

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from app.core.logging import logger
from app.schemas.chat_schema import (
    ChatHistoryResponse,
    ChatHistoryTurn,
    ChatRequest,
    ChatResponse,
    SourceReferenceResponse,
)
from app.services.rag_service import RagService

router = APIRouter(prefix="/api/chat", tags=["chat"])

_rag_service = RagService()


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Ask a question against the uploaded document knowledge base (waits for the full answer).

    Prefer POST /api/chat/stream from a browser -- it returns the same information but
    starts sending bytes back immediately as the answer is generated, which avoids
    504 Gateway Timeout errors on slow answers and lets the UI show the answer typing
    out in real time. This endpoint is kept for simple API/script consumers that just
    want one JSON response.
    """
    logger.info(f"Chat request: '{request.question[:80]}' (conversation_id={request.conversation_id})")

    conversation_id, answer, sources = await _rag_service.answer_question(
        question=request.question,
        conversation_id=request.conversation_id,
        top_k=request.top_k,
        response_language=request.response_language,
    )

    return ChatResponse(
        conversation_id=conversation_id,
        question=request.question,
        answer=answer,
        sources=[
            SourceReferenceResponse(
                document_id=s.document_id,
                filename=s.filename,
                chunk_id=s.chunk_id,
                chunk_text=s.chunk_text,
                score=s.score,
            )
            for s in sources
        ],
    )


@router.post("/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Ask a question and stream the answer back as Server-Sent Events, token by token.

    Each event is one line of the form `data: {...}\\n\\n`, where the JSON payload is
    one of:
      {"type": "token", "content": "..."}                        -- an answer fragment
      {"type": "done", "conversation_id": "...", "sources": [...]} -- always sent last on success
      {"type": "error", "detail": "..."}                          -- sent instead of "done" on failure

    Streaming the HTTP response itself (not just internally between this backend and the
    AI provider) is what actually prevents 504 Gateway Timeout errors from any reverse
    proxy in front of this app (Render, nginx, etc.): those proxies time out requests that
    go silent for too long, and a real byte-by-byte stream never looks silent to them,
    even if the full answer takes a couple of minutes on slower hardware.
    """
    logger.info(
        f"Streaming chat request: '{request.question[:80]}' (conversation_id={request.conversation_id})"
    )

    async def event_source():
        async for event in _rag_service.stream_answer(
            question=request.question,
            conversation_id=request.conversation_id,
            top_k=request.top_k,
            response_language=request.response_language,
        ):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # tell nginx-style proxies not to buffer this response
        },
    )


@router.get("/history/{conversation_id}", response_model=ChatHistoryResponse)
async def get_chat_history(conversation_id: str) -> ChatHistoryResponse:
    """Retrieve the full conversation history for a given conversation_id."""
    turns = await _rag_service.get_history(conversation_id)
    if not turns:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No conversation history found for id '{conversation_id}'",
        )

    return ChatHistoryResponse(
        conversation_id=conversation_id,
        turns=[
            ChatHistoryTurn(
                question=t.question,
                answer=t.answer,
                sources=[
                    SourceReferenceResponse(
                        document_id=s.document_id,
                        filename=s.filename,
                        chunk_id=s.chunk_id,
                        chunk_text=s.chunk_text,
                        score=s.score,
                    )
                    for s in t.sources
                ],
                timestamp=t.timestamp,
            )
            for t in turns
        ],
    )
