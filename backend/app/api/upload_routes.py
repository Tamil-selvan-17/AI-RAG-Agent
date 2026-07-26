"""Document upload and listing endpoints."""

from fastapi import APIRouter, UploadFile, status

from app.core.logging import logger
from app.core.security import validate_extension, validate_upload_size
from app.schemas.upload_schema import (
    DocumentListItem,
    DocumentListResponse,
    DocumentUploadResponse,
)
from app.services.rag_service import RagService

router = APIRouter(prefix="/api/documents", tags=["documents"])

_rag_service = RagService()


@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile) -> DocumentUploadResponse:
    """
    Upload a document (PDF, DOCX, TXT, or CSV).

    Runs the full ingestion pipeline: validate -> extract -> chunk -> embed -> store.
    """
    validate_extension(file.filename or "")
    contents = await validate_upload_size(file)

    logger.info(f"Received upload: {file.filename} ({len(contents)} bytes)")

    document = await _rag_service.ingest_document(file.filename or "upload", contents)

    return DocumentUploadResponse(
        document_id=document.document_id,
        filename=document.filename,
        status=document.status.value,
        chunk_count=document.chunk_count,
        message=f"Document processed successfully into {document.chunk_count} chunks",
    )


@router.get("", response_model=DocumentListResponse)
async def list_documents() -> DocumentListResponse:
    """List all uploaded documents and their processing status."""
    documents = await _rag_service.list_documents()
    items = [
        DocumentListItem(
            document_id=d.document_id,
            filename=d.filename,
            file_extension=d.file_extension,
            file_size_bytes=d.file_size_bytes,
            status=d.status.value,
            chunk_count=d.chunk_count,
            created_date=d.created_date,
            error_message=d.error_message,
        )
        for d in documents
    ]
    return DocumentListResponse(total=len(items), documents=items)


@router.delete("/{document_id}", status_code=status.HTTP_200_OK)
async def delete_document(document_id: str) -> dict:
    """Delete a single document: its vectors, its registry entry, and its file on disk."""
    document = await _rag_service.delete_document(document_id)
    return {"message": f"Deleted document '{document.filename}'", "document_id": document_id}


@router.delete("", status_code=status.HTTP_200_OK)
async def delete_all_documents() -> dict:
    """Delete every uploaded document: all vectors, all registry entries, all files."""
    count = await _rag_service.delete_all_documents()
    return {"message": f"Deleted {count} document(s)", "count": count}
