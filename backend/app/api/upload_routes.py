"""Document upload and listing endpoints."""

from fastapi import APIRouter, BackgroundTasks, Depends, Response, UploadFile, status

from app.core.logging import logger
from app.core.security import validate_extension, validate_upload_size
from app.schemas.upload_schema import (
    DocumentListItem,
    DocumentListResponse,
    DocumentUploadResponse,
)
from app.services.container import get_rag_service
from app.services.rag_service import RagService

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={201: {"description": "Processed synchronously (wait=true)"}},
)
async def upload_document(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    response: Response,
    wait: bool = False,
    rag: RagService = Depends(get_rag_service),
) -> DocumentUploadResponse:
    """
    Upload a document (PDF, DOCX, TXT, CSV) or an image (JPG, PNG, WEBP, GIF, BMP).

    By default the file is saved and the request returns immediately (202) with
    status "processing"; extraction, embedding and indexing continue in the
    background. Poll GET /api/documents until its status is "ready" or "failed".
    Pass `?wait=true` to block until processing finishes instead (201).

    Images are described/OCR'd via the active AI provider's vision capability and
    indexed the same way as extracted document text.
    """
    validate_extension(file.filename or "")
    contents = await validate_upload_size(file)

    logger.info(f"Received upload: {file.filename} ({len(contents)} bytes)")

    document = await rag.register_upload(file.filename or "upload", contents)

    if wait:
        document = await rag.process_document(document, contents)
        response.status_code = status.HTTP_201_CREATED
        message = f"Document processed successfully into {document.chunk_count} chunks"
    else:
        background_tasks.add_task(rag.process_document_in_background, document, contents)
        message = "Upload received; the document is being indexed in the background"

    return DocumentUploadResponse(
        document_id=document.document_id,
        filename=document.filename,
        status=document.status.value,
        chunk_count=document.chunk_count,
        message=message,
    )


@router.get("", response_model=DocumentListResponse)
async def list_documents(rag: RagService = Depends(get_rag_service)) -> DocumentListResponse:
    """List all uploaded documents and their processing status."""
    documents = await rag.list_documents()
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
async def delete_document(document_id: str, rag: RagService = Depends(get_rag_service)) -> dict:
    """Delete a single document: its vectors, its registry entry, and its file on disk."""
    document = await rag.delete_document(document_id)
    return {"message": f"Deleted document '{document.filename}'", "document_id": document_id}


@router.delete("", status_code=status.HTTP_200_OK)
async def delete_all_documents(rag: RagService = Depends(get_rag_service)) -> dict:
    """Delete every uploaded document: all vectors, all registry entries, all files."""
    count = await rag.delete_all_documents()
    return {"message": f"Deleted {count} document(s)", "count": count}
