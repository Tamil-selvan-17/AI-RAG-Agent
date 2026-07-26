"""API request/response schemas for the document upload endpoints."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class DocumentUploadResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    document_id: str
    filename: str
    status: str
    chunk_count: int
    message: str


class DocumentListItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    document_id: str
    filename: str
    file_extension: str
    file_size_bytes: int
    status: str
    chunk_count: int
    created_date: datetime
    error_message: str | None = None


class DocumentListResponse(BaseModel):
    total: int
    documents: list[DocumentListItem]
