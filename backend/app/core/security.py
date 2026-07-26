"""
Security-related helpers: filename sanitization, extension/MIME allow-listing,
and file size validation. These guard the upload endpoint against path
traversal, disallowed file types, and oversized uploads.
"""

import re
import uuid
from pathlib import PurePosixPath

from fastapi import HTTPException, UploadFile, status

from app.core.config import get_settings

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".csv"}

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.\-]")


def sanitize_filename(filename: str) -> str:
    """Strip path components and unsafe characters from a client-supplied filename."""
    name = PurePosixPath(filename).name  # drop any directory components
    name = _SAFE_NAME_RE.sub("_", name)
    if not name:
        name = "file"
    return name


def validate_extension(filename: str) -> str:
    """Validate the file extension is in the allow-list. Returns lowercase extension."""
    ext = PurePosixPath(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file type '{ext}'. "
                f"Allowed types: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )
    return ext


async def validate_upload_size(file: UploadFile) -> bytes:
    """Read the full upload into memory while enforcing the max size limit."""
    settings = get_settings()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024

    contents = await file.read()
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {settings.max_upload_size_mb}MB",
        )
    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty",
        )
    return contents


def generate_document_id() -> str:
    return str(uuid.uuid4())
