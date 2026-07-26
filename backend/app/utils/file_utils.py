"""Filesystem helpers used by the document service."""

from pathlib import Path

from app.core.logging import logger


def save_file_to_disk(directory: Path, filename: str, contents: bytes) -> Path:
    """Persist uploaded bytes to disk and return the resulting path."""
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / filename

    # Avoid overwriting an existing file with the same name.
    if destination.exists():
        stem, suffix = destination.stem, destination.suffix
        counter = 1
        while destination.exists():
            destination = directory / f"{stem}_{counter}{suffix}"
            counter += 1

    destination.write_bytes(contents)
    logger.info(f"Saved uploaded file to {destination}")
    return destination


def clean_text(raw_text: str) -> str:
    """Normalize whitespace and strip control characters from extracted text."""
    if not raw_text:
        return ""

    # Normalize line endings and collapse excessive blank lines / whitespace.
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    cleaned = "\n".join(lines)

    # Collapse repeated spaces.
    while "  " in cleaned:
        cleaned = cleaned.replace("  ", " ")

    return cleaned.strip()
