"""
Advanced chunking service.

Implements a recursive character-based text splitter with configurable
overlap, similar in spirit to LangChain's RecursiveCharacterTextSplitter,
but self-contained (no external dependency) for full control and stability.

Sizes are measured in approximate tokens using a simple heuristic
(1 token ~= 4 characters), which is accurate enough for chunk sizing
purposes without requiring a tokenizer model.
"""

import re
import uuid
from datetime import UTC, datetime

from app.core.config import get_settings
from app.core.logging import logger
from app.models.document import DocumentChunk

# Separators tried in order, from largest semantic unit to smallest.
_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]

_CHARS_PER_TOKEN = 4

# Page markers written by DocumentService's PDF extractor ("[Page 3]").
_PAGE_MARKER_RE = re.compile(r"\[Page (\d+)\]")

# Heading styles recognised for contextual chunk headers.
_MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")
# "1. nPMS - Project Management ..." -- a numbered *title* has a " - " style separator,
# unlike a numbered instruction step ("3. Register it in ...").
_NUMBERED_TITLE_RE = re.compile(r"^\d{1,2}\.\s+\S.{0,40}\s[-–|:]\s.{2,80}$")
_SPACED_LETTERS_RE = re.compile(r"^(?:\w ){2,}\w(?:\s{2,}(?:\w ){1,}\w)*$")  # "S k i l l s"
_MAX_HEADING_CHARS = 90


def contextual_text(filename: str, section: str | None, chunk_text: str) -> str:
    """The text that gets embedded for a chunk: a short header naming its document
    and section, then the chunk itself.

    A chunk like "Developed and maintained a vendor management platform..." never
    says it belongs to the nVMS project or to a resume -- the header does, so a
    question like "what did he build in nVMS?" matches it much more strongly.
    """
    header = f"Document: {filename}"
    if section:
        header += f"\nSection: {section}"
    return f"{header}\n\n{chunk_text}"


def _detect_heading(line: str, previous_line: str = "") -> tuple[int, str] | None:
    """Return (level, title) if the line looks like a heading, else None.

    Level 1 = section (markdown #/##, ALL-CAPS line, spaced-letter heading),
    level 2 = subsection (markdown ### and deeper, numbered titles like "1. Project X").
    A line continuing the previous one (which ended in "," "&" etc.) is never a heading.
    """
    line = line.strip()
    if not line or len(line) > _MAX_HEADING_CHARS or _PAGE_MARKER_RE.fullmatch(line):
        return None

    markdown = _MARKDOWN_HEADING_RE.match(line)
    if markdown:
        return (1 if len(markdown.group(1)) <= 2 else 2), markdown.group(2).strip()

    previous = previous_line.strip()
    if previous.endswith((",", "&", "/", "(", " and", " or")) or (
        previous.endswith(("-", "–")) and not set(previous) <= {"-", "–", " "}  # "---" is a rule, not a wrap
    ):
        return None

    if _SPACED_LETTERS_RE.match(line):
        words = re.split(r"\s{2,}", line)
        return 1, " ".join(w.replace(" ", "") for w in words).upper()

    letters = [c for c in line if c.isalpha()]
    if (
        # Short single words like "XML" / "AWS" are acronyms, not headings.
        (len(letters) >= 5 or len(line.split()) >= 2)
        and sum(c.isupper() for c in letters) / len(letters) >= 0.9
        and not line.endswith((".", ",", ";", ":"))
    ):
        return 1, line

    if _NUMBERED_TITLE_RE.match(line) and not line.endswith((".", ",", ";")):
        return 2, line
    return None


class ChunkingService:
    """Splits cleaned document text into overlapping, semantically-aware chunks."""

    def __init__(self, chunk_size_tokens: int | None = None, chunk_overlap_tokens: int | None = None):
        settings = get_settings()
        self.chunk_size_tokens = chunk_size_tokens if chunk_size_tokens is not None else settings.chunk_size
        self.chunk_overlap_tokens = (
            chunk_overlap_tokens if chunk_overlap_tokens is not None else settings.chunk_overlap
        )
        self.chunk_size_chars = self.chunk_size_tokens * _CHARS_PER_TOKEN
        self.chunk_overlap_chars = self.chunk_overlap_tokens * _CHARS_PER_TOKEN

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using the first separator that yields small-enough pieces."""
        if len(text) <= self.chunk_size_chars:
            return [text] if text.strip() else []

        if not separators:
            # Base case: hard split by character length.
            return [
                text[i : i + self.chunk_size_chars]
                for i in range(0, len(text), self.chunk_size_chars)
            ]

        sep, remaining_separators = separators[0], separators[1:]
        if sep == "":
            parts = list(text)
        else:
            parts = text.split(sep)

        # Reassemble parts into chunks close to chunk_size_chars.
        chunks: list[str] = []
        current = ""
        for part in parts:
            candidate = current + (sep if current else "") + part if sep else current + part
            if len(candidate) <= self.chunk_size_chars:
                current = candidate
            else:
                if current.strip():
                    chunks.append(current)
                if len(part) > self.chunk_size_chars:
                    # Part itself too large; recurse with remaining separators.
                    chunks.extend(self._split_recursive(part, remaining_separators))
                    current = ""
                else:
                    current = part
        if current.strip():
            chunks.append(current)

        return chunks

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        """Prepend a trailing slice of the previous chunk to each chunk for context continuity."""
        if self.chunk_overlap_chars <= 0 or len(chunks) <= 1:
            return chunks

        overlapped: list[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-self.chunk_overlap_chars :]
            # Start the overlap at a word boundary so chunks don't begin with half a word.
            if len(chunks[i - 1]) > self.chunk_overlap_chars:
                boundary = re.search(r"\s", prev_tail)
                prev_tail = prev_tail[boundary.end():] if boundary else prev_tail
            overlapped.append(f"{prev_tail.strip()} {chunks[i]}".strip())
        return overlapped

    @staticmethod
    def _page_ranges(pieces: list[str]) -> list[tuple[int | None, int | None]]:
        """(first_page, last_page) for each chunk, from "[Page N]" markers in order.

        A chunk that contains no marker is on the page of the most recent marker
        seen before it. Text without any markers (TXT, DOCX, CSV) gets (None, None).
        """
        ranges: list[tuple[int | None, int | None]] = []
        current: int | None = None
        for piece in pieces:
            markers = [int(m) for m in _PAGE_MARKER_RE.findall(piece)]
            starts_with_marker = bool(_PAGE_MARKER_RE.match(piece))
            first = markers[0] if starts_with_marker or current is None and markers else current
            if markers:
                current = markers[-1]
            last = current
            ranges.append((first, last if last is not None else first))
        return ranges

    def split_text(self, text: str) -> list[str]:
        """Split raw text into overlapping chunks bounded by chunk_size_tokens."""
        if not text or not text.strip():
            return []

        return self._apply_overlap(self._split_without_overlap(text))

    def _split_without_overlap(self, text: str) -> list[str]:
        """The consecutive, non-overlapping pieces that split_text adds overlap to."""
        if not text or not text.strip():
            return []
        raw_chunks = self._split_recursive(text, _SEPARATORS)
        raw_chunks = [c.strip() for c in raw_chunks if c.strip()]
        return self._merge_lone_page_markers(raw_chunks)

    @staticmethod
    def _merge_lone_page_markers(chunks: list[str]) -> list[str]:
        """Attach a chunk that is only a "[Page N]" marker to the chunk after it."""
        merged: list[str] = []
        pending = ""
        for chunk in chunks:
            if _PAGE_MARKER_RE.fullmatch(chunk):
                pending = f"{pending} {chunk}".strip()
                continue
            merged.append(f"{pending}\n{chunk}" if pending else chunk)
            pending = ""
        if pending:
            merged.append(pending)
        return merged

    @staticmethod
    def _section_paths(pieces: list[str]) -> list[str | None]:
        """The section each chunk's content belongs to, e.g. "KEY PROJECTS > 2. nVMS - ...".

        `pieces` must be the non-overlapping pieces (an overlapped chunk starts with
        a copy of the previous chunk's tail, which would mislabel it). Headings are
        tracked in order across pieces, so a piece in the middle of a section still
        knows its section; the path is taken at the piece's first content line.
        Fenced code blocks are skipped (a "# comment" there isn't a heading).
        """
        paths: list[str | None] = []
        section: str | None = None
        subsection: str | None = None
        in_code_block = False
        previous_line = ""
        for piece in pieces:
            path_at_start: str | None = None
            found_content = False
            for line in piece.splitlines():
                if line.strip().startswith("```"):
                    in_code_block = not in_code_block
                    continue
                heading = None if in_code_block else _detect_heading(line, previous_line)
                if line.strip():
                    previous_line = line
                if heading:
                    level, title = heading
                    if level == 1:
                        section, subsection = title, None
                    else:
                        subsection = title
                    continue
                if line.strip() and not found_content and not _PAGE_MARKER_RE.fullmatch(line.strip()):
                    found_content = True
                    path_at_start = " > ".join(p for p in (section, subsection) if p) or None
            if not found_content:
                path_at_start = " > ".join(p for p in (section, subsection) if p) or None
            paths.append(path_at_start)
        return paths

    def chunk_document(
        self,
        text: str,
        document_id: str,
        filename: str,
        extra_metadata: dict | None = None,
    ) -> list[DocumentChunk]:
        """Split document text and wrap each piece as a DocumentChunk with metadata."""
        raw_pieces = self._split_without_overlap(text)
        pieces = self._apply_overlap(raw_pieces)
        total = len(pieces)
        now = datetime.now(UTC)
        page_ranges = self._page_ranges(pieces)
        sections = self._section_paths(raw_pieces)

        chunks = [
            DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                document_id=document_id,
                filename=filename,
                chunk_text=piece,
                chunk_index=idx,
                total_chunks=total,
                metadata={
                    **(extra_metadata or {}),
                    "char_count": len(piece),
                    **(
                        {"page_start": page_ranges[idx][0], "page_end": page_ranges[idx][1]}
                        if page_ranges[idx][0] is not None
                        else {}
                    ),
                    **({"section": sections[idx]} if sections[idx] else {}),
                },
                created_date=now,
            )
            for idx, piece in enumerate(pieces)
        ]

        logger.info(f"Chunked '{filename}' into {total} chunks (document_id={document_id})")
        return chunks
