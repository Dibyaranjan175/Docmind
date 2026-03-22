"""
DocMind Ingestion Module
Handles loading PDF / TXT / Markdown files and splitting them into
overlapping text chunks ready for embedding.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from src.config import cfg
from src.logger import get_logger

logger = get_logger(__name__)


# ── Data models ──────────────────────────────────────────────────────────────

@dataclass
class TextChunk:
    """A single chunk of text extracted from a document."""
    chunk_id: str            # Stable hash-based ID
    text: str                # Chunk content
    source: str              # Original filename
    page: int                # Page number (0-indexed; 0 for plain text)
    chunk_index: int         # Position within the source document
    char_start: int          # Character offset in the full document text


# ── Loaders ──────────────────────────────────────────────────────────────────

def _load_pdf(path: Path) -> str:
    """Extract all text from a PDF using pypdf (pure-Python, no system deps)."""
    try:
        import pypdf  # lazy import — only needed for PDFs
    except ImportError:
        raise ImportError(
            "pypdf is required for PDF ingestion. "
            "Install it with:  pip install pypdf"
        )

    pages: list[str] = []
    with open(path, "rb") as fh:
        reader = pypdf.PdfReader(fh)
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)

    logger.info(f"Loaded PDF '{path.name}' — {len(pages)} page(s)")
    return "\n".join(pages)


def _load_text(path: Path) -> str:
    """Load a plain-text or Markdown file."""
    text = path.read_text(encoding="utf-8", errors="replace")
    logger.info(f"Loaded text file '{path.name}' — {len(text):,} characters")
    return text


def load_document(path: Path) -> str:
    """
    Dispatch to the correct loader based on file extension.

    Supported: .pdf  .txt  .md  .markdown
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    elif suffix in {".txt", ".md", ".markdown", ".rst"}:
        return _load_text(path)
    else:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            "Supported: .pdf, .txt, .md, .markdown, .rst"
        )


# ── Chunker ──────────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """Normalise whitespace without destroying paragraph breaks."""
    # Collapse runs of spaces/tabs to a single space
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse 3+ consecutive newlines to two (one blank line)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _make_chunk_id(source: str, chunk_index: int, text: str) -> str:
    """Create a stable, unique ID for a chunk."""
    raw = f"{source}::{chunk_index}::{text[:64]}"
    return hashlib.md5(raw.encode()).hexdigest()


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = cfg.chunk_size,
    chunk_overlap: int = cfg.chunk_overlap,
) -> list[TextChunk]:
    """
    Split *text* into overlapping chunks of approximately *chunk_size* characters.

    Strategy
    --------
    1. Prefer splitting on paragraph boundaries (\n\n).
    2. If a paragraph is larger than chunk_size, fall back to sentence boundaries.
    3. Accumulate sentences/paragraphs until the chunk would exceed chunk_size,
       then emit the chunk and slide the window back by *chunk_overlap* characters.

    This approach keeps coherent sentences together rather than splitting mid-word.
    """
    text = _clean_text(text)
    if not text:
        return []

    # Split into atomic units (paragraphs, then sentences inside large paragraphs)
    units: list[str] = []
    for para in re.split(r"\n\n+", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= chunk_size:
            units.append(para)
        else:
            # Break large paragraph into sentences
            sentences = re.split(r"(?<=[.!?])\s+", para)
            units.extend(s.strip() for s in sentences if s.strip())

    chunks: list[TextChunk] = []
    current_parts: list[str] = []
    current_len = 0
    char_cursor = 0

    def _flush(idx: int) -> int:
        """Emit the current buffer as a chunk; return the new char_cursor."""
        nonlocal char_cursor
        chunk_text_val = " ".join(current_parts)
        cid = _make_chunk_id(source, idx, chunk_text_val)
        chunks.append(
            TextChunk(
                chunk_id=cid,
                text=chunk_text_val,
                source=source,
                page=0,
                chunk_index=idx,
                char_start=char_cursor,
            )
        )
        return char_cursor + len(chunk_text_val)

    for unit in units:
        unit_len = len(unit)

        if current_len + unit_len + 1 > chunk_size and current_parts:
            char_cursor = _flush(len(chunks))

            # Slide window: keep the last *chunk_overlap* characters worth of units
            overlap_text = ""
            overlap_parts: list[str] = []
            for part in reversed(current_parts):
                if len(overlap_text) + len(part) <= chunk_overlap:
                    overlap_parts.insert(0, part)
                    overlap_text = part + " " + overlap_text
                else:
                    break

            current_parts = overlap_parts
            current_len = len(" ".join(current_parts))

        current_parts.append(unit)
        current_len += unit_len + 1  # +1 for the joining space

    # Flush remaining content
    if current_parts:
        _flush(len(chunks))

    logger.info(
        f"Chunked '{source}' → {len(chunks)} chunk(s) "
        f"(size={chunk_size}, overlap={chunk_overlap})"
    )
    return chunks


# ── Directory scanner ────────────────────────────────────────────────────────

def ingest_directory(
    directory: str | Path = cfg.documents_dir,
    chunk_size: int = cfg.chunk_size,
    chunk_overlap: int = cfg.chunk_overlap,
) -> list[TextChunk]:
    """
    Scan *directory* for supported documents and return all chunks.

    Walks non-recursively (top-level files only by default).
    """
    doc_dir = Path(directory)
    if not doc_dir.exists():
        raise FileNotFoundError(f"Documents directory not found: {doc_dir}")

    supported = {".pdf", ".txt", ".md", ".markdown", ".rst"}
    files = [f for f in sorted(doc_dir.iterdir()) if f.suffix.lower() in supported]

    if not files:
        logger.warning(f"No supported documents found in '{doc_dir}'")
        return []

    logger.info(f"Found {len(files)} document(s) in '{doc_dir}'")

    all_chunks: list[TextChunk] = []
    for filepath in files:
        try:
            raw_text = load_document(filepath)
            chunks = chunk_text(
                text=raw_text,
                source=filepath.name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            all_chunks.extend(chunks)
        except Exception as exc:
            logger.error(f"Failed to process '{filepath.name}': {exc}")

    logger.info(f"Total chunks ready for embedding: {len(all_chunks)}")
    return all_chunks
