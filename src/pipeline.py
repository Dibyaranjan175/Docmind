"""
DocMind Pipeline Orchestrator
Runs the full ingestion pipeline: load → chunk → embed → store in Endee.
Can be called programmatically or via  python -m src.pipeline
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from src.config import cfg
from src.embedding import embed_chunks
from src.ingest import TextChunk, ingest_directory, load_document, chunk_text
from src.logger import get_logger
from src.vectorstore import describe_index, upsert_chunks

logger = get_logger(__name__)


def run_ingestion(
    source: str | Path | None = None,
    chunk_size: int = cfg.chunk_size,
    chunk_overlap: int = cfg.chunk_overlap,
) -> dict:
    """
    Full ingestion pipeline.

    Parameters
    ----------
    source  : Path to a single file OR a directory of documents.
              Defaults to cfg.documents_dir when None.
    Returns
    -------
    Summary dict with document_count, chunk_count, vector_count.
    """
    source_path = Path(source) if source else Path(cfg.documents_dir)

    # ── 1. Load & chunk ───────────────────────────────────────────────────────
    if source_path.is_file():
        logger.info(f"Ingesting single file: '{source_path}'")
        raw_text = load_document(source_path)
        chunks = chunk_text(
            text=raw_text,
            source=source_path.name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        doc_count = 1
    elif source_path.is_dir():
        logger.info(f"Ingesting directory: '{source_path}'")
        chunks = ingest_directory(
            directory=source_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        doc_count = len({c.source for c in chunks})
    else:
        raise FileNotFoundError(f"Source not found: {source_path}")

    if not chunks:
        logger.warning("No chunks produced — nothing to store.")
        return {"document_count": 0, "chunk_count": 0, "vector_count": 0}

    # ── 2. Embed ──────────────────────────────────────────────────────────────
    logger.info(f"Embedding {len(chunks)} chunk(s)…")
    chunk_vector_pairs = embed_chunks(chunks)

    # ── 3. Store in Endee ─────────────────────────────────────────────────────
    logger.info("Storing vectors in Endee…")
    stored = upsert_chunks(chunk_vector_pairs)

    # ── 4. Summary ────────────────────────────────────────────────────────────
    summary = {
        "document_count": doc_count,
        "chunk_count": len(chunks),
        "vector_count": stored,
    }
    logger.info(
        f"Ingestion complete — "
        f"{doc_count} doc(s), {len(chunks)} chunks, {stored} vectors"
    )
    return summary


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="DocMind — Ingest documents into Endee vector database"
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to a file or directory (default: data/documents/)",
    )
    parser.add_argument("--chunk-size", type=int, default=cfg.chunk_size)
    parser.add_argument("--chunk-overlap", type=int, default=cfg.chunk_overlap)
    args = parser.parse_args()

    result = run_ingestion(
        source=args.source,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print("\n✅ Ingestion Summary")
    print(f"   Documents : {result['document_count']}")
    print(f"   Chunks    : {result['chunk_count']}")
    print(f"   Vectors   : {result['vector_count']}")

    try:
        info = describe_index()
        print(f"   Index info: {info}")
    except Exception:
        pass
