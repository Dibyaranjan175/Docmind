"""
DocMind Embedding Module
Converts text (chunks or queries) into dense vector embeddings using
SentenceTransformers — no API key required, runs fully locally.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.config import cfg
from src.logger import get_logger

if TYPE_CHECKING:
    from src.ingest import TextChunk

logger = get_logger(__name__)


# ── Model singleton ──────────────────────────────────────────────────────────

_model = None  # Lazy-loaded on first use


def _get_model():
    """Load the SentenceTransformer model once and cache it."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. "
                "Install it with:  pip install sentence-transformers"
            )

        logger.info(f"Loading embedding model: '{cfg.embedding_model}'")
        _model = SentenceTransformer(cfg.embedding_model)
        logger.info(
            f"Model loaded — output dimension: {cfg.embedding_dimension}"
        )
    return _model


# ── Public API ───────────────────────────────────────────────────────────────

def embed_texts(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """
    Embed a list of strings and return a list of float vectors.

    Parameters
    ----------
    texts       : List of strings to embed.
    batch_size  : Number of texts to encode per forward pass.

    Returns
    -------
    List of embedding vectors (one per input text).
    """
    if not texts:
        return []

    model = _get_model()
    logger.debug(f"Embedding {len(texts)} text(s) in batches of {batch_size}")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 50,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalise → cosine ≡ dot product
    )

    # Convert numpy array rows to plain Python lists for JSON serialisation
    vectors = [emb.tolist() for emb in embeddings]
    logger.debug(f"Embedding complete — dim={len(vectors[0]) if vectors else 0}")
    return vectors


def embed_chunks(chunks: "list[TextChunk]") -> list[tuple["TextChunk", list[float]]]:
    """
    Embed a list of TextChunk objects.

    Returns
    -------
    List of (chunk, vector) tuples, preserving order.
    """
    texts = [chunk.text for chunk in chunks]
    vectors = embed_texts(texts)
    return list(zip(chunks, vectors))


def embed_query(query: str) -> list[float]:
    """
    Embed a single query string.

    Uses the same model and normalisation as chunk embedding so that
    cosine similarity is meaningful between query and stored vectors.
    """
    vectors = embed_texts([query])
    return vectors[0]
