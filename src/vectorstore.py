"""
DocMind Vector Store — Endee Integration
All interactions with the Endee vector database are centralised here.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any
from src.config import cfg
from src.logger import get_logger

if TYPE_CHECKING:
    from src.ingest import TextChunk

logger = get_logger(__name__)

_client = None
_index = None


def _get_client():
    global _client
    if _client is None:
        from endee import Endee
        logger.info(f"Connecting to Endee at {cfg.endee_base_url}")
        _client = Endee(cfg.endee_auth_token or "")
        _client.set_base_url(cfg.endee_base_url)
        logger.info("Endee client initialised")
    return _client


def _get_index():
    global _index
    if _index is None:
        _index = get_or_create_index()
    return _index


def get_or_create_index(name: str = cfg.endee_index_name, dimension: int = cfg.embedding_dimension):
    from endee import Precision
    client = _get_client()
    try:
        client.create_index(name=name, dimension=dimension, space_type="cosine", precision=Precision.INT8)
        logger.info(f"Index '{name}' created successfully")
    except Exception:
        logger.info(f"Index '{name}' already exists, reusing it")
    return client.get_index(name=name)


def delete_index(name: str = cfg.endee_index_name) -> None:
    global _index
    client = _get_client()
    client.delete_index(name)
    _index = None
    logger.info(f"Index '{name}' deleted")


def describe_index(name: str = cfg.endee_index_name) -> dict[str, Any]:
    index = _get_index()
    info = index.describe()
    logger.info(f"Index info: {info}")
    return info


def upsert_chunks(chunk_vector_pairs: list[tuple["TextChunk", list[float]]], batch_size: int = cfg.upsert_batch_size) -> int:
    index = _get_index()
    total = 0
    for start in range(0, len(chunk_vector_pairs), batch_size):
        batch = chunk_vector_pairs[start: start + batch_size]
        records = [
            {
                "id": chunk.chunk_id,
                "vector": vector,
                "meta": {
                    "text": chunk.text,
                    "source": chunk.source,
                    "page": chunk.page,
                    "chunk_index": chunk.chunk_index
                },
                "filter": {"source": chunk.source},
            }
            for chunk, vector in batch
        ]
        index.upsert(records)
        total += len(records)
    logger.info(f"Upsert complete — {total} vectors stored in Endee")
    return total


def query_index(query_vector: list[float], top_k: int = cfg.top_k, ef: int = cfg.ef_search, source_filter: str | None = None) -> list[dict[str, Any]]:
    index = _get_index()
    query_kwargs: dict[str, Any] = {
        "vector": query_vector,
        "top_k": top_k,
        "ef": ef,
        "include_vectors": False
    }
    if source_filter:
        query_kwargs["filter"] = [{"source": {"$eq": source_filter}}]
    raw_results = index.query(**query_kwargs)
    results = []
    for item in raw_results:
        meta = item.get("meta", {})
        results.append({
            "id": item["id"],
            "similarity": round(float(item["similarity"]), 4),
            "text": meta.get("text", ""),
            "source": meta.get("source", "unknown"),
            "chunk_index": meta.get("chunk_index", 0),
            "page": meta.get("page", 0),
        })
    return results