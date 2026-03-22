"""
DocMind Configuration
All settings loaded from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass


@dataclass
class Config:
    # ── Endee Vector Database ────────────────────────────────────────────────
    endee_base_url: str = os.getenv("ENDEE_BASE_URL", "http://localhost:8080/api/v1")
    endee_auth_token: str = os.getenv("ENDEE_AUTH_TOKEN", "")
    endee_index_name: str = os.getenv("ENDEE_INDEX_NAME", "docmind_index")

    # ── Embedding Model ──────────────────────────────────────────────────────
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "384"))

    # ── Text Chunking ────────────────────────────────────────────────────────
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "512"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "64"))

    # ── Retrieval ────────────────────────────────────────────────────────────
    top_k: int = int(os.getenv("TOP_K", "5"))
    ef_search: int = int(os.getenv("EF_SEARCH", "128"))

    # ── LLM ─────────────────────────────────────────────────────────────────
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai")   # "openai" | "anthropic"
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))

    # ── Data ─────────────────────────────────────────────────────────────────
    documents_dir: str = os.getenv("DOCUMENTS_DIR", "data/documents")
    logs_dir: str = os.getenv("LOGS_DIR", "logs")

    # ── Batch processing ─────────────────────────────────────────────────────
    upsert_batch_size: int = int(os.getenv("UPSERT_BATCH_SIZE", "100"))


# Singleton instance used across all modules
cfg = Config()
