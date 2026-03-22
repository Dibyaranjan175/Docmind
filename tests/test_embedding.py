"""
Unit tests for the DocMind embedding module.
These tests mock the SentenceTransformer model so no GPU or download is needed.
Run with:  pytest tests/
"""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np


@pytest.fixture(autouse=True)
def mock_sentence_transformer():
    """Patch SentenceTransformer so tests run without downloading the model."""
    with patch("src.embedding._model", None):
        with patch("src.embedding._get_model") as mock_get:
            fake_model = MagicMock()
            fake_model.encode.side_effect = lambda texts, **kw: np.random.rand(
                len(texts), 384
            ).astype(np.float32)
            mock_get.return_value = fake_model
            yield mock_get


def test_embed_texts_returns_correct_length():
    from src.embedding import embed_texts

    texts = ["hello", "world", "docmind"]
    vectors = embed_texts(texts)
    assert len(vectors) == 3


def test_embed_texts_correct_dimension():
    from src.embedding import embed_texts

    vectors = embed_texts(["test sentence"])
    assert len(vectors[0]) == 384


def test_embed_texts_returns_list_of_floats():
    from src.embedding import embed_texts

    vectors = embed_texts(["sample text"])
    assert isinstance(vectors, list)
    assert isinstance(vectors[0], list)
    assert all(isinstance(v, float) for v in vectors[0])


def test_embed_texts_empty():
    from src.embedding import embed_texts

    result = embed_texts([])
    assert result == []


def test_embed_query_single_vector():
    from src.embedding import embed_query

    vector = embed_query("what is AI?")
    assert isinstance(vector, list)
    assert len(vector) == 384


def test_embed_chunks_pairs():
    from src.embedding import embed_chunks
    from src.ingest import TextChunk

    chunks = [
        TextChunk(
            chunk_id="abc",
            text="Some text",
            source="test.txt",
            page=0,
            chunk_index=0,
            char_start=0,
        )
    ]
    pairs = embed_chunks(chunks)
    assert len(pairs) == 1
    chunk, vector = pairs[0]
    assert chunk.text == "Some text"
    assert len(vector) == 384
