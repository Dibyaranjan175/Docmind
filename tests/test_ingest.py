"""
Unit tests for DocMind ingestion and chunking.
Run with:  pytest tests/
"""

import pytest
from src.ingest import chunk_text, _clean_text, _make_chunk_id


# ── _clean_text ──────────────────────────────────────────────────────────────

def test_clean_text_collapses_spaces():
    result = _clean_text("hello   world")
    assert result == "hello world"


def test_clean_text_collapses_blank_lines():
    result = _clean_text("para1\n\n\n\npara2")
    assert result == "para1\n\npara2"


def test_clean_text_strips():
    result = _clean_text("  hello  ")
    assert result == "hello"


# ── chunk_text ───────────────────────────────────────────────────────────────

SAMPLE = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump! "
    "The five boxing wizards jump quickly. "
    "Sphinx of black quartz, judge my vow."
)


def test_chunk_text_returns_list():
    chunks = chunk_text(SAMPLE, source="test.txt", chunk_size=100, chunk_overlap=20)
    assert isinstance(chunks, list)
    assert len(chunks) >= 1


def test_chunk_text_ids_are_unique():
    chunks = chunk_text(SAMPLE, source="test.txt", chunk_size=50, chunk_overlap=10)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "Chunk IDs must be unique"


def test_chunk_text_source_preserved():
    chunks = chunk_text(SAMPLE, source="myfile.pdf", chunk_size=100, chunk_overlap=20)
    for chunk in chunks:
        assert chunk.source == "myfile.pdf"


def test_chunk_text_empty_input():
    chunks = chunk_text("", source="empty.txt")
    assert chunks == []


def test_chunk_text_no_loss_of_content():
    """All characters in the original text should appear in at least one chunk."""
    chunks = chunk_text(SAMPLE, source="test.txt", chunk_size=80, chunk_overlap=20)
    combined = " ".join(c.text for c in chunks)
    # Every word from SAMPLE should appear somewhere
    for word in SAMPLE.split():
        assert word in combined, f"Word '{word}' missing from chunks"


def test_chunk_text_respects_overlap():
    """With overlap, adjacent chunks should share some content."""
    long_text = " ".join([f"Sentence number {i} is here." for i in range(50)])
    chunks = chunk_text(long_text, source="test.txt", chunk_size=100, chunk_overlap=40)
    if len(chunks) >= 2:
        # Some words from chunk[0] should appear in chunk[1]
        words_0 = set(chunks[0].text.split())
        words_1 = set(chunks[1].text.split())
        assert len(words_0 & words_1) > 0, "Overlap chunks share no content"


# ── _make_chunk_id ────────────────────────────────────────────────────────────

def test_chunk_id_deterministic():
    id1 = _make_chunk_id("doc.pdf", 0, "hello world")
    id2 = _make_chunk_id("doc.pdf", 0, "hello world")
    assert id1 == id2


def test_chunk_id_differs_by_source():
    id1 = _make_chunk_id("doc1.pdf", 0, "hello world")
    id2 = _make_chunk_id("doc2.pdf", 0, "hello world")
    assert id1 != id2


def test_chunk_id_differs_by_index():
    id1 = _make_chunk_id("doc.pdf", 0, "hello world")
    id2 = _make_chunk_id("doc.pdf", 1, "hello world")
    assert id1 != id2
