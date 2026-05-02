"""Tests for src.rag.chunker — paragraph-aware character splitter."""
from __future__ import annotations

import pytest

from src.rag.chunker import chunk_text


class TestChunkText:
    def test_short_text_returns_single_chunk(self) -> None:
        out = chunk_text("hello world", max_chars=100, overlap=10)
        assert out == ["hello world"]

    def test_empty_text_returns_empty_list(self) -> None:
        assert chunk_text("", max_chars=100, overlap=10) == []
        assert chunk_text("   \n\n  ", max_chars=100, overlap=10) == []

    def test_long_text_splits_into_multiple_chunks(self) -> None:
        text = "a" * 250
        out = chunk_text(text, max_chars=100, overlap=10)
        assert len(out) >= 3
        # every chunk respects max_chars
        for c in out:
            assert len(c) <= 100

    def test_overlap_between_chunks(self) -> None:
        text = "abcdefghij" * 30  # 300 chars, no natural break
        out = chunk_text(text, max_chars=100, overlap=20)
        # consecutive chunks share at least some characters
        for i in range(len(out) - 1):
            assert out[i][-10:] in out[i + 1] or out[i + 1][:10] in out[i]

    def test_paragraph_boundary_preferred(self) -> None:
        # First paragraph fits, second doesn't — split at \n\n
        para_a = "First paragraph content."
        para_b = "Second paragraph content " * 10
        text = f"{para_a}\n\n{para_b}"
        out = chunk_text(text, max_chars=100, overlap=10)
        # first chunk should end at the paragraph boundary, not mid-word
        assert para_a in out[0]
