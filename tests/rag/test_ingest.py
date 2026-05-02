"""Tests for src.rag.ingest — walk a directory, chunk, embed, persist."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src.rag.ingest import ingest_directory
from src.rag.store import FAISSStore


_FIXTURE_KB = Path(__file__).parent.parent / "fixtures" / "kb_sample"


class TestIngestDirectory:
    def test_ingests_markdown_files(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "idx"
        n = ingest_directory(_FIXTURE_KB, out_dir)
        assert n > 0  # at least one chunk per fixture file
        assert (out_dir / "index.bin").exists()
        assert (out_dir / "chunks.json").exists()

    def test_loaded_store_is_searchable(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "idx"
        ingest_directory(_FIXTURE_KB, out_dir)
        from src.rag.embed import EMBEDDING_DIM
        store = FAISSStore.load(out_dir, dim=EMBEDDING_DIM)
        assert len(store) > 0
        # chunks have source metadata
        assert all("source" in c for c in store._chunks)
        assert all("text" in c for c in store._chunks)

    def test_empty_directory_creates_empty_index(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty_kb"
        empty.mkdir()
        out_dir = tmp_path / "idx"
        n = ingest_directory(empty, out_dir)
        assert n == 0
        assert (out_dir / "index.bin").exists()
