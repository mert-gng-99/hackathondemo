"""Tests for src.rag.store — FAISS vector store with metadata."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.rag.store import FAISSStore


def _rand_vecs(n: int, d: int = 4, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d), dtype=np.float32)


class TestFAISSStore:
    def test_add_then_search(self) -> None:
        store = FAISSStore(dim=4)
        vecs = _rand_vecs(3)
        chunks = [{"text": f"chunk-{i}", "source": "test.md"} for i in range(3)]
        store.add(vecs, chunks)
        results = store.search(vecs[0], k=2)
        assert len(results) == 2
        # the closest hit is the chunk we used as the query (cosine ~1.0)
        top_chunk, top_score = results[0]
        assert top_chunk["text"] == "chunk-0"
        assert top_score > 0.99

    def test_add_size_mismatch_raises(self) -> None:
        store = FAISSStore(dim=4)
        with pytest.raises(ValueError, match="size mismatch"):
            store.add(_rand_vecs(3), [{"text": "only-one"}])

    def test_search_k_larger_than_corpus(self) -> None:
        store = FAISSStore(dim=4)
        store.add(_rand_vecs(2), [{"text": f"c{i}"} for i in range(2)])
        results = store.search(_rand_vecs(1)[0], k=10)
        assert len(results) == 2

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        store = FAISSStore(dim=4)
        vecs = _rand_vecs(3)
        chunks = [{"text": f"chunk-{i}", "source": "test.md"} for i in range(3)]
        store.add(vecs, chunks)
        store.save(tmp_path / "idx")

        restored = FAISSStore.load(tmp_path / "idx", dim=4)
        results = restored.search(vecs[0], k=1)
        assert results[0][0]["text"] == "chunk-0"

    def test_search_on_empty_store_returns_empty(self) -> None:
        store = FAISSStore(dim=4)
        assert store.search(_rand_vecs(1)[0], k=5) == []

    def test_add_does_not_mutate_caller_vectors(self) -> None:
        store = FAISSStore(dim=4)
        vecs = _rand_vecs(3)
        original = vecs.copy()
        store.add(vecs, [{"text": f"c{i}"} for i in range(3)])
        # Caller's array must be unchanged after add() (faiss.normalize_L2 is in-place)
        assert np.allclose(vecs, original), "store.add() mutated caller's vectors"

    def test_search_does_not_mutate_caller_query(self) -> None:
        store = FAISSStore(dim=4)
        store.add(_rand_vecs(3), [{"text": f"c{i}"} for i in range(3)])
        query = _rand_vecs(1)[0]
        original_query = query.copy()
        store.search(query, k=2)
        assert np.allclose(query, original_query), "store.search() mutated caller's query"
