"""Tests for src.rag.embed — fastembed wrapper."""
from __future__ import annotations

import numpy as np
import pytest

from src.rag.embed import Embedder, EMBEDDING_DIM


class TestEmbedder:
    @pytest.fixture(scope="class")
    def embedder(self) -> Embedder:
        return Embedder()

    def test_dim_constant_matches_model(self, embedder: Embedder) -> None:
        out = embedder.encode(["hello"])
        assert out.shape == (1, EMBEDDING_DIM)

    def test_batch_encoding(self, embedder: Embedder) -> None:
        out = embedder.encode(["hello", "world", "blood-brain barrier"])
        assert out.shape == (3, EMBEDDING_DIM)
        assert out.dtype == np.float32

    def test_empty_list_returns_empty_array(self, embedder: Embedder) -> None:
        out = embedder.encode([])
        assert out.shape == (0, EMBEDDING_DIM)

    def test_similar_strings_have_higher_similarity_than_dissimilar(
        self, embedder: Embedder
    ) -> None:
        vecs = embedder.encode([
            "blood-brain barrier permeability",
            "BBB drug penetration",
            "MRI multi-site harmonization",
        ])
        # cosine similarity (vectors should be normalized for stable comparison)
        from numpy.linalg import norm
        def cos(a, b):
            return float(np.dot(a, b) / (norm(a) * norm(b)))
        sim_ab = cos(vecs[0], vecs[1])
        sim_ac = cos(vecs[0], vecs[2])
        assert sim_ab > sim_ac, f"Expected BBB-related strings closer; got {sim_ab=} vs {sim_ac=}"
