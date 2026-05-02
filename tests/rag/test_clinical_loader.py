"""Tests for src.rag.clinical.loader."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.rag.clinical import loader
from tests.fixtures.build_tiny_clinical_index import build as build_tiny


class TestLoadIndex:
    def test_load_returns_payload_with_expected_keys(self, tmp_path: Path) -> None:
        idx_path = build_tiny(tmp_path / "tiny.pkl")
        payload = loader.load_index(idx_path)
        assert {"chunks", "vectorizer", "matrix"} <= set(payload)
        assert len(payload["chunks"]) == 4

    def test_missing_index_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="clinical RAG index not found"):
            loader.load_index(tmp_path / "nope.pkl")

    def test_unique_sources(self, tmp_path: Path) -> None:
        idx_path = build_tiny(tmp_path / "tiny.pkl")
        payload = loader.load_index(idx_path)
        sources = {c.source for c in payload["chunks"]}
        assert sources == {
            "alzheimers_lifestyle.pdf", "parkinsons_motor.pdf",
            "alzheimers_mci.pdf", "parkinsons_nutrition.pdf",
        }
