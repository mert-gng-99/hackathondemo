"""Tests for src.rag.clinical.retrieve."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.rag.clinical.retrieve import retrieve_clinical
from src.rag.clinical.loader import load_index
from tests.fixtures.build_tiny_clinical_index import build as build_tiny


class TestRetrieve:
    def test_alzheimer_query_picks_alzheimer_chunks(self, tmp_path: Path) -> None:
        payload = load_index(build_tiny(tmp_path / "tiny.pkl"))
        result = retrieve_clinical(payload, query="exercise and Alzheimer's", top_k=2)
        sources = {ev.source for ev in result.evidence}
        assert any("alzheimers" in s for s in sources)

    def test_parkinson_query_picks_parkinson_chunks(self, tmp_path: Path) -> None:
        payload = load_index(build_tiny(tmp_path / "tiny.pkl"))
        result = retrieve_clinical(payload, query="Parkinson levodopa", top_k=2)
        sources = {ev.source for ev in result.evidence}
        assert any("parkinsons" in s for s in sources)

    def test_turkish_keyword_routes_via_expansion(self, tmp_path: Path) -> None:
        payload = load_index(build_tiny(tmp_path / "tiny.pkl"))
        result = retrieve_clinical(payload, query="egzersiz Alzheimer", top_k=2)
        assert any("alzheimers_lifestyle" in ev.source for ev in result.evidence)

    def test_summary_text_contains_citations(self, tmp_path: Path) -> None:
        payload = load_index(build_tiny(tmp_path / "tiny.pkl"))
        result = retrieve_clinical(payload, query="diet and Parkinson", top_k=2)
        assert any(ev.source in result.summary_text for ev in result.evidence)

    def test_empty_query_returns_empty_evidence(self, tmp_path: Path) -> None:
        payload = load_index(build_tiny(tmp_path / "tiny.pkl"))
        result = retrieve_clinical(payload, query="", top_k=2)
        assert result.evidence == []
