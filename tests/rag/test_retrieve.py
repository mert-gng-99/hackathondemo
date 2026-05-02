"""Tests for src.rag.retrieve — query → top-k chunks."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.rag.ingest import ingest_directory
from src.rag.retrieve import RAGRetriever


_FIXTURE_KB = Path(__file__).parent.parent / "fixtures" / "kb_sample"


class TestRAGRetriever:
    @pytest.fixture(scope="class")
    def retriever(self, tmp_path_factory: pytest.TempPathFactory) -> RAGRetriever:
        idx_dir = tmp_path_factory.mktemp("rag_idx")
        ingest_directory(_FIXTURE_KB, idx_dir)
        return RAGRetriever.load(idx_dir)

    def test_bbb_query_returns_lipinski_chunk(self, retriever: RAGRetriever) -> None:
        hits = retriever.search("Why does ethanol cross the blood-brain barrier?", k=3)
        assert len(hits) == 3
        sources = [h["source"] for h in hits]
        assert "lipinski_rule_of_five.md" in sources
        # top hit should be from lipinski
        assert hits[0]["source"] == "lipinski_rule_of_five.md"

    def test_combat_query_returns_combat_chunk(self, retriever: RAGRetriever) -> None:
        hits = retriever.search("How does ComBat remove scanner bias from MRI data?", k=2)
        assert hits[0]["source"] == "combat_harmonization_primer.md"

    def test_eeg_query_returns_ica_chunk(self, retriever: RAGRetriever) -> None:
        hits = retriever.search("How do you remove eye blink artifacts from EEG?", k=2)
        assert hits[0]["source"] == "mne_ica_basics.md"

    def test_search_includes_score_and_text(self, retriever: RAGRetriever) -> None:
        hits = retriever.search("BBB permeability", k=1)
        h = hits[0]
        assert "text" in h
        assert "source" in h
        assert "score" in h
        assert isinstance(h["score"], float)
        assert 0.0 <= h["score"] <= 1.0
