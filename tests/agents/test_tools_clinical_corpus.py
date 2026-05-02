"""Tests: retrieve_context tool dispatches by `corpus`."""
from __future__ import annotations

from pathlib import Path

from src.agents.tools import build_default_tools
from tests.fixtures.build_tiny_clinical_index import build as build_tiny


class TestClinicalCorpus:
    def test_default_corpus_is_reference(self, tmp_path: Path) -> None:
        clinical_idx = build_tiny(tmp_path / "tiny.pkl")
        tools = {t.name: t for t in build_default_tools(
            rag_index_dir=None,
            clinical_rag_index_path=clinical_idx,
        )}
        tool = tools["retrieve_context"]
        out = tool.execute(tool.input_model.model_validate({"query": "test query"}))
        assert hasattr(out, "chunks")
        # rag_index_dir=None means reference returns empty.
        assert out.chunks == []

    def test_clinical_corpus_returns_evidence(self, tmp_path: Path) -> None:
        clinical_idx = build_tiny(tmp_path / "tiny.pkl")
        tools = {t.name: t for t in build_default_tools(
            rag_index_dir=None,
            clinical_rag_index_path=clinical_idx,
        )}
        tool = tools["retrieve_context"]
        out = tool.execute(tool.input_model.model_validate({
            "query": "exercise and Alzheimer",
            "corpus": "clinical",
        }))
        assert len(out.chunks) > 0
        for c in out.chunks:
            assert "source" in c and "text" in c

    def test_clinical_corpus_without_index_returns_empty(self, tmp_path: Path) -> None:
        # No clinical index path configured.
        tools = {t.name: t for t in build_default_tools(
            rag_index_dir=None,
            clinical_rag_index_path=None,
        )}
        tool = tools["retrieve_context"]
        out = tool.execute(tool.input_model.model_validate({
            "query": "egzersiz Alzheimer",
            "corpus": "clinical",
        }))
        assert out.chunks == []
