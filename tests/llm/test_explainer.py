"""Tests for src.llm.explainer.

The deterministic template path is exhaustively tested here. The LLM
path is exercised only by env-gated integration tests in
test_explainer_integration.py (NOT run in CI by default).
"""
from __future__ import annotations

import os

import pytest

from src.llm.explainer import ExplainPayload, explain


def _payload(**overrides) -> ExplainPayload:
    """Build a representative ExplainPayload; overrides win."""
    base: ExplainPayload = {
        "smiles": "CCO",
        "label": 1,
        "label_text": "permeable",
        "confidence": 0.82,
        "top_features": [
            {"feature": "fp_341", "shap_value": 0.045},
            {"feature": "fp_902", "shap_value": -0.031},
            {"feature": "fp_77", "shap_value": 0.022},
        ],
        "calibration": {"threshold": 0.80, "precision": 0.92, "support": 18},
        "drift_z": 0.42,
        "user_question": "Why was this molecule predicted as permeable?",
    }
    base.update(overrides)
    return base


class TestTemplateExplain:
    """Day-7 T3A: deterministic-template path of the explainer."""

    def test_template_path_is_deterministic(self, monkeypatch):
        """Same input → byte-identical rationale string. No randomness."""
        monkeypatch.setenv("NEUROBRIDGE_DISABLE_LLM", "1")
        out_a = explain(_payload())
        out_b = explain(_payload())
        assert out_a["rationale"] == out_b["rationale"]
        assert out_a["source"] == "template"
        assert out_b["source"] == "template"
        assert out_a["model"] is None

    def test_template_includes_top_feature_names(self, monkeypatch):
        """Rationale must mention the SHAP features so jurors see attribution."""
        monkeypatch.setenv("NEUROBRIDGE_DISABLE_LLM", "1")
        result = explain(_payload())
        for feat in ("fp_341", "fp_902", "fp_77"):
            assert feat in result["rationale"], (
                f"expected feature {feat!r} in rationale, got {result['rationale']!r}"
            )

    def test_template_includes_label_text(self, monkeypatch):
        """The verdict word ('permeable' / 'non-permeable') must appear."""
        monkeypatch.setenv("NEUROBRIDGE_DISABLE_LLM", "1")
        result = explain(_payload(label=0, label_text="non-permeable"))
        assert "non-permeable" in result["rationale"]

    def test_disable_flag_forces_template_even_with_key_set(self, monkeypatch):
        """NEUROBRIDGE_DISABLE_LLM=1 wins over OPENROUTER_API_KEY presence."""
        monkeypatch.setenv("NEUROBRIDGE_DISABLE_LLM", "1")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-fake-not-used")
        result = explain(_payload())
        assert result["source"] == "template"
        assert result["model"] is None
