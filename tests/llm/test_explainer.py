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


class TestEEGTemplate:
    """Day-8 T1A: deterministic EEG template path."""

    def test_eeg_template_uses_pipeline_metrics(self, monkeypatch):
        monkeypatch.setenv("NEUROBRIDGE_DISABLE_LLM", "1")
        payload = {
            "rows": 30,
            "columns": 95,
            "duration_sec": 4.32,
            "mlflow_run_id": "abc12345",
            "user_question": "Why were epochs dropped?",
        }
        result = explain(payload, modality="eeg")
        assert result["source"] == "template"
        assert result["model"] is None
        rationale = result["rationale"]
        assert "30" in rationale, "epoch count must appear"
        assert "95" in rationale, "feature count must appear"
        assert "4.3" in rationale, "duration must appear (1-decimal)"


class TestMRITemplate:
    """Day-8 T1A: deterministic MRI template path."""

    def test_mri_template_uses_combat_metrics(self, monkeypatch):
        monkeypatch.setenv("NEUROBRIDGE_DISABLE_LLM", "1")
        payload = {
            "site_gap_pre": 5.0004,
            "site_gap_post": 0.0015,
            "reduction_factor": 3290.0,
            "n_subjects": 6,
            "user_question": "Why does ComBat matter?",
        }
        result = explain(payload, modality="mri")
        assert result["source"] == "template"
        rationale = result["rationale"]
        assert "5.00" in rationale or "5.0" in rationale, "pre-gap must appear"
        assert "3290" in rationale or "3290×" in rationale, "reduction factor must appear"
        assert "6" in rationale, "n_subjects must appear"


class TestModalityDispatch:
    """Day-8 T1A: explain(modality=…) routes to the right template."""

    def test_unknown_modality_falls_back_to_bbb_template(self, monkeypatch):
        """Defensive: an unknown modality string degrades gracefully (warn + bbb-style template)."""
        monkeypatch.setenv("NEUROBRIDGE_DISABLE_LLM", "1")
        payload = {
            "smiles": "CCO",
            "label": 1,
            "label_text": "permeable",
            "confidence": 0.82,
            "top_features": [{"feature": "fp_1", "shap_value": 0.05}],
        }
        result = explain(payload, modality="unknown_xyz")
        # Should not raise; should produce a non-empty rationale
        assert result["source"] == "template"
        assert result["rationale"], "rationale must be non-empty"
