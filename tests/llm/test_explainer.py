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


class TestAuthFailureShortCircuits:
    """A 401 from OpenRouter means the key is unauthorized — every model
    in the chain will fail the same way, so we must short-circuit instead
    of burning the full chain on every request."""

    def test_401_short_circuits_to_template_after_one_attempt(self, monkeypatch):
        from src.llm import explainer as ex
        from openai import APIStatusError
        import httpx

        monkeypatch.delenv("NEUROBRIDGE_DISABLE_LLM", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-deliberately-bad")

        attempts: list[str] = []

        def _raise_401(**kwargs):
            attempts.append(kwargs["model"])
            req = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
            resp = httpx.Response(status_code=401, request=req)
            raise APIStatusError(message="No auth credentials found", response=resp, body={})

        class _StubCompletions:
            create = staticmethod(_raise_401)

        class _StubChat:
            completions = _StubCompletions()

        class _StubClient:
            chat = _StubChat()
            def __init__(self, **kwargs):
                pass

        # Must patch on the `openai` module — the explainer does
        # `from openai import OpenAI` *inside* the function (see
        # src/llm/explainer.py:269-275), so any module-level attribute
        # on `src.llm.explainer` would be a no-op.
        monkeypatch.setattr("openai.OpenAI", _StubClient)

        out = ex._llm_explain(_payload(), modality="bbb")

        assert out is None, "401 must surface as a None return (caller falls back to template)"
        assert len(attempts) == 1, f"401 must short-circuit; tried {len(attempts)} models: {attempts}"

    def test_explain_returns_template_source_on_401(self, monkeypatch):
        from src.llm import explainer as ex
        from openai import APIStatusError
        import httpx

        monkeypatch.delenv("NEUROBRIDGE_DISABLE_LLM", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-deliberately-bad")

        def _raise_401(**kwargs):
            req = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
            raise APIStatusError(
                message="auth",
                response=httpx.Response(401, request=req),
                body={},
            )

        class _Comp:
            create = staticmethod(_raise_401)

        class _Chat:
            completions = _Comp()

        class _Client:
            chat = _Chat()
            def __init__(self, **kwargs):
                pass

        monkeypatch.setattr("openai.OpenAI", _Client)

        result = ex.explain(_payload(), modality="bbb")

        assert result["source"] == "template"
        assert result["model"] is None
        assert result["rationale"], "rationale must never be empty"

    def test_400_advances_to_next_model_instead_of_short_circuiting(self, monkeypatch):
        """A 400 from one model is a prompt-shape mismatch with THAT model
        (some models reject system roles, etc.) — try the next, don't give up."""
        from src.llm import explainer as ex
        from openai import APIStatusError
        import httpx

        monkeypatch.delenv("NEUROBRIDGE_DISABLE_LLM", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-anything")

        attempts: list[str] = []
        # Force a known multi-model chain so we can count attempts deterministically
        monkeypatch.setenv("OPENROUTER_FREE_MODELS", "model-a:free,model-b:free,model-c:free")

        def _raise_400(**kwargs):
            attempts.append(kwargs["model"])
            req = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
            raise APIStatusError(
                message="bad request",
                response=httpx.Response(400, request=req),
                body={},
            )

        class _Comp:
            create = staticmethod(_raise_400)

        class _Chat:
            completions = _Comp()

        class _Client:
            chat = _Chat()
            def __init__(self, **kwargs):
                pass

        monkeypatch.setattr("openai.OpenAI", _Client)

        out = ex._llm_explain(_payload(), modality="bbb")

        assert out is None, "all models 400'd → must return None for template fallback"
        assert attempts == ["model-a:free", "model-b:free", "model-c:free"], (
            f"400 must advance to next model; got attempts={attempts}"
        )


@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set — skipping live LLM integration test",
)
@pytest.mark.skipif(
    os.environ.get("NEUROBRIDGE_DISABLE_LLM") == "1",
    reason="NEUROBRIDGE_DISABLE_LLM=1 — skipping live LLM integration test",
)
class TestLiveOpenRouterLLM:
    """End-to-end: hit a real OpenRouter free-tier model and assert
    `explain()` returns source='llm' with non-empty content. Skipped
    when no key is set or the kill-switch is on.

    Marked `slow` because it makes a real network round-trip
    (worst case ~80s if the entire chain is unreachable). Run with
    `pytest -m slow` or include it in the default suite by not passing
    `-m "not slow"`."""

    def test_bbb_explain_returns_llm_source_with_real_key(self, caplog):
        import logging
        from src.llm import explainer as ex

        # The explainer's logger has propagate=False (see src/core/logger.py),
        # so caplog's root-level handler never sees its records. Attach the
        # caplog handler directly to bypass propagation.
        ex.logger.addHandler(caplog.handler)
        try:
            with caplog.at_level(logging.INFO, logger="src.llm.explainer"):
                result = ex.explain(_payload(), modality="bbb")
        finally:
            ex.logger.removeHandler(caplog.handler)

        # Flaky-network safety net: only skip when we have evidence the
        # template fallback fired due to transient infra (rate-limit,
        # 5xx, network). If the fallback fired silently — no infra-error
        # log line — that's a real regression we want to fail loud.
        if result["source"] == "template":
            log_text = " ".join(r.getMessage() for r in caplog.records)
            transient_signals = (
                "429", "OpenRouter 5", "OpenRouter 4",  # status-code log lines
                "connection error", "timeout",          # transport-error log lines
                "All free models exhausted",            # chain-end log line
            )
            had_infra_evidence = any(s.lower() in log_text.lower() for s in transient_signals)
            if not had_infra_evidence:
                pytest.fail(
                    "explain() fell back to template with NO infra-error log "
                    "line — this is a real regression, not a network blip. "
                    f"Captured logs: {log_text!r}"
                )
            pytest.skip(
                "All free models in the chain were rate-limited or unreachable "
                "at test time. Re-run later or run scripts/diagnose_openrouter.py."
            )

        assert result["source"] == "llm"
        assert result["model"] is not None and result["model"].endswith(":free"), (
            f"unexpected model id (must end with ':free' to ensure no paid model "
            f"snuck into the chain): {result['model']!r}"
        )
        assert result["rationale"].strip(), "LLM returned empty rationale"

        # Refusal/safety-filter sanity: catch the common patterns instead
        # of just one prefix.
        lowered = result["rationale"].lower()
        refusal_signals = (
            "i cannot",
            "i can't",
            "i'm sorry, but i",
            "i'm sorry, i can't",
            "as an ai",
            "as a language model",
            "i'm unable to",
            "i do not have the ability",
        )
        assert not any(lowered.startswith(s) for s in refusal_signals), (
            f"LLM refused (matched refusal pattern): {result['rationale']!r}"
        )

        # Positive on-topic assertion: the rationale must reference at least
        # one of the SHAP feature names from the payload, OR the verdict
        # word ("permeable" / "non-permeable"). A model that produced
        # off-topic small-talk would fail here.
        payload = _payload()
        feature_names = [f["feature"] for f in payload["top_features"]]
        verdict = payload["label_text"].lower()
        on_topic_anchors = [f.lower() for f in feature_names] + [verdict]
        assert any(anchor in lowered for anchor in on_topic_anchors), (
            f"rationale appears off-topic (no mention of SHAP features "
            f"{feature_names!r} or verdict {verdict!r}): {result['rationale']!r}"
        )
