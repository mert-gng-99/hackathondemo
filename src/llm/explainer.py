"""Natural-language rationale for a single BBB prediction.

Public entry point: `explain(payload)`. Always returns a usable
ExplainResult — never raises. Tries OpenRouter first when a key is set
and the kill-switch is off; falls back to a deterministic template on
any failure (network, auth, rate limit, malformed response).

Test discipline: deterministic template path is the source of truth.
LLM path is env-gated and exercised by integration tests only.
"""
from __future__ import annotations

import os
from typing import Any, TypedDict

from src.core.logger import get_logger

logger = get_logger(__name__)


class FeatureRow(TypedDict):
    feature: str
    shap_value: float


class CalibrationDict(TypedDict):
    threshold: float
    precision: float
    support: int


class ExplainPayload(TypedDict, total=False):
    smiles: str
    label: int
    label_text: str
    confidence: float
    top_features: list[FeatureRow]
    calibration: CalibrationDict | None
    drift_z: float | None
    user_question: str


class ExplainResult(TypedDict):
    rationale: str
    source: str          # "llm" | "template"
    model: str | None    # llm model name when source="llm", else None


_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_DEFAULT_MODEL = "meta-llama/llama-3.2-3b-instruct:free"
_LLM_TIMEOUT_SECONDS = 8.0
_LLM_MAX_TOKENS = 256
_LLM_TEMPERATURE = 0.3


def _should_use_llm() -> bool:
    """Gate: env kill-switch off AND key present."""
    if os.environ.get("NEUROBRIDGE_DISABLE_LLM") == "1":
        return False
    if not os.environ.get("OPENROUTER_API_KEY"):
        return False
    return True


def _drift_interpretation(drift_z: float | None) -> str:
    if drift_z is None:
        return "drift unavailable"
    mag = abs(drift_z)
    if mag < 1.0:
        return "within expected range"
    if mag < 2.0:
        return "mild distribution shift"
    return "significant shift, retrain recommended"


def _template_explain(payload: ExplainPayload) -> str:
    """Deterministic, jury-friendly rationale. Never raises."""
    label_text = payload.get("label_text", "unknown")
    confidence = float(payload.get("confidence", 0.0))
    top_features = payload.get("top_features") or []

    # Sentence 1
    sentences = [
        f"Predicted **{label_text}** with {confidence * 100:.0f}% confidence."
    ]

    # Sentence 2 (calibration, optional)
    cal = payload.get("calibration")
    if cal is not None:
        thr_pct = float(cal["threshold"]) * 100
        prec_pct = float(cal["precision"]) * 100
        support = int(cal["support"])
        if support > 0:
            sentences.append(
                f"Calibration: predictions in the ≥{thr_pct:.0f}% bin are "
                f"correct {prec_pct:.0f}% of the time on held-out data "
                f"(n={support})."
            )

    # Sentence 3 (top-3 SHAP features)
    if top_features:
        feat_strs = [
            f"{row['feature']} (Δ{float(row['shap_value']):+.3f})"
            for row in top_features[:3]
        ]
        sentences.append(
            f"Top SHAP attributions toward this label: {', '.join(feat_strs)}."
        )

    # Sentence 4 (drift, optional)
    drift_z = payload.get("drift_z")
    if drift_z is not None:
        interp = _drift_interpretation(drift_z)
        sentences.append(
            f"Drift signal: trailing-100 confidence median is "
            f"{float(drift_z):+.2f}σ from training distribution ({interp})."
        )

    return " ".join(sentences)


def _build_llm_prompt(payload: ExplainPayload) -> str:
    """Format the payload + user question into a single LLM prompt."""
    top_features = payload.get("top_features") or []
    top_lines = "\n".join(
        f"  - {row['feature']}: Δ{float(row['shap_value']):+.3f}"
        for row in top_features[:5]
    ) or "  - (none)"
    drift_z = payload.get("drift_z")
    drift_str = "n/a" if drift_z is None else f"{float(drift_z):+.2f}"
    user_q = payload.get("user_question") or (
        "Explain the prediction in 2-4 sentences."
    )
    return (
        "You are a clinical-ML explainer for a B2B blood-brain-barrier "
        "permeability tool. Given the prediction details below, write a "
        "2-4 sentence rationale a researcher could paste into a paper. "
        "Use the SHAP attributions to justify the verdict. Mention drift "
        "if abnormal. Avoid hedging; be specific about the numbers.\n\n"
        f"Prediction:\n"
        f"- SMILES: {payload.get('smiles', '?')}\n"
        f"- Verdict: {payload.get('label_text', '?')} "
        f"({float(payload.get('confidence', 0.0)) * 100:.0f}% confident)\n"
        f"- Top SHAP features (positive = pushed toward verdict):\n"
        f"{top_lines}\n"
        f"- Drift z-score: {drift_str}\n"
        f"\nUser question: {user_q}\n"
        f"\nRespond with the rationale only, no preamble."
    )


def _llm_explain(payload: ExplainPayload) -> tuple[str, str] | None:
    """Try the OpenRouter chat completion. Return (rationale, model) or None."""
    try:
        # Local import — keeps this dep optional at module load time.
        from openai import OpenAI
    except ImportError as e:
        logger.warning("openai SDK not importable: %s", e)
        return None

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return None

    client = OpenAI(
        base_url=_OPENROUTER_BASE_URL,
        api_key=api_key,
        timeout=_LLM_TIMEOUT_SECONDS,
    )
    prompt = _build_llm_prompt(payload)
    try:
        completion = client.chat.completions.create(
            model=_DEFAULT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=_LLM_MAX_TOKENS,
            temperature=_LLM_TEMPERATURE,
        )
    except Exception as e:  # broad: APITimeoutError, APIConnectionError, RateLimitError, ...
        logger.warning("LLM call failed (%s); falling back to template.", type(e).__name__)
        return None

    try:
        text = completion.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as e:
        logger.warning("LLM response malformed (%s); falling back to template.", e)
        return None

    if not text or not text.strip():
        logger.warning("LLM returned empty rationale; falling back to template.")
        return None

    return text.strip(), _DEFAULT_MODEL


def explain(payload: ExplainPayload) -> ExplainResult:
    """Return a natural-language rationale for a BBB prediction.

    Tries the LLM first when env-permitted; falls back to a deterministic
    template on any failure. Never raises.
    """
    if _should_use_llm():
        llm_out: Any = _llm_explain(payload)
        if llm_out is not None:
            rationale, model = llm_out
            return ExplainResult(rationale=rationale, source="llm", model=model)
        # else: fall through to template
    return ExplainResult(
        rationale=_template_explain(payload),
        source="template",
        model=None,
    )
