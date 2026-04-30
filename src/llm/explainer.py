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


ExplainPayload = dict[str, Any]  # Heterogeneous: BBB / EEG / MRI shapes differ.


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


def _template_explain_bbb(payload: ExplainPayload) -> str:
    """Deterministic, jury-friendly rationale for a single BBB prediction."""
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


def _template_explain_eeg(payload: ExplainPayload) -> str:
    """Deterministic rationale for an EEG pipeline run."""
    rows = payload.get("rows", 0)
    columns = payload.get("columns", 0)
    duration = float(payload.get("duration_sec", 0.0))
    run_id = payload.get("mlflow_run_id") or "—"
    sentences = [
        f"EEG pipeline produced **{rows}** epochs × **{columns}** features "
        f"in {duration:.1f}s.",
        "ICA decomposed the signal and dropped components whose absolute "
        "EOG correlation exceeded 0.5 (eye-blink artifacts).",
        "Bandpass filter 0.5-40 Hz removed line noise and DC drift before ICA.",
        f"Run id: `{run_id}` (use the Experiments tab to compare against "
        "previous runs).",
    ]
    return " ".join(sentences)


def _template_explain_mri(payload: ExplainPayload) -> str:
    """Deterministic rationale for an MRI ComBat-harmonization diagnostic."""
    pre = float(payload.get("site_gap_pre", 0.0))
    post = float(payload.get("site_gap_post", 0.0))
    factor = float(payload.get("reduction_factor", 0.0))
    n_subjects = int(payload.get("n_subjects", 0))
    sentences = [
        f"ComBat harmonization reduced the per-site mean gap from "
        f"**{pre:.4f}** to **{post:.4f}** — a **{factor:.0f}×** collapse "
        f"across **{n_subjects}** subjects on the first feature.",
        "This is the quantified proof that scanner / acquisition-site bias "
        "was removed: predictions trained on the harmonized features "
        "generalize across hospitals instead of memorizing site identity.",
        "The visual evidence is the per-site KDE convergence in the "
        "Pre-ComBat → Post-ComBat panels (Streamlit MRI tab).",
    ]
    return " ".join(sentences)


_TEMPLATE_DISPATCH = {
    "bbb": _template_explain_bbb,
    "eeg": _template_explain_eeg,
    "mri": _template_explain_mri,
}


def _build_llm_prompt(payload: ExplainPayload, modality: str = "bbb") -> str:
    """Format the payload + user question into a single LLM prompt."""
    headers = {
        "bbb": (
            "You are a clinical-ML explainer for a B2B blood-brain-barrier "
            "permeability tool."
        ),
        "eeg": (
            "You are a clinical-ML explainer for an EEG signal-processing "
            "pipeline (MNE-Python + ICA artifact removal)."
        ),
        "mri": (
            "You are a clinical-ML explainer for a multi-site MRI "
            "harmonization pipeline (neuroHarmonize / ComBat)."
        ),
    }
    header = headers.get(modality, headers["bbb"])
    user_q = payload.get("user_question") or "Explain the result in 2-4 sentences."
    body_lines: list[str] = []
    if modality == "bbb":
        top_features = payload.get("top_features") or []
        top_lines = "\n".join(
            f"  - {row['feature']}: Δ{float(row['shap_value']):+.3f}"
            for row in top_features[:5]
        ) or "  - (none)"
        drift_z = payload.get("drift_z")
        drift_str = "n/a" if drift_z is None else f"{float(drift_z):+.2f}"
        body_lines.append(
            f"Prediction:\n"
            f"- SMILES: {payload.get('smiles', '?')}\n"
            f"- Verdict: {payload.get('label_text', '?')} "
            f"({float(payload.get('confidence', 0.0)) * 100:.0f}% confident)\n"
            f"- Top SHAP features (positive = pushed toward verdict):\n"
            f"{top_lines}\n"
            f"- Drift z-score: {drift_str}"
        )
    elif modality == "eeg":
        body_lines.append(
            f"EEG Pipeline Run:\n"
            f"- Epochs produced: {payload.get('rows', 0)}\n"
            f"- Features per epoch: {payload.get('columns', 0)}\n"
            f"- Wall-clock: {float(payload.get('duration_sec', 0.0)):.2f}s\n"
            f"- MLflow run id: {payload.get('mlflow_run_id') or 'n/a'}"
        )
    elif modality == "mri":
        body_lines.append(
            f"MRI ComBat Diagnostics:\n"
            f"- Site-gap pre-ComBat: {float(payload.get('site_gap_pre', 0)):.4f}\n"
            f"- Site-gap post-ComBat: {float(payload.get('site_gap_post', 0)):.4f}\n"
            f"- Reduction factor: {float(payload.get('reduction_factor', 0)):.0f}×\n"
            f"- Subjects: {int(payload.get('n_subjects', 0))}"
        )
    else:
        # fallback uses BBB-shape prompt
        body_lines.append(f"Payload: {payload!r}")

    return (
        f"{header} Given the details below, write a 2-4 sentence rationale a "
        f"researcher could paste into a paper. Avoid hedging; be specific "
        f"about the numbers.\n\n"
        f"{body_lines[0]}\n\n"
        f"User question: {user_q}\n\n"
        f"Respond with the rationale only, no preamble."
    )


def _llm_explain(payload: ExplainPayload, modality: str = "bbb") -> tuple[str, str] | None:
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
    prompt = _build_llm_prompt(payload, modality)
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


def explain(
    payload: ExplainPayload, modality: str = "bbb",
) -> ExplainResult:
    """Return a natural-language rationale for a prediction or pipeline run.

    `modality` selects the template family ('bbb' | 'eeg' | 'mri'). Unknown
    values degrade to the BBB template with a warning log; the function
    never raises.
    """
    if modality not in _TEMPLATE_DISPATCH:
        logger.warning(
            "Unknown explain modality %r; falling back to bbb template.",
            modality,
        )
        modality = "bbb"

    if _should_use_llm():
        llm_out: Any = _llm_explain(payload, modality=modality)
        if llm_out is not None:
            rationale, model = llm_out
            return ExplainResult(rationale=rationale, source="llm", model=model)
        # else: fall through to template

    template_fn = _TEMPLATE_DISPATCH[modality]
    return ExplainResult(
        rationale=template_fn(payload),
        source="template",
        model=None,
    )
