"""POST /pipeline/{bbb,eeg,mri} routes — thin dispatchers over the pipelines.

Each route validates its request body via Pydantic, invokes the pipeline,
reads back the produced Parquet to populate row/column counts, and returns
a uniform PipelineResponse. Pipeline-domain errors map to standard HTTP
codes: FileNotFoundError -> 404, ValueError -> 400, anything else -> 500.
"""
from __future__ import annotations

import os
import time
from collections import deque
from pathlib import Path
from typing import Callable

import mlflow
import pandas as pd
from fastapi import APIRouter, HTTPException

from src.api.schemas import (
    BBBExplainRequest,
    BBBExplainResponse,
    BBBPredictRequest,
    BBBPredictResponse,
    BBBRequest,
    CalibrationContext,
    EEGExplainRequest,
    EEGExplainResponse,
    EEGRequest,
    FeatureAttribution,
    HarmonizationRow,
    ModelProvenance,
    MRIDiagnosticsRequest,
    MRIDiagnosticsResponse,
    MRIExplainRequest,
    MRIExplainResponse,
    MRIRequest,
    PipelineResponse,
)
from src.core.logger import get_logger
from src.llm import explainer as llm_explainer
from src.models import bbb_model
from src.pipelines import bbb_pipeline, eeg_pipeline, mri_pipeline

logger = get_logger(__name__)
router = APIRouter(prefix="/pipeline")
predict_router = APIRouter(prefix="/predict")
explain_router = APIRouter(prefix="/explain")


def _wrap(
    experiment_name: str,
    output_path: Path,
    fn: Callable[[], None],
) -> PipelineResponse:
    """Run `fn()` (the pipeline call), gather metrics, return PipelineResponse."""
    started = time.perf_counter()
    try:
        fn()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except (ValueError, KeyError) as e:
        # KeyError: MRI pipeline raises this when sites_csv is missing a
        # site assignment for a subject — a user data problem, not a 500.
        raise HTTPException(status_code=400, detail=str(e))
    duration_sec = time.perf_counter() - started

    df = pd.read_parquet(output_path)
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        max_results=1,
        order_by=["start_time DESC"],
    )
    run_id = runs.iloc[0]["run_id"] if len(runs) else None

    return PipelineResponse(
        status="ok",
        output_path=str(output_path),
        rows=len(df),
        columns=df.shape[1],
        duration_sec=duration_sec,
        mlflow_run_id=run_id,
    )


@router.post("/bbb", response_model=PipelineResponse)
def run_bbb(req: BBBRequest) -> PipelineResponse:
    """Run the BBB pipeline; return rows/cols/duration + the MLflow run id."""
    return _wrap(
        "bbb_pipeline",
        Path(req.output_path),
        lambda: bbb_pipeline.run_pipeline(
            input_path=Path(req.input_path),
            output_path=Path(req.output_path),
            smiles_col=req.smiles_col,
            n_bits=req.n_bits,
            radius=req.radius,
        ),
    )


@router.post("/eeg", response_model=PipelineResponse)
def run_eeg(req: EEGRequest) -> PipelineResponse:
    """Run the EEG pipeline; return rows/cols/duration + the MLflow run id."""
    return _wrap(
        "eeg_pipeline",
        Path(req.output_path),
        lambda: eeg_pipeline.run_pipeline(
            input_path=Path(req.input_path),
            output_path=Path(req.output_path),
            epoch_duration_s=req.epoch_duration_s,
            eog_ch_name=req.eog_ch_name,
            n_components=req.n_components,
            random_state=req.random_state,
        ),
    )


@router.post("/mri", response_model=PipelineResponse)
def run_mri(req: MRIRequest) -> PipelineResponse:
    """Run the MRI pipeline; return rows/cols/duration + the MLflow run id."""
    return _wrap(
        "mri_pipeline",
        Path(req.output_path),
        lambda: mri_pipeline.run_pipeline(
            input_dir=Path(req.input_dir),
            sites_csv=Path(req.sites_csv),
            output_path=Path(req.output_path),
        ),
    )


# Default artifact location. Overridable via BBB_MODEL_PATH env var so tests
# can point at a tmp-built model without touching production paths.
_DEFAULT_BBB_MODEL_PATH = Path("data/processed/bbb_model.joblib")


def _bbb_model_path() -> Path:
    """Return the BBB model artifact path, overridable via BBB_MODEL_PATH env var."""
    return Path(os.environ.get("BBB_MODEL_PATH", str(_DEFAULT_BBB_MODEL_PATH)))


# Per-worker rolling window of recent prediction confidences.
# Cleared on worker restart; multi-worker setups have independent windows.
WORKER_CONFIDENCE_DEQUE: deque[float] = deque(maxlen=100)
_DRIFT_MIN_SAMPLES = 10


def _compute_drift_z(model, confidence: float) -> tuple[float | None, int]:
    """Append `confidence` to the worker deque and compute the drift z-score.

    Returns (drift_z, rolling_n). drift_z is None until both:
      (1) the deque has at least `_DRIFT_MIN_SAMPLES` samples, AND
      (2) the model has `_neurobridge_train_stats` attached.

    z = (rolling_median - train_median) / max(train_std, 1e-9)
    """
    import statistics

    WORKER_CONFIDENCE_DEQUE.append(float(confidence))
    rolling_n = len(WORKER_CONFIDENCE_DEQUE)
    stats = getattr(model, "_neurobridge_train_stats", None)
    if rolling_n < _DRIFT_MIN_SAMPLES or stats is None:
        return None, rolling_n
    rolling_median = statistics.median(WORKER_CONFIDENCE_DEQUE)
    train_median = float(stats["median"])
    train_std = max(float(stats["std"]), 1e-9)
    drift_z = (rolling_median - train_median) / train_std
    return float(drift_z), rolling_n


_PROVENANCE_CACHE: ModelProvenance | None = None
_MODEL_VERSION = "v1"  # bump manually per train cycle


def _build_provenance(model) -> ModelProvenance:
    """Look up the most recent BBB MLflow run; build a ModelProvenance.

    Cached at module level so we hit MLflow once per worker. Failures (no
    runs found, MLflow unreachable, NEUROBRIDGE_DISABLE_MLFLOW=1) all
    degrade to a partial ModelProvenance with mlflow_run_id=None — the
    badge still renders, just without a run id.
    """
    global _PROVENANCE_CACHE
    if _PROVENANCE_CACHE is not None:
        # Refresh n_examples each call from the model (cheap lookup).
        n_train = None
        stats = getattr(model, "_neurobridge_train_stats", None)
        if stats is not None:
            n_train = int(stats.get("n_train", 0)) or None
        return _PROVENANCE_CACHE.model_copy(update={"n_examples": n_train})

    run_id: str | None = None
    train_date: str | None = None
    if os.environ.get("NEUROBRIDGE_DISABLE_MLFLOW") != "1":
        try:
            runs = mlflow.search_runs(
                experiment_names=["bbb_pipeline"],
                max_results=1,
                order_by=["start_time DESC"],
            )
            if len(runs):
                row = runs.iloc[0]
                run_id = str(row["run_id"])
                ts = row.get("start_time")
                if ts is not None:
                    train_date = str(pd.Timestamp(ts).isoformat())
        except Exception as e:  # broad: MLflow store unreachable, schema mismatch, etc.
            logger.warning("MLflow provenance lookup failed: %s", e)

    n_train = None
    stats = getattr(model, "_neurobridge_train_stats", None)
    if stats is not None:
        n_train = int(stats.get("n_train", 0)) or None

    _PROVENANCE_CACHE = ModelProvenance(
        mlflow_run_id=run_id,
        model_version=_MODEL_VERSION,
        train_date=train_date,
        n_examples=n_train,
    )
    return _PROVENANCE_CACHE


def _matching_calibration_bin(model, confidence: float) -> CalibrationContext | None:
    """Pick the highest-threshold bin whose threshold <= confidence. None if no match or no metadata."""
    bins = getattr(model, "_neurobridge_calibration", None)
    if not bins:
        return None
    matched = None
    for bin_ in bins:
        if bin_["threshold"] <= confidence:
            matched = bin_
        else:
            break
    if matched is None:
        return None
    return CalibrationContext(
        threshold=matched["threshold"],
        precision=matched["precision"],
        support=matched["support"],
    )


@predict_router.post("/bbb", response_model=BBBPredictResponse)
def predict_bbb(req: BBBPredictRequest) -> BBBPredictResponse:
    """Predict BBB permeability + return SHAP attributions for one SMILES.

    Returns 503 if the model artifact is missing (operator hasn't run the
    trainer CLI yet); 400 on invalid SMILES; 200 with the decision payload
    on success.
    """
    artifact = _bbb_model_path()
    if not artifact.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"BBB model artifact not available at {artifact}. "
                f"Run `python -m src.models.bbb_model` to train it."
            ),
        )
    try:
        model = bbb_model.load(artifact)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    try:
        pred = bbb_model.predict_with_proba(model, req.smiles)
        attributions = bbb_model.explain_prediction(model, req.smiles, top_k=req.top_k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    label_text = "permeable" if pred["label"] == 1 else "non-permeable"
    calibration = _matching_calibration_bin(model, pred["confidence"])
    drift_z, rolling_n = _compute_drift_z(model, pred["confidence"])
    provenance = _build_provenance(model)
    return BBBPredictResponse(
        label=pred["label"],
        label_text=label_text,
        confidence=pred["confidence"],
        top_features=[FeatureAttribution(**a) for a in attributions],
        calibration=calibration,
        drift_z=drift_z,
        rolling_n=rolling_n,
        provenance=provenance,
    )


@router.post("/mri/diagnostics", response_model=MRIDiagnosticsResponse)
def mri_diagnostics(req: MRIDiagnosticsRequest) -> MRIDiagnosticsResponse:
    """Run the MRI pipeline twice and return pre/post ComBat data + site-gap KPIs."""
    input_dir = Path(req.input_dir)
    sites_csv = Path(req.sites_csv)
    try:
        df = mri_pipeline.compute_harmonization_diagnostics(
            input_dir=input_dir, sites_csv=sites_csv,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if df.empty:
        return MRIDiagnosticsResponse(
            rows=[], site_gap_pre=0.0, site_gap_post=0.0, reduction_factor=0.0,
        )

    # Site-gap KPI on the first feature, averaged per site
    feat = df["feature"].iloc[0]
    feat_df = df[df["feature"] == feat]
    pre_means = feat_df[feat_df["harmonization_state"] == "Pre-ComBat"].groupby(
        "site"
    )["feature_value"].mean()
    post_means = feat_df[feat_df["harmonization_state"] == "Post-ComBat"].groupby(
        "site"
    )["feature_value"].mean()
    site_gap_pre = float(pre_means.max() - pre_means.min())
    site_gap_post = float(post_means.max() - post_means.min())
    eps = 1e-9
    reduction_factor = site_gap_pre / max(site_gap_post, eps)

    rows = [
        HarmonizationRow(**rec) for rec in df.to_dict(orient="records")
    ]
    return MRIDiagnosticsResponse(
        rows=rows,
        site_gap_pre=site_gap_pre,
        site_gap_post=site_gap_post,
        reduction_factor=reduction_factor,
    )


@explain_router.post("/bbb", response_model=BBBExplainResponse)
def explain_bbb(req: BBBExplainRequest) -> BBBExplainResponse:
    """Natural-language rationale for a single BBB prediction.

    Always returns 200 — the explainer is guaranteed to produce a
    rationale via deterministic-template fallback. Pydantic enforces
    a non-empty top_features list; an empty list returns 422 from
    FastAPI before this handler runs.
    """
    payload: llm_explainer.ExplainPayload = {
        "smiles": req.smiles,
        "label": req.label,
        "label_text": req.label_text,
        "confidence": req.confidence,
        "top_features": [
            {"feature": f.feature, "shap_value": f.shap_value}
            for f in req.top_features
        ],
        "calibration": (
            None
            if req.calibration is None
            else {
                "threshold": req.calibration.threshold,
                "precision": req.calibration.precision,
                "support": req.calibration.support,
            }
        ),
        "drift_z": req.drift_z,
        "user_question": req.user_question or "",
    }
    result = llm_explainer.explain(payload)
    return BBBExplainResponse(
        rationale=result["rationale"],
        source=result["source"],
        model=result["model"],
    )


@explain_router.post("/eeg", response_model=EEGExplainResponse)
def explain_eeg(req: EEGExplainRequest) -> EEGExplainResponse:
    """Natural-language rationale for an EEG pipeline run."""
    payload = {
        "rows": req.rows,
        "columns": req.columns,
        "duration_sec": req.duration_sec,
        "mlflow_run_id": req.mlflow_run_id,
        "user_question": req.user_question or "",
    }
    result = llm_explainer.explain(payload, modality="eeg")
    return EEGExplainResponse(
        rationale=result["rationale"],
        source=result["source"],
        model=result["model"],
    )


@explain_router.post("/mri", response_model=MRIExplainResponse)
def explain_mri(req: MRIExplainRequest) -> MRIExplainResponse:
    """Natural-language rationale for an MRI ComBat diagnostic run."""
    payload = {
        "site_gap_pre": req.site_gap_pre,
        "site_gap_post": req.site_gap_post,
        "reduction_factor": req.reduction_factor,
        "n_subjects": req.n_subjects,
        "user_question": req.user_question or "",
    }
    result = llm_explainer.explain(payload, modality="mri")
    return MRIExplainResponse(
        rationale=result["rationale"],
        source=result["source"],
        model=result["model"],
    )
