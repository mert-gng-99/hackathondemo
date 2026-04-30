"""POST /pipeline/{bbb,eeg,mri} routes — thin dispatchers over the pipelines.

Each route validates its request body via Pydantic, invokes the pipeline,
reads back the produced Parquet to populate row/column counts, and returns
a uniform PipelineResponse. Pipeline-domain errors map to standard HTTP
codes: FileNotFoundError -> 404, ValueError -> 400, anything else -> 500.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable

import mlflow
import pandas as pd
from fastapi import APIRouter, HTTPException

from src.api.schemas import (
    BBBPredictRequest,
    BBBPredictResponse,
    BBBRequest,
    CalibrationContext,
    EEGRequest,
    FeatureAttribution,
    MRIRequest,
    PipelineResponse,
)
from src.core.logger import get_logger
from src.models import bbb_model
from src.pipelines import bbb_pipeline, eeg_pipeline, mri_pipeline

logger = get_logger(__name__)
router = APIRouter(prefix="/pipeline")
predict_router = APIRouter(prefix="/predict")


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
    return BBBPredictResponse(
        label=pred["label"],
        label_text=label_text,
        confidence=pred["confidence"],
        top_features=[FeatureAttribution(**a) for a in attributions],
        calibration=calibration,
    )
