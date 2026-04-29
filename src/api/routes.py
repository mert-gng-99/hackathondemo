"""POST /pipeline/{bbb,eeg,mri} routes — thin dispatchers over the pipelines.

Each route validates its request body via Pydantic, invokes the pipeline,
reads back the produced Parquet to populate row/column counts, and returns
a uniform PipelineResponse. Pipeline-domain errors map to standard HTTP
codes: FileNotFoundError -> 404, ValueError -> 400, anything else -> 500.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import mlflow
import pandas as pd
from fastapi import APIRouter, HTTPException

from src.api.schemas import (
    BBBRequest,
    EEGRequest,
    MRIRequest,
    PipelineResponse,
)
from src.core.logger import get_logger
from src.pipelines import bbb_pipeline, eeg_pipeline, mri_pipeline

logger = get_logger(__name__)
router = APIRouter(prefix="/pipeline")


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
