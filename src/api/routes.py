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
    AgentRunRequest,
    AgentRunResponse,
    AgentToolTraceItem,
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
    MLflowRunsResponse,
    MLflowRunSummary,
    ModelProvenance,
    MRIDiagnosticsRequest,
    MRIDiagnosticsResponse,
    MRIClassProbability,
    MRIExplainRequest,
    MRIExplainResponse,
    MRIPredictRequest,
    MRIPredictResponse,
    MRIRequest,
    PipelineResponse,
    RunDiffRequest,
    RunDiffResponse,
    RunDiffRow,
)
from src.core.logger import get_logger
from src.llm import explainer as llm_explainer
from src.models import bbb_model, mri_model
from src.pipelines import bbb_pipeline, eeg_pipeline, mri_pipeline

logger = get_logger(__name__)
router = APIRouter(prefix="/pipeline")
predict_router = APIRouter(prefix="/predict")
explain_router = APIRouter(prefix="/explain")
experiments_router = APIRouter(prefix="/experiments")


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
    run_id = _latest_mlflow_run_id(experiment_name)

    return PipelineResponse(
        status="ok",
        output_path=str(output_path),
        rows=len(df),
        columns=df.shape[1],
        duration_sec=duration_sec,
        mlflow_run_id=run_id,
    )


def _latest_mlflow_run_id(experiment_name: str) -> str | None:
    """Return the newest MLflow run id, degrading to None when tracking is off."""
    if os.environ.get("NEUROBRIDGE_DISABLE_MLFLOW") == "1":
        return None
    try:
        runs = mlflow.search_runs(
            experiment_names=[experiment_name],
            max_results=1,
            order_by=["start_time DESC"],
        )
    except Exception as e:
        logger.warning("MLflow run lookup failed for %s: %s", experiment_name, e)
        return None
    return str(runs.iloc[0]["run_id"]) if len(runs) else None


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
_DEFAULT_MRI_MODEL_PATH = Path("data/processed/mri_model.onnx")


def _bbb_model_path() -> Path:
    """Return the BBB model artifact path, overridable via BBB_MODEL_PATH env var."""
    return Path(os.environ.get("BBB_MODEL_PATH", str(_DEFAULT_BBB_MODEL_PATH)))


def _mri_model_path() -> Path:
    """Return the MRI ONNX model artifact path, overridable via MRI_MODEL_PATH."""
    return Path(os.environ.get("MRI_MODEL_PATH", str(_DEFAULT_MRI_MODEL_PATH)))


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


@predict_router.post("/mri", response_model=MRIPredictResponse)
def predict_mri(req: MRIPredictRequest) -> MRIPredictResponse:
    """Predict from one MRI NIfTI image using an externally-trained ONNX model."""
    artifact = _mri_model_path()
    if not artifact.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"MRI model artifact not available at {artifact}. "
                "Export the trained volumetric model to ONNX and place it there, "
                "or set MRI_MODEL_PATH."
            ),
        )
    try:
        model = mri_model.load(artifact)
        pred = mri_model.predict_nifti(
            model,
            Path(req.input_path),
            target_shape=req.target_shape,
            label_names=req.label_names,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return MRIPredictResponse(
        label=int(pred["label"]),
        label_text=str(pred["label_text"]),
        confidence=float(pred["confidence"]),
        probabilities=[
            MRIClassProbability(**p)
            for p in pred["probabilities"]
        ],
        input_path=req.input_path,
        model_path=str(artifact),
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


@experiments_router.get("/runs", response_model=MLflowRunsResponse)
def list_runs(limit: int = 50) -> MLflowRunsResponse:
    """List recent MLflow runs across known experiments.

    Returns an empty list when MLflow is disabled or unreachable.
    """
    if os.environ.get("NEUROBRIDGE_DISABLE_MLFLOW") == "1":
        return MLflowRunsResponse(runs=[])

    summaries: list[MLflowRunSummary] = []
    for exp_name in ("bbb_pipeline", "eeg_pipeline", "mri_pipeline"):
        try:
            df = mlflow.search_runs(
                experiment_names=[exp_name],
                max_results=limit,
                order_by=["start_time DESC"],
            )
        except Exception as e:  # broad: MLflow store unreachable / not found
            logger.warning("MLflow lookup failed for %s: %s", exp_name, e)
            continue
        for _, row in df.iterrows():
            metrics = {
                col[len("metrics."):]: float(row[col])
                for col in df.columns
                if col.startswith("metrics.") and pd.notna(row[col])
            }
            params = {
                col[len("params."):]: str(row[col])
                for col in df.columns
                if col.startswith("params.") and pd.notna(row[col])
            }
            summaries.append(
                MLflowRunSummary(
                    run_id=str(row["run_id"]),
                    experiment_name=exp_name,
                    start_time=str(pd.Timestamp(row["start_time"]).isoformat())
                    if pd.notna(row.get("start_time"))
                    else "",
                    status=str(row.get("status", "UNKNOWN")),
                    metrics=metrics,
                    params=params,
                )
            )
    summaries.sort(key=lambda s: s.start_time, reverse=True)
    return MLflowRunsResponse(runs=summaries[:limit])


@experiments_router.post("/diff", response_model=RunDiffResponse)
def diff_runs(req: RunDiffRequest) -> RunDiffResponse:
    """Side-by-side diff of two MLflow runs (metrics + params).

    Returns 404 if either run id is not found in the local MLflow store.
    Returns 200 with an empty rows list when MLflow is disabled.
    """
    if os.environ.get("NEUROBRIDGE_DISABLE_MLFLOW") == "1":
        return RunDiffResponse(rows=[])

    try:
        run_a = mlflow.get_run(req.run_id_a)
        run_b = mlflow.get_run(req.run_id_b)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Run not found: {e}")

    metrics_a = run_a.data.metrics
    metrics_b = run_b.data.metrics
    params_a = run_a.data.params
    params_b = run_b.data.params

    rows: list[RunDiffRow] = []
    for key in sorted(set(metrics_a) | set(metrics_b)):
        va = metrics_a.get(key)
        vb = metrics_b.get(key)
        rows.append(
            RunDiffRow(
                key=key, kind="metric",
                value_a=None if va is None else f"{va:.6g}",
                value_b=None if vb is None else f"{vb:.6g}",
                differs=(va != vb),
            )
        )
    for key in sorted(set(params_a) | set(params_b)):
        va = params_a.get(key)
        vb = params_b.get(key)
        rows.append(
            RunDiffRow(
                key=key, kind="param",
                value_a=va, value_b=vb, differs=(va != vb),
            )
        )
    return RunDiffResponse(rows=rows)


# --- Agent router ----------------------------------------------------------

agent_router = APIRouter(prefix="/agent")


_DEFAULT_RAG_INDEX_DIR = Path("data/processed/faiss_index")
_AGENT_MODEL_ENV = "NEUROBRIDGE_AGENT_MODEL"
_AGENT_DEFAULT_MODEL = "google/gemini-2.0-flash-exp:free"


def _build_orchestrator():
    """Construct the default orchestrator. Patchable in tests."""
    from openai import OpenAI

    from src.agents.orchestrator import Orchestrator
    from src.agents.prompts import ORCHESTRATOR_SYSTEM_PROMPT
    from src.agents.routing import build_retrieval_query, route_pipeline_input
    from src.agents.tools import build_default_tools

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="OPENROUTER_API_KEY not set; agent surface unavailable.",
        )
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        timeout=30.0,
    )
    rag_dir = _DEFAULT_RAG_INDEX_DIR if _DEFAULT_RAG_INDEX_DIR.exists() else None
    tools = build_default_tools(rag_index_dir=rag_dir)
    model = os.environ.get(_AGENT_MODEL_ENV, _AGENT_DEFAULT_MODEL)
    return Orchestrator(
        llm_client=client,
        tools=tools,
        system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
        model=model,
        max_steps=5,
        enforce_workflow=True,
        workflow_pipeline_tools={
            "run_bbb_pipeline",
            "run_eeg_pipeline",
            "run_mri_pipeline",
        },
        workflow_retrieval_tool="retrieve_context",
        workflow_router=route_pipeline_input,
        workflow_query_builder=build_retrieval_query,
    )


@agent_router.post("/run", response_model=AgentRunResponse)
def run_agent(req: AgentRunRequest) -> AgentRunResponse:
    """Run the orchestrator on `user_input`. Picks a pipeline + grounds via RAG."""
    orch = _build_orchestrator()
    user_text = req.user_input
    if req.user_question:
        user_text = f"{req.user_input}\n\nUser question: {req.user_question}"
    result = orch.run(user_text, context={"sites_csv": req.sites_csv})
    return AgentRunResponse(
        text=result.text,
        trace=[
            AgentToolTraceItem(name=t.name, args=t.args, result=t.result, error=t.error)
            for t in result.trace
        ],
        model=result.model,
        finish_reason=result.finish_reason,
    )
