"""Pydantic request / response models for the NeuroBridge FastAPI surface.

Each pipeline accepts its own request schema (BBBRequest / EEGRequest /
MRIRequest) but they all return a unified PipelineResponse — the dashboard
can render a single result card regardless of modality.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class BBBRequest(BaseModel):
    input_path: str = Field(..., description="CSV path with a 'smiles' column")
    output_path: str = Field(..., description="Parquet output path")
    smiles_col: str = "smiles"
    n_bits: int = 2048
    radius: int = 2


class EEGRequest(BaseModel):
    """Field names mirror eeg_pipeline.run_pipeline kwargs exactly."""
    input_path: str = Field(..., description="FIF or EDF file")
    output_path: str = Field(..., description="Parquet output path")
    epoch_duration_s: float = 2.0
    eog_ch_name: str | None = None
    n_components: int = 15
    random_state: int = 97


class MRIRequest(BaseModel):
    input_dir: str = Field(..., description="Directory of .nii.gz files")
    sites_csv: str = Field(..., description="CSV mapping subject_id → site")
    output_path: str = Field(..., description="Parquet output path")


class PipelineResponse(BaseModel):
    """Uniform response for every pipeline route."""
    status: str
    output_path: str
    rows: int
    columns: int
    duration_sec: float
    mlflow_run_id: str | None = None


class HealthResponse(BaseModel):
    status: str
    pipelines: list[str]


class BBBPredictRequest(BaseModel):
    """Single-molecule BBB-permeability prediction request."""
    smiles: str = Field(..., description="SMILES string; e.g. 'CCO' for ethanol")
    top_k: int = Field(5, ge=1, le=20, description="Top-k SHAP features to return")


class FeatureAttribution(BaseModel):
    """A single SHAP attribution: which fingerprint bit contributed and by how much."""
    feature: str = Field(..., description="Fingerprint column name, e.g. 'fp_1234'")
    shap_value: float = Field(
        ...,
        description="Signed SHAP value for the predicted class (positive pushed model toward, negative away)",
    )


class CalibrationContext(BaseModel):
    """Precision-at-confidence-threshold bin matched to a single prediction."""
    threshold: float = Field(..., description="Lowest confidence threshold this bin covers (0.0-1.0)")
    precision: float = Field(..., description="Precision on the held-out test set among predictions ≥ threshold")
    support: int = Field(..., description="Number of held-out predictions falling in this bin")


class ModelProvenance(BaseModel):
    """Auditable provenance of the BBB model that produced a prediction."""
    # Disable the `model_` protected-namespace check so `model_version` doesn't
    # trip Pydantic v2's UserWarning (which our DoD gate escalates to error).
    model_config = ConfigDict(protected_namespaces=())

    mlflow_run_id: str | None = Field(None, description="MLflow run id of the most recent training run, if any")
    model_version: str = Field("v1", description="Manually-bumped model version label")
    train_date: str | None = Field(None, description="ISO 8601 train timestamp from MLflow run start_time")
    n_examples: int | None = Field(None, description="Training set size (from model._neurobridge_train_stats[\"n_train\"])")


class BBBPredictResponse(BaseModel):
    """Decision-system payload: prediction + uncertainty + explanation + drift."""
    label: int
    label_text: str = Field(..., description="'permeable' or 'non-permeable'")
    confidence: float
    top_features: list[FeatureAttribution]
    calibration: CalibrationContext | None = Field(
        None,
        description="Statistical context: how often the model is right when this confident on held-out data.",
    )
    drift_z: float | None = Field(
        None,
        description=(
            "Z-score of the trailing-100 confidence median against the "
            "train-time median; None when warming up (<10 samples) or "
            "when the model lacks _neurobridge_train_stats."
        ),
    )
    rolling_n: int = Field(
        0,
        description=(
            "Number of confidence samples currently buffered in the worker's "
            "rolling window (max 100). Zero on a fresh worker."
        ),
    )
    provenance: ModelProvenance | None = Field(
        None,
        description="Auditing metadata (MLflow run id, train date, n_examples).",
    )


class EEGPredictRequest(BaseModel):
    """Single-subject EEG-features prediction request."""
    features: list[float] = Field(
        ..., min_length=1,
        description="EEG features matching the classifier's training-time feature count.",
    )


class EEGClassProbability(BaseModel):
    """One EEG model class probability."""
    label: int
    label_text: str
    probability: float


class EEGPredictResponse(BaseModel):
    """EEG prediction payload — same shape as MRIPredictResponse minus model_path."""
    label: int
    label_text: str
    confidence: float
    probabilities: list[EEGClassProbability]


class MRIPredictRequest(BaseModel):
    """Single-subject MRI image prediction request."""
    input_path: str = Field(
        ...,
        description=(
            "Path to MRI input. With MRI_MODEL_KIND=volumetric_onnx (default), "
            "expects a .nii/.nii.gz volume. With MRI_MODEL_KIND=resnet18_2d, "
            "expects a 2D image (.png/.jpg)."
        ),
    )
    target_shape: tuple[int, int, int] = Field(
        (64, 64, 64),
        description="Model preprocessing resize target as (D, H, W)",
    )
    label_names: list[str] | None = Field(
        None,
        description="Optional class labels matching ONNX output order",
    )


class MRIClassProbability(BaseModel):
    """One MRI model class probability."""
    label: int
    label_text: str
    probability: float


class MRIPredictResponse(BaseModel):
    """MRI DL decision payload from a volumetric ONNX model."""
    model_config = ConfigDict(protected_namespaces=())

    label: int
    label_text: str
    confidence: float
    probabilities: list[MRIClassProbability]
    input_path: str
    model_path: str


class MRIDiagnosticsRequest(BaseModel):
    """Request body for /pipeline/mri/diagnostics — same as MRIRequest minus output_path."""
    input_dir: str = Field(..., description="Directory of .nii.gz files")
    sites_csv: str = Field(..., description="CSV mapping subject_id → site")


class HarmonizationRow(BaseModel):
    subject_id: str
    site: str
    feature: str
    feature_value: float
    harmonization_state: str


class MRIDiagnosticsResponse(BaseModel):
    """Long-format pre/post ComBat data for visualization."""
    rows: list[HarmonizationRow]
    site_gap_pre: float = Field(..., description="Range of per-site means before ComBat")
    site_gap_post: float = Field(..., description="Range of per-site means after ComBat")
    reduction_factor: float = Field(..., description="site_gap_pre / max(site_gap_post, eps)")


class BBBExplainRequest(BaseModel):
    """Day-7 T3B: payload for POST /explain/bbb (chat-style explainer)."""
    smiles: str = Field(..., description="SMILES string of the molecule")
    label: int = Field(..., description="Predicted label (0 = non-permeable, 1 = permeable)")
    label_text: str = Field(..., description="'permeable' or 'non-permeable'")
    confidence: float = Field(..., ge=0.0, le=1.0)
    top_features: list[FeatureAttribution] = Field(
        ..., min_length=1,
        description="Non-empty list of SHAP attributions; an empty list returns 400.",
    )
    calibration: CalibrationContext | None = None
    drift_z: float | None = None
    user_question: str | None = Field(
        None,
        description="Optional question from the user; passed to the LLM prompt only.",
    )


class BBBExplainResponse(BaseModel):
    """Day-7 T3B: response from POST /explain/bbb."""
    rationale: str = Field(..., description="2-4 sentence natural-language explanation")
    source: str = Field(..., description="'llm' or 'template'")
    model: str | None = Field(
        None,
        description="LLM model name when source='llm'; None when source='template'",
    )


class EEGExplainRequest(BaseModel):
    """Day-8 T1B: payload for POST /explain/eeg."""
    rows: int = Field(..., ge=0, description="Number of epochs produced")
    columns: int = Field(..., ge=0, description="Number of features per epoch")
    duration_sec: float = Field(..., ge=0.0, description="Pipeline wall-clock seconds")
    mlflow_run_id: str | None = Field(None, description="MLflow run id, if available")
    user_question: str | None = Field(None, description="Optional user question for the LLM prompt")


class EEGExplainResponse(BaseModel):
    """Day-8 T1B: response from POST /explain/eeg."""
    rationale: str
    source: str
    model: str | None = None


class MRIExplainRequest(BaseModel):
    """Day-8 T1B: payload for POST /explain/mri."""
    site_gap_pre: float = Field(..., ge=0.0)
    site_gap_post: float = Field(..., ge=0.0)
    reduction_factor: float = Field(..., ge=0.0)
    n_subjects: int = Field(..., ge=0)
    user_question: str | None = None


class MRIExplainResponse(BaseModel):
    """Day-8 T1B: response from POST /explain/mri."""
    rationale: str
    source: str
    model: str | None = None


class MLflowRunSummary(BaseModel):
    """One MLflow run row for the Experiments tab table."""
    run_id: str
    experiment_name: str
    start_time: str  # ISO 8601
    status: str
    metrics: dict[str, float] = Field(default_factory=dict)
    params: dict[str, str] = Field(default_factory=dict)


class MLflowRunsResponse(BaseModel):
    """Response for GET /experiments/runs."""
    runs: list[MLflowRunSummary]


class RunDiffRequest(BaseModel):
    """Request body for POST /experiments/diff."""
    run_id_a: str
    run_id_b: str


class RunDiffRow(BaseModel):
    """One row of a run-vs-run diff: metric/param key + value pair."""
    key: str
    kind: str  # "metric" | "param"
    value_a: str | None
    value_b: str | None
    differs: bool


class RunDiffResponse(BaseModel):
    """Response for POST /experiments/diff: side-by-side metric/param diff."""
    rows: list[RunDiffRow]


# --- Agent surface (orchestrator + RAG) ------------------------------------

class AgentRunRequest(BaseModel):
    """User input to the orchestrator."""
    user_input: str = Field(..., min_length=1, description="SMILES, file path, or directory path")
    user_question: str | None = Field(
        None, description="Optional natural-language question to language-match the response"
    )
    sites_csv: str | None = Field(
        None,
        description="Optional MRI sites CSV. Defaults to <user_input>/sites.csv for directory inputs.",
    )


class AgentToolTraceItem(BaseModel):
    name: str
    args: dict = Field(default_factory=dict)
    result: dict | None = None
    error: str | None = None


class AgentRunResponse(BaseModel):
    text: str
    trace: list[AgentToolTraceItem] = Field(default_factory=list)
    model: str | None = None
    finish_reason: str = "complete"


# --- Fusion engine surface --------------------------------------------------

# Re-export the fusion types so the API surface lives in one file but the
# implementation stays in src/fusion. This keeps `from src.api.schemas import *`
# style imports stable for the frontend layer.
from src.fusion.types import (  # noqa: E402,F401
    ClinicalScores as FusionClinicalScores,
    FusionInput as FusionRequest,
    FusionOutput as FusionResponse,
    ModalityPrediction as FusionModalityPrediction,
)
