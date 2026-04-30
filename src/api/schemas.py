"""Pydantic request / response models for the NeuroBridge FastAPI surface.

Each pipeline accepts its own request schema (BBBRequest / EEGRequest /
MRIRequest) but they all return a unified PipelineResponse — the dashboard
can render a single result card regardless of modality.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


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
