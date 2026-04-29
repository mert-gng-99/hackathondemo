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
