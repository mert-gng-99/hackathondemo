"""Pydantic input/output schemas for orchestrator tools and the agent result.

These schemas double as OpenAI function-calling parameter definitions
(via `model_json_schema()`) and as runtime validation gates. Keep field
names lowercase + snake_case so prompts and JSON outputs align.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# --- Pipeline tool inputs ---------------------------------------------------

class BBBPipelineInput(BaseModel):
    """Input for `run_bbb_pipeline` — a single SMILES string."""
    smiles: str = Field(..., description="A single molecular SMILES string, e.g. 'CCO'")
    top_k: int = Field(5, ge=1, le=20, description="Top-k SHAP attributions to return")


class EEGPipelineInput(BaseModel):
    """Input for `run_eeg_pipeline` — path to an EEG file (.fif or .edf)."""
    input_path: str = Field(..., description="Path to EEG recording file (.fif or .edf)")
    epoch_duration_s: float = Field(2.0, gt=0.1, le=60.0)


class MRIPipelineInput(BaseModel):
    """Input for `run_mri_pipeline` — directory of NIfTI files + sites CSV."""
    input_dir: str = Field(..., description="Directory containing .nii.gz volumes")
    sites_csv: str | None = Field(
        None,
        description="CSV mapping subject_id → site; defaults to <input_dir>/sites.csv",
    )


class BBBPermeabilityMapInput(BaseModel):
    """Input for `compute_bbb_leakage_score` — MRI input + scoring mode."""
    input_path: str = Field(..., description="Path to MRI input (2D image for heuristic_proxy; 4D NIfTI for dce_onnx).")
    mode: Literal["heuristic_proxy", "dce_onnx"] = Field(
        "heuristic_proxy",
        description="'heuristic_proxy' (default) | 'dce_onnx' (real DCE artifact)",
    )


class BBBPermeabilityMapOutput(BaseModel):
    permeability_score: float
    interpretation: str
    method: str
    voxel_map_available: bool


class DrugDoseAdjustmentInput(BaseModel):
    """Input for `adjust_drug_dose` — baseline + patient + drug profile."""
    baseline_dose_mg: float = Field(..., gt=0.0)
    bbb_permeability_score: float = Field(..., ge=0.0, le=1.0)
    drug_bbb_permeable: bool | None = None
    smiles: str | None = Field(
        None, description="Optional SMILES; auto-resolves drug_bbb_permeable when given.",
    )


class DrugDoseAdjustmentOutput(BaseModel):
    recommended_dose_mg: float
    adjustment_factor: float
    risk_level: str
    rationale: str
    drug_bbb_permeable: bool | None = None


class RetrieveContextInput(BaseModel):
    """Input for `retrieve_context` — natural-language query into the KB."""
    query: str = Field(..., min_length=2, description="Search query for the knowledge base")
    k: int = Field(4, ge=1, le=10, description="Number of chunks to return")
    corpus: Literal["reference", "clinical"] = Field(
        "reference",
        description=(
            "Which corpus to query. 'reference' = curated FAISS index (default). "
            "'clinical' = TF-IDF index over peer-reviewed Alzheimer's/Parkinson's "
            "papers with Turkish+English query expansion."
        ),
    )


# --- Pipeline tool outputs --------------------------------------------------

class BBBPipelineOutput(BaseModel):
    smiles: str
    label: int
    label_text: str
    confidence: float
    top_features: list[dict[str, Any]]
    drift_z: float | None = None


class EEGPipelineOutput(BaseModel):
    input_path: str
    output_path: str
    rows: int
    columns: int
    duration_sec: float


class MRIPipelineOutput(BaseModel):
    input_dir: str
    output_path: str
    rows: int
    columns: int
    duration_sec: float


class RetrieveContextOutput(BaseModel):
    query: str
    chunks: list[dict[str, Any]]


# --- Agent result -----------------------------------------------------------

class ToolTraceItem(BaseModel):
    """One step in the orchestrator's tool-call trace."""
    name: str
    args: dict[str, Any]
    result: dict[str, Any] | None = None
    error: str | None = None


class AgentResult(BaseModel):
    """Final orchestrator response: synthesized text + full trace."""
    text: str
    trace: list[ToolTraceItem] = Field(default_factory=list)
    model: str | None = None
    finish_reason: str = "complete"  # complete | max_steps | error
