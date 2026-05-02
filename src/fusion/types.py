"""Pydantic data contract for the multi-modal fusion engine."""
from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class ModalityClassProb(BaseModel):
    label_text: str
    probability: float = Field(..., ge=0.0, le=1.0)


class ModalityPrediction(BaseModel):
    """One modality's classifier output (MRI or EEG)."""
    model_config = ConfigDict(protected_namespaces=())

    label_text: str
    label: int = Field(..., ge=0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: list[ModalityClassProb] = Field(..., min_length=1)


class ClinicalScores(BaseModel):
    """Doctor-entered extra-test scores. Each is optional."""
    mmse: Annotated[float, Field(ge=0.0, le=30.0)] | None = None
    moca: Annotated[float, Field(ge=0.0, le=30.0)] | None = None
    updrs: Annotated[float, Field(ge=0.0, le=199.0)] | None = None
    gait_speed_m_s: Annotated[float, Field(ge=0.0, le=2.5)] | None = None
    age_years: Annotated[float, Field(ge=0.0, le=120.0)] | None = None


class FusionInput(BaseModel):
    mri: ModalityPrediction | None = None
    eeg: ModalityPrediction | None = None
    clinical: ClinicalScores = Field(default_factory=ClinicalScores)


class ModalityContribution(BaseModel):
    """One row of the attribution table for a single disease score."""
    modality: str  # "mri" | "eeg" | "clinical_<name>"
    weight: float
    signal: float = Field(..., ge=-1.0, le=1.0)
    delta_logit: float


class DiseaseScore(BaseModel):
    disease: str
    probability: float = Field(..., ge=0.0, le=1.0)
    contributions: list[ModalityContribution]


class FusionOutput(BaseModel):
    diseases: list[DiseaseScore]
    top_disease: str
    missing_inputs: list[str] = Field(default_factory=list)
