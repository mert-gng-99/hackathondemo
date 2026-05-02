"""Multi-modal fusion engine — combines MRI, EEG, and clinical signals into
per-disease confidence with full attribution.
"""
from __future__ import annotations

import math
from typing import Callable

from src.core.logger import get_logger
from src.fusion import clinical as clinical_signals
from src.fusion import weights as weight_registry
from src.fusion.modality import signal_for_disease
from src.fusion.types import (
    ClinicalScores,
    DiseaseScore,
    FusionInput,
    FusionOutput,
    ModalityContribution,
    ModalityPrediction,
)

logger = get_logger(__name__)

_LOGIT_SCALE = 4.0  # tuned so a single saturated modality maps to ~0.88


# Clinical-test name -> (signal_fn, attribute_on_ClinicalScores)
_CLINICAL_FNS: dict[str, tuple[Callable[[float], float], str]] = {
    "clinical_mmse":  (clinical_signals.mmse_to_signal,  "mmse"),
    "clinical_moca":  (clinical_signals.moca_to_signal,  "moca"),
    "clinical_updrs": (clinical_signals.updrs_to_signal, "updrs"),
    "clinical_gait":  (clinical_signals.gait_to_signal,  "gait_speed_m_s"),
    "clinical_age":   (clinical_signals.age_to_signal,   "age_years"),
}


def fuse(inp: FusionInput) -> FusionOutput:
    """Combine all available modalities into a per-disease confidence."""
    missing: list[str] = []
    if inp.mri is None:
        missing.append("mri")
    if inp.eeg is None:
        missing.append("eeg")

    diseases: list[DiseaseScore] = []
    for disease in weight_registry.available_diseases():
        diseases.append(_score_one_disease(disease, inp))

    top = max(diseases, key=lambda d: d.probability).disease
    return FusionOutput(diseases=diseases, top_disease=top, missing_inputs=missing)


def _score_one_disease(disease: str, inp: FusionInput) -> DiseaseScore:
    weights = weight_registry.get_weights(disease)
    contributions: list[ModalityContribution] = []

    for modality_key, weight in weights.items():
        signal = _signal_for_modality(modality_key, disease, inp.mri, inp.eeg, inp.clinical)
        if signal is None:
            continue
        contributions.append(ModalityContribution(
            modality=modality_key,
            weight=weight,
            signal=signal,
            delta_logit=weight * signal,
        ))

    if not contributions:
        logger.info("no signals available for disease=%s; returning baseline 0.5", disease)
        return DiseaseScore(disease=disease, probability=0.5, contributions=[])

    logit = sum(c.delta_logit for c in contributions)
    probability = _sigmoid(_LOGIT_SCALE * logit)
    return DiseaseScore(
        disease=disease,
        probability=probability,
        contributions=contributions,
    )


def _signal_for_modality(
    modality_key: str,
    disease: str,
    mri: ModalityPrediction | None,
    eeg: ModalityPrediction | None,
    clinical: ClinicalScores,
) -> float | None:
    if modality_key == "mri":
        return signal_for_disease(mri, disease) if mri is not None else None
    if modality_key == "eeg":
        return signal_for_disease(eeg, disease) if eeg is not None else None
    if modality_key in _CLINICAL_FNS:
        fn, attr = _CLINICAL_FNS[modality_key]
        value = getattr(clinical, attr, None)
        return fn(value) if value is not None else None
    logger.warning("unknown modality key in weights table: %s", modality_key)
    return None


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)
