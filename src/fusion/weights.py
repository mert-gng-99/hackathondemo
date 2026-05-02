"""Disease × modality weight registry for the fusion engine.

Heuristic preset weights — tune offline as clinician feedback arrives.
"""
from __future__ import annotations

from typing import Mapping

DEFAULT_WEIGHTS: dict[str, dict[str, float]] = {
    "alzheimers": {
        "mri":             0.35,
        "eeg":             0.20,
        "clinical_mmse":   0.20,
        "clinical_moca":   0.15,
        "clinical_age":    0.10,
    },
    "parkinsons": {
        "mri":             0.20,
        "eeg":             0.30,
        "clinical_updrs":  0.30,
        "clinical_gait":   0.15,
        "clinical_age":    0.05,
    },
    "other": {
        "mri": 0.50,
        "eeg": 0.50,
    },
}


def available_diseases() -> list[str]:
    return sorted(DEFAULT_WEIGHTS.keys())


def available_clinical_tests() -> list[str]:
    """Return the bare clinical-test names (without the 'clinical_' prefix)."""
    names: set[str] = set()
    for table in DEFAULT_WEIGHTS.values():
        for key in table:
            if key.startswith("clinical_"):
                names.add(key[len("clinical_"):])
    return sorted(names)


def get_weights(disease: str) -> Mapping[str, float]:
    if disease not in DEFAULT_WEIGHTS:
        raise KeyError(f"unknown disease: {disease!r}")
    return DEFAULT_WEIGHTS[disease]
