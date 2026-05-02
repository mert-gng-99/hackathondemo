"""Convert a modality classifier's probability vector into a signed signal."""
from __future__ import annotations

from src.fusion.types import ModalityPrediction


def signal_for_disease(pred: ModalityPrediction, disease: str) -> float | None:
    """Return signal in [-1, 1] for `disease`, or None if the model has no
    matching class.

    A class matches if its `label_text` equals `disease` case-insensitively.
    Signal = 2 * P(disease) - 1.
    """
    target = disease.strip().lower()
    for cls in pred.probabilities:
        if cls.label_text.strip().lower() == target:
            return 2.0 * cls.probability - 1.0
    return None
