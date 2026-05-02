"""EEG classifier inference utilities.

Loads any sklearn-style classifier (object with `predict_proba`) from joblib
and emits the same dict shape as src.models.mri_model.predict_with_proba so
the API surface and fusion engine treat MRI and EEG predictions identically.

The real pretrained artifact swaps in at data/processed/eeg_clf.joblib (or
override via EEG_CLF_ARTIFACT env). Tests use a stub fixture; the real model
drops in without code changes.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np

from src.core.logger import get_logger

logger = get_logger(__name__)

DEFAULT_LABELS: tuple[str, ...] = ("control", "alzheimers")


def _resolve_labels() -> tuple[str, ...]:
    raw = os.environ.get("EEG_CLF_LABELS")
    if not raw:
        return DEFAULT_LABELS
    return tuple(s.strip() for s in raw.split(",") if s.strip())


def load(path: Path) -> Any:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"EEG classifier artifact not found: {path}")
    return joblib.load(str(path))


def predict_features(
    model: Any,
    features: np.ndarray,
    labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run inference on one row of EEG features."""
    arr = np.asarray(features, dtype=np.float32).reshape(-1)
    expected = int(getattr(model, "n_features_in_", arr.size))
    if arr.size != expected:
        raise ValueError(
            f"EEG feature count mismatch: model expects {expected}, got {arr.size}"
        )

    proba = np.asarray(model.predict_proba(arr.reshape(1, -1))[0], dtype=np.float32)
    label_names = tuple(labels or _resolve_labels())
    if len(label_names) != proba.shape[0]:
        logger.warning(
            "EEG label count (%d) != model output dim (%d); falling back to class_0..N",
            len(label_names), proba.shape[0],
        )
        label_names = tuple(f"class_{i}" for i in range(proba.shape[0]))

    label_idx = int(np.argmax(proba))
    return {
        "label": label_idx,
        "label_text": label_names[label_idx],
        "confidence": float(proba[label_idx]),
        "probabilities": [
            {"label": i, "label_text": label_names[i], "probability": float(p)}
            for i, p in enumerate(proba)
        ],
    }
