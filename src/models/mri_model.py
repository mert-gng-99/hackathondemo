"""MRI image deep-learning inference utilities.

This module is the decision-layer bridge for an externally-trained volumetric
MRI model. The training code can live outside this repo; production only needs
an ONNX artifact plus the preprocessing contract below.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import nibabel as nib
import numpy as np
from scipy import ndimage as scipy_ndimage

from src.core.logger import get_logger
from src.pipelines.mri_pipeline import is_valid_volume

logger = get_logger(__name__)

DEFAULT_MODEL_PATH = Path("data/processed/mri_model.onnx")
DEFAULT_TARGET_SHAPE: tuple[int, int, int] = (64, 64, 64)
DEFAULT_LABEL_NAMES: tuple[str, ...] = ("class_0", "class_1")
_MIN_STD = 1e-6


def load(path: Path) -> Any:
    """Load an ONNX MRI model artifact.

    Args:
        path: Path to an externally-trained `.onnx` artifact.

    Returns:
        An `onnxruntime.InferenceSession`.

    Raises:
        FileNotFoundError: if the artifact does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MRI model artifact not found: {path}")
    import onnxruntime as ort

    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def load_nifti_volume(path: Path) -> np.ndarray:
    """Read a NIfTI volume from disk as float32."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MRI input not found: {path}")
    img = nib.load(str(path))
    return np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32)


def preprocess_volume(
    volume: np.ndarray,
    target_shape: tuple[int, int, int] = DEFAULT_TARGET_SHAPE,
) -> np.ndarray:
    """Convert a 3-D MRI volume into model input `[1, 1, D, H, W]`.

    The external trainer must use the same contract: trilinear resize to
    `target_shape`, z-score over non-zero voxels when present, then add batch
    and channel dimensions.
    """
    if not is_valid_volume(volume):
        raise ValueError("MRI volume must be a finite numeric 3-D array")
    if len(target_shape) != 3 or any(int(x) <= 0 for x in target_shape):
        raise ValueError(f"target_shape must contain three positive integers: {target_shape}")

    resized = _resize_volume(np.asarray(volume, dtype=np.float32), target_shape)
    normalized = _zscore_volume(resized)
    return normalized[np.newaxis, np.newaxis, :, :, :].astype(np.float32, copy=False)


def preprocess_nifti(
    input_path: Path,
    target_shape: tuple[int, int, int] = DEFAULT_TARGET_SHAPE,
) -> np.ndarray:
    """Read and preprocess one NIfTI file for ONNX inference."""
    return preprocess_volume(load_nifti_volume(input_path), target_shape=target_shape)


def predict_with_proba(
    model: Any,
    model_input: np.ndarray,
    label_names: Sequence[str] | None = None,
) -> dict[str, object]:
    """Run an ONNX model and return label, confidence, and per-class probabilities."""
    labels = tuple(label_names or DEFAULT_LABEL_NAMES)
    if model_input.ndim != 5:
        raise ValueError(f"model_input must have shape [1, 1, D, H, W], got {model_input.shape}")

    input_name = model.get_inputs()[0].name
    output = model.run(None, {input_name: model_input.astype(np.float32, copy=False)})[0]
    proba = _as_probabilities(np.asarray(output, dtype=np.float32))
    if len(labels) != proba.shape[0]:
        labels = tuple(f"class_{i}" for i in range(proba.shape[0]))

    label_idx = int(np.argmax(proba))
    return {
        "label": label_idx,
        "label_text": labels[label_idx],
        "confidence": float(proba[label_idx]),
        "probabilities": [
            {"label": i, "label_text": labels[i], "probability": float(p)}
            for i, p in enumerate(proba)
        ],
    }


def predict_nifti(
    model: Any,
    input_path: Path,
    target_shape: tuple[int, int, int] = DEFAULT_TARGET_SHAPE,
    label_names: Sequence[str] | None = None,
) -> dict[str, object]:
    """Preprocess one NIfTI image and run MRI model inference."""
    model_input = preprocess_nifti(input_path, target_shape=target_shape)
    return predict_with_proba(model, model_input, label_names=label_names)


def _resize_volume(volume: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    zoom = tuple(t / s for t, s in zip(target_shape, volume.shape, strict=True))
    return scipy_ndimage.zoom(volume, zoom=zoom, order=1).astype(np.float32, copy=False)


def _zscore_volume(volume: np.ndarray) -> np.ndarray:
    mask = volume != 0
    ref = volume[mask] if np.any(mask) else volume.reshape(-1)
    mean = float(ref.mean())
    std = float(ref.std())
    if std < _MIN_STD:
        return np.zeros_like(volume, dtype=np.float32)
    return ((volume - mean) / std).astype(np.float32, copy=False)


def _as_probabilities(raw_output: np.ndarray) -> np.ndarray:
    logits = np.squeeze(raw_output)
    if logits.ndim != 1:
        raise ValueError(f"MRI model output must be one class vector, got shape {raw_output.shape}")
    if logits.size < 2:
        raise ValueError("MRI model output must contain at least two class scores")

    if np.all(logits >= 0.0) and np.all(logits <= 1.0) and np.isclose(logits.sum(), 1.0, atol=1e-4):
        return logits.astype(np.float32, copy=False)
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return (exp / exp.sum()).astype(np.float32, copy=False)
