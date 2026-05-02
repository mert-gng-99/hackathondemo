"""Env-var-driven dispatch between volumetric ONNX and 2D resnet18 MRI models."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.core.logger import get_logger
from src.models import mri_dl_2d, mri_model

logger = get_logger(__name__)

VALID_KINDS = ("volumetric_onnx", "resnet18_2d")
_DEFAULT_KIND = "volumetric_onnx"


def current_kind() -> str:
    kind = os.environ.get("MRI_MODEL_KIND", _DEFAULT_KIND)
    if kind not in VALID_KINDS:
        raise ValueError(f"unknown MRI_MODEL_KIND={kind!r}; expected one of {VALID_KINDS}")
    return kind


def label_names_for_kind(kind: str) -> tuple[str, ...]:
    if kind == "resnet18_2d":
        return tuple(mri_dl_2d.IDX_TO_CLASS[i] for i in range(len(mri_dl_2d.CLASS_TO_IDX)))
    return mri_model.DEFAULT_LABEL_NAMES


def predict(
    input_path: Path,
    checkpoint_path: Path,
    target_shape: tuple[int, int, int] | None = None,
    label_names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Run the active MRI model on one input. Returns the unified prediction dict."""
    kind = current_kind()
    logger.info("dispatching MRI prediction kind=%s input=%s", kind, input_path)
    if kind == "resnet18_2d":
        model = mri_dl_2d.load(checkpoint_path)
        return mri_dl_2d.predict_image(model, input_path)
    model = mri_model.load(checkpoint_path)
    return mri_model.predict_nifti(
        model,
        input_path,
        target_shape=target_shape or mri_model.DEFAULT_TARGET_SHAPE,
        label_names=label_names,
    )
