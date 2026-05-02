"""BBB Permeability Map / Score from MRI input.

Two modes:
- `heuristic_proxy` (default, demo-ready): reuses the 2D resnet18 4-class
  Alzheimer's classifier output. Score = 1 - P(NonDemented). Anchored in the
  published correlation between disease severity and BBB breakdown.
- `dce_onnx` (real-DCE artifact, future): loads an ONNX model trained on
  4D DCE-MRI inputs that emit per-voxel Ktrans maps. Stub for now; works
  when the artifact drops in.

Researcher-persona module. Does NOT feed into the fusion engine — fusion's
'BBB is NOT a modality' rule is preserved. The only legitimate place where
BBB and MRI couple in this platform.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import numpy as np

from src.core.logger import get_logger

logger = get_logger(__name__)

PermeabilityMode = Literal["heuristic_proxy", "dce_onnx"]
DEFAULT_MODE: PermeabilityMode = "heuristic_proxy"


def _interpret(score: float) -> str:
    if score < 0.2:
        return "BBB intact"
    if score < 0.4:
        return "mild leakage"
    if score < 0.7:
        return "moderate leakage"
    return "severe leakage"


def compute_from_classifier_probs(probabilities: list[dict[str, Any]]) -> float:
    """Heuristic: 1 - P(NonDemented) from a 4-class probability list.

    Accepts the standard prediction-dict shape used across this repo
    (`probabilities=[{"label_text", "probability"}, ...]`).
    """
    p_nondemented = 0.0
    for entry in probabilities:
        if str(entry.get("label_text", "")).lower() == "nondemented":
            p_nondemented = float(entry.get("probability", 0.0))
            break
    score = max(0.0, min(1.0, 1.0 - p_nondemented))
    return float(score)


def heuristic_proxy_score(input_path: Path, checkpoint_path: Path) -> dict[str, Any]:
    """Run the 2D resnet18 classifier and derive the permeability score from
    its NonDemented probability.
    """
    from src.models import mri_dl_2d

    model = mri_dl_2d.load(checkpoint_path)
    pred = mri_dl_2d.predict_image(model, input_path)
    score = compute_from_classifier_probs(pred["probabilities"])
    return {
        "permeability_score": score,
        "interpretation": _interpret(score),
        "method": "heuristic_proxy",
        "voxel_map_available": False,
        "source_class_probabilities": pred["probabilities"],
    }


def dce_onnx_score(input_path: Path, checkpoint_path: Path) -> dict[str, Any]:
    """Load a DCE-MRI ONNX model and compute per-voxel Ktrans → scalar score.

    Contract for the future artifact:
    - Input: 4D NIfTI `(X, Y, Z, T)` with at least one timepoint.
    - Output: 3D Ktrans map (mL/min/100g). We normalise to `[0, 1]` by
      dividing by a clinically-conservative cap (`_DCE_KTRANS_CAP_MAX`)
      and clipping. Mean over the brain mask becomes the scalar score.

    No real artifact ships with the repo — this raises a clear error
    explaining the contract until one lands at `data/processed/bbb_permeability_dce.onnx`
    (override via `BBB_PERMEABILITY_DCE_PATH`).
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"DCE-MRI BBB permeability artifact not found at {checkpoint_path}. "
            "Train and export an ONNX model that consumes a 4D DCE volume "
            "(X, Y, Z, T) and emits a 3D Ktrans map; drop it at this path or "
            "set BBB_PERMEABILITY_DCE_PATH."
        )
    import nibabel as nib
    import onnxruntime as ort

    img = nib.load(str(input_path))
    arr = np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32)
    if arr.ndim != 4:
        raise ValueError(
            f"DCE-MRI mode expects a 4D NIfTI (X,Y,Z,T); got shape {arr.shape}."
        )
    session = ort.InferenceSession(str(checkpoint_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    ktrans = session.run(None, {input_name: arr[np.newaxis, ...]})[0]
    # Normalise to [0, 1]; clinically-conservative cap of 0.5 mL/min/100g.
    _DCE_KTRANS_CAP = 0.5
    normalised = np.clip(ktrans / _DCE_KTRANS_CAP, 0.0, 1.0).astype(np.float32)
    score = float(np.mean(normalised))
    return {
        "permeability_score": score,
        "interpretation": _interpret(score),
        "method": "dce_onnx",
        "voxel_map_available": True,
        "voxel_map_shape": list(normalised.shape),
    }


def compute_permeability(
    input_path: Path,
    mode: PermeabilityMode = DEFAULT_MODE,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    """Dispatch to the requested mode and return a unified payload."""
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"MRI input not found: {input_path}")

    if mode == "heuristic_proxy":
        ckpt = checkpoint_path or Path(os.environ.get(
            "MRI_MODEL_PATH_2D", "data/processed/mri_dl_2d/best_model.pt",
        ))
        return heuristic_proxy_score(input_path, ckpt)

    if mode == "dce_onnx":
        ckpt = checkpoint_path or Path(os.environ.get(
            "BBB_PERMEABILITY_DCE_PATH",
            "data/processed/bbb_permeability_dce.onnx",
        ))
        return dce_onnx_score(input_path, ckpt)

    raise ValueError(
        f"unknown BBB permeability mode={mode!r}; expected one of "
        "('heuristic_proxy', 'dce_onnx')"
    )
