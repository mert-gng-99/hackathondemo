"""MRI (magnetic resonance imaging) pipeline.

Loads NIfTI volumes (`.nii` / `.nii.gz`), applies a brain mask, harmonizes
across sites with ComBat (`neuroHarmonize`), and writes per-subject ROI
statistics as a model-ready Parquet at `data/processed/mri_features.parquet`.

Follows the Data Readiness contract in AGENTS.md §4 and the Parquet storage
convention in §6: schema validity, domain validity (drop NaN/inf volumes
with a logged WARNING), determinism (ComBat is RNG-free given fixed input),
traceability (in/out/dropped counts at INFO), and idempotent overwrite.
"""
from __future__ import annotations

import os

import nibabel as nib
import numpy as np
import pyarrow as pa
from scipy import ndimage as scipy_ndimage

from src.core.logger import get_logger

logger = get_logger(__name__)

# Pin BLAS / OpenMP / pyarrow to single-threaded mode so byte-determinism
# (AGENTS.md §4 rule 3) holds across hardware. Without this, multi-threaded
# floating-point reductions can reorder and produce non-bit-identical output.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
pa.set_cpu_count(1)
pa.set_io_thread_count(1)


def is_valid_volume(volume: np.ndarray | None) -> bool:
    """Return True iff `volume` is a non-empty 3-D numeric array with no NaN/inf.

    Used to drop corrupted volumes before masking + feature extraction.
    Defensive against the full set of garbage we expect from real archives:
    lists, None, NaN/inf samples, zero-sized arrays, string-dtype arrays.
    """
    if not isinstance(volume, np.ndarray):
        return False
    if volume.ndim != 3:
        return False
    if volume.size == 0:
        return False
    if not np.issubdtype(volume.dtype, np.number):
        return False
    if not np.all(np.isfinite(volume)):
        return False
    return True


def mask_brain(
    volume: np.ndarray,
    intensity_threshold: float | None = None,
) -> np.ndarray:
    """Build a brain mask from a 3-D MRI volume.

    Two-step pipeline:
      1. Intensity threshold: keep voxels above `intensity_threshold`. When
         `None`, use the volume's mean as a robust auto-threshold (works on
         the synthetic fixture where brain ≫ background; for real data the
         caller should pass an Otsu or BET-derived threshold explicitly).
      2. Morphological opening (`scipy.ndimage.binary_opening`) to remove
         isolated noise voxels and disconnected fragments.

    Args:
        volume: 3-D numeric `np.ndarray` (must satisfy `is_valid_volume`).
        intensity_threshold: Voxel-intensity floor. `None` → use `volume.mean()`.

    Returns:
        A boolean `np.ndarray` of the same shape as `volume`. True = brain.
    """
    if intensity_threshold is None:
        intensity_threshold = float(volume.mean())

    raw = volume > intensity_threshold
    cleaned = scipy_ndimage.binary_opening(raw, iterations=1)
    return cleaned.astype(bool)
