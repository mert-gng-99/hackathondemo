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
import pandas as pd
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
      2. Morphological opening (`scipy.ndimage.binary_opening`, 6-connectivity,
         iterations=1) to remove isolated noise voxels and disconnected
         fragments. Note: thin features (< 3 voxels wide along any axis pair)
         may be eroded entirely; for production data with cortical sheets or
         sulcal bridges, prefer 26-connectivity or pass `iterations=0` upstream.

    If the resulting mask is all-False (e.g. caller passed a threshold above
    the volume's max intensity, or the volume is constant-valued), a WARNING
    is emitted so silent feature-zeroing is visible in production logs.

    Args:
        volume: 3-D numeric `np.ndarray` (must satisfy `is_valid_volume`).
        intensity_threshold: Voxel-intensity floor. `None` → use `volume.mean()`.

    Returns:
        A boolean `np.ndarray` of the same shape as `volume`. True = brain.
    """
    if intensity_threshold is None:
        intensity_threshold = float(volume.mean())

    raw = volume > intensity_threshold
    cleaned = scipy_ndimage.binary_opening(raw, iterations=1).astype(bool)
    if not cleaned.any():
        logger.warning(
            "mask_brain produced an all-False mask "
            "(volume min=%.4f, max=%.4f, threshold=%.4f); "
            "downstream features for this volume will be all-zero.",
            float(volume.min()), float(volume.max()), intensity_threshold,
        )
    return cleaned


# Default ROI partition: split a (D, H, W) volume into 2×2×2 = 8 octant ROIs.
# Octant index follows binary (z, y, x) ordering: 0..7.
DEFAULT_N_ROI_AXES: tuple[int, int, int] = (2, 2, 2)


def _roi_slices(
    shape: tuple[int, int, int],
    n_roi_axes: tuple[int, int, int],
) -> list[tuple[slice, slice, slice]]:
    """Generate the ROI slice list in deterministic (z, y, x) octant order."""
    nz, ny, nx = n_roi_axes
    dz, dy, dx = shape
    bins_z = np.array_split(np.arange(dz), nz)
    bins_y = np.array_split(np.arange(dy), ny)
    bins_x = np.array_split(np.arange(dx), nx)
    out: list[tuple[slice, slice, slice]] = []
    for bz in bins_z:
        for by in bins_y:
            for bx in bins_x:
                out.append((
                    slice(bz[0], bz[-1] + 1),
                    slice(by[0], by[-1] + 1),
                    slice(bx[0], bx[-1] + 1),
                ))
    return out


# Statistical functions, bound to their column-label names. The `ROI_STATS`
# tuple below is derived from this list so labels and computations cannot
# drift out of sync (a class of bug the prior parallel-list design was
# vulnerable to — same pattern as EEG's _STATS_FUNCS).
#
# `mean`/`std` use NumPy with `ddof=0` (biased / population estimators).
# `p10`/`p50`/`p90` use `np.percentile` default linear interpolation.
# `voxel_count` is stored as float for column-uniformity in the eventual
# Parquet, but always represents a whole number (assertable via
# `v == float(int(v))`).
_ROI_STATS_FUNCS: tuple[tuple[str, "object"], ...] = (
    ("mean", lambda v: float(v.mean())),
    ("std", lambda v: float(v.std())),
    ("p10", lambda v: float(np.percentile(v, 10))),
    ("p50", lambda v: float(np.percentile(v, 50))),
    ("p90", lambda v: float(np.percentile(v, 90))),
    ("voxel_count", lambda v: float(v.size)),
)
ROI_STATS: tuple[str, ...] = tuple(name for name, _ in _ROI_STATS_FUNCS)


def _roi_stats_for(values: np.ndarray) -> dict[str, float]:
    """Compute the ROI stats. Empty array → all 0.0 (no-NaN contract)."""
    if values.size == 0:
        return {name: 0.0 for name, _ in _ROI_STATS_FUNCS}
    return {name: fn(values) for name, fn in _ROI_STATS_FUNCS}


def extract_features_from_volume(
    volume: np.ndarray,
    mask: np.ndarray,
    n_roi_axes: tuple[int, int, int] = DEFAULT_N_ROI_AXES,
) -> dict[str, float]:
    """Compute per-ROI summary statistics from a masked volume.

    The volume is partitioned into ``prod(n_roi_axes)`` axis-aligned octants
    in deterministic (z, y, x) order. For each ROI, intensity values from
    voxels where `mask` is True are summarized via mean / std / 10th, 50th,
    90th percentile / voxel count. Empty ROIs (no mask voxels) report all
    zeros so the resulting Parquet has no NaN values.

    Statistical conventions:
      - ``mean`` / ``std`` use ``ddof=0`` (biased / population estimators).
      - ``p10`` / ``p50`` / ``p90`` use ``np.percentile`` with the default
        linear interpolation.
      - ``voxel_count`` is stored as float for column uniformity but always
        represents a whole number.

    Args:
        volume: 3-D numeric `np.ndarray` (already validated).
        mask: Boolean `np.ndarray` of the same shape (from `mask_brain`).
        n_roi_axes: ROI grid along (z, y, x). Default `(2, 2, 2)` → 8 ROIs.

    Returns:
        Flat dict `{"feat_roi{i}_{stat}": float}` of length
        ``prod(n_roi_axes) * len(ROI_STATS)``.

    Raises:
        ValueError: if `volume.shape` and `mask.shape` differ.
    """
    if volume.shape != mask.shape:
        raise ValueError(
            f"volume.shape {volume.shape} != mask.shape {mask.shape}"
        )

    feats: dict[str, float] = {}
    slices = _roi_slices(volume.shape, n_roi_axes)
    for i, sl in enumerate(slices):
        roi_values = volume[sl][mask[sl]]
        stats = _roi_stats_for(roi_values)
        for stat_name, stat_val in stats.items():
            feats[f"feat_roi{i}_{stat_name}"] = stat_val
    return feats


def harmonize_combat(
    features: pd.DataFrame,
    sites: pd.Series,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Apply ComBat harmonization across sites to remove site-level domain shift.

    Wraps `neuroHarmonize.harmonizationLearn` which fits a parametric ComBat
    model (no internal RNG → byte-deterministic given fixed input). Only
    `feature_cols` are harmonized; other columns in `features` (e.g.
    metadata) are not touched by this function — callers should join after.

    Args:
        features: DataFrame with at least the columns listed in `feature_cols`.
        sites: Site label per row (length must match `len(features)`).
        feature_cols: Names of the columns to harmonize.

    Returns:
        A new DataFrame of identical shape & column order to
        `features[feature_cols]`, with ComBat-harmonized values.

    Raises:
        ValueError: if fewer than 2 distinct sites are present.
    """
    from neuroHarmonize import harmonizationLearn

    if not feature_cols:
        raise ValueError("feature_cols must be a non-empty list")
    if len(features) != len(sites):
        raise ValueError(
            f"features has {len(features)} rows but sites has {len(sites)} elements"
        )

    if sites.nunique() < 2:
        raise ValueError(
            f"ComBat requires at least 2 sites; got {sites.nunique()} "
            f"({sites.unique().tolist()})"
        )

    matrix = features[feature_cols].to_numpy(dtype=np.float64)
    covars = pd.DataFrame({"SITE": sites.to_numpy()})

    _, harmonized = harmonizationLearn(matrix, covars)
    # Defensive: with OMP/OPENBLAS/MKL_NUM_THREADS=1 (set at module import,
    # per AGENTS.md §4), harmonizationLearn is already bit-identical across
    # calls. np.round(14) provides an additional determinism boundary for
    # environments where those env pins are overridden before module load
    # (e.g. a sub-process that re-exports a thread count). It discards ~5
    # trailing-mantissa bits, which is well below ComBat's biological
    # effect-size precision floor.
    out = pd.DataFrame(
        np.round(np.asarray(harmonized, dtype=np.float64), 14),
        columns=list(feature_cols),
        index=features.index,
    )
    logger.info(
        "ComBat harmonized %d rows × %d features across %d sites",
        len(out), len(feature_cols), sites.nunique(),
    )
    return out
