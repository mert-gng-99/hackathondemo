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

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage as scipy_ndimage

from src.core.determinism import pin_threads
from src.core.logger import get_logger
from src.core.storage import write_parquet

logger = get_logger(__name__)

# Pin BLAS / OpenMP / pyarrow to single-threaded mode so byte-determinism
# (AGENTS.md §4 rule 3) holds across hardware. See src.core.determinism.
pin_threads()


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


# Default I/O paths for the MRI pipeline. Override via run_pipeline() args.
DEFAULT_INPUT = Path("data/raw/mri")
DEFAULT_OUTPUT = Path("data/processed/mri_features.parquet")


# Variance floor used to decide whether a feature column is "constant" for
# ComBat. Strict ``std() > 0`` would still send near-zero-variance columns
# (e.g. ULP-level differences) into ComBat, where var_pooled ≈ 0 produces
# NaN. 1e-8 is well above machine epsilon and far below any biologically
# meaningful signal variance.
_MIN_VAR_THRESHOLD: float = 1e-8


def _list_nifti_volumes(input_dir: Path) -> list[Path]:
    """Return sorted list of .nii / .nii.gz files in `input_dir`."""
    return sorted(
        p for p in input_dir.iterdir()
        if p.suffix == ".nii" or p.name.endswith(".nii.gz")
    )


def run_pipeline(
    input_dir: Path = DEFAULT_INPUT,
    sites_csv: Path | None = None,
    output_path: Path = DEFAULT_OUTPUT,
    intensity_threshold: float | None = None,
    n_roi_axes: tuple[int, int, int] = DEFAULT_N_ROI_AXES,
) -> None:
    """Run the MRI pipeline end-to-end: NIfTI directory → harmonized Parquet.

    For each `subject_id.nii(.gz)` in `input_dir`, validates the volume,
    masks the brain, computes per-ROI statistics, then harmonizes across
    sites (column "site" of `sites_csv`, joined on "subject_id") via ComBat.
    Output is float64 Parquet at `output_path`.

    Args:
        input_dir: Directory containing one NIfTI per subject and a
            `sites.csv` (or `sites_csv` override) with columns
            `subject_id, site`.
        sites_csv: Path to the site-covariates CSV. If `None`, defaults to
            `input_dir / "sites.csv"`.
        output_path: Where to write the processed feature Parquet file.
        intensity_threshold: Brain-mask intensity floor. `None` → per-volume
            mean (see `mask_brain`).
        n_roi_axes: ROI grid (z, y, x).

    Raises:
        FileNotFoundError: if `input_dir` does not exist.
        IsADirectoryError: if `output_path` resolves to an existing directory.
        KeyError: if `sites_csv` is missing a site for some subject.
    """
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    if not input_dir.exists():
        raise FileNotFoundError(f"MRI input directory not found: {input_dir}")
    sites_csv = Path(sites_csv) if sites_csv is not None else input_dir / "sites.csv"
    if not sites_csv.exists():
        raise FileNotFoundError(f"sites_csv not found: {sites_csv}")

    logger.info("Reading MRI volumes from %s", input_dir)
    nifti_paths = _list_nifti_volumes(input_dir)
    sites_df = pd.read_csv(sites_csv)

    rows: list[dict[str, float | str]] = []
    invalid_subject_ids: list[str] = []
    for path in nifti_paths:
        subject_id = path.name.removesuffix(".nii.gz").removesuffix(".nii")
        volume = nib.load(path).get_fdata()
        if not is_valid_volume(volume):
            invalid_subject_ids.append(subject_id)
            continue
        mask = mask_brain(volume, intensity_threshold=intensity_threshold)
        feats = extract_features_from_volume(volume, mask, n_roi_axes=n_roi_axes)
        rows.append({"subject_id": subject_id, **feats})

    n_total = len(nifti_paths)
    n_dropped = len(invalid_subject_ids)
    if n_dropped:
        display = invalid_subject_ids[:10]
        suffix = (
            f"... (+{n_dropped - 10} more)" if n_dropped > 10 else ""
        )
        logger.warning(
            "Dropping %d/%d volumes with invalid samples (subjects=%s%s)",
            n_dropped, n_total, display, suffix,
        )

    feature_cols = [
        f"feat_roi{i}_{stat}"
        for i in range(int(np.prod(n_roi_axes)))
        for stat in ROI_STATS
    ]

    if not rows:
        logger.info(
            "Feature extraction complete: in=%d, out=0, dropped=%d (%.2f%%)",
            n_total, n_dropped, 100.0 * n_dropped / max(n_total, 1),
        )
        empty = pd.DataFrame(
            columns=["subject_id", "site", *feature_cols]
        ).astype({c: np.float64 for c in feature_cols})
        write_parquet(empty, output_path)
        return

    raw_features = pd.DataFrame(rows)
    raw_features = raw_features.merge(sites_df, on="subject_id", how="left")
    if raw_features["site"].isna().any():
        missing = raw_features.loc[raw_features["site"].isna(), "subject_id"].tolist()
        raise KeyError(
            f"sites_csv missing site assignment for subjects: {missing}"
        )

    # ComBat cannot handle (near-)zero-variance columns: var_pooled ≈ 0 produces
    # NaN. Split feature_cols on a strictly-positive variance floor so ULP-level
    # noise is treated as constant.
    col_std = raw_features[feature_cols].std()
    var_feature_cols = [c for c in feature_cols if col_std[c] > _MIN_VAR_THRESHOLD]
    zero_var_cols = [c for c in feature_cols if col_std[c] <= _MIN_VAR_THRESHOLD]

    if not var_feature_cols:
        # Degenerate dataset: every feature is essentially constant. ComBat has
        # no signal to harmonize on; pass all columns through and warn.
        logger.warning(
            "All %d feature columns have variance ≤ %.1e; ComBat skipped "
            "(output contains unharmonized features).",
            len(feature_cols), _MIN_VAR_THRESHOLD,
        )
        harmonized = raw_features[feature_cols].copy()
    else:
        harmonized = harmonize_combat(
            raw_features, raw_features["site"], var_feature_cols,
        )
        # Re-attach zero-variance columns (unchanged) and restore the original
        # column order.
        for c in zero_var_cols:
            harmonized[c] = raw_features[c].to_numpy()
        harmonized = harmonized[feature_cols]

    final = pd.concat(
        [raw_features[["subject_id", "site"]].reset_index(drop=True),
         harmonized.reset_index(drop=True)],
        axis=1,
    )

    logger.info(
        "Feature extraction complete: in=%d, out=%d, dropped=%d (%.2f%%)",
        n_total, len(final), n_dropped, 100.0 * n_dropped / max(n_total, 1),
    )

    # Parquet preserves dtypes (float64 features stay float64) and is
    # byte-deterministic with single-threaded snappy. AGENTS.md §6.
    write_parquet(final, output_path)
    logger.info(
        "Wrote processed features to %s (rows=%d, cols=%d)",
        output_path, len(final), final.shape[1],
    )


if __name__ == "__main__":
    # Day-3 CLI entrypoint — runs with default paths against `data/raw/mri/`.
    # Expects `data/raw/mri/sites.csv` with columns `subject_id, site`.
    # Argument parsing (argparse / click) will land in a later task.
    #   python -m src.pipelines.mri_pipeline
    run_pipeline()
