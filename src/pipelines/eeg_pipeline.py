"""EEG (electroencephalography) pipeline.

Loads raw recordings (FIF/EDF), bandpass-filters, removes EOG artifacts via
ICA, slices into fixed-duration epochs, computes per-band PSD + statistical
features, flattens to a 2D table, and writes a model-ready Parquet at
`data/processed/eeg_features.parquet`.

Follows the Data Readiness contract in AGENTS.md §4 and the Parquet storage
convention in §6: schema validity, domain validity (drop NaN/inf epochs with
a logged WARNING), determinism (seeded ICA + sklearn RNG), traceability
(in/out/dropped counts at INFO), and idempotent overwrite output.
"""
from __future__ import annotations

import mne
import numpy as np

from src.core.logger import get_logger

logger = get_logger(__name__)


def is_valid_epoch(epoch: np.ndarray | None) -> bool:
    """Return True iff `epoch` is a non-empty 2-D numeric array with no NaN/inf.

    The annotation is the *expected* input class; the implementation defensively
    rejects any other garbage (lists, scalars, string dtypes, zero-sized arrays)
    without raising — matching the BBB pipeline's `is_valid_smiles` pattern.
    """
    if not isinstance(epoch, np.ndarray):
        return False
    if epoch.ndim != 2:
        return False
    if epoch.size == 0:
        return False
    if not np.issubdtype(epoch.dtype, np.number):
        return False
    if not np.all(np.isfinite(epoch)):
        return False
    return True


def bandpass_filter(
    raw: mne.io.BaseRaw,
    l_freq: float = 1.0,
    h_freq: float = 40.0,
) -> mne.io.BaseRaw:
    """Apply a non-mutating bandpass filter to an MNE Raw.

    Default 1-40 Hz removes drift below 1 Hz and high-frequency noise / line
    artifacts above 40 Hz. Returns a copy; the input `raw` is unchanged.

    Args:
        raw: Loaded `mne.io.BaseRaw` (call `.load_data()` first if from disk).
        l_freq: Low-cut frequency in Hz.
        h_freq: High-cut frequency in Hz.

    Returns:
        A filtered copy of `raw`.
    """
    out = raw.copy()
    out.filter(l_freq=l_freq, h_freq=h_freq, picks="all", verbose="ERROR")
    logger.info("Bandpass filter applied: %.1f-%.1f Hz", l_freq, h_freq)
    return out
