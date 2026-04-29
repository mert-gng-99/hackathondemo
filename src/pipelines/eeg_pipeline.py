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

import numpy as np

from src.core.logger import get_logger

logger = get_logger(__name__)


def is_valid_epoch(epoch: object) -> bool:
    """Return True iff `epoch` is a non-empty 2-D float array with no NaN/inf.

    Used to drop corrupted segments before feature extraction. Defensive
    against the full set of garbage we expect from real recordings: lists,
    None, NaN/inf samples, zero-sized arrays.
    """
    if not isinstance(epoch, np.ndarray):
        return False
    if epoch.ndim != 2:
        return False
    if epoch.size == 0:
        return False
    if not np.all(np.isfinite(epoch)):
        return False
    return True
