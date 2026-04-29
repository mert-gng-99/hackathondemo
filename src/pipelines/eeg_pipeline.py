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
from mne.preprocessing import ICA

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
        l_freq: Low-cut frequency in Hz. Must be strictly less than `h_freq`.
        h_freq: High-cut frequency in Hz.

    Returns:
        A filtered copy of `raw`.

    Raises:
        ValueError: if `l_freq >= h_freq`. MNE silently produces a corrupted
            band-stop-like result on inverted inputs, so we guard up front.
    """
    if l_freq >= h_freq:
        raise ValueError(
            f"l_freq ({l_freq}) must be strictly less than h_freq ({h_freq})"
        )

    out = raw.copy()
    # picks="all" includes the EOG channel so the ICA step in
    # remove_artifacts_with_ica sees a consistently-filtered EOG reference.
    out.filter(l_freq=l_freq, h_freq=h_freq, picks="all", verbose="ERROR")
    logger.info("Bandpass filter applied: %.1f-%.1f Hz", l_freq, h_freq)
    return out


def remove_artifacts_with_ica(
    raw: mne.io.BaseRaw,
    eog_ch_name: str | None = None,
    n_components: int = 15,
    random_state: int = 97,
) -> mne.io.BaseRaw:
    """Remove EOG-like artifacts using MNE's ICA + EOG correlation.

    Fits an ICA decomposition on `raw`, finds components whose time courses
    correlate (Pearson) with the named EOG channel via `find_bads_eog` using
    `measure="correlation"`, marks them as "bad" and reconstructs the signal
    without them. Returns a copy; the input `raw` is unchanged.

    If `eog_ch_name` is None or no bad components are found, returns a copy of
    `raw` unchanged. This keeps the function safe to call on recordings
    without an EOG reference.

    Args:
        raw: Loaded, ideally bandpass-filtered, `mne.io.BaseRaw`.
        eog_ch_name: Name of the EOG channel for correlation-based detection.
            None disables auto-rejection.
        n_components: Cap on ICA components. For small recordings, MNE will
            silently cap this at the rank of the data.
        random_state: Seed for ICA's underlying solver. Required for §4
            Determinism.

    Returns:
        A copy of `raw` with EOG-correlated ICA components removed.
    """
    out = raw.copy()
    if eog_ch_name is None or eog_ch_name not in out.ch_names:
        logger.info("ICA skipped: no EOG channel reference provided")
        return out

    # Cap n_components at the rank of the data to avoid solver complaints
    # on small synthetic fixtures.
    n_eeg = len(mne.pick_types(out.info, eeg=True, meg=False))
    safe_n = min(n_components, max(n_eeg - 1, 1))

    ica = ICA(
        n_components=safe_n,
        random_state=random_state,
        max_iter="auto",
        method="fastica",
        verbose="ERROR",
    )
    ica.fit(out, picks="eeg", verbose="ERROR")
    # Use raw correlation (not z-score) so we can reliably flag artifact
    # components on small recordings where n_components < 10 makes the
    # default z-score threshold algebraically unreachable.
    bad_idx, _ = ica.find_bads_eog(
        out,
        ch_name=eog_ch_name,
        measure="correlation",
        threshold=0.9,
        verbose="ERROR",
    )
    ica.exclude = list(bad_idx)
    logger.info(
        "ICA fit: n_components=%d, EOG-correlated rejected=%d",
        safe_n, len(ica.exclude),
    )
    ica.apply(out, verbose="ERROR")
    return out
