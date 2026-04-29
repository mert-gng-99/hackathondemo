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
import pandas as pd
from mne.preprocessing import ICA
from scipy import signal as scipy_signal
from scipy import stats as scipy_stats

from src.core.logger import get_logger

logger = get_logger(__name__)

# Pearson-correlation threshold for EOG-component rejection in ICA.
# Real-world EOG components typically score 0.8-0.95 against the EOG channel;
# 0.9 is a conservative floor that avoids false positives at the cost of
# missing weak artifacts. Lower (0.7-0.8) for noisier recordings.
_EOG_CORR_THRESHOLD: float = 0.9


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

    If `eog_ch_name` is None or not present in the recording's channels,
    ICA is skipped entirely and a copy of `raw` is returned unchanged.

    Args:
        raw: Loaded, ideally bandpass-filtered, `mne.io.BaseRaw`.
        eog_ch_name: Name of the EOG channel for correlation-based detection.
            None disables auto-rejection; a string that is not in the recording's
            channel list logs a WARNING and skips ICA.
        n_components: Cap on ICA components. If this exceeds the number of EEG
            channels, MNE raises ValueError, so the implementation internally
            caps it at `max(n_eeg - 1, 1)` before fitting.
        random_state: Seed for ICA's underlying solver. Required for §4
            Determinism.

    Returns:
        A copy of `raw` with EOG-correlated ICA components removed (or an
        unchanged copy if ICA was skipped).

    Raises:
        ValueError: if the EEG data is rank-deficient (all-zero or constant
            channels) and `mne.preprocessing.ICA.fit` cannot converge.
    """
    out = raw.copy()
    if eog_ch_name is None:
        logger.info("ICA skipped: eog_ch_name not provided")
        return out
    if eog_ch_name not in out.ch_names:
        logger.warning(
            "ICA skipped: eog_ch_name=%r not found in channels %s",
            eog_ch_name, out.ch_names,
        )
        return out

    # Guard: ICA.fit cannot handle NaN/inf in the data (scipy SVD will raise).
    # If the raw contains non-finite samples, skip ICA so the NaN propagates
    # to the epoch-level validity check in extract_features_from_recording
    # where it will be cleanly dropped with a WARNING.
    eeg_picks_check = mne.pick_types(out.info, eeg=True, meg=False)
    if not np.all(np.isfinite(out.get_data(picks=eeg_picks_check))):
        logger.warning(
            "ICA skipped: EEG data contains NaN/inf values; "
            "invalid epochs will be dropped downstream"
        )
        return out

    # Cap n_components at rank-1. Average reference (if applied) reduces rank
    # to n_eeg - 1; using that as the ceiling is safe for both referenced and
    # unreferenced data and avoids ValueError from ICA.fit on small recordings.
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
        threshold=_EOG_CORR_THRESHOLD,
        verbose="ERROR",
    )
    ica.exclude = list(bad_idx)
    logger.info(
        "ICA fit: n_components=%d, EOG-correlated rejected=%d (indices=%s)",
        safe_n, len(ica.exclude), ica.exclude,
    )
    ica.apply(out, verbose="ERROR")
    return out


EEG_BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 40.0),
}


def _band_power(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> float:
    """Mean PSD value within the [lo, hi) frequency band."""
    mask = (freqs >= lo) & (freqs < hi)
    if not mask.any():
        return 0.0
    return float(psd[mask].mean())


# Statistical-moment functions, bound to their column-label names. The
# `STATS` tuple below is derived from this list so labels and computations
# can never drift out of sync (a class of bug the original parallel-list
# design was vulnerable to).
_STATS_FUNCS: tuple[tuple[str, "_StatFn"], ...]  # populated below
_StatFn = "callable that maps a 1-D channel array to a single float"


def _stat_mean(x: np.ndarray) -> float:
    return float(np.mean(x))


def _stat_std(x: np.ndarray) -> float:
    return float(np.std(x))


def _stat_var(x: np.ndarray) -> float:
    return float(np.var(x))


def _stat_skew(x: np.ndarray) -> float:
    return float(scipy_stats.skew(x))


def _stat_kurtosis(x: np.ndarray) -> float:
    return float(scipy_stats.kurtosis(x))


_STATS_FUNCS = (
    ("mean", _stat_mean),
    ("std", _stat_std),
    ("var", _stat_var),
    ("skew", _stat_skew),
    ("kurtosis", _stat_kurtosis),
)
STATS: tuple[str, ...] = tuple(name for name, _ in _STATS_FUNCS)


def compute_features_from_epoch(epoch: np.ndarray, sfreq: float) -> np.ndarray:
    """Compute PSD-band + statistical features for one epoch.

    Per channel, the feature block is:
      [psd_delta, psd_theta, psd_alpha, psd_beta, psd_gamma,
       mean, std, var, skew, kurtosis]
    Channels are stacked in their input order. The resulting 1-D vector has
    length ``n_channels * (len(EEG_BANDS) + len(STATS))``.

    PSD uses Welch's method (`scipy.signal.welch`, `nperseg=min(256, n_samples)`).
    For meaningful Welch averaging, the epoch should contain at least
    `2 * nperseg` samples (e.g. ≥2 seconds at 256 Hz); shorter epochs degrade
    to a single-segment periodogram with high estimation variance.

    Statistical conventions:
      - ``mean``, ``std``, ``var`` use NumPy with ``ddof=0`` (biased / population
        estimators). For sample statistics callers must apply ``ddof=1`` adjustment
        downstream.
      - ``skew`` uses ``scipy.stats.skew(bias=True)`` (biased estimator).
      - ``kurtosis`` uses ``scipy.stats.kurtosis(fisher=True, bias=True)`` —
        Fisher's *excess* kurtosis (Gaussian → 0, not 3). Add 3 if Pearson
        kurtosis is required downstream.
      - For constant-valued channels (zero variance), ``skew`` and
        ``kurtosis`` are mathematically undefined and scipy returns NaN.
        We post-process the feature vector with ``np.nan_to_num`` to map
        any NaN/inf to 0.0, preserving the "no NaN survives" Parquet
        contract from AGENTS.md §6.

    Precondition: `epoch` must be finite (no NaN/inf). Filter via
    `is_valid_epoch` before calling — feature values are NaN-propagating.

    Args:
        epoch: A 2-D array shape (n_channels, n_samples), all-finite.
        sfreq: Sampling rate in Hz.

    Returns:
        A 1-D `np.ndarray` of dtype float64.
    """
    n_channels, n_samples = epoch.shape
    nperseg = min(256, n_samples)
    feats: list[float] = []
    for ch in range(n_channels):
        x = epoch[ch]
        freqs, psd = scipy_signal.welch(x, fs=sfreq, nperseg=nperseg)
        for _band, (lo, hi) in EEG_BANDS.items():
            feats.append(_band_power(freqs, psd, lo, hi))
        for _name, fn in _STATS_FUNCS:
            feats.append(fn(x))
    arr = np.asarray(feats, dtype=np.float64)
    # Constant-valued / zero-variance channels (e.g., disconnected electrodes)
    # make scipy.stats.skew / kurtosis return NaN. Map those to 0.0 so the
    # downstream Parquet contract ("no NaN in feature table") holds.
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _build_feature_columns(eeg_ch_names: list[str]) -> list[str]:
    """Generate the deterministic, in-channel-order column ordering."""
    cols: list[str] = []
    for ch in eeg_ch_names:
        for band in EEG_BANDS:
            cols.append(f"feat_{ch}_psd_{band}")
        for stat in STATS:
            cols.append(f"feat_{ch}_{stat}")
    return cols


def extract_features_from_recording(
    raw: mne.io.BaseRaw,
    epoch_duration_s: float = 2.0,
    eog_ch_name: str | None = None,
    n_components: int = 15,
    random_state: int = 97,
) -> pd.DataFrame:
    """Run the EEG pipeline on a Raw and return a 2-D feature DataFrame.

    Steps:
      1. Bandpass filter (1-40 Hz).
      2. ICA-based EOG artifact rejection (skipped if `eog_ch_name` is None).
      3. Slice into fixed-duration epochs.
      4. Drop any epoch with NaN/inf samples (logged WARNING).
      5. Compute features per epoch and stack into a DataFrame whose columns
         are `feat_<channel>_psd_<band>` and `feat_<channel>_<stat>`.

    Args:
        raw: Loaded `mne.io.BaseRaw` (must be `.load_data()`'d).
        epoch_duration_s: Length of each fixed-duration epoch in seconds.
        eog_ch_name: Name of EOG reference channel for ICA. None disables ICA.
        n_components: Cap on ICA components.
        random_state: Seed for ICA's solver (determinism).

    Returns:
        A `pd.DataFrame` with one row per valid epoch and ``n_eeg_channels *
        (len(EEG_BANDS) + len(STATS))`` ``feat_*`` columns.

    Raises:
        ValueError: if `epoch_duration_s * sfreq` rounds to less than 1 sample.
            (Other ValueError sources can propagate from `bandpass_filter`
            and `remove_artifacts_with_ica`; see their respective docstrings.)
    """
    filtered = bandpass_filter(raw, l_freq=1.0, h_freq=40.0)
    cleaned = remove_artifacts_with_ica(
        filtered,
        eog_ch_name=eog_ch_name,
        n_components=n_components,
        random_state=random_state,
    )

    sfreq = float(cleaned.info["sfreq"])
    n_samples_per_epoch = int(round(epoch_duration_s * sfreq))
    if n_samples_per_epoch < 1:
        raise ValueError(
            f"epoch_duration_s={epoch_duration_s!r} at sfreq={sfreq} Hz produces "
            f"{n_samples_per_epoch} samples per epoch (must be >= 1)"
        )
    eeg_picks = mne.pick_types(cleaned.info, eeg=True, meg=False, eog=False)
    eeg_names = [cleaned.ch_names[i] for i in eeg_picks]
    data = cleaned.get_data(picks=eeg_picks)  # shape (n_eeg, n_times)
    _, n_times = data.shape
    n_total_epochs = n_times // n_samples_per_epoch

    feature_cols = _build_feature_columns(eeg_names)
    rows: list[np.ndarray] = []
    invalid_indices: list[int] = []
    for ep in range(n_total_epochs):
        start = ep * n_samples_per_epoch
        end = start + n_samples_per_epoch
        epoch = data[:, start:end]
        if not is_valid_epoch(epoch):
            invalid_indices.append(ep)
            continue
        rows.append(compute_features_from_epoch(epoch, sfreq=sfreq))

    n_dropped = len(invalid_indices)
    if n_dropped:
        display = invalid_indices[:10]
        suffix = (
            f"... (+{n_dropped - 10} more)" if n_dropped > 10 else ""
        )
        logger.warning(
            "Dropping %d/%d epochs with invalid samples (indices=%s%s)",
            n_dropped, n_total_epochs, display, suffix,
        )

    if not rows:
        logger.info(
            "Feature extraction complete: in=%d, out=0, dropped=%d (%.2f%%)",
            n_total_epochs, n_dropped,
            100.0 * n_dropped / max(n_total_epochs, 1),
        )
        return pd.DataFrame(columns=feature_cols).astype(np.float64)

    matrix = np.vstack(rows)
    out = pd.DataFrame(matrix, columns=feature_cols, dtype=np.float64)
    logger.info(
        "Feature extraction complete: in=%d, out=%d, dropped=%d (%.2f%%)",
        n_total_epochs, len(out), n_dropped,
        100.0 * n_dropped / max(n_total_epochs, 1),
    )
    return out
