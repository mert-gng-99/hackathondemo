"""Unit + integration tests for the EEG pipeline."""
from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pytest

from src.pipelines.eeg_pipeline import (
    bandpass_filter,
    is_valid_epoch,
)


FIXTURE = Path(__file__).parent.parent / "fixtures" / "eeg_sample.fif"


class TestIsValidEpoch:
    def test_accepts_2d_finite_array(self) -> None:
        epoch = np.zeros((4, 256), dtype=np.float64)
        assert is_valid_epoch(epoch) is True

    def test_rejects_wrong_dimension(self) -> None:
        assert is_valid_epoch(np.zeros((4,))) is False
        assert is_valid_epoch(np.zeros((4, 256, 2))) is False

    def test_rejects_nan(self) -> None:
        epoch = np.zeros((4, 256))
        epoch[0, 10] = np.nan
        assert is_valid_epoch(epoch) is False

    def test_rejects_inf(self) -> None:
        epoch = np.zeros((4, 256))
        epoch[1, 5] = np.inf
        assert is_valid_epoch(epoch) is False
        epoch[1, 5] = -np.inf
        assert is_valid_epoch(epoch) is False

    def test_rejects_empty(self) -> None:
        assert is_valid_epoch(np.zeros((0, 256))) is False
        assert is_valid_epoch(np.zeros((4, 0))) is False

    def test_rejects_non_array(self) -> None:
        assert is_valid_epoch([[1, 2, 3]]) is False
        assert is_valid_epoch(None) is False

    def test_rejects_non_numeric_dtype(self) -> None:
        """String / object dtype arrays must be rejected without raising."""
        epoch = np.array([["a", "b"], ["c", "d"]])
        assert is_valid_epoch(epoch) is False


class TestBandpassFilter:
    def _load(self) -> mne.io.BaseRaw:
        return mne.io.read_raw_fif(FIXTURE, preload=True, verbose="ERROR")

    def test_returns_raw_instance(self) -> None:
        raw = self._load()
        out = bandpass_filter(raw, l_freq=1.0, h_freq=40.0)
        assert isinstance(out, mne.io.BaseRaw)

    def test_preserves_shape(self) -> None:
        raw = self._load()
        n_ch_before, n_t_before = raw.get_data().shape
        out = bandpass_filter(raw, l_freq=1.0, h_freq=40.0)
        assert out.get_data().shape == (n_ch_before, n_t_before)

    def test_attenuates_dc_component(self) -> None:
        """A bandpass with l_freq=1.0 must remove a DC offset."""
        raw = self._load()
        # Inject a large DC offset on every channel.
        data = raw.get_data() + 1e-3
        raw_dc = mne.io.RawArray(data, raw.info, verbose="ERROR")
        out = bandpass_filter(raw_dc, l_freq=1.0, h_freq=40.0)
        # Mean on each channel should be near zero (much smaller than 1e-3).
        assert np.all(np.abs(out.get_data().mean(axis=1)) < 1e-4)

    def test_does_not_mutate_input(self) -> None:
        raw = self._load()
        original_mean = raw.get_data().mean()
        _ = bandpass_filter(raw, l_freq=1.0, h_freq=40.0)
        assert raw.get_data().mean() == pytest.approx(original_mean, rel=1e-12)
