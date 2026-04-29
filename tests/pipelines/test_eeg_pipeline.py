"""Unit + integration tests for the EEG pipeline."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.pipelines.eeg_pipeline import is_valid_epoch


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

    def test_rejects_empty(self) -> None:
        assert is_valid_epoch(np.zeros((0, 256))) is False
        assert is_valid_epoch(np.zeros((4, 0))) is False

    def test_rejects_non_array(self) -> None:
        assert is_valid_epoch([[1, 2, 3]]) is False
        assert is_valid_epoch(None) is False
