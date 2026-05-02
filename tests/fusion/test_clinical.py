"""Tests for src.fusion.clinical — per-test signal normalisers.

Convention: signal in [-1, 1] where +1 = strong evidence the disease IS
present and -1 = strong evidence it is NOT present.
"""
from __future__ import annotations

import pytest

from src.fusion import clinical


class TestMMSE:
    def test_perfect_score_signals_no_alzheimers(self) -> None:
        assert clinical.mmse_to_signal(30.0) == pytest.approx(-1.0)

    def test_severely_impaired_signals_alzheimers(self) -> None:
        assert clinical.mmse_to_signal(0.0) == pytest.approx(1.0)

    def test_borderline_24_is_near_neutral_slightly_positive(self) -> None:
        sig = clinical.mmse_to_signal(24.0)
        assert -0.1 < sig < 0.5


class TestMoCA:
    def test_perfect_signals_negative(self) -> None:
        assert clinical.moca_to_signal(30.0) == pytest.approx(-1.0)

    def test_zero_signals_positive(self) -> None:
        assert clinical.moca_to_signal(0.0) == pytest.approx(1.0)


class TestUPDRS:
    def test_zero_signals_no_parkinsons(self) -> None:
        assert clinical.updrs_to_signal(0.0) == pytest.approx(-1.0)

    def test_max_signals_parkinsons(self) -> None:
        assert clinical.updrs_to_signal(199.0) == pytest.approx(1.0, abs=1e-3)


class TestGait:
    def test_fast_walker_signals_negative(self) -> None:
        assert clinical.gait_to_signal(1.4) < -0.4

    def test_slow_walker_signals_positive(self) -> None:
        assert clinical.gait_to_signal(0.3) > 0.4


class TestAge:
    def test_young_signals_negative(self) -> None:
        assert clinical.age_to_signal(30.0) < -0.4

    def test_elderly_signals_positive(self) -> None:
        assert clinical.age_to_signal(85.0) > 0.4
