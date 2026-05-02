"""Map raw clinical-test scores to a unitless signal in [-1, 1].

+1 means the test strongly supports the disease being present.
-1 means it strongly supports the disease being absent.
"""
from __future__ import annotations


def _linear_map(value: float, low: float, high: float, *, invert: bool) -> float:
    """Map `value` from [low, high] to [-1, 1]. If invert, flip sign."""
    if high == low:
        return 0.0
    clipped = max(low, min(high, value))
    norm = (clipped - low) / (high - low)
    signal = 2.0 * norm - 1.0
    return -signal if invert else signal


def mmse_to_signal(score: float) -> float:
    """MMSE: 30 = healthy (-1), 18 = severe (+1), 24 = mild cutoff (0)."""
    return _linear_map(score, low=18.0, high=30.0, invert=True)


def moca_to_signal(score: float) -> float:
    """MoCA: same scale as MMSE."""
    return _linear_map(score, low=18.0, high=30.0, invert=True)


def updrs_to_signal(score: float) -> float:
    return _linear_map(score, low=0.0, high=199.0, invert=False)


def gait_to_signal(speed_m_s: float) -> float:
    return _linear_map(speed_m_s, low=0.0, high=1.4, invert=True)


def age_to_signal(years: float) -> float:
    return _linear_map(years, low=30.0, high=90.0, invert=False)
