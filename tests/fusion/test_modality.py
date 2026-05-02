"""Tests for src.fusion.modality — turn ModalityPrediction into a per-disease signal."""
from __future__ import annotations

import pytest

from src.fusion.modality import signal_for_disease
from src.fusion.types import ModalityClassProb, ModalityPrediction


def _pred(probs: dict[str, float]) -> ModalityPrediction:
    items = [ModalityClassProb(label_text=k, probability=v) for k, v in probs.items()]
    top = max(items, key=lambda p: p.probability)
    return ModalityPrediction(
        label_text=top.label_text,
        label=list(probs).index(top.label_text),
        confidence=top.probability,
        probabilities=items,
    )


class TestSignalForDisease:
    def test_disease_class_present_high_prob(self) -> None:
        pred = _pred({"control": 0.1, "alzheimers": 0.9})
        sig = signal_for_disease(pred, disease="alzheimers")
        assert sig == pytest.approx(0.8)

    def test_disease_class_present_low_prob(self) -> None:
        pred = _pred({"control": 0.95, "alzheimers": 0.05})
        sig = signal_for_disease(pred, disease="alzheimers")
        assert sig == pytest.approx(-0.9)

    def test_disease_class_absent_returns_none(self) -> None:
        pred = _pred({"control": 0.4, "parkinsons": 0.6})
        sig = signal_for_disease(pred, disease="alzheimers")
        assert sig is None

    def test_label_alias_matches_case_insensitively(self) -> None:
        pred = _pred({"Control": 0.2, "ALZHEIMERS": 0.8})
        sig = signal_for_disease(pred, disease="alzheimers")
        assert sig == pytest.approx(0.6)
