"""Tests for src.fusion.engine.fuse — the core multi-modal combiner."""
from __future__ import annotations

import logging
from typing import Any

import pytest

from src.fusion import engine
from src.fusion.types import (
    ClinicalScores,
    FusionInput,
    ModalityClassProb,
    ModalityPrediction,
)


def _mri(prob_alz: float, prob_pd: float = 0.0) -> ModalityPrediction:
    p_other = max(0.0, 1.0 - prob_alz - prob_pd)
    items = [
        ModalityClassProb(label_text="control", probability=p_other),
        ModalityClassProb(label_text="alzheimers", probability=prob_alz),
        ModalityClassProb(label_text="parkinsons", probability=prob_pd),
    ]
    top = max(items, key=lambda p: p.probability)
    return ModalityPrediction(
        label_text=top.label_text,
        label=[p.label_text for p in items].index(top.label_text),
        confidence=top.probability,
        probabilities=items,
    )


class TestFuse:
    def test_empty_input_returns_baseline_with_missing_listed(self) -> None:
        out = engine.fuse(FusionInput())
        assert {d.disease for d in out.diseases} >= {"alzheimers", "parkinsons", "other"}
        for ds in out.diseases:
            assert ds.probability == pytest.approx(0.5, abs=1e-6)
            assert ds.contributions == []
        assert "mri" in out.missing_inputs
        assert "eeg" in out.missing_inputs

    def test_mri_only_alzheimers_high(self) -> None:
        inp = FusionInput(mri=_mri(prob_alz=0.9))
        out = engine.fuse(inp)
        alz = next(d for d in out.diseases if d.disease == "alzheimers")
        assert alz.probability > 0.7
        assert any(c.modality == "mri" for c in alz.contributions)
        assert out.top_disease == "alzheimers"

    def test_mri_eeg_agreement_boosts_above_either_alone(self) -> None:
        only_mri = engine.fuse(FusionInput(mri=_mri(prob_alz=0.8)))
        only_eeg = engine.fuse(FusionInput(eeg=_mri(prob_alz=0.8)))
        both = engine.fuse(FusionInput(
            mri=_mri(prob_alz=0.8), eeg=_mri(prob_alz=0.8),
        ))

        def alz(out: Any) -> float:
            return next(d for d in out.diseases if d.disease == "alzheimers").probability

        assert alz(both) > alz(only_mri)
        assert alz(both) > alz(only_eeg)

    def test_clinical_only_low_mmse_raises_alzheimers(self) -> None:
        out = engine.fuse(FusionInput(clinical=ClinicalScores(mmse=10.0)))
        alz = next(d for d in out.diseases if d.disease == "alzheimers")
        assert alz.probability > 0.55
        assert any(c.modality == "clinical_mmse" for c in alz.contributions)

    def test_disagreement_moderates_confidence(self) -> None:
        out = engine.fuse(FusionInput(
            mri=_mri(prob_alz=0.85),
            clinical=ClinicalScores(mmse=30.0),
        ))
        alz = next(d for d in out.diseases if d.disease == "alzheimers")
        assert 0.5 < alz.probability < 0.78

    def test_unknown_clinical_field_is_ignored_safely(self) -> None:
        out = engine.fuse(FusionInput(clinical=ClinicalScores(age_years=80.0)))
        assert out.top_disease in {"alzheimers", "parkinsons", "other"}

    def test_engine_does_not_depend_on_bbb(self) -> None:
        # Independence regression: fusion must not couple to BBB. A patient
        # with only MRI/EEG/clinical data must produce a valid output even
        # though no BBB module is involved.
        import inspect
        import src.fusion.engine as engine_mod
        import src.fusion.weights as weights_mod
        assert "bbb" not in inspect.getsource(engine_mod).lower()
        for disease in weights_mod.available_diseases():
            for key in weights_mod.get_weights(disease):
                assert "bbb" not in key.lower(), (disease, key)

    def test_warning_logged_when_disease_has_no_signals(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # 'other' disease with no MRI/EEG inputs -> no signals available.
        engine.logger.addHandler(caplog.handler)
        caplog.handler.setLevel(logging.INFO)
        try:
            out = engine.fuse(FusionInput(clinical=ClinicalScores(mmse=10.0)))
        finally:
            engine.logger.removeHandler(caplog.handler)
        other = next(d for d in out.diseases if d.disease == "other")
        assert other.probability == pytest.approx(0.5, abs=1e-6)
        assert other.contributions == []
