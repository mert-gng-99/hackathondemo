"""Tests for src.fusion.types — pydantic contract for fusion I/O."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.fusion.types import (
    ClinicalScores,
    DiseaseScore,
    FusionInput,
    FusionOutput,
    ModalityContribution,
    ModalityPrediction,
)


class TestModalityPrediction:
    def test_minimal_round_trip(self) -> None:
        pred = ModalityPrediction(
            label_text="alzheimers", label=1, confidence=0.81,
            probabilities=[
                {"label_text": "control", "probability": 0.19},
                {"label_text": "alzheimers", "probability": 0.81},
            ],
        )
        assert pred.label == 1
        assert pred.probabilities[1].probability == pytest.approx(0.81)

    def test_probabilities_must_be_non_empty(self) -> None:
        with pytest.raises(ValidationError):
            ModalityPrediction(label_text="x", label=0, confidence=0.5, probabilities=[])

    def test_rejects_probability_above_one(self) -> None:
        with pytest.raises(ValidationError):
            ModalityPrediction(
                label_text="x", label=0, confidence=0.5,
                probabilities=[{"label_text": "x", "probability": 1.5}],
            )

    def test_rejects_negative_label(self) -> None:
        with pytest.raises(ValidationError):
            ModalityPrediction(
                label_text="x", label=-1, confidence=0.5,
                probabilities=[{"label_text": "x", "probability": 0.5}],
            )


class TestClinicalScores:
    def test_all_optional(self) -> None:
        s = ClinicalScores()
        assert s.mmse is None and s.age_years is None

    def test_rejects_out_of_range_mmse(self) -> None:
        with pytest.raises(ValidationError):
            ClinicalScores(mmse=42.0)

    def test_rejects_negative_mmse(self) -> None:
        with pytest.raises(ValidationError):
            ClinicalScores(mmse=-1.0)

    def test_rejects_out_of_range_updrs(self) -> None:
        with pytest.raises(ValidationError):
            ClinicalScores(updrs=200.0)


class TestFusionInputOutput:
    def test_fusion_input_allows_no_modalities(self) -> None:
        # Caller may pass nothing — engine returns baseline scores.
        f = FusionInput()
        assert f.mri is None and f.eeg is None
        assert f.clinical == ClinicalScores()

    def test_fusion_output_round_trip(self) -> None:
        out = FusionOutput(
            diseases=[
                DiseaseScore(
                    disease="alzheimers",
                    probability=0.7,
                    contributions=[
                        ModalityContribution(
                            modality="mri", weight=0.35, signal=0.6, delta_logit=0.21,
                        )
                    ],
                )
            ],
            top_disease="alzheimers",
            missing_inputs=["eeg"],
        )
        assert out.top_disease == "alzheimers"
        assert out.diseases[0].contributions[0].delta_logit == pytest.approx(0.21)
