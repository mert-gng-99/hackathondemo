"""Tests for src.fusion.weights — disease/modality weight registry."""
from __future__ import annotations

import pytest

from src.fusion import weights


class TestWeights:
    def test_available_diseases_includes_known(self) -> None:
        diseases = set(weights.available_diseases())
        assert {"alzheimers", "parkinsons", "other"} <= diseases

    def test_available_clinical_tests_includes_each_named_input(self) -> None:
        tests = set(weights.available_clinical_tests())
        assert {"mmse", "moca", "updrs", "gait", "age"} <= tests

    def test_get_weights_returns_nonempty_mapping(self) -> None:
        ws = weights.get_weights("alzheimers")
        assert ws["mri"] > 0
        assert sum(ws.values()) == pytest.approx(1.0, abs=1e-6)

    def test_get_weights_unknown_disease_raises(self) -> None:
        with pytest.raises(KeyError, match="unknown disease"):
            weights.get_weights("invented_disease")

    def test_each_disease_weight_table_sums_to_one(self) -> None:
        for d in weights.available_diseases():
            assert sum(weights.get_weights(d).values()) == pytest.approx(1.0, abs=1e-6), d
