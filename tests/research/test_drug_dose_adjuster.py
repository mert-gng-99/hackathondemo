"""Tests for src.research.drug_dose_adjuster — pure-function dose revision."""
from __future__ import annotations

import pytest

from src.research.drug_dose_adjuster import adjust


class TestAdjust:
    # --- BBB intact ---------------------------------------------------------

    def test_intact_bbb_no_change_permeable_drug(self) -> None:
        out = adjust(baseline_dose_mg=100.0, bbb_permeability_score=0.05, drug_bbb_permeable=True)
        assert out.recommended_dose_mg == pytest.approx(100.0)
        assert out.adjustment_factor == 1.0
        assert out.risk_level == "low"

    def test_intact_bbb_no_change_non_permeable_drug(self) -> None:
        out = adjust(baseline_dose_mg=200.0, bbb_permeability_score=0.0, drug_bbb_permeable=False)
        assert out.adjustment_factor == 1.0
        assert out.risk_level == "low"

    # --- BBB leaky, drug permeable -----------------------------------------

    def test_leaky_bbb_permeable_drug_reduces_dose(self) -> None:
        out = adjust(baseline_dose_mg=100.0, bbb_permeability_score=0.5, drug_bbb_permeable=True)
        # factor = max(0.3, 1 - 0.7*0.5) = max(0.3, 0.65) = 0.65
        assert out.adjustment_factor == pytest.approx(0.65)
        assert out.recommended_dose_mg == pytest.approx(65.0)
        assert out.risk_level == "moderate"

    def test_severe_leakage_permeable_drug_high_risk(self) -> None:
        out = adjust(baseline_dose_mg=100.0, bbb_permeability_score=0.8, drug_bbb_permeable=True)
        assert out.risk_level == "high"
        # factor = max(0.3, 1 - 0.7*0.8) = max(0.3, 0.44) = 0.44
        assert out.adjustment_factor == pytest.approx(0.44)

    def test_safety_floor_not_breached(self) -> None:
        # At perm=1.0: 1 - 0.7 = 0.3 — the floor.
        out = adjust(baseline_dose_mg=100.0, bbb_permeability_score=1.0, drug_bbb_permeable=True)
        assert out.adjustment_factor >= 0.3 - 1e-9
        # And anything above 1.0 (clipped by validator) — won't get there.

    # --- BBB leaky, drug NOT permeable -------------------------------------

    def test_leaky_bbb_non_permeable_drug_mild_reduction(self) -> None:
        out = adjust(baseline_dose_mg=100.0, bbb_permeability_score=0.5, drug_bbb_permeable=False)
        # factor = max(0.6, 1 - 0.4*0.5) = max(0.6, 0.8) = 0.8
        assert out.adjustment_factor == pytest.approx(0.8)
        assert out.risk_level == "moderate"

    def test_non_permeable_drug_safety_floor(self) -> None:
        out = adjust(baseline_dose_mg=100.0, bbb_permeability_score=1.0, drug_bbb_permeable=False)
        assert out.adjustment_factor == pytest.approx(0.6)

    # --- Drug permeability unknown -----------------------------------------

    def test_unknown_permeability_treated_as_permeable(self) -> None:
        out_unknown = adjust(baseline_dose_mg=100.0, bbb_permeability_score=0.5, drug_bbb_permeable=None)
        out_perm    = adjust(baseline_dose_mg=100.0, bbb_permeability_score=0.5, drug_bbb_permeable=True)
        assert out_unknown.adjustment_factor == pytest.approx(out_perm.adjustment_factor)

    def test_unknown_permeability_rationale_says_so(self) -> None:
        out = adjust(baseline_dose_mg=100.0, bbb_permeability_score=0.5, drug_bbb_permeable=None)
        assert "unknown" in out.rationale.lower()

    # --- Validation ---------------------------------------------------------

    def test_zero_baseline_raises(self) -> None:
        with pytest.raises(ValueError, match="baseline_dose_mg must be > 0"):
            adjust(baseline_dose_mg=0.0, bbb_permeability_score=0.5)

    def test_negative_baseline_raises(self) -> None:
        with pytest.raises(ValueError, match="baseline_dose_mg must be > 0"):
            adjust(baseline_dose_mg=-10.0, bbb_permeability_score=0.5)

    def test_score_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="bbb_permeability_score must be in"):
            adjust(baseline_dose_mg=100.0, bbb_permeability_score=1.5)
        with pytest.raises(ValueError, match="bbb_permeability_score must be in"):
            adjust(baseline_dose_mg=100.0, bbb_permeability_score=-0.1)

    # --- Monotonicity ------------------------------------------------------

    def test_increasing_leakage_lowers_recommended_dose_for_permeable_drug(self) -> None:
        scores = [0.25, 0.4, 0.55, 0.7, 0.85]
        doses = [
            adjust(100.0, s, drug_bbb_permeable=True).recommended_dose_mg
            for s in scores
        ]
        # Strictly non-increasing.
        assert all(a >= b - 1e-9 for a, b in zip(doses, doses[1:]))

    # --- Rationale always says it's not medical advice --------------------

    def test_rationale_disclaims_medical_advice(self) -> None:
        for score in [0.0, 0.3, 0.7, 1.0]:
            out = adjust(100.0, score, drug_bbb_permeable=True)
            assert "research suggestion" in out.rationale.lower()
            assert "medical advice" in out.rationale.lower()
