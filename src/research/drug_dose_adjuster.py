"""Drug-dose adjustment given patient BBB permeability + drug BBB classification.

Pure-function logic. Researcher-persona module — does NOT touch the fusion
engine. Output is a research suggestion, NOT medical advice.

Decision matrix:

    BBB intact (perm < 0.2)              -> factor=1.0, risk=low
    BBB leaky, drug BBB-permeable        -> factor=max(0.3, 1 - 0.7*perm),
                                             risk=high if perm > 0.6 else moderate
    BBB leaky, drug NOT permeable        -> factor=max(0.6, 1 - 0.4*perm),
                                             risk=moderate
    drug_permeable=None (unknown)        -> treat as permeable (safer)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RiskLevel = Literal["low", "moderate", "high"]


@dataclass(frozen=True)
class DoseAdjustment:
    recommended_dose_mg: float
    adjustment_factor: float  # 0..1; 1.0 = no change
    risk_level: RiskLevel
    rationale: str


# Threshold below which BBB is considered intact and no adjustment applies.
_INTACT_THRESHOLD = 0.2

# Floor on the adjustment factor (we never recommend going below this fraction
# of the baseline dose — anything more aggressive belongs to a clinician).
_PERMEABLE_DRUG_MIN_FACTOR = 0.3
_NON_PERMEABLE_DRUG_MIN_FACTOR = 0.6

# Slope on the linear adjustment.
_PERMEABLE_DRUG_SLOPE = 0.7
_NON_PERMEABLE_DRUG_SLOPE = 0.4


def adjust(
    baseline_dose_mg: float,
    bbb_permeability_score: float,
    drug_bbb_permeable: bool | None = None,
) -> DoseAdjustment:
    """Return a DoseAdjustment for the (patient, drug) pair."""
    if baseline_dose_mg <= 0.0:
        raise ValueError(f"baseline_dose_mg must be > 0; got {baseline_dose_mg}")
    if not 0.0 <= bbb_permeability_score <= 1.0:
        raise ValueError(
            f"bbb_permeability_score must be in [0, 1]; got {bbb_permeability_score}"
        )

    if bbb_permeability_score < _INTACT_THRESHOLD:
        return DoseAdjustment(
            recommended_dose_mg=baseline_dose_mg,
            adjustment_factor=1.0,
            risk_level="low",
            rationale=(
                f"BBB permeability score {bbb_permeability_score:.2f} is below the "
                f"intact threshold ({_INTACT_THRESHOLD:.2f}). No dose adjustment "
                "recommended. Research suggestion, not medical advice."
            ),
        )

    # Treat unknown permeability conservatively as 'permeable'.
    permeable = True if drug_bbb_permeable is None else bool(drug_bbb_permeable)

    if permeable:
        factor = max(
            _PERMEABLE_DRUG_MIN_FACTOR,
            1.0 - _PERMEABLE_DRUG_SLOPE * bbb_permeability_score,
        )
        risk: RiskLevel = "high" if bbb_permeability_score > 0.6 else "moderate"
        permeable_note = (
            "Drug is BBB-permeable" if drug_bbb_permeable is True
            else "Drug BBB permeability is unknown (treated as permeable for safety)"
        )
    else:
        factor = max(
            _NON_PERMEABLE_DRUG_MIN_FACTOR,
            1.0 - _NON_PERMEABLE_DRUG_SLOPE * bbb_permeability_score,
        )
        risk = "moderate"
        permeable_note = "Drug is normally BBB-excluded"

    recommended = baseline_dose_mg * factor
    rationale = (
        f"BBB permeability score {bbb_permeability_score:.2f} indicates "
        f"compromised barrier. {permeable_note}; recommended dose reduced "
        f"to {factor*100:.0f}% of baseline ({recommended:.1f} mg of "
        f"{baseline_dose_mg:.1f} mg). Risk level: {risk}. "
        "Research suggestion, not medical advice."
    )
    return DoseAdjustment(
        recommended_dose_mg=float(recommended),
        adjustment_factor=float(factor),
        risk_level=risk,
        rationale=rationale,
    )
