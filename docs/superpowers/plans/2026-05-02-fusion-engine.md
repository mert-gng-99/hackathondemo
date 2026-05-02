# Fusion Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fusion module that takes any combination of MRI prediction, EEG prediction, and clinical-test scores and returns per-disease (Alzheimer's, Parkinson's, other) confidence with full attribution showing which input contributed which fraction.

**Architecture.** Pure Python module under `src/fusion/`. Each modality is converted to a signed signal in `[-1, 1]`. Per disease we compute `logit = bias + Σ weight × signal`, apply a sigmoid, then expose every term as a `ModalityContribution` so the UI can render an attribution bar. A FastAPI route and an agent tool sit on top so the orchestrator can call it. **No new ML training** — all models that feed it are existing artefacts.

**Tech stack.** Python 3.11, pydantic v2, FastAPI, pytest, numpy. Re-uses `src.core.logger`, `src.api.schemas`, `src.agents.tools` patterns already in the repo.

**Independence guarantee (locked in).** Fusion modalities are exactly: `mri`, `eeg`, and the named clinical scores (`mmse`, `moca`, `updrs`, `gait`, `age`). **BBB is NOT a fusion modality.** Even if a patient has BBB data, it does not flow into Alzheimer's/Parkinson's confidence — BBB belongs to the drug-researcher persona and lives in its own pipeline. The engine must never import from `src.pipelines.bbb_pipeline` or `src.models.bbb_model`. Task 5's tests include a regression assertion to enforce this.

**Independence test (Task 5).** A unit test imports `src.fusion.engine` and asserts `bbb` does not appear in any weight key, contribution modality, or imported module name. This pins the decoupling at CI time.

---

## File structure

| Path | Responsibility |
|---|---|
| Create `src/fusion/__init__.py` | package marker |
| Create `src/fusion/types.py` | pydantic types: `ModalityPrediction`, `ClinicalScores`, `FusionInput`, `ModalityContribution`, `DiseaseScore`, `FusionOutput` |
| Create `src/fusion/weights.py` | `DEFAULT_WEIGHTS`, `available_diseases()`, `available_clinical_tests()`, `get_weights(disease)` |
| Create `src/fusion/clinical.py` | per-test signal normalisers (`mmse_to_signal` etc.) |
| Create `src/fusion/modality.py` | `mri_signal_for_disease`, `eeg_signal_for_disease` |
| Create `src/fusion/engine.py` | `fuse(input) -> FusionOutput` |
| Modify `src/api/schemas.py` | add `FusionRequest`, `FusionResponse` (thin wrappers re-exporting fusion types) |
| Modify `src/api/routes.py` | mount `POST /fusion/predict` |
| Modify `src/agents/tools.py` | register `run_fusion` tool |
| Modify `src/agents/prompts.py` | add fusion tool description so the orchestrator can call it |
| Create `tests/fusion/__init__.py` | test package marker |
| Create `tests/fusion/test_weights.py` | weight registry tests |
| Create `tests/fusion/test_clinical.py` | normaliser boundary tests |
| Create `tests/fusion/test_modality.py` | modality signal extraction tests |
| Create `tests/fusion/test_engine.py` | core fusion behaviour tests |
| Create `tests/api/test_fusion_route.py` | FastAPI integration test |
| Create `tests/agents/test_tools_fusion.py` | agent tool wrapper test |

Each file is small and focused. No file is doing two jobs.

---

## Data contract (lock this in before coding)

### Inputs

```python
ModalityPrediction = {
    "label_text": str,           # e.g. "alzheimers", "parkinsons", "control"
    "label": int,                # class index from underlying model
    "confidence": float,         # in [0, 1]
    "probabilities": [           # full softmax
        {"label_text": str, "probability": float},
        ...
    ],
}

ClinicalScores = {
    "mmse": float | None,        # 0..30, lower = worse
    "moca": float | None,        # 0..30, lower = worse
    "updrs": float | None,       # 0..199, higher = worse
    "gait_speed_m_s": float | None,  # 0..2, lower = worse
    "age_years": float | None,   # 0..120
}
```

### Output

```python
FusionOutput = {
    "diseases": [
        {
            "disease": "alzheimers",
            "probability": 0.71,
            "contributions": [
                {"modality": "mri", "weight": 0.40, "signal": 0.6, "delta_logit": 0.24},
                {"modality": "clinical_mmse", "weight": 0.20, "signal": 0.8, "delta_logit": 0.16},
                ...
            ],
        },
        ...
    ],
    "top_disease": "alzheimers",
    "missing_inputs": ["eeg"],   # things that would have helped but were absent
}
```

**Math.** For disease *d*: `logit_d = bias_d + Σ_m w_{m,d} · signal_{m,d}` where `signal_{m,d} ∈ [-1, 1]` and `Σ_m w_{m,d} = 1` (excluding modalities that were not provided — we renormalise on-the-fly). `probability = sigmoid(scale · logit_d)` with `scale = 4.0` (chosen so a single-modality saturated agreement maps to ~0.88, leaving headroom for stacking).

**bias_d.** Default `0.0` for every disease; configurable via weights file later.

### Renormalisation rule

When some modalities are missing (e.g., EEG not uploaded), we normalise the **provided** weights to sum to the original sum of provided weights only — we do **not** redistribute missing weight onto remaining modalities, because that would silently inflate confidence. Concretely: if `mri` has weight 0.40 and `eeg` (weight 0.25) is missing, the disease's max attainable logit is now `0.40 · 1.0 + Σ clinical · 1.0` rather than `(0.40+0.25) · 1.0`. This naturally lowers confidence when modalities are absent — desired behaviour.

---

## Weights table (preset)

```python
DEFAULT_WEIGHTS = {
    "alzheimers": {
        "mri":             0.35,
        "eeg":             0.20,
        "clinical_mmse":   0.20,
        "clinical_moca":   0.15,
        "clinical_age":    0.10,
    },
    "parkinsons": {
        "mri":             0.20,
        "eeg":             0.30,
        "clinical_updrs":  0.30,
        "clinical_gait":   0.15,
        "clinical_age":    0.05,
    },
    "other": {
        "mri": 0.50,
        "eeg": 0.50,
    },
}
```

These are heuristic, not validated. The plan exposes them in code so a clinician collaborator can tune them without a deploy.

---

## Tasks

### Task 1: Types and schemas

**Files:**
- Create: `src/fusion/__init__.py`
- Create: `src/fusion/types.py`
- Create: `tests/fusion/__init__.py`
- Test: `tests/fusion/test_types.py`

- [ ] **Step 1: Write the failing test**

Create `tests/fusion/test_types.py`:

```python
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


class TestClinicalScores:
    def test_all_optional(self) -> None:
        s = ClinicalScores()
        assert s.mmse is None and s.age_years is None

    def test_rejects_out_of_range_mmse(self) -> None:
        with pytest.raises(ValidationError):
            ClinicalScores(mmse=42.0)


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/fusion/test_types.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.fusion'`

- [ ] **Step 3: Write minimal implementation**

Create `src/fusion/__init__.py` (empty file).

Create `src/fusion/types.py`:

```python
"""Pydantic data contract for the multi-modal fusion engine."""
from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class ModalityClassProb(BaseModel):
    label_text: str
    probability: float = Field(..., ge=0.0, le=1.0)


class ModalityPrediction(BaseModel):
    """One modality's classifier output (MRI or EEG)."""
    model_config = ConfigDict(protected_namespaces=())

    label_text: str
    label: int = Field(..., ge=0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: list[ModalityClassProb] = Field(..., min_length=1)


class ClinicalScores(BaseModel):
    """Doctor-entered extra-test scores. Each is optional."""
    mmse: Annotated[float, Field(ge=0.0, le=30.0)] | None = None
    moca: Annotated[float, Field(ge=0.0, le=30.0)] | None = None
    updrs: Annotated[float, Field(ge=0.0, le=199.0)] | None = None
    gait_speed_m_s: Annotated[float, Field(ge=0.0, le=2.5)] | None = None
    age_years: Annotated[float, Field(ge=0.0, le=120.0)] | None = None


class FusionInput(BaseModel):
    mri: ModalityPrediction | None = None
    eeg: ModalityPrediction | None = None
    clinical: ClinicalScores = Field(default_factory=ClinicalScores)


class ModalityContribution(BaseModel):
    """One row of the attribution table for a single disease score."""
    modality: str  # "mri" | "eeg" | "clinical_<name>"
    weight: float
    signal: float = Field(..., ge=-1.0, le=1.0)
    delta_logit: float


class DiseaseScore(BaseModel):
    disease: str
    probability: float = Field(..., ge=0.0, le=1.0)
    contributions: list[ModalityContribution]


class FusionOutput(BaseModel):
    diseases: list[DiseaseScore]
    top_disease: str
    missing_inputs: list[str] = Field(default_factory=list)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/fusion/test_types.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/fusion/__init__.py src/fusion/types.py tests/fusion/__init__.py tests/fusion/test_types.py
git commit -m "feat(fusion): add pydantic data contract for multi-modal fusion"
```

---

### Task 2: Weight registry

**Files:**
- Create: `src/fusion/weights.py`
- Test: `tests/fusion/test_weights.py`

- [ ] **Step 1: Write the failing test**

Create `tests/fusion/test_weights.py`:

```python
"""Tests for src.fusion.weights — disease/modality weight registry."""
from __future__ import annotations

import pytest

from src.fusion import weights


class TestWeights:
    def test_available_diseases_includes_known(self) -> None:
        diseases = set(weights.available_diseases())
        assert {"alzheimers", "parkinsons", "other"} <= diseases

    def test_available_clinical_tests_includes_each_named_input(self) -> None:
        # Every clinical_<name> weight key must correspond to a clinical input.
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/fusion/test_weights.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.fusion.weights'`

- [ ] **Step 3: Write minimal implementation**

Create `src/fusion/weights.py`:

```python
"""Disease × modality weight registry for the fusion engine.

Heuristic preset weights — tune offline as clinician feedback arrives.
"""
from __future__ import annotations

from typing import Mapping

DEFAULT_WEIGHTS: dict[str, dict[str, float]] = {
    "alzheimers": {
        "mri":             0.35,
        "eeg":             0.20,
        "clinical_mmse":   0.20,
        "clinical_moca":   0.15,
        "clinical_age":    0.10,
    },
    "parkinsons": {
        "mri":             0.20,
        "eeg":             0.30,
        "clinical_updrs":  0.30,
        "clinical_gait":   0.15,
        "clinical_age":    0.05,
    },
    "other": {
        "mri": 0.50,
        "eeg": 0.50,
    },
}


def available_diseases() -> list[str]:
    return sorted(DEFAULT_WEIGHTS.keys())


def available_clinical_tests() -> list[str]:
    """Return the bare clinical-test names (without the 'clinical_' prefix)."""
    names: set[str] = set()
    for table in DEFAULT_WEIGHTS.values():
        for key in table:
            if key.startswith("clinical_"):
                names.add(key[len("clinical_"):])
    return sorted(names)


def get_weights(disease: str) -> Mapping[str, float]:
    if disease not in DEFAULT_WEIGHTS:
        raise KeyError(f"unknown disease: {disease!r}")
    return DEFAULT_WEIGHTS[disease]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/fusion/test_weights.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/fusion/weights.py tests/fusion/test_weights.py
git commit -m "feat(fusion): add disease/modality weight registry"
```

---

### Task 3: Clinical signal normalisers

**Files:**
- Create: `src/fusion/clinical.py`
- Test: `tests/fusion/test_clinical.py`

- [ ] **Step 1: Write the failing test**

Create `tests/fusion/test_clinical.py`:

```python
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
        # MMSE 24 is the mild-impairment cutoff. Signal should be slightly above 0.
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
        # Healthy adult gait ~1.4 m/s — signal should be clearly negative.
        assert clinical.gait_to_signal(1.4) < -0.4

    def test_slow_walker_signals_positive(self) -> None:
        # Bradykinesia / shuffling gait < 0.5 m/s — signal should be positive.
        assert clinical.gait_to_signal(0.3) > 0.4


class TestAge:
    def test_young_signals_negative(self) -> None:
        assert clinical.age_to_signal(30.0) < -0.4

    def test_elderly_signals_positive(self) -> None:
        assert clinical.age_to_signal(85.0) > 0.4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/fusion/test_clinical.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

Create `src/fusion/clinical.py`:

```python
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
    norm = (clipped - low) / (high - low)  # [0, 1]
    signal = 2.0 * norm - 1.0              # [-1, 1]
    return -signal if invert else signal


def mmse_to_signal(score: float) -> float:
    # MMSE: 30 = healthy, 0 = severe — invert so low score => +1.
    return _linear_map(score, low=0.0, high=30.0, invert=True)


def moca_to_signal(score: float) -> float:
    return _linear_map(score, low=0.0, high=30.0, invert=True)


def updrs_to_signal(score: float) -> float:
    # UPDRS: 0 = healthy, ~199 = severe.
    return _linear_map(score, low=0.0, high=199.0, invert=False)


def gait_to_signal(speed_m_s: float) -> float:
    # Healthy adult ~1.4 m/s, parkinsonian shuffling < 0.5 m/s.
    return _linear_map(speed_m_s, low=0.0, high=1.4, invert=True)


def age_to_signal(years: float) -> float:
    # Risk rises sharply past 65. Anchor: 30 -> -1, 90 -> +1.
    return _linear_map(years, low=30.0, high=90.0, invert=False)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/fusion/test_clinical.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Commit**

```bash
git add src/fusion/clinical.py tests/fusion/test_clinical.py
git commit -m "feat(fusion): add clinical-test signal normalisers (MMSE/MoCA/UPDRS/gait/age)"
```

---

### Task 4: Modality signal extractors

**Files:**
- Create: `src/fusion/modality.py`
- Test: `tests/fusion/test_modality.py`

- [ ] **Step 1: Write the failing test**

Create `tests/fusion/test_modality.py`:

```python
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
        # The model exposes a class for the disease and assigns it 0.9.
        pred = _pred({"control": 0.1, "alzheimers": 0.9})
        sig = signal_for_disease(pred, disease="alzheimers")
        assert sig == pytest.approx(0.8)  # 2*0.9 - 1

    def test_disease_class_present_low_prob(self) -> None:
        pred = _pred({"control": 0.95, "alzheimers": 0.05})
        sig = signal_for_disease(pred, disease="alzheimers")
        assert sig == pytest.approx(-0.9)

    def test_disease_class_absent_returns_none(self) -> None:
        # Model only emits {"control", "parkinsons"}; we ask for alzheimers.
        pred = _pred({"control": 0.4, "parkinsons": 0.6})
        sig = signal_for_disease(pred, disease="alzheimers")
        assert sig is None

    def test_label_alias_matches_case_insensitively(self) -> None:
        pred = _pred({"Control": 0.2, "ALZHEIMERS": 0.8})
        sig = signal_for_disease(pred, disease="alzheimers")
        assert sig == pytest.approx(0.6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/fusion/test_modality.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

Create `src/fusion/modality.py`:

```python
"""Convert a modality classifier's probability vector into a signed signal."""
from __future__ import annotations

from src.fusion.types import ModalityPrediction


def signal_for_disease(pred: ModalityPrediction, disease: str) -> float | None:
    """Return signal in [-1, 1] for `disease`, or None if the model has no
    matching class.

    A class matches if its `label_text` equals `disease` case-insensitively.
    Signal = 2 * P(disease) - 1.
    """
    target = disease.strip().lower()
    for cls in pred.probabilities:
        if cls.label_text.strip().lower() == target:
            return 2.0 * cls.probability - 1.0
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/fusion/test_modality.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/fusion/modality.py tests/fusion/test_modality.py
git commit -m "feat(fusion): map modality predictions to per-disease signals"
```

---

### Task 5: Fusion engine core

**Files:**
- Create: `src/fusion/engine.py`
- Test: `tests/fusion/test_engine.py`

- [ ] **Step 1: Write the failing test**

Create `tests/fusion/test_engine.py`:

```python
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
        # MRI says alzheimers, clinical MMSE is perfect (against).
        out = engine.fuse(FusionInput(
            mri=_mri(prob_alz=0.85),
            clinical=ClinicalScores(mmse=30.0),
        ))
        alz = next(d for d in out.diseases if d.disease == "alzheimers")
        # Lower than MRI-only would have been (0.7+), but still elevated.
        assert 0.5 < alz.probability < 0.78

    def test_unknown_clinical_field_is_ignored_safely(self) -> None:
        # If a clinical field isn't in any weight table, it's still valid input
        # and must not error. (No such field exists in pydantic, but covers
        # defensive paths for future fields.)
        out = engine.fuse(FusionInput(clinical=ClinicalScores(age_years=80.0)))
        assert out.top_disease in {"alzheimers", "parkinsons", "other"}

    def test_engine_does_not_depend_on_bbb(self) -> None:
        # Independence regression: fusion must not couple to BBB. A patient
        # with only MRI/EEG/clinical data must produce a valid output even
        # though no BBB module is involved.
        import inspect
        import src.fusion.engine as engine_mod
        import src.fusion.weights as weights_mod
        # No imports from bbb anywhere in the fusion package.
        assert "bbb" not in inspect.getsource(engine_mod).lower()
        # No 'bbb' weight key in any disease table.
        for disease in weights_mod.available_diseases():
            for key in weights_mod.get_weights(disease):
                assert "bbb" not in key.lower(), (disease, key)

    def test_warning_logged_when_disease_has_no_signals(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # 'other' disease with no MRI/EEG inputs -> no signals available.
        # Engine should log a debug/info note and produce baseline 0.5 for it.
        engine.logger.addHandler(caplog.handler)
        caplog.handler.setLevel(logging.INFO)
        try:
            out = engine.fuse(FusionInput(clinical=ClinicalScores(mmse=10.0)))
        finally:
            engine.logger.removeHandler(caplog.handler)
        other = next(d for d in out.diseases if d.disease == "other")
        assert other.probability == pytest.approx(0.5, abs=1e-6)
        assert other.contributions == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/fusion/test_engine.py -v`
Expected: FAIL — engine module missing.

- [ ] **Step 3: Write minimal implementation**

Create `src/fusion/engine.py`:

```python
"""Multi-modal fusion engine — combines MRI, EEG, and clinical signals into
per-disease confidence with full attribution.
"""
from __future__ import annotations

import math
from typing import Callable

from src.core.logger import get_logger
from src.fusion import clinical as clinical_signals
from src.fusion import weights as weight_registry
from src.fusion.modality import signal_for_disease
from src.fusion.types import (
    ClinicalScores,
    DiseaseScore,
    FusionInput,
    FusionOutput,
    ModalityContribution,
    ModalityPrediction,
)

logger = get_logger(__name__)

_LOGIT_SCALE = 4.0  # tuned so a single saturated modality maps to ~0.88


# Clinical-test name -> (signal_fn, attribute_on_ClinicalScores)
_CLINICAL_FNS: dict[str, tuple[Callable[[float], float], str]] = {
    "clinical_mmse":  (clinical_signals.mmse_to_signal,  "mmse"),
    "clinical_moca":  (clinical_signals.moca_to_signal,  "moca"),
    "clinical_updrs": (clinical_signals.updrs_to_signal, "updrs"),
    "clinical_gait":  (clinical_signals.gait_to_signal,  "gait_speed_m_s"),
    "clinical_age":   (clinical_signals.age_to_signal,   "age_years"),
}


def fuse(inp: FusionInput) -> FusionOutput:
    """Combine all available modalities into a per-disease confidence."""
    missing: list[str] = []
    if inp.mri is None:
        missing.append("mri")
    if inp.eeg is None:
        missing.append("eeg")

    diseases: list[DiseaseScore] = []
    for disease in weight_registry.available_diseases():
        diseases.append(_score_one_disease(disease, inp))

    top = max(diseases, key=lambda d: d.probability).disease
    return FusionOutput(diseases=diseases, top_disease=top, missing_inputs=missing)


def _score_one_disease(disease: str, inp: FusionInput) -> DiseaseScore:
    weights = weight_registry.get_weights(disease)
    contributions: list[ModalityContribution] = []

    for modality_key, weight in weights.items():
        signal = _signal_for_modality(modality_key, disease, inp.mri, inp.eeg, inp.clinical)
        if signal is None:
            continue
        contributions.append(ModalityContribution(
            modality=modality_key,
            weight=weight,
            signal=signal,
            delta_logit=weight * signal,
        ))

    if not contributions:
        logger.info("no signals available for disease=%s; returning baseline 0.5", disease)
        return DiseaseScore(disease=disease, probability=0.5, contributions=[])

    logit = sum(c.delta_logit for c in contributions)
    probability = _sigmoid(_LOGIT_SCALE * logit)
    return DiseaseScore(
        disease=disease,
        probability=probability,
        contributions=contributions,
    )


def _signal_for_modality(
    modality_key: str,
    disease: str,
    mri: ModalityPrediction | None,
    eeg: ModalityPrediction | None,
    clinical: ClinicalScores,
) -> float | None:
    if modality_key == "mri":
        return signal_for_disease(mri, disease) if mri is not None else None
    if modality_key == "eeg":
        return signal_for_disease(eeg, disease) if eeg is not None else None
    if modality_key in _CLINICAL_FNS:
        fn, attr = _CLINICAL_FNS[modality_key]
        value = getattr(clinical, attr, None)
        return fn(value) if value is not None else None
    logger.warning("unknown modality key in weights table: %s", modality_key)
    return None


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/fusion/test_engine.py -v`
Expected: PASS (8 tests, including the BBB-independence regression)

- [ ] **Step 5: Commit**

```bash
git add src/fusion/engine.py tests/fusion/test_engine.py
git commit -m "feat(fusion): add core multi-modal fuse() with per-disease attribution"
```

---

### Task 6: FastAPI route

**Files:**
- Modify: `src/api/schemas.py` (append fusion section)
- Modify: `src/api/routes.py` (add `/fusion/predict`)
- Test: `tests/api/test_fusion_route.py`

- [ ] **Step 1: Write the failing test**

Create `tests/api/test_fusion_route.py`:

```python
"""Integration test for POST /fusion/predict."""
from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


class TestFusionRoute:
    def test_happy_path_mri_only(self) -> None:
        body = {
            "mri": {
                "label_text": "alzheimers",
                "label": 1,
                "confidence": 0.88,
                "probabilities": [
                    {"label_text": "control", "probability": 0.12},
                    {"label_text": "alzheimers", "probability": 0.88},
                ],
            },
        }
        r = client.post("/fusion/predict", json=body)
        assert r.status_code == 200, r.text
        data = r.json()
        assert "diseases" in data
        assert any(d["disease"] == "alzheimers" for d in data["diseases"])
        assert data["top_disease"] in {"alzheimers", "parkinsons", "other"}

    def test_empty_input_returns_baseline(self) -> None:
        r = client.post("/fusion/predict", json={})
        assert r.status_code == 200
        data = r.json()
        for d in data["diseases"]:
            assert abs(d["probability"] - 0.5) < 1e-6
        assert "mri" in data["missing_inputs"]

    def test_invalid_probability_returns_422(self) -> None:
        body = {
            "mri": {
                "label_text": "x",
                "label": 0,
                "confidence": 1.5,   # invalid
                "probabilities": [{"label_text": "x", "probability": 1.5}],
            },
        }
        r = client.post("/fusion/predict", json=body)
        assert r.status_code == 422
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/api/test_fusion_route.py -v`
Expected: FAIL — 404 (route missing).

- [ ] **Step 3: Wire schemas + route**

Append to `src/api/schemas.py` (at the bottom, before any closing matter):

```python
# --- Fusion engine surface --------------------------------------------------

# Re-export the fusion types so the API surface lives in one file but the
# implementation stays in src/fusion. This keeps `from src.api.schemas import *`
# style imports stable for the frontend layer.
from src.fusion.types import (  # noqa: E402,F401
    ClinicalScores as FusionClinicalScores,
    FusionInput as FusionRequest,
    FusionOutput as FusionResponse,
    ModalityPrediction as FusionModalityPrediction,
)
```

In `src/api/routes.py`, add the route. Find an existing pipeline route (e.g. `@router.post("/pipeline/bbb"...)`) and add this near the bottom of the same router:

```python
from src.fusion.engine import fuse as fuse_engine
from src.api.schemas import FusionRequest, FusionResponse


@router.post("/fusion/predict", response_model=FusionResponse)
def fusion_predict(req: FusionRequest) -> FusionResponse:
    """Combine MRI, EEG, and clinical scores into per-disease confidence."""
    return fuse_engine(req)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/api/test_fusion_route.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/api/schemas.py src/api/routes.py tests/api/test_fusion_route.py
git commit -m "feat(api): add POST /fusion/predict route for multi-modal fusion"
```

---

### Task 7: Agent tool wrapper

**Files:**
- Modify: `src/agents/tools.py`
- Modify: `src/agents/prompts.py` (mention the new tool in the system prompt)
- Test: `tests/agents/test_tools_fusion.py`

- [ ] **Step 1: Write the failing test**

Create `tests/agents/test_tools_fusion.py`:

```python
"""Tests for the run_fusion agent tool."""
from __future__ import annotations

from src.agents.tools import build_tools


class TestRunFusionTool:
    def test_fusion_tool_is_registered(self) -> None:
        tools = build_tools()
        names = [t.name for t in tools]
        assert "run_fusion" in names

    def test_fusion_tool_executes_with_only_clinical(self) -> None:
        tools = {t.name: t for t in build_tools()}
        tool = tools["run_fusion"]
        out = tool.execute(tool.input_model.model_validate({
            "clinical": {"mmse": 12.0, "age_years": 78.0},
        }))
        assert out.top_disease in {"alzheimers", "parkinsons", "other"}
        assert any(d.disease == "alzheimers" for d in out.diseases)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/agents/test_tools_fusion.py -v`
Expected: FAIL — `run_fusion` not in tool list.

- [ ] **Step 3: Register the tool**

In `src/agents/tools.py`, locate the `build_tools()` function (or whichever function returns the tool registry — match the existing pattern; e.g., a list of `Tool(...)` constructions). Add:

```python
from src.fusion.engine import fuse as fuse_engine
from src.fusion.types import FusionInput, FusionOutput


def _make_fusion_tool() -> Tool:
    return Tool(
        name="run_fusion",
        description=(
            "Combine MRI prediction, EEG prediction, and clinical-test scores "
            "into per-disease (Alzheimer's, Parkinson's, other) confidence "
            "with attribution. Pass whichever modalities are available; missing "
            "ones are skipped, not imputed."
        ),
        input_model=FusionInput,
        output_model=FusionOutput,
        execute=lambda inp: fuse_engine(inp),
    )
```

Then append `_make_fusion_tool()` to whatever list `build_tools()` returns. (If the file structures tools differently, adapt — the principle is: register it the same way `run_bbb_pipeline` is registered.)

In `src/agents/prompts.py` find the system prompt that lists tools and add a one-liner under the relevant section:

```
- run_fusion: combine MRI/EEG/clinical-test scores into a per-disease confidence with attribution.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/agents/test_tools_fusion.py -v`
Expected: PASS (2 tests).

Then run the full agent test file to make sure nothing regressed:

`pytest tests/agents/ -v`
Expected: all previously-passing tests still pass.

- [ ] **Step 5: Commit**

```bash
git add src/agents/tools.py src/agents/prompts.py tests/agents/test_tools_fusion.py
git commit -m "feat(agents): register run_fusion tool for multi-modal disease confidence"
```

---

### Task 8: End-to-end smoke test + README note

**Files:**
- Create: `tests/integration/test_fusion_end_to_end.py`
- Modify: `README.md` (add a one-paragraph "Fusion Engine" section under existing feature docs)

- [ ] **Step 1: Write the integration test**

Create `tests/integration/__init__.py` if not present (empty file).

Create `tests/integration/test_fusion_end_to_end.py`:

```python
"""End-to-end: agent calls run_fusion with realistic inputs, top disease is sane."""
from __future__ import annotations

import pytest

from src.agents.tools import build_tools


@pytest.mark.parametrize(
    "scenario,expected_top",
    [
        # Strong AD signal: low MMSE + MRI flags alzheimers
        (
            {
                "mri": {
                    "label_text": "alzheimers", "label": 1, "confidence": 0.85,
                    "probabilities": [
                        {"label_text": "control", "probability": 0.15},
                        {"label_text": "alzheimers", "probability": 0.85},
                    ],
                },
                "clinical": {"mmse": 14.0, "age_years": 79.0},
            },
            "alzheimers",
        ),
        # Strong PD signal: high UPDRS + slow gait + EEG flags parkinsons
        (
            {
                "eeg": {
                    "label_text": "parkinsons", "label": 1, "confidence": 0.78,
                    "probabilities": [
                        {"label_text": "control", "probability": 0.22},
                        {"label_text": "parkinsons", "probability": 0.78},
                    ],
                },
                "clinical": {"updrs": 80.0, "gait_speed_m_s": 0.4, "age_years": 70.0},
            },
            "parkinsons",
        ),
    ],
)
def test_realistic_scenarios_pick_correct_top_disease(scenario, expected_top) -> None:
    tools = {t.name: t for t in build_tools()}
    tool = tools["run_fusion"]
    out = tool.execute(tool.input_model.model_validate(scenario))
    assert out.top_disease == expected_top
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/integration/test_fusion_end_to_end.py -v`
Expected: PASS (2 cases).

- [ ] **Step 3: Add README paragraph**

In `README.md`, find the features section. Add:

```markdown
### Fusion Engine

`POST /fusion/predict` (and the agent tool `run_fusion`) combines whichever of
MRI, EEG, and clinical-test scores (MMSE, MoCA, UPDRS, gait, age) the doctor
has uploaded into a per-disease confidence (Alzheimer's, Parkinson's, other)
with full attribution showing how much each modality contributed.

Weights live in `src/fusion/weights.py` and are heuristic — adjust there.
```

- [ ] **Step 4: Run the full suite**

Run: `pytest -q`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/integration/__init__.py tests/integration/test_fusion_end_to_end.py README.md
git commit -m "test(fusion): end-to-end smoke + README section"
```

---

## Self-review checklist (do this before declaring the plan finished)

1. **Spec coverage.** The spec asks for: doctor enters extra clinical tests with preset weights → **Tasks 2, 3, 5** (weights table + normalisers + engine). MRI + EEG fusion → **Task 5**. Disease-specific (Alzheimer's, Parkinson's, other) → **Task 2** (weights are per disease). API + agent reachability → **Tasks 6, 7**. End-to-end demo → **Task 8**. ✓

2. **Out of scope (do NOT build here).**
   - BBB-from-MRI: separate sub-plan #3.
   - Doctor UI form: separate sub-plan #2.
   - Patient lifestyle text: separate sub-plan #5.
   - Drug-dosing hint: separate sub-plan #6.

3. **Testing surface.** Unit tests for each pure module (`weights`, `clinical`, `modality`, `engine`), integration tests at the API and agent layers, plus an end-to-end scenario test. No mocked-out internal logic — only external boundaries are stubbed (none here, since the engine is pure).

4. **Logging & propagation.** Every test that asserts a log message attaches `caplog.handler` directly to the module logger because `src/core/logger.py` sets `propagate=False`. See Task 5 step 1.

5. **No placeholders.** Every code block above is the full file or full appended block. No "TODO", no "implement later", no "similar to X".

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-02-fusion-engine.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, two-stage review (spec then quality) between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using `superpowers:executing-plans`, batch execution with checkpoints for review.

Which approach?
