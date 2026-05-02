# OASIS Tabular Classifier — Fusion Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: `superpowers:subagent-driven-development`. TDD throughout.

## ⚠️ Important context — read before executing

The user said "I have the pretrained model for eeg, integrate it into the eeg pipeline. its the ipynb file named detecting-early-alzheimers...".

The notebook (`/Users/mertgungor/Downloads/rag/detecting-early-alzheimer-s (1).ipynb`) is **NOT an EEG model**. It is an sklearn ensemble (LogReg / SVM / DT / RF / AdaBoost) trained on the OASIS longitudinal **tabular** dataset — features are MMSE, eTIV, nWBV, ASF, EDUC, SES, M/F, Age. Zero EEG signal processing. Zero saved model artifact (the notebook trains in-memory only).

This plan therefore has **two branches**. Pick one with the user before executing.

### Branch 3a — Train + integrate the OASIS *tabular* classifier as a fusion feature

We re-train the best variant (Random Forest, AUC 84.4 % per the notebook) from the OASIS CSV, save a `joblib` artifact, and expose it as a fusion-engine modality named `tabular_oasis`. The fusion engine already handles arbitrary modality keys; this plugs in cleanly.

**Demo value:** When a doctor has only OASIS-style biomarkers (MMSE / eTIV / nWBV / ASF / Age / EDUC / SES / M/F) but no MRI image, the fusion engine still produces an Alzheimer's confidence with attribution.

### Branch 3b — User has a real EEG model elsewhere

If the user can point us to a checkpoint that consumes raw FIF / EDF EEG data (e.g., a `.pt`, `.pth`, `.h5`, `.onnx`, or `.joblib` file) and emits Alzheimer's class probabilities, this plan is rewritten around that artifact: signature, expected input shape, label order. We replace `src/models/eeg_model.py` (currently absent — `eeg_pipeline.py` only does signal processing) with a new module similar to `mri_dl_2d.py`.

**The user must pick a branch** before any task starts. The default below is **Branch 3a**, because the notebook is what's actually on disk.

---

## Branch 3a (default): OASIS tabular classifier as fusion modality

**Goal.** Save a Random Forest trained on OASIS biomarkers; wire it into the fusion engine as a new modality `tabular_oasis`. The doctor enters MMSE/eTIV/nWBV/ASF (fusion already takes MMSE; this extends to the other three) and gets an Alzheimer's signal that flows through the existing logit/sigmoid combiner.

**Architecture.** New module `src/models/tabular_oasis.py` trains-or-loads a `joblib`-pickled `Pipeline(scaler -> RandomForestClassifier)`. The fusion engine grows one entry in `_CLINICAL_FNS` (or, more cleanly, a sibling `_TABULAR_FNS`) so the model's class probability for `Demented=1` becomes a signed signal. New API route `POST /predict/tabular_oasis` lets the frontend call it directly. All optional — if the OASIS CSV is absent, the module degrades gracefully and fusion ignores the modality.

**Tech stack.** scikit-learn (already in deps), pandas, joblib (likely in deps via sklearn).

---

## Prerequisite (controller blocker)

The OASIS dataset is not in this repo. Two acquisition options:

1. **Download from Kaggle** (https://www.kaggle.com/datasets/jboysen/mri-and-alzheimers, file `oasis_longitudinal.csv`). Save to `data/external/oasis_longitudinal.csv`. Gitignore (already covered by `data/external_rag/` if you broaden it; otherwise add `data/external/`).

2. **Use a local copy** if the user already downloaded it for the notebook. Same destination.

If the dataset is unavailable, **stop and surface to the user**. The classifier cannot be trained without it; we will not fabricate synthetic OASIS-shaped data for a clinical demo.

---

## File structure

| Path | Responsibility |
|---|---|
| Modify `requirements.txt` | confirm `joblib` (sklearn pulls it transitively but pin explicitly is safer) |
| Modify `.gitignore` | ensure `data/external/` is ignored |
| Create `src/models/tabular_oasis.py` | train + persist + load + predict the OASIS RF classifier |
| Create `scripts/train_oasis.py` | one-shot CLI: trains and saves the model artifact |
| Modify `src/fusion/types.py` | extend `ClinicalScores` with `etiv`, `nwbv`, `asf`, `educ`, `ses`, `is_male` |
| Modify `src/fusion/weights.py` | add `tabular_oasis` weight key for `alzheimers` |
| Modify `src/fusion/engine.py` | add `tabular_oasis` to the modality dispatch |
| Modify `src/api/routes.py` | new route `POST /predict/tabular_oasis` |
| Modify `src/api/schemas.py` | request/response for the new route |
| Create `tests/models/test_tabular_oasis.py` | training + persistence + prediction tests |
| Create `tests/fixtures/build_synthetic_oasis.py` | synthetic OASIS-shaped CSV for tests (clearly labelled non-clinical) |
| Create `tests/fusion/test_tabular_oasis_modality.py` | fusion-side integration |
| Create `tests/api/test_tabular_oasis_route.py` | API integration |
| Modify `README.md` | document the modality + how to acquire the OASIS CSV |

---

## Tasks

### Task 0: Deps + ignore

**Files:** `requirements.txt`, `.gitignore`

- [ ] **Step 1:** verify `joblib` and `pandas` are in `requirements.txt`. `pandas` already is (used by every pipeline). Add `joblib>=1.3,<2.0` if not pinned.

- [ ] **Step 2:** `.gitignore` should cover `data/external/`. Add it if needed.

- [ ] **Step 3:** `pytest -q` baseline. Commit: `chore(oasis): pin joblib; gitignore external dataset dir`.

---

### Task 1: Training + persistence module

**Files:**
- Create: `src/models/tabular_oasis.py`
- Create: `scripts/train_oasis.py`
- Create: `tests/fixtures/build_synthetic_oasis.py`
- Create: `tests/models/test_tabular_oasis.py`

- [ ] **Step 1: Synthetic-fixture helper** (clearly synthetic — never confused with real clinical data):

`tests/fixtures/build_synthetic_oasis.py`:

```python
"""Build a synthetic OASIS-shaped CSV for tests. NON-CLINICAL data."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def build(path: Path, n: int = 200, seed: int = 42) -> Path:
    """Save a synthetic CSV at `path` with the columns the trainer expects."""
    path = Path(path)
    if path.exists():
        return path
    rng = np.random.default_rng(seed)
    n_dem = n // 2

    # Demented half — lower MMSE, higher CDR, smaller nWBV.
    dem = pd.DataFrame({
        "Group":   ["Demented"] * n_dem,
        "M/F":     rng.choice(["M", "F"], n_dem),
        "Age":     rng.integers(70, 95, n_dem),
        "EDUC":    rng.integers(8, 18, n_dem),
        "SES":     rng.integers(1, 5, n_dem),
        "MMSE":    rng.integers(15, 26, n_dem),
        "CDR":     rng.choice([0.5, 1.0], n_dem),
        "eTIV":    rng.integers(1200, 1700, n_dem),
        "nWBV":    rng.uniform(0.65, 0.74, n_dem),
        "ASF":     rng.uniform(1.0, 1.4, n_dem),
        "Visit":   1,
        "Hand":    "R",
    })
    nondem = pd.DataFrame({
        "Group":   ["Nondemented"] * (n - n_dem),
        "M/F":     rng.choice(["M", "F"], n - n_dem),
        "Age":     rng.integers(60, 90, n - n_dem),
        "EDUC":    rng.integers(10, 22, n - n_dem),
        "SES":     rng.integers(1, 5, n - n_dem),
        "MMSE":    rng.integers(26, 31, n - n_dem),
        "CDR":     rng.choice([0.0], n - n_dem),
        "eTIV":    rng.integers(1300, 1900, n - n_dem),
        "nWBV":    rng.uniform(0.70, 0.83, n - n_dem),
        "ASF":     rng.uniform(0.9, 1.5, n - n_dem),
        "Visit":   1,
        "Hand":    "R",
    })

    pd.concat([dem, nondem], ignore_index=True).to_csv(path, index=False)
    return path
```

- [ ] **Step 2: Failing test.**

`tests/models/test_tabular_oasis.py`:

```python
"""Tests for src.models.tabular_oasis."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.models import tabular_oasis
from tests.fixtures.build_synthetic_oasis import build as build_synth


class TestTrainAndPredict:
    def test_train_persists_loadable_artifact(self, tmp_path: Path) -> None:
        csv = build_synth(tmp_path / "oasis.csv")
        artifact = tabular_oasis.train_from_csv(csv, tmp_path / "rf.joblib")
        assert artifact.exists()
        loaded = tabular_oasis.load(artifact)
        assert hasattr(loaded, "predict_proba")

    def test_predict_returns_full_dict(self, tmp_path: Path) -> None:
        csv = build_synth(tmp_path / "oasis.csv")
        artifact = tabular_oasis.train_from_csv(csv, tmp_path / "rf.joblib")
        model = tabular_oasis.load(artifact)
        out = tabular_oasis.predict_one(model, {
            "is_male": 1, "age": 80, "educ": 10, "ses": 3.0,
            "mmse": 18.0, "etiv": 1500.0, "nwbv": 0.68, "asf": 1.2,
        })
        assert set(out) == {"label", "label_text", "confidence", "probabilities"}
        assert out["label"] in {0, 1}
        assert out["label_text"] in {"Nondemented", "Demented"}
        assert 0.0 <= out["confidence"] <= 1.0
        probs = out["probabilities"]
        assert len(probs) == 2
        assert abs(sum(p["probability"] for p in probs) - 1.0) < 1e-5

    def test_predict_with_synthetic_demented_profile_yields_demented_label(self, tmp_path: Path) -> None:
        # The synthetic data has clean separation, so a clearly-demented profile
        # (MMSE=15, low nWBV, age 88) should classify as Demented.
        csv = build_synth(tmp_path / "oasis.csv")
        artifact = tabular_oasis.train_from_csv(csv, tmp_path / "rf.joblib")
        model = tabular_oasis.load(artifact)
        out = tabular_oasis.predict_one(model, {
            "is_male": 1, "age": 88, "educ": 8, "ses": 3.0,
            "mmse": 15.0, "etiv": 1300.0, "nwbv": 0.66, "asf": 1.3,
        })
        assert out["label_text"] == "Demented"

    def test_load_missing_artifact_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="OASIS classifier artifact not found"):
            tabular_oasis.load(tmp_path / "missing.joblib")
```

Run → ImportError.

- [ ] **Step 3: Minimal impl.**

`src/models/tabular_oasis.py`:

```python
"""OASIS tabular Alzheimer's classifier — Random Forest with full pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from src.core.logger import get_logger

logger = get_logger(__name__)

FEATURE_ORDER: tuple[str, ...] = (
    "is_male", "age", "educ", "ses", "mmse", "etiv", "nwbv", "asf",
)
LABEL_NAMES: tuple[str, ...] = ("Nondemented", "Demented")


def _df_from_oasis_csv(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Replicate the notebook's preprocessing: first visit only, M/F encoded,
    Converted-as-Demented, drop unused columns, median-impute SES on EDUC."""
    df = pd.read_csv(csv_path)
    df = df.loc[df["Visit"] == 1].reset_index(drop=True)
    df["M/F"] = df["M/F"].replace({"F": 0, "M": 1})
    df["Group"] = df["Group"].replace({"Converted": "Demented"}).replace(
        {"Demented": 1, "Nondemented": 0}
    )
    df = df.drop(columns=[c for c in ("MRI ID", "Visit", "Hand") if c in df.columns])
    df["SES"] = df["SES"].fillna(df.groupby("EDUC")["SES"].transform("median"))

    feature_df = pd.DataFrame({
        "is_male": df["M/F"].astype(float),
        "age":     df["Age"].astype(float),
        "educ":    df["EDUC"].astype(float),
        "ses":     df["SES"].astype(float),
        "mmse":    df["MMSE"].astype(float),
        "etiv":    df["eTIV"].astype(float),
        "nwbv":    df["nWBV"].astype(float),
        "asf":     df["ASF"].astype(float),
    })[list(FEATURE_ORDER)]
    return feature_df, df["Group"].astype(int)


def train_from_csv(csv_path: Path, artifact_path: Path) -> Path:
    """Train and persist a MinMaxScaler→RandomForest pipeline. Returns artifact path."""
    csv_path = Path(csv_path)
    artifact_path = Path(artifact_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"OASIS CSV not found: {csv_path}")

    X, y = _df_from_oasis_csv(csv_path)
    pipeline = Pipeline([
        ("scaler", MinMaxScaler()),
        ("rf",     RandomForestClassifier(
            n_estimators=12, max_depth=8, max_features=8,
            n_jobs=4, random_state=0,
        )),
    ])
    pipeline.fit(X, y)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, artifact_path)
    logger.info("trained OASIS RF: n=%d, artifact=%s", len(X), artifact_path)
    return artifact_path


def load(artifact_path: Path) -> Pipeline:
    p = Path(artifact_path)
    if not p.exists():
        raise FileNotFoundError(f"OASIS classifier artifact not found: {p}")
    return joblib.load(p)


def predict_one(model: Pipeline, features: dict[str, float]) -> dict[str, Any]:
    """Predict for a single subject. `features` must have all FEATURE_ORDER keys."""
    missing = [k for k in FEATURE_ORDER if k not in features]
    if missing:
        raise ValueError(f"OASIS prediction missing features: {missing}")
    row = pd.DataFrame([{k: float(features[k]) for k in FEATURE_ORDER}])
    probs = np.asarray(model.predict_proba(row))[0]
    label_idx = int(np.argmax(probs))
    return {
        "label": label_idx,
        "label_text": LABEL_NAMES[label_idx],
        "confidence": float(probs[label_idx]),
        "probabilities": [
            {"label": i, "label_text": LABEL_NAMES[i], "probability": float(p)}
            for i, p in enumerate(probs)
        ],
    }
```

`scripts/train_oasis.py`:

```python
"""CLI: train the OASIS RF classifier and save it.

Usage:
    python scripts/train_oasis.py data/external/oasis_longitudinal.csv data/processed/oasis_rf.joblib
"""
from __future__ import annotations

import sys
from pathlib import Path

from src.models.tabular_oasis import train_from_csv


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    csv = Path(sys.argv[1])
    out = Path(sys.argv[2])
    train_from_csv(csv, out)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
```

Run tests → 4 passed.

- [ ] **Step 4:** commit: `feat(models): OASIS tabular Alzheimer's RF classifier (joblib + train CLI)`.

---

### Task 2: Extend fusion's clinical inputs

**Files:**
- Modify: `src/fusion/types.py` (extend `ClinicalScores`)
- Modify: `src/fusion/clinical.py` (add normalisers for the new fields)
- Modify: `tests/fusion/test_types.py` (loosen / extend bound tests)
- Modify: `tests/fusion/test_clinical.py` (add new normaliser tests)

- [ ] **Step 1: Failing test for new ClinicalScores fields.**

In `tests/fusion/test_types.py`, append:

```python
class TestExtendedClinicalScores:
    def test_etiv_in_range(self) -> None:
        s = ClinicalScores(etiv=1500.0)
        assert s.etiv == pytest.approx(1500.0)

    def test_etiv_out_of_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ClinicalScores(etiv=5000.0)

    def test_nwbv_in_range(self) -> None:
        s = ClinicalScores(nwbv=0.72)
        assert s.nwbv == pytest.approx(0.72)
```

- [ ] **Step 2: Update `src/fusion/types.py` ClinicalScores.**

Add fields (preserve existing ones):

```python
class ClinicalScores(BaseModel):
    mmse: Annotated[float, Field(ge=0.0, le=30.0)] | None = None
    moca: Annotated[float, Field(ge=0.0, le=30.0)] | None = None
    updrs: Annotated[float, Field(ge=0.0, le=199.0)] | None = None
    gait_speed_m_s: Annotated[float, Field(ge=0.0, le=2.5)] | None = None
    age_years: Annotated[float, Field(ge=0.0, le=120.0)] | None = None
    # OASIS biomarkers — used by the tabular_oasis modality.
    etiv: Annotated[float, Field(ge=900.0, le=2200.0)] | None = None
    nwbv: Annotated[float, Field(ge=0.5, le=0.95)] | None = None
    asf:  Annotated[float, Field(ge=0.5, le=2.0)]  | None = None
    educ: Annotated[float, Field(ge=0.0, le=30.0)] | None = None
    ses:  Annotated[float, Field(ge=1.0, le=5.0)]  | None = None
    is_male: Annotated[int, Field(ge=0, le=1)]     | None = None
```

- [ ] **Step 3:** the tests should pass after the type change. `pytest tests/fusion/test_types.py -v`.

- [ ] **Step 4:** commit: `feat(fusion): extend ClinicalScores with OASIS biomarker fields`.

---

### Task 3: Wire `tabular_oasis` modality into the fusion engine

**Files:**
- Modify: `src/fusion/weights.py`
- Modify: `src/fusion/engine.py`
- Create: `tests/fusion/test_tabular_oasis_modality.py`

- [ ] **Step 1: Update weights.**

`src/fusion/weights.py`, in the `alzheimers` table:

```python
"alzheimers": {
    "mri":              0.25,   # was 0.35
    "eeg":              0.15,   # was 0.20
    "tabular_oasis":    0.20,   # new
    "clinical_mmse":    0.20,
    "clinical_moca":    0.10,   # was 0.15
    "clinical_age":     0.10,
},
```

Re-balance so the table still sums to 1.0. Add a comment that re-balancing changed the existing tests' tolerances — verify which tests need updating.

- [ ] **Step 2: Failing fusion-modality test.**

`tests/fusion/test_tabular_oasis_modality.py`:

```python
"""Tests: tabular_oasis modality contributes to alzheimers fusion score."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.fusion import engine
from src.fusion.types import ClinicalScores, FusionInput
from src.models.tabular_oasis import train_from_csv
from tests.fixtures.build_synthetic_oasis import build as build_synth


@pytest.fixture()
def trained_artifact(tmp_path: Path, monkeypatch) -> Path:
    csv = build_synth(tmp_path / "oasis.csv")
    art = train_from_csv(csv, tmp_path / "rf.joblib")
    monkeypatch.setenv("OASIS_RF_ARTIFACT", str(art))
    return art


class TestTabularOasisModality:
    def test_demented_profile_raises_alzheimers(self, trained_artifact: Path) -> None:
        out = engine.fuse(FusionInput(clinical=ClinicalScores(
            is_male=1, age_years=88, educ=8, ses=3.0,
            mmse=15.0, etiv=1300.0, nwbv=0.66, asf=1.3,
        )))
        alz = next(d for d in out.diseases if d.disease == "alzheimers")
        assert alz.probability > 0.6
        assert any(c.modality == "tabular_oasis" for c in alz.contributions)

    def test_missing_oasis_inputs_skips_modality(self, trained_artifact: Path) -> None:
        # MMSE alone but no etiv/nwbv → tabular_oasis should be skipped, not error.
        out = engine.fuse(FusionInput(clinical=ClinicalScores(mmse=12.0)))
        alz = next(d for d in out.diseases if d.disease == "alzheimers")
        names = {c.modality for c in alz.contributions}
        assert "tabular_oasis" not in names
```

- [ ] **Step 3: Update the engine.**

In `src/fusion/engine.py`, add a tabular-modality dispatcher that lazy-loads the joblib artifact once and treats the OASIS classifier's `P(Demented)` as the alzheimers signal `2*P-1`:

```python
import os

_oasis_cache: dict[str, Any] = {}


def _signal_for_tabular_oasis(disease: str, clinical: ClinicalScores) -> float | None:
    if disease != "alzheimers":
        return None
    required = ("is_male", "age_years", "educ", "ses", "mmse", "etiv", "nwbv", "asf")
    if any(getattr(clinical, k, None) is None for k in required):
        return None
    artifact = os.environ.get("OASIS_RF_ARTIFACT", "data/processed/oasis_rf.joblib")
    artifact_path = Path(artifact)
    if not artifact_path.exists():
        logger.warning("tabular_oasis artifact missing at %s; skipping modality", artifact_path)
        return None
    if "model" not in _oasis_cache:
        from src.models.tabular_oasis import load
        _oasis_cache["model"] = load(artifact_path)
    from src.models.tabular_oasis import predict_one
    feats = {
        "is_male": int(clinical.is_male),
        "age":     float(clinical.age_years),
        "educ":    float(clinical.educ),
        "ses":     float(clinical.ses),
        "mmse":    float(clinical.mmse),
        "etiv":    float(clinical.etiv),
        "nwbv":    float(clinical.nwbv),
        "asf":     float(clinical.asf),
    }
    pred = predict_one(_oasis_cache["model"], feats)
    p_dem = next(p["probability"] for p in pred["probabilities"] if p["label_text"] == "Demented")
    return 2.0 * p_dem - 1.0
```

In `_signal_for_modality`, add the dispatch:

```python
if modality_key == "tabular_oasis":
    return _signal_for_tabular_oasis(disease, clinical)
```

- [ ] **Step 4:** `pytest tests/fusion/ -v` — expect re-balancing to perturb a couple of existing thresholds. Adjust thresholds in the affected tests (e.g., the disagreement test) so they still hold with the new weights, OR adjust the new weights so existing tests still pass within tolerance. Prefer the latter — existing thresholds were chosen carefully.

- [ ] **Step 5:** commit: `feat(fusion): add tabular_oasis modality with lazy joblib load`.

---

### Task 4: API + Streamlit + README

**Files:**
- Modify: `src/api/routes.py` — add `POST /predict/tabular_oasis`
- Modify: `src/api/schemas.py` — request/response schemas
- Modify: `src/frontend/app.py` — extend the Doctor view's clinical-input form with eTIV / nWBV / ASF / EDUC / SES
- Modify: `README.md` — describe the new modality and the OASIS dataset path

- [ ] **Step 1: New schemas.**

`src/api/schemas.py`:

```python
class TabularOasisRequest(BaseModel):
    is_male: int = Field(..., ge=0, le=1)
    age: float = Field(..., ge=0.0, le=120.0)
    educ: float = Field(..., ge=0.0, le=30.0)
    ses: float = Field(..., ge=1.0, le=5.0)
    mmse: float = Field(..., ge=0.0, le=30.0)
    etiv: float = Field(..., ge=900.0, le=2200.0)
    nwbv: float = Field(..., ge=0.5, le=0.95)
    asf: float = Field(..., ge=0.5, le=2.0)


class TabularOasisProbability(BaseModel):
    label: int
    label_text: str
    probability: float


class TabularOasisResponse(BaseModel):
    label: int
    label_text: str
    confidence: float
    probabilities: list[TabularOasisProbability]
```

- [ ] **Step 2: Route.**

`src/api/routes.py`:

```python
@predict_router.post("/tabular_oasis", response_model=TabularOasisResponse)
def predict_tabular_oasis(req: TabularOasisRequest) -> TabularOasisResponse:
    from src.models.tabular_oasis import load, predict_one
    artifact = Path(os.environ.get("OASIS_RF_ARTIFACT", "data/processed/oasis_rf.joblib"))
    model = load(artifact)
    out = predict_one(model, req.model_dump())
    return TabularOasisResponse(**out)
```

- [ ] **Step 3: Test (`tests/api/test_tabular_oasis_route.py`).**

```python
"""Integration: POST /predict/tabular_oasis."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.models.tabular_oasis import train_from_csv
from tests.fixtures.build_synthetic_oasis import build as build_synth


@pytest.fixture()
def client(monkeypatch, tmp_path):
    csv = build_synth(tmp_path / "oasis.csv")
    artifact = train_from_csv(csv, tmp_path / "rf.joblib")
    monkeypatch.setenv("OASIS_RF_ARTIFACT", str(artifact))
    return TestClient(app)


def test_predict_tabular_oasis_demented_profile(client):
    body = {
        "is_male": 1, "age": 88, "educ": 8, "ses": 3.0,
        "mmse": 15.0, "etiv": 1300.0, "nwbv": 0.66, "asf": 1.3,
    }
    r = client.post("/predict/tabular_oasis", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["label_text"] == "Demented"
```

- [ ] **Step 4:** Streamlit form extension. In `src/frontend/app.py`, find the clinical-inputs section the doctor view exposes (likely under a "Clinical scores" expander; if absent, add it under the fusion tab). Add number_input widgets for the seven new fields (`is_male`, `age`, `educ`, `ses`, `etiv`, `nwbv`, `asf`) that flow into the existing `/fusion/predict` payload's `clinical` block.

- [ ] **Step 5:** README. Append:

```markdown
### OASIS Tabular Alzheimer's Classifier

A scikit-learn Random Forest trained on the OASIS longitudinal dataset (https://www.oasis-brains.org/) classifies Demented vs Nondemented from 8 biomarkers (sex, age, education, SES, MMSE, eTIV, nWBV, ASF). It contributes to the fusion engine as modality `tabular_oasis` (weight 0.20 for Alzheimer's).

To use: download `oasis_longitudinal.csv` from Kaggle, save to `data/external/oasis_longitudinal.csv`, then:

```bash
python scripts/train_oasis.py data/external/oasis_longitudinal.csv data/processed/oasis_rf.joblib
export OASIS_RF_ARTIFACT=data/processed/oasis_rf.joblib
```

The fusion engine and `POST /predict/tabular_oasis` will pick it up. If the artifact is missing, the modality is skipped — fusion still works.
```

- [ ] **Step 6:** commit: `feat(oasis): /predict/tabular_oasis route + Streamlit form + README`.

---

## Self-review checklist

1. **Independence.** OASIS classifier and fusion remain decoupled when the artifact is absent (`OASIS_RF_ARTIFACT` unset → modality skipped). ✓
2. **No real-data fabrication.** Tests use a clearly-labelled synthetic CSV. The real OASIS dataset is never committed. ✓
3. **Backward compatibility.** Existing `ClinicalScores` fields untouched. New fields are all `Optional`. ✓
4. **Branch 3a vs 3b.** This plan is Branch 3a. If the user picks Branch 3b, this plan is replaced wholesale.

---

## Execution handoff

Save and choose: subagent-driven (recommended) or inline executing-plans.

**Reminder to controller:** before starting any task, confirm with the user: "Do you have a real EEG checkpoint I'm missing, or shall I proceed with Branch 3a (OASIS tabular Alzheimer's classifier)?"
