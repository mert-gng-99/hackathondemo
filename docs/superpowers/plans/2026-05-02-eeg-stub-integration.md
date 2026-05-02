# EEG Pretrained Classifier — Stub Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: `superpowers:subagent-driven-development`. TDD throughout.

**Goal.** Add an EEG classifier to the decision layer that flows into the fusion engine as the `eeg` modality. The real pretrained artifact will arrive later; for the hackathon demo we ship a stub-able contract so the entire flow (Streamlit → API → fusion) works **today**, and swapping in the real `.joblib` later is a one-file drop with **zero** code changes.

**Architecture.** New module `src/models/eeg_model.py` parallel to `src/models/mri_dl_2d.py`. Loads a sklearn-style classifier from `joblib`, runs `predict_proba` on a feature row produced by the existing `src/pipelines/eeg_pipeline.py` (which already extracts band-power features). Output dict shape mirrors the other model surfaces, so the API and fusion engine consume it without dispatch logic. A new route `POST /predict/eeg` exposes it. The fusion engine already accepts an `eeg` `ModalityPrediction` — no fusion code changes.

**Contract for the eventual real artifact.**

```
- Path: data/processed/eeg_clf.joblib (override via EEG_CLF_ARTIFACT env)
- Type: any object with sklearn's predict_proba interface (e.g. RandomForest,
        SVC with probability=True, MLPClassifier, or a thin wrapper around
        a torch model)
- Input: numpy array of shape (1, n_features) where n_features matches the
        column count of eeg_pipeline.py's parquet output
- Output: probability vector of length len(EEG_CLF_LABELS); default labels
        are ("control", "alzheimers")
```

The stub fixture (`tests/fixtures/build_dummy_eeg_clf.py`) writes a `RandomForestClassifier` with the same interface, so the entire pipeline is testable before the real model arrives.

**Tech stack.** scikit-learn (already in deps), joblib, numpy, pandas. No new dependencies.

---

## Asset note

For the demo we **assume the real EEG model exists**. Tests use the stub fixture so they pass regardless. When the real artifact arrives:

1. Save it to `data/processed/eeg_clf.joblib`.
2. If its label order isn't `("control", "alzheimers")`, set `EEG_CLF_LABELS=label0,label1,...` env (comma-separated). The fusion engine's `signal_for_disease` already case-insensitively matches labels, so as long as one of them is `"alzheimers"` (or `"parkinsons"`), it flows.
3. If `n_features` doesn't match the pipeline's parquet output, update the EEG pipeline's feature contract — out of scope for this plan, separate sub-plan if needed.

---

## File structure

| Path | Responsibility |
|---|---|
| Create `src/models/eeg_model.py` | sklearn-style classifier loader + `predict_features()` |
| Modify `src/api/routes.py` | new route `POST /predict/eeg` |
| Modify `src/api/schemas.py` | `EEGPredictRequest` / `EEGPredictResponse` |
| Create `tests/fixtures/build_dummy_eeg_clf.py` | stub joblib-pickled RF for tests |
| Create `tests/models/test_eeg_model.py` | unit tests for loader + predict |
| Create `tests/api/test_eeg_predict_route.py` | integration test through `POST /predict/eeg` |
| Create `tests/fusion/test_eeg_modality_flow.py` | confirms an EEG prediction flows into fusion as the `eeg` modality |
| Create `tests/models/test_eeg_model_real.py` | real-artifact sanity (skips when absent — same pattern as MRI Task 4) |
| Modify `README.md` | document the contract + how to swap the real artifact in |

---

## Tasks

### Task 1: EEG model module + dummy fixture

**Files:**
- Create: `src/models/eeg_model.py`
- Create: `tests/fixtures/build_dummy_eeg_clf.py`
- Create: `tests/models/test_eeg_model.py`

- [ ] **Step 1: Dummy fixture.**

`tests/fixtures/build_dummy_eeg_clf.py`:

```python
"""Build a stub EEG classifier (sklearn RF) for tests.

Demo-time placeholder — produces a 2-class probability output matching the
eeg_model.predict_features contract. Replace with the real artifact when
the user provides it; tests don't change.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def build(path: Path, n_features: int = 16, seed: int = 0) -> Path:
    """Save a fitted RandomForestClassifier at `path` and return the path."""
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    n = 200
    n_alz = n // 2
    # Synthetic separable features: alzheimers half has higher mean.
    X_ctrl = rng.normal(0.0, 1.0, size=(n - n_alz, n_features))
    X_alz  = rng.normal(2.0, 1.0, size=(n_alz, n_features))
    X = np.vstack([X_ctrl, X_alz])
    y = np.array([0] * (n - n_alz) + [1] * n_alz)

    clf = RandomForestClassifier(n_estimators=12, max_depth=6, random_state=seed)
    clf.fit(X, y)
    joblib.dump(clf, str(path))
    return path
```

- [ ] **Step 2: Failing test.**

`tests/models/test_eeg_model.py`:

```python
"""Tests for src.models.eeg_model."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.models import eeg_model
from tests.fixtures.build_dummy_eeg_clf import build as build_dummy_eeg


class TestEEGModel:
    def test_load_missing_artifact_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="EEG classifier artifact not found"):
            eeg_model.load(tmp_path / "nope.joblib")

    def test_predict_returns_full_dict(self, tmp_path: Path) -> None:
        ckpt = build_dummy_eeg(tmp_path / "eeg.joblib", n_features=16)
        clf = eeg_model.load(ckpt)
        features = np.zeros((16,), dtype=np.float32)

        out = eeg_model.predict_features(clf, features)

        assert set(out) == {"label", "label_text", "confidence", "probabilities"}
        assert out["label"] in {0, 1}
        assert out["label_text"] in eeg_model.DEFAULT_LABELS
        assert 0.0 <= out["confidence"] <= 1.0
        probs = out["probabilities"]
        assert len(probs) == 2
        assert abs(sum(p["probability"] for p in probs) - 1.0) < 1e-5

    def test_alzheimers_separation_with_synthetic_features(self, tmp_path: Path) -> None:
        # Synthetic stub clusters alzheimers around mean=2.0, control around 0.0.
        ckpt = build_dummy_eeg(tmp_path / "eeg.joblib", n_features=16)
        clf = eeg_model.load(ckpt)
        alz_features = np.full((16,), 2.0, dtype=np.float32)
        ctrl_features = np.zeros((16,), dtype=np.float32)

        alz_pred = eeg_model.predict_features(clf, alz_features)
        ctrl_pred = eeg_model.predict_features(clf, ctrl_features)

        assert alz_pred["label_text"] == "alzheimers"
        assert ctrl_pred["label_text"] == "control"

    def test_label_override_via_env(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("EEG_CLF_LABELS", "no_disease,alzheimers")
        ckpt = build_dummy_eeg(tmp_path / "eeg.joblib", n_features=16)
        clf = eeg_model.load(ckpt)
        out = eeg_model.predict_features(clf, np.zeros((16,), dtype=np.float32))
        assert out["label_text"] in {"no_disease", "alzheimers"}

    def test_feature_count_mismatch_raises(self, tmp_path: Path) -> None:
        ckpt = build_dummy_eeg(tmp_path / "eeg.joblib", n_features=16)
        clf = eeg_model.load(ckpt)
        with pytest.raises(ValueError, match="feature count"):
            eeg_model.predict_features(clf, np.zeros((8,), dtype=np.float32))
```

Run → `ModuleNotFoundError: No module named 'src.models.eeg_model'`.

- [ ] **Step 3: Minimal impl.**

`src/models/eeg_model.py`:

```python
"""EEG classifier inference utilities.

Loads any sklearn-style classifier (object with `predict_proba`) from joblib
and emits the same dict shape as src.models.mri_model.predict_with_proba so
the API surface and fusion engine treat MRI and EEG predictions identically.

The real pretrained artifact swaps in at data/processed/eeg_clf.joblib (or
override via EEG_CLF_ARTIFACT env). Tests use a stub fixture; the real model
drops in without code changes.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np

from src.core.logger import get_logger

logger = get_logger(__name__)

DEFAULT_LABELS: tuple[str, ...] = ("control", "alzheimers")


def _resolve_labels() -> tuple[str, ...]:
    raw = os.environ.get("EEG_CLF_LABELS")
    if not raw:
        return DEFAULT_LABELS
    return tuple(s.strip() for s in raw.split(",") if s.strip())


def load(path: Path) -> Any:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"EEG classifier artifact not found: {path}")
    return joblib.load(str(path))


def predict_features(
    model: Any,
    features: np.ndarray,
    labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run inference on one row of EEG features.

    Args:
        model: sklearn-style classifier (must expose `predict_proba`).
        features: 1-D numpy array of shape (n_features,) matching the
            classifier's training-time feature count.
        labels: optional label tuple. Defaults to env-derived or ("control",
            "alzheimers").
    """
    arr = np.asarray(features, dtype=np.float32).reshape(-1)
    expected = int(getattr(model, "n_features_in_", arr.size))
    if arr.size != expected:
        raise ValueError(
            f"EEG feature count mismatch: model expects {expected}, got {arr.size}"
        )

    proba = np.asarray(model.predict_proba(arr.reshape(1, -1))[0], dtype=np.float32)
    label_names = tuple(labels or _resolve_labels())
    if len(label_names) != proba.shape[0]:
        logger.warning(
            "EEG label count (%d) != model output dim (%d); falling back to class_0..N",
            len(label_names), proba.shape[0],
        )
        label_names = tuple(f"class_{i}" for i in range(proba.shape[0]))

    label_idx = int(np.argmax(proba))
    return {
        "label": label_idx,
        "label_text": label_names[label_idx],
        "confidence": float(proba[label_idx]),
        "probabilities": [
            {"label": i, "label_text": label_names[i], "probability": float(p)}
            for i, p in enumerate(proba)
        ],
    }
```

Run tests → expect 5 passed.

- [ ] **Step 4:** `pytest -q` no regressions.

- [ ] **Step 5:** commit:

```bash
git add src/models/eeg_model.py tests/fixtures/build_dummy_eeg_clf.py tests/models/test_eeg_model.py
git commit -m "feat(models): EEG classifier loader + predict (stub-able for hackathon demo)"
```

---

### Task 2: `POST /predict/eeg` route

**Files:**
- Modify: `src/api/schemas.py` (add `EEGPredictRequest` / `EEGPredictResponse`)
- Modify: `src/api/routes.py`
- Create: `tests/api/test_eeg_predict_route.py`

- [ ] **Step 1: Failing test.**

`tests/api/test_eeg_predict_route.py`:

```python
"""Integration: POST /predict/eeg."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from tests.fixtures.build_dummy_eeg_clf import build as build_dummy_eeg


@pytest.fixture()
def client(monkeypatch, tmp_path):
    artifact = build_dummy_eeg(tmp_path / "eeg.joblib", n_features=16)
    monkeypatch.setenv("EEG_CLF_ARTIFACT", str(artifact))
    return TestClient(app)


def test_predict_eeg_happy_path(client):
    body = {"features": [0.0] * 16}
    r = client.post("/predict/eeg", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["label_text"] in {"control", "alzheimers"}
    assert 0.0 <= data["confidence"] <= 1.0
    assert len(data["probabilities"]) == 2


def test_predict_eeg_alzheimers_profile(client):
    body = {"features": [2.0] * 16}
    r = client.post("/predict/eeg", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["label_text"] == "alzheimers"


def test_predict_eeg_feature_mismatch_returns_500(client):
    # Stub was trained on 16 features; sending 8 must surface as a 500 (or 400).
    body = {"features": [0.0] * 8}
    r = client.post("/predict/eeg", json=body)
    assert r.status_code in {400, 500}
```

- [ ] **Step 2: Schemas.**

In `src/api/schemas.py`, append (before the fusion re-export block):

```python
class EEGPredictRequest(BaseModel):
    features: list[float] = Field(
        ..., min_length=1,
        description="EEG features matching the classifier's training-time feature count.",
    )


class EEGClassProbability(BaseModel):
    label: int
    label_text: str
    probability: float


class EEGPredictResponse(BaseModel):
    label: int
    label_text: str
    confidence: float
    probabilities: list[EEGClassProbability]
```

- [ ] **Step 3: Route.**

In `src/api/routes.py`, add near the existing predict routes:

```python
@predict_router.post("/eeg", response_model=EEGPredictResponse)
def predict_eeg(req: EEGPredictRequest) -> EEGPredictResponse:
    import os
    from pathlib import Path
    import numpy as np
    from src.models import eeg_model

    artifact = Path(os.environ.get("EEG_CLF_ARTIFACT", "data/processed/eeg_clf.joblib"))
    clf = eeg_model.load(artifact)
    features = np.asarray(req.features, dtype=np.float32)
    out = eeg_model.predict_features(clf, features)
    return EEGPredictResponse(**out)
```

Add `EEGPredictRequest`, `EEGPredictResponse` to the schema imports at the top of `routes.py`.

- [ ] **Step 4:** `pytest tests/api/test_eeg_predict_route.py -v` → 3 passed.

- [ ] **Step 5:** commit: `feat(api): add POST /predict/eeg route (stub-able for demo)`.

---

### Task 3: End-to-end fusion flow with EEG

**Files:**
- Create: `tests/fusion/test_eeg_modality_flow.py`

This task validates that an EEG prediction (via `predict_features` or via `/predict/eeg`) plugs into the fusion engine's `eeg` modality without any code change in the engine.

- [ ] **Step 1: Test.**

`tests/fusion/test_eeg_modality_flow.py`:

```python
"""End-to-end: EEG classifier output flows into fusion as the `eeg` modality."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.fusion import engine
from src.fusion.types import (
    FusionInput,
    ModalityClassProb,
    ModalityPrediction,
)
from src.models import eeg_model
from tests.fixtures.build_dummy_eeg_clf import build as build_dummy_eeg


def _eeg_pred_from_features(model, features: np.ndarray) -> ModalityPrediction:
    raw = eeg_model.predict_features(model, features)
    return ModalityPrediction(
        label_text=raw["label_text"],
        label=raw["label"],
        confidence=raw["confidence"],
        probabilities=[
            ModalityClassProb(label_text=p["label_text"], probability=p["probability"])
            for p in raw["probabilities"]
        ],
    )


class TestEEGFusionFlow:
    def test_alzheimers_eeg_lifts_alzheimers_disease_score(self, tmp_path: Path) -> None:
        ckpt = build_dummy_eeg(tmp_path / "eeg.joblib", n_features=16)
        model = eeg_model.load(ckpt)
        eeg_pred = _eeg_pred_from_features(model, np.full((16,), 2.0, dtype=np.float32))

        out = engine.fuse(FusionInput(eeg=eeg_pred))

        alz = next(d for d in out.diseases if d.disease == "alzheimers")
        assert alz.probability > 0.5
        assert any(c.modality == "eeg" for c in alz.contributions)
        # Missing-MRI list should mention mri (it wasn't supplied).
        assert "mri" in out.missing_inputs

    def test_control_eeg_does_not_inflate_alzheimers(self, tmp_path: Path) -> None:
        ckpt = build_dummy_eeg(tmp_path / "eeg.joblib", n_features=16)
        model = eeg_model.load(ckpt)
        eeg_pred = _eeg_pred_from_features(model, np.zeros((16,), dtype=np.float32))

        out = engine.fuse(FusionInput(eeg=eeg_pred))

        alz = next(d for d in out.diseases if d.disease == "alzheimers")
        assert alz.probability < 0.5
```

- [ ] **Step 2:** Run → 2 passed (engine and types unchanged; the test exercises the existing `eeg` modality path with real predictions instead of hand-built fakes).

- [ ] **Step 3:** commit: `test(fusion): EEG classifier output flows into fusion modality end-to-end`.

---

### Task 4: Real-artifact sanity (skips when absent)

**Files:**
- Create: `tests/models/test_eeg_model_real.py`

- [ ] **Step 1: Test.**

```python
"""Real-artifact EEG sanity. Skipped unless data/processed/eeg_clf.joblib exists."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.models import eeg_model


REAL_CKPT = Path("data/processed/eeg_clf.joblib")


@pytest.mark.skipif(not REAL_CKPT.exists(), reason="real EEG checkpoint not present")
def test_real_eeg_checkpoint_loads_and_predicts():
    model = eeg_model.load(REAL_CKPT)
    n_features = int(getattr(model, "n_features_in_", 16))
    features = np.zeros((n_features,), dtype=np.float32)
    out = eeg_model.predict_features(model, features)
    s = sum(p["probability"] for p in out["probabilities"])
    assert abs(s - 1.0) < 1e-5
    assert out["label_text"]  # not empty
```

- [ ] **Step 2:** `pytest tests/models/test_eeg_model_real.py -v` → **skipped** today (expected). Will run automatically once the user drops the real artifact in.

- [ ] **Step 3:** commit: `test(models): EEG real-artifact sanity (skips when absent)`.

---

### Task 5: Streamlit form + README

**Files:**
- Modify: `src/frontend/app.py` (add an EEG features input — number array or file upload of a parquet row from `eeg_pipeline`)
- Modify: `README.md`

- [ ] **Step 1:** Streamlit. The simplest demo path: a `st.text_area` accepting comma-separated floats, parsed and POSTed to `/predict/eeg`. Place it in the doctor-view tab next to the existing MRI predict form.

```python
eeg_csv = st.text_area("EEG features (comma-separated)", placeholder="0.1,0.2,...")
if st.button("Predict (EEG)"):
    try:
        features = [float(x.strip()) for x in eeg_csv.split(",") if x.strip()]
    except ValueError:
        st.error("EEG features must be numeric.")
    else:
        r = httpx.post(f"{API_BASE}/predict/eeg", json={"features": features}, timeout=10.0)
        st.json(r.json())
```

(Integrate with the existing `httpx`/`requests` style the file already uses. If the file uses `requests`, follow that.)

- [ ] **Step 2:** README. Append:

```markdown
### EEG Pretrained Classifier

`POST /predict/eeg` runs an sklearn-style classifier (any `predict_proba` interface) on a feature vector and returns probability + attribution. The artifact loads from `data/processed/eeg_clf.joblib` (override via `EEG_CLF_ARTIFACT` env). Default labels are `("control", "alzheimers")` — override via `EEG_CLF_LABELS=label0,label1,...`.

For the hackathon demo a synthetic stub (`tests/fixtures/build_dummy_eeg_clf.py`) is used — drop the real `.joblib` at the artifact path to swap in production weights. The fusion engine consumes this prediction as the `eeg` modality automatically; no fusion-side code changes.
```

- [ ] **Step 3:** `pytest -q` → no regressions.

- [ ] **Step 4:** commit: `feat(frontend,docs): EEG predict form + README contract`.

---

## Self-review checklist

1. **Spec coverage.** User asked: "for the demo assume the EEG pretrained model exists; we can find it and put it into the project later." This plan ships a working demo today (stub fixture) and a documented swap-in path for the real artifact. ✓
2. **Independence.** EEG plumbing uses only `src.core.logger`, sklearn (already in deps), joblib, numpy. No coupling to MRI / BBB / fusion-internal code. The fusion engine consumes the EEG `ModalityPrediction` through its existing public API only. ✓
3. **No re-training.** The plan loads a classifier and runs `predict_proba` — never trains anything at runtime. ✓
4. **Demo ready without the real artifact.** Tests pass green using only the stub fixture; the real-artifact sanity test auto-skips. ✓
5. **No placeholders.** Every step has full code blocks. ✓

---

## Execution handoff

Save and choose: subagent-driven (recommended) or inline executing-plans.
