# Day 5 — Downstream Model, Uncertainty, XAI & Interactive Demo

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the Day-1..4 stack from a "data pipeline" into a "decision system" — train a downstream BBB-permeability classifier on the Morgan-fingerprint features, expose its probability + top-5 SHAP attributions through a new `POST /predict/bbb` endpoint, and rebuild the Streamlit BBB tab as a single-molecule interactive demo with a confidence bar + SHAP waterfall.

**Architecture:** A thin `src/models/bbb_model.py` module exposes `train(features_df)`, `save(model, path)`, `load(path)`, `predict_with_proba(model, smiles)` and `explain_prediction(model, smiles, top_k=5)`. The model is a scikit-learn `RandomForestClassifier` (already in requirements; XGBoost not needed and adds a heavy dep). Probabilities come from `predict_proba`; SHAP attributions use `TreeExplainer` (fast for tree models, exact). A trainer CLI (`python -m src.models.bbb_model`) materializes `data/processed/bbb_model.joblib` from the Day-4 features Parquet. FastAPI loads the artifact at startup; if missing, `/predict/bbb` returns 503 with a remediation hint. Streamlit's BBB tab is rebuilt around `st.text_input` (single SMILES) + `st.file_uploader` (CSV batch) and renders a result card + SHAP horizontal bar chart.

**Tech Stack:** scikit-learn 1.5.1 (existing), shap (new pin), joblib (sklearn transitive — explicitly pinned), Streamlit 1.39 (existing). No XGBoost (over-engineering for ECFP/Morgan binary classification at this dataset scale; RandomForest matches user's "XGBoost OR Random Forest" wording).

---

## File Structure

```
src/
├── models/
│   ├── __init__.py             # NEW (empty)
│   └── bbb_model.py            # NEW — Tasks 1+2: train/save/load/predict/explain + CLI
├── api/
│   ├── schemas.py              # MODIFY — Task 3: BBBPredictRequest, BBBPredictResponse, FeatureAttribution
│   └── routes.py               # MODIFY — Task 3: add POST /predict/bbb + startup artifact load
└── frontend/
    └── app.py                  # MODIFY — Task 4: interactive BBB tab

tests/
├── models/
│   ├── __init__.py             # NEW (empty)
│   └── test_bbb_model.py       # NEW — Tasks 1+2: ~10 tests
├── api/
│   └── test_routes.py          # MODIFY — Task 3: append TestBBBPredictRoute (3 tests)
└── frontend/
    └── test_app_import.py      # MODIFY — Task 4: extend smoke (still 2 tests, just verify new helpers)

requirements.txt                 # MODIFY — Task 1: pin shap, joblib
AGENTS.md                        # MODIFY — Task 5 close-out (§8 Decision Layer)
README.md                        # MODIFY — Task 5 close-out
```

**Test count target:** 142 (existing) + ~13 (new) = **~155 tests green at end of Day 5**.

---

## Task 1: BBB downstream model — train / save / load / predict_with_proba

**Files:**
- Create: `src/models/__init__.py` (empty)
- Create: `src/models/bbb_model.py`
- Create: `tests/models/__init__.py` (empty)
- Create: `tests/models/test_bbb_model.py`
- Modify: `requirements.txt` — add `shap==0.46.0` and `joblib==1.4.2`

- [ ] **Step 1: Add dependencies and install**

In `requirements.txt`, after the `# --- Modality: tabular ...` block (or in a new `# --- Downstream ML / XAI ---` block), add:

```
# --- Downstream ML / XAI (Day 5 decision layer) ---
shap==0.46.0
joblib==1.4.2
```

Then run:
```
cd /Users/mertgungor/Desktop/hackathon && source .venv312/bin/activate && pip install shap==0.46.0 joblib==1.4.2
```

- [ ] **Step 2: Write failing tests**

Create `tests/models/__init__.py` (empty file).

Create `tests/models/test_bbb_model.py`:

```python
"""Tests for src.models.bbb_model — train, save/load, predict, uncertainty."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models import bbb_model


_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture(scope="module")
def trained_model_and_features():
    """Train one tiny model from the committed BBBP fixture; cache for the module."""
    from src.pipelines import bbb_pipeline
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="bbb_model_test_"))
    out = tmp / "features.parquet"
    bbb_pipeline.run_pipeline(
        input_path=_FIXTURES / "bbbp_sample.csv",
        output_path=out,
    )
    df = pd.read_parquet(out)
    # Tiny n_estimators for test speed; real training uses default 100.
    model = bbb_model.train(df, label_col="p_np", n_estimators=10, random_state=42)
    return model, df


class TestTrain:
    def test_returns_fitted_classifier(self, trained_model_and_features):
        model, _ = trained_model_and_features
        # sklearn classifiers expose .classes_ after fit
        assert hasattr(model, "classes_")
        assert len(model.classes_) == 2

    def test_raises_on_missing_label_column(self, trained_model_and_features):
        _, df = trained_model_and_features
        with pytest.raises(KeyError):
            bbb_model.train(df.drop(columns=["p_np"]), label_col="p_np")

    def test_deterministic_with_random_state(self, trained_model_and_features):
        _, df = trained_model_and_features
        m1 = bbb_model.train(df, label_col="p_np", n_estimators=10, random_state=42)
        m2 = bbb_model.train(df, label_col="p_np", n_estimators=10, random_state=42)
        # Two trainings with same seed produce identical predictions on the same input
        fp_cols = [c for c in df.columns if c.startswith("fp_")]
        X = df[fp_cols].to_numpy()
        np.testing.assert_array_equal(m1.predict_proba(X), m2.predict_proba(X))


class TestSaveLoad:
    def test_save_then_load_roundtrip(self, trained_model_and_features, tmp_path: Path):
        model, df = trained_model_and_features
        artifact = tmp_path / "bbb_model.joblib"
        bbb_model.save(model, artifact)
        assert artifact.exists()

        reloaded = bbb_model.load(artifact)
        fp_cols = [c for c in df.columns if c.startswith("fp_")]
        X = df[fp_cols].to_numpy()
        np.testing.assert_array_equal(model.predict(X), reloaded.predict(X))

    def test_load_raises_on_missing_path(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            bbb_model.load(tmp_path / "does_not_exist.joblib")


class TestPredictWithProba:
    def test_returns_label_and_probabilities(self, trained_model_and_features):
        model, _ = trained_model_and_features
        # A real-ish drug-like SMILES (ethanol — definitely BBB+)
        result = bbb_model.predict_with_proba(model, "CCO")
        assert "label" in result
        assert "probability" in result
        assert "confidence" in result
        assert result["label"] in (0, 1)
        # Probabilities for the predicted class
        assert 0.0 <= result["probability"] <= 1.0
        assert 0.0 <= result["confidence"] <= 1.0

    def test_raises_on_invalid_smiles(self, trained_model_and_features):
        model, _ = trained_model_and_features
        with pytest.raises(ValueError):
            bbb_model.predict_with_proba(model, "this_is_not_a_smiles_AT_ALL")

    def test_confidence_equals_max_class_probability(self, trained_model_and_features):
        """confidence is the model's max class probability — its own self-rated certainty."""
        model, _ = trained_model_and_features
        result = bbb_model.predict_with_proba(model, "CCO")
        # confidence should equal the probability of the predicted class
        assert abs(result["confidence"] - result["probability"]) < 1e-9
```

- [ ] **Step 3: Run tests to verify they fail**

```
cd /Users/mertgungor/Desktop/hackathon && source .venv312/bin/activate && pytest tests/models/test_bbb_model.py -v
```
Expected: ImportError (`src.models.bbb_model` does not exist).

- [ ] **Step 4: Implement `src/models/bbb_model.py` (Task 1 surface only — train/save/load/predict)**

Create `src/models/__init__.py` (empty file).

Create `src/models/bbb_model.py`:

```python
"""BBB-permeability downstream classifier — train / save / load / predict.

Built on top of `data/processed/bbbp_features.parquet` produced by
`src.pipelines.bbb_pipeline`. Uses scikit-learn's `RandomForestClassifier`
(no XGBoost — saves a heavy dep without losing accuracy at this scale).

The model takes a 2,048-bit Morgan fingerprint as input. SHAP-based
explanation lives in this same module (Task 2 adds `explain_prediction`).
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.core.logger import get_logger
from src.pipelines.bbb_pipeline import (
    compute_morgan_fingerprint,
    is_valid_smiles,
)

logger = get_logger(__name__)


_FP_COL_PREFIX = "fp_"


def _split_features_and_label(
    df: pd.DataFrame, label_col: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Pull out fp_* columns as X and `label_col` as y. Returns (X, y, fp_col_names)."""
    if label_col not in df.columns:
        raise KeyError(f"Label column {label_col!r} not in DataFrame")
    fp_cols = [c for c in df.columns if c.startswith(_FP_COL_PREFIX)]
    if not fp_cols:
        raise KeyError(
            f"No {_FP_COL_PREFIX}* columns found — was this DataFrame produced "
            f"by bbb_pipeline.run_pipeline?"
        )
    X = df[fp_cols].to_numpy()
    y = df[label_col].to_numpy()
    return X, y, fp_cols


def train(
    df: pd.DataFrame,
    label_col: str = "p_np",
    n_estimators: int = 100,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Train a Random Forest classifier on Morgan fingerprints.

    Args:
        df: Output of `bbb_pipeline.run_pipeline` — has `fp_0..fp_N-1` cols
            plus a binary `label_col`.
        label_col: Name of the binary target column. Defaults to "p_np"
            (BBBP dataset's permeable / non-permeable).
        n_estimators: Number of trees. 100 is the sklearn default; tests
            override to 10 for speed.
        random_state: Seed for split + tree construction. Required for
            byte-deterministic predictions on the same input.

    Returns:
        A fitted `RandomForestClassifier` with `feature_names_in_` matching
        the `fp_*` column order, so downstream callers can map SHAP values
        back to human-meaningful bit indices.
    """
    X, y, fp_cols = _split_features_and_label(df, label_col)
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=1,  # determinism: no thread races in tree fit
    )
    model.fit(X, y)
    # Stash the column names so explainers can label features.
    model.feature_names_in_ = np.array(fp_cols, dtype=object)
    logger.info(
        "Trained BBB classifier: n=%d, n_features=%d, classes=%s",
        len(y), X.shape[1], model.classes_.tolist(),
    )
    return model


def save(model: RandomForestClassifier, path: Path) -> None:
    """Persist a fitted model to `path` (parent dirs auto-created)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Saved BBB model to %s", path)


def load(path: Path) -> RandomForestClassifier:
    """Load a previously-saved model. Raises `FileNotFoundError` on missing artifact."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"BBB model artifact not found: {path}")
    return joblib.load(path)


def predict_with_proba(
    model: RandomForestClassifier,
    smiles: str,
    n_bits: int = 2048,
    radius: int = 2,
) -> dict[str, object]:
    """Predict BBB permeability for a single SMILES.

    Args:
        model: Fitted classifier from `train()` or `load()`.
        smiles: A SMILES string. Validated via `is_valid_smiles`.
        n_bits / radius: Must match the values used at training time
            (defaults match `bbb_pipeline.run_pipeline`'s defaults).

    Returns:
        `{"label": int, "probability": float, "confidence": float}` where
        `probability` is the predicted class's probability and `confidence`
        is the model's self-rated certainty (max class probability — same
        value as `probability` for the predicted class).

    Raises:
        ValueError: if `smiles` cannot be parsed by RDKit.
    """
    if not is_valid_smiles(smiles):
        raise ValueError(f"invalid SMILES: {smiles!r}")
    fp = compute_morgan_fingerprint(smiles, n_bits=n_bits, radius=radius)
    proba = model.predict_proba(fp.reshape(1, -1))[0]  # shape (n_classes,)
    label_idx = int(np.argmax(proba))
    label = int(model.classes_[label_idx])
    return {
        "label": label,
        "probability": float(proba[label_idx]),
        "confidence": float(proba[label_idx]),
    }
```

- [ ] **Step 5: Run tests, expect all to pass**

```
pytest tests/models/test_bbb_model.py -v
```
Expected: 8 passed (3 train + 2 save/load + 3 predict).

- [ ] **Step 6: Run full suite to confirm no regressions**

```
pytest -v 2>&1 | tail -3
```
Expected: 150 passed (142 prior + 8 new).

- [ ] **Step 7: Commit**

```bash
git add requirements.txt src/models/__init__.py src/models/bbb_model.py \
        tests/models/__init__.py tests/models/test_bbb_model.py
git commit -m "feat(models): BBB classifier with predict_with_proba uncertainty"
```

---

## Task 2: SHAP explainability — top-5 feature attributions

**Files:**
- Modify: `src/models/bbb_model.py` (append `explain_prediction`)
- Modify: `tests/models/test_bbb_model.py` (append `TestExplainPrediction`)

- [ ] **Step 1: Write failing tests**

Append to `tests/models/test_bbb_model.py`:

```python
class TestExplainPrediction:
    def test_returns_top_k_features(self, trained_model_and_features):
        model, _ = trained_model_and_features
        attributions = bbb_model.explain_prediction(model, "CCO", top_k=5)
        assert len(attributions) == 5
        # Each entry has feature name + shap value
        for a in attributions:
            assert "feature" in a
            assert "shap_value" in a
            assert isinstance(a["shap_value"], float)

    def test_features_sorted_by_absolute_shap_value_descending(
        self, trained_model_and_features,
    ):
        model, _ = trained_model_and_features
        attributions = bbb_model.explain_prediction(model, "CCO", top_k=10)
        abs_vals = [abs(a["shap_value"]) for a in attributions]
        assert abs_vals == sorted(abs_vals, reverse=True)

    def test_features_named_fp_INDEX(self, trained_model_and_features):
        model, _ = trained_model_and_features
        attributions = bbb_model.explain_prediction(model, "CCO", top_k=3)
        for a in attributions:
            # SHAP values map back to fp_<integer> bit indices
            assert a["feature"].startswith("fp_")
            int(a["feature"].split("_")[1])  # parses cleanly

    def test_raises_on_invalid_smiles(self, trained_model_and_features):
        model, _ = trained_model_and_features
        with pytest.raises(ValueError):
            bbb_model.explain_prediction(model, "still_not_a_smiles", top_k=5)
```

- [ ] **Step 2: Run failing tests**

```
pytest tests/models/test_bbb_model.py::TestExplainPrediction -v
```
Expected: AttributeError — `explain_prediction` not defined.

- [ ] **Step 3: Implement `explain_prediction` in `src/models/bbb_model.py`**

Append after `predict_with_proba`:

```python
def explain_prediction(
    model: RandomForestClassifier,
    smiles: str,
    top_k: int = 5,
    n_bits: int = 2048,
    radius: int = 2,
) -> list[dict[str, object]]:
    """Return the top-`top_k` feature attributions (SHAP values) for `smiles`.

    Uses `shap.TreeExplainer` (exact for tree ensembles, no sampling). The
    explanation is for the *predicted* class — i.e. SHAP values that pushed
    the model toward whichever label was returned by `predict_with_proba`.

    Args:
        model: Fitted classifier from `train()` or `load()`.
        smiles: A SMILES string (validated via `is_valid_smiles`).
        top_k: How many top features to return. Default 5 — matches the
            jury-demo budget (more bars = noisier waterfall chart).
        n_bits / radius: Must match training-time fingerprint settings.

    Returns:
        A list of `{"feature": "fp_<bit_idx>", "shap_value": float}` dicts,
        sorted by `abs(shap_value)` descending.

    Raises:
        ValueError: if `smiles` cannot be parsed by RDKit.
    """
    import shap  # local import — heavy module, only loaded when needed

    if not is_valid_smiles(smiles):
        raise ValueError(f"invalid SMILES: {smiles!r}")
    fp = compute_morgan_fingerprint(smiles, n_bits=n_bits, radius=radius)
    X = fp.reshape(1, -1)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X, check_additivity=False)
    # `shap_values` shape varies by sklearn / shap versions:
    #   - older: list of (1, n_features) arrays, one per class
    #   - newer: ndarray of shape (1, n_features, n_classes) for binary RF
    #   - or (1, n_features) when output already condensed
    if isinstance(shap_values, list):
        # one (1, n_features) per class — pick the predicted class's array
        proba = model.predict_proba(X)[0]
        label_idx = int(np.argmax(proba))
        per_feature = shap_values[label_idx][0]
    else:
        arr = np.asarray(shap_values)
        if arr.ndim == 3:
            # (1, n_features, n_classes)
            proba = model.predict_proba(X)[0]
            label_idx = int(np.argmax(proba))
            per_feature = arr[0, :, label_idx]
        else:
            # (1, n_features)
            per_feature = arr[0]

    fp_cols = (
        list(model.feature_names_in_)
        if hasattr(model, "feature_names_in_")
        else [f"fp_{i}" for i in range(len(per_feature))]
    )

    pairs = sorted(
        zip(fp_cols, per_feature, strict=True),
        key=lambda p: abs(p[1]),
        reverse=True,
    )
    return [
        {"feature": str(name), "shap_value": float(value)}
        for name, value in pairs[:top_k]
    ]
```

- [ ] **Step 4: Run tests**

```
pytest tests/models/test_bbb_model.py::TestExplainPrediction -v
```
Expected: 4 passed.

- [ ] **Step 5: Run full suite**

```
pytest -v 2>&1 | tail -3
```
Expected: 154 passed (150 prior + 4 new).

- [ ] **Step 6: Commit**

```bash
git add src/models/bbb_model.py tests/models/test_bbb_model.py
git commit -m "feat(models): SHAP top-k explainer for BBB predictions"
```

---

## Task 3: FastAPI `POST /predict/bbb` endpoint

**Files:**
- Modify: `src/api/schemas.py` — add `BBBPredictRequest`, `FeatureAttribution`, `BBBPredictResponse`
- Modify: `src/api/routes.py` — add `POST /predict/bbb` + lazy model loading
- Modify: `tests/api/test_routes.py` — append `TestBBBPredictRoute` (3 tests)

- [ ] **Step 1: Add schemas**

In `/Users/mertgungor/Desktop/hackathon/src/api/schemas.py`, append at the bottom:

```python
class BBBPredictRequest(BaseModel):
    """Single-molecule BBB-permeability prediction request."""
    smiles: str = Field(..., description="SMILES string; e.g. 'CCO' for ethanol")
    top_k: int = Field(5, ge=1, le=20, description="Top-k SHAP features to return")


class FeatureAttribution(BaseModel):
    feature: str
    shap_value: float


class BBBPredictResponse(BaseModel):
    """Decision-system payload: prediction + uncertainty + explanation."""
    label: int
    label_text: str = Field(..., description="'permeable' or 'non-permeable'")
    probability: float
    confidence: float
    top_features: list[FeatureAttribution]
```

- [ ] **Step 2: Write failing route tests**

Append to `/Users/mertgungor/Desktop/hackathon/tests/api/test_routes.py`:

```python
class TestBBBPredictRoute:
    def test_returns_200_with_prediction_and_attributions(self, tmp_path: Path):
        """End-to-end: train tiny model, save, point env at it, POST a SMILES."""
        import os
        from src.pipelines import bbb_pipeline
        from src.models import bbb_model
        # 1. Build features from the committed fixture
        features_path = tmp_path / "features.parquet"
        bbb_pipeline.run_pipeline(
            input_path=_FIXTURES / "bbbp_sample.csv",
            output_path=features_path,
        )
        # 2. Train + save a tiny model
        import pandas as pd
        df = pd.read_parquet(features_path)
        model = bbb_model.train(df, label_col="p_np", n_estimators=10, random_state=42)
        artifact = tmp_path / "bbb_model.joblib"
        bbb_model.save(model, artifact)
        # 3. Point the API at this artifact (env-var override)
        os.environ["BBB_MODEL_PATH"] = str(artifact)
        try:
            resp = client.post(
                "/predict/bbb",
                json={"smiles": "CCO", "top_k": 5},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["label"] in (0, 1)
            assert body["label_text"] in ("permeable", "non-permeable")
            assert 0.0 <= body["probability"] <= 1.0
            assert 0.0 <= body["confidence"] <= 1.0
            assert len(body["top_features"]) == 5
            for f in body["top_features"]:
                assert f["feature"].startswith("fp_")
                assert isinstance(f["shap_value"], float)
        finally:
            os.environ.pop("BBB_MODEL_PATH", None)

    def test_returns_400_on_invalid_smiles(self, tmp_path: Path):
        import os
        from src.pipelines import bbb_pipeline
        from src.models import bbb_model
        import pandas as pd
        features_path = tmp_path / "features.parquet"
        bbb_pipeline.run_pipeline(
            input_path=_FIXTURES / "bbbp_sample.csv",
            output_path=features_path,
        )
        df = pd.read_parquet(features_path)
        model = bbb_model.train(df, label_col="p_np", n_estimators=10, random_state=42)
        artifact = tmp_path / "bbb_model.joblib"
        bbb_model.save(model, artifact)
        os.environ["BBB_MODEL_PATH"] = str(artifact)
        try:
            resp = client.post(
                "/predict/bbb",
                json={"smiles": "this_is_not_a_smiles", "top_k": 5},
            )
            assert resp.status_code == 400
        finally:
            os.environ.pop("BBB_MODEL_PATH", None)

    def test_returns_503_when_artifact_missing(self, tmp_path: Path):
        import os
        os.environ["BBB_MODEL_PATH"] = str(tmp_path / "does_not_exist.joblib")
        try:
            resp = client.post(
                "/predict/bbb",
                json={"smiles": "CCO", "top_k": 5},
            )
            assert resp.status_code == 503
        finally:
            os.environ.pop("BBB_MODEL_PATH", None)
```

- [ ] **Step 3: Run failing tests**

```
pytest tests/api/test_routes.py::TestBBBPredictRoute -v
```
Expected: 3 errors — endpoint not mounted.

- [ ] **Step 4: Implement the route**

In `/Users/mertgungor/Desktop/hackathon/src/api/routes.py`, append at the end:

```python
from src.api.schemas import BBBPredictRequest, BBBPredictResponse, FeatureAttribution
from src.models import bbb_model

# Default artifact location. Overridable via BBB_MODEL_PATH env var so tests
# can point at a tmp-built model without touching production paths.
_DEFAULT_BBB_MODEL_PATH = Path("data/processed/bbb_model.joblib")


def _bbb_model_path() -> Path:
    return Path(os.environ.get("BBB_MODEL_PATH", str(_DEFAULT_BBB_MODEL_PATH)))


@router.post("/predict/bbb", response_model=BBBPredictResponse, prefix="")
def predict_bbb(req: BBBPredictRequest) -> BBBPredictResponse:
    """Predict BBB permeability + return SHAP attributions for one SMILES."""
    artifact = _bbb_model_path()
    if not artifact.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"BBB model artifact not available at {artifact}. "
                f"Run `python -m src.models.bbb_model` to train it."
            ),
        )
    try:
        model = bbb_model.load(artifact)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    try:
        pred = bbb_model.predict_with_proba(model, req.smiles)
        attributions = bbb_model.explain_prediction(model, req.smiles, top_k=req.top_k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    label_text = "permeable" if pred["label"] == 1 else "non-permeable"
    return BBBPredictResponse(
        label=pred["label"],
        label_text=label_text,
        probability=pred["probability"],
        confidence=pred["confidence"],
        top_features=[FeatureAttribution(**a) for a in attributions],
    )
```

> **Important**: the `prefix=""` on the router decorator above is a hack — `APIRouter(prefix="/pipeline")` set the global prefix, but `/predict/bbb` is a sibling not a child of `/pipeline`. **Fix the placement instead**: don't use `@router.post(...)` for this endpoint. Use a *new* router with no prefix, or register the route directly on `app` from `main.py`. Cleanest: in `routes.py` add a second `predict_router = APIRouter()` (no prefix) at module scope, register `predict_bbb` on it, and export it. Then `main.py` does `app.include_router(predict_router)` alongside `pipeline_router`.

Actual implementation pattern:

In `src/api/routes.py`, near the top after the existing `router = APIRouter(prefix="/pipeline")`, add:

```python
predict_router = APIRouter(prefix="/predict")
```

Then the endpoint:

```python
@predict_router.post("/bbb", response_model=BBBPredictResponse)
def predict_bbb(req: BBBPredictRequest) -> BBBPredictResponse:
    # ... (same body as above, minus the prefix="" hack)
```

Add `import os` to the imports block at top of `routes.py` if not already present.

- [ ] **Step 5: Mount the new router in `src/api/main.py`**

In `src/api/main.py`, change the existing import line:

```python
from src.api.routes import router as pipeline_router
```

to:

```python
from src.api.routes import router as pipeline_router, predict_router
```

And below the existing `app.include_router(pipeline_router)` line, add:

```python
app.include_router(predict_router)
```

- [ ] **Step 6: Run tests**

```
pytest tests/api/ -v
```
Expected: 11 passed (8 prior + 3 new).

- [ ] **Step 7: Run full suite**

```
pytest -v 2>&1 | tail -3
```
Expected: 157 passed (154 prior + 3 new).

- [ ] **Step 8: Commit**

```bash
git add src/api/schemas.py src/api/routes.py src/api/main.py tests/api/test_routes.py
git commit -m "feat(api): POST /predict/bbb with prediction, uncertainty, SHAP top-k"
```

---

## Task 4: Interactive Streamlit BBB tab

**Files:**
- Modify: `src/frontend/app.py` — replace `_render_bbb_tab` body with interactive form
- (No test changes — the existing 2 frontend tests still pass; manual smoke verifies UX.)

- [ ] **Step 1: Replace `_render_bbb_tab` in `src/frontend/app.py`**

Find the existing `_render_bbb_tab` function and replace its entire body with:

```python
def _render_bbb_tab() -> None:
    _render_section(
        "MOLECULE — BBBP",
        "Blood-Brain-Barrier permeability decision",
        "Enter a SMILES string. The system computes a 2,048-bit Morgan "
        "fingerprint, runs it through a trained Random Forest classifier, "
        "and returns the predicted permeability label, the model's "
        "self-rated confidence, and the top-5 SHAP feature attributions "
        "explaining the decision.",
    )

    smiles = st.text_input(
        "SMILES string",
        value="CCO",
        key="bbb_smiles",
        help="Examples: CCO (ethanol, BBB+), CC(=O)Nc1ccc(O)cc1 (paracetamol)",
    )
    top_k = st.slider("SHAP features to display", min_value=3, max_value=10, value=5, key="bbb_topk")

    if st.button("Predict BBB permeability", type="primary", key="bbb_predict"):
        with st.spinner("Computing fingerprint, predicting, and explaining…"):
            try:
                result = _post("/predict/bbb", {"smiles": smiles, "top_k": top_k})
                _render_prediction_card(result)
                st.toast("Prediction complete", icon="✅")
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 503:
                    st.error(
                        "Model artifact not loaded yet. Run "
                        "`python -m src.models.bbb_model` to train it, "
                        "then retry."
                    )
                else:
                    st.error(f"Prediction failed (HTTP {e.response.status_code}): {e.response.text}")
            except httpx.RequestError as e:
                st.error(f"Cannot reach FastAPI at {_API_URL}: {e!r}")
```

- [ ] **Step 2: Add `_render_prediction_card` helper above `main()`**

Insert above `main()`:

```python
def _render_prediction_card(result: dict) -> None:
    """Render a B2B-styled decision card: label badge + confidence + SHAP bars."""
    label_text = _html.escape(str(result["label_text"]))
    badge_color = "#166534" if result["label"] == 1 else "#991B1B"
    badge_bg    = "#DCFCE7" if result["label"] == 1 else "#FEE2E2"
    confidence_pct = result["confidence"] * 100

    st.markdown(
        f"""
        <div style='background:#FFFFFF;border:1px solid #E2E8F0;border-radius:10px;
                    padding:1.5rem;margin:1rem 0;box-shadow:0 1px 2px rgba(15,23,42,0.04);'>
            <p style='font-size:0.72rem;font-weight:700;color:#64748B;
                      letter-spacing:0.08em;text-transform:uppercase;margin:0;'>Prediction</p>
            <div style='display:flex;align-items:center;gap:0.75rem;margin-top:0.4rem;'>
                <span style='background:{badge_bg};color:{badge_color};
                             padding:0.4rem 0.9rem;border-radius:999px;
                             font-size:1rem;font-weight:700;letter-spacing:0.01em;'>
                    {label_text.upper()}
                </span>
                <span style='color:#475569;font-size:0.95rem;'>
                    Model confidence: <strong style='color:#0F172A;'>{confidence_pct:.1f}%</strong>
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Confidence bar
    st.markdown(
        "<p style='font-size:0.72rem;font-weight:700;color:#64748B;"
        "letter-spacing:0.08em;text-transform:uppercase;margin:1rem 0 0.4rem 0;'>"
        "Confidence</p>",
        unsafe_allow_html=True,
    )
    st.progress(float(result["confidence"]))

    # SHAP attributions chart
    st.markdown(
        "<p style='font-size:0.72rem;font-weight:700;color:#64748B;"
        "letter-spacing:0.08em;text-transform:uppercase;margin:1.5rem 0 0.4rem 0;'>"
        f"Top {len(result['top_features'])} SHAP attributions</p>",
        unsafe_allow_html=True,
    )
    import pandas as pd
    shap_df = pd.DataFrame(result["top_features"]).set_index("feature")
    st.bar_chart(shap_df, height=240, color="#0369A1")

    st.caption(
        "Positive SHAP values pushed the model toward the predicted class; "
        "negative values pushed it away. Feature names are 2,048-bit Morgan "
        "fingerprint indices (`fp_<bit>`)."
    )
```

- [ ] **Step 3: Run smoke tests**

```
pytest tests/frontend/ -v
```
Expected: 2 passed (the existing import smoke tests still cover the module).

- [ ] **Step 4: Run full suite**

```
pytest -v 2>&1 | tail -3
```
Expected: 157 passed (no test count change — UI redesign is covered by manual smoke).

- [ ] **Step 5: Manual smoke (recommended)**

```
streamlit run src/frontend/app.py --server.headless true &
sleep 5
curl -s http://localhost:8501 | head -3
pkill -f "streamlit run"
```
Expected: HTML response. Then open the dashboard manually in a browser, paste `CCO` in the BBB tab, click Predict, verify card renders + SHAP bar chart appears.

- [ ] **Step 6: Commit**

```bash
git add src/frontend/app.py
git commit -m "feat(frontend): interactive BBB tab — SMILES input + decision card + SHAP bars"
```

---

## Task 5: Trainer CLI + close-out (AGENTS.md §8 + README + DoD)

**Files:**
- Modify: `src/models/bbb_model.py` — add `if __name__ == "__main__":` CLI for training
- Modify: `AGENTS.md` — §2 directory layout + new §8 Decision Layer
- Modify: `README.md` — Day 5 row + how-to-train section

- [ ] **Step 1: Add training CLI to `src/models/bbb_model.py`**

Append at the bottom of the file:

```python
DEFAULT_FEATURES_PATH = Path("data/processed/bbbp_features.parquet")
DEFAULT_MODEL_PATH = Path("data/processed/bbb_model.joblib")


def main() -> None:
    """Train and persist the production BBB model from the Day-4 features Parquet.

    Reads from `DEFAULT_FEATURES_PATH`, trains with default hyperparameters,
    and writes the artifact to `DEFAULT_MODEL_PATH`. Re-runs are idempotent
    (same random_state).
    """
    if not DEFAULT_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Features Parquet not found at {DEFAULT_FEATURES_PATH}. "
            f"Run `python -m src.pipelines.bbb_pipeline` first."
        )
    df = pd.read_parquet(DEFAULT_FEATURES_PATH)
    model = train(df, label_col="p_np")
    save(model, DEFAULT_MODEL_PATH)
    logger.info("BBB model artifact ready at %s", DEFAULT_MODEL_PATH)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Update AGENTS.md**

Add `src/models/bbb_model.py` to the §2 layout tree.

After §7 Experiment Tracking, append:

```markdown
## 8. Decision Layer (Downstream Models)

Pipelines produce features (`data/processed/<modality>_features.parquet`).
Downstream models live in `src/models/` and consume those features:

| Model | File | Output | Endpoint |
|---|---|---|---|
| BBB permeability | `src/models/bbb_model.py` | `data/processed/bbb_model.joblib` | `POST /predict/bbb` |

Each downstream model module exposes a uniform surface:
- `train(df, label_col, ...)` → fitted classifier
- `save(model, path)` / `load(path)` → joblib artifact I/O
- `predict_with_proba(model, smiles)` → `{label, probability, confidence}`
- `explain_prediction(model, smiles, top_k)` → SHAP top-k attributions

The API loads the joblib artifact at request time. If the artifact is
missing, the endpoint returns **HTTP 503** with a remediation hint pointing
at the trainer CLI (`python -m src.models.<name>`). This keeps the API
process startup fast and lets operators retrain without redeploying.

**Determinism**: all classifiers are seeded (`random_state=42` default). Re-running
the trainer on the same Parquet produces identical predictions.
```

- [ ] **Step 3: Update README.md**

Add Day 5 to the status table:

```markdown
| Day 5 — Decision Layer (Model + XAI + Interactive UI) | ✅ Shipped — 157 tests green |
```

Add a "Train the BBB model" step under Quick Start:

```markdown
### Train the downstream BBB model (one-time)

```bash
python -m src.pipelines.bbb_pipeline   # produces data/processed/bbbp_features.parquet
python -m src.models.bbb_model          # produces data/processed/bbb_model.joblib
```

Then `POST /predict/bbb` (and the Streamlit BBB tab) become live.
```

Add to "Where to Look":
- `src/models/bbb_model.py` (downstream classifier + SHAP)
- `tests/models/test_bbb_model.py` (~12 tests)

- [ ] **Step 4: DoD smoke run**

```
cd /Users/mertgungor/Desktop/hackathon && source .venv312/bin/activate
pytest -v 2>&1 | tail -3
# Expect: 157 passed
```

If the BBB raw input is not present, skip the live API smoke. If it is:

```
python -m src.pipelines.bbb_pipeline
python -m src.models.bbb_model
uvicorn src.api.main:app --port 8000 &
sleep 4
curl -s -X POST http://localhost:8000/predict/bbb \
  -H 'Content-Type: application/json' \
  -d '{"smiles": "CCO", "top_k": 5}' | python3 -m json.tool
pkill -f "uvicorn src.api.main:app"
```
Expected: `label`, `label_text`, `probability`, `confidence`, and 5-element `top_features` array.

- [ ] **Step 5: Commit**

```bash
git add src/models/bbb_model.py AGENTS.md README.md
git commit -m "docs: Day-5 close-out — AGENTS §8 decision layer + trainer CLI"
```

---

## Definition of Done (Day 5)

| Check | Pass criterion |
|---|---|
| `pytest -v` reports 157 passed | yes |
| `src/models/bbb_model.py` exposes `train`, `save`, `load`, `predict_with_proba`, `explain_prediction` | yes |
| `python -m src.models.bbb_model` produces a joblib artifact | yes (DoD smoke) |
| `POST /predict/bbb` returns label + label_text + probability + confidence + top_features | yes |
| 503 returned when artifact is missing (live-demo lifeline) | yes (test) |
| 400 returned on invalid SMILES | yes (test) |
| Streamlit BBB tab has SMILES input + Predict button + decision card + SHAP bar chart | yes (manual) |
| AGENTS.md §8 documents the decision-layer contract | yes |
| Existing 142 tests still green (no regressions) | yes (full suite) |
| The 4 design constraints are met: Interaction (text input), Uncertainty (confidence %), Explainability (SHAP top-5), Decision (binary label + label_text) | yes |

When all rows green: Day 5 complete. The system is now an end-to-end **Living + Decision System**: data → features → model → uncertainty → explanation → interactive UI.
