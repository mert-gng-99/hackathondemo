# Day 7 — The Final 5% (Drift, Traceability & Agents) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the "Adapt Over Time" gap and add a Track-1 "AI Lab Agents" surface (chat-style explainer) without breaking the 165-test green floor. Test target: **165 → 175 passed** (+10 new tests).

**Architecture:** Drift = train-time stats baked into `model._neurobridge_train_stats` + module-level `collections.deque(maxlen=100)` per FastAPI worker. LLM explainer = thin abstraction (`src/llm/explainer.py`) with deterministic-template fallback and OpenRouter (via `openai==1.51.0` SDK) hybrid. Hard kill-switch: `NEUROBRIDGE_DISABLE_LLM=1` forces template path. Spec source of truth: [docs/superpowers/specs/2026-05-05-day7-drift-traceability-agents-design.md](docs/superpowers/specs/2026-05-05-day7-drift-traceability-agents-design.md) (commit `09dd9c3`).

**Tech Stack:** Python 3.12 · sklearn 1.5.1 (existing) · FastAPI + Pydantic (existing) · Streamlit + altair (existing) · MLflow 2.16.0 (existing) · **`openai==1.51.0` (NEW pip dep)**.

---

## File Structure

```
src/
├── models/
│   └── bbb_model.py            # MODIFY — T1A: stash _neurobridge_train_stats
├── api/
│   ├── schemas.py              # MODIFY — T1B: drift_z + rolling_n; T2: ModelProvenance; T3B: BBBExplainRequest/Response
│   └── routes.py               # MODIFY — T1B: deque + drift helper; T2: provenance lookup; T3B: explain_router + /explain/bbb
├── llm/                        # NEW dir
│   ├── __init__.py             # CREATE
│   └── explainer.py            # CREATE — T3A: explain() public API + template + openrouter
└── frontend/
    └── app.py                  # MODIFY — T1C: drift line; T2: provenance badge; T3C: AI Assistant tab

tests/
├── models/
│   └── test_bbb_model.py       # MODIFY — T1A: TestTrainStatsMetadata (+2)
├── api/
│   └── test_routes.py          # MODIFY — T1B: extend TestBBBPredictRoute (+2); T2: extend with provenance (+1); T3B: TestExplainBBBRoute (+1)
└── llm/                        # NEW dir
    ├── __init__.py             # CREATE
    └── test_explainer.py       # CREATE — T3A: TestTemplateExplain (+4)

requirements.txt                 # MODIFY — add openai==1.51.0
AGENTS.md                        # MODIFY — T4: §10 Drift Surface, §11 LLM Explainer Surface
README.md                        # MODIFY — T4: Day 7 row + curl recipe
```

**Test count growth:** 2 (T1A) + 2 (T1B drift) + 1 (T2 provenance) + 4 (T3A template) + 1 (T3B route) = **+10 → 175 passed**.

---

## Pre-Flight Verification

- [ ] **Step 0: Confirm clean baseline**

```bash
cd /Users/mertgungor/Desktop/hackathon
source .venv312/bin/activate
git status                      # Expect: clean tree on main
git log --oneline -1            # Expect: 09dd9c3 docs(spec): Day-7 final-5% design …
pytest -q 2>&1 | tail -3        # Expect: 165 passed
```

If any of these fail, STOP and resolve before proceeding.

---

## Task 1A — Train-Time Stats Metadata

**Why:** Drift z-score requires a frozen "training distribution" reference (median + std of the model's own confidence on the train set). We bake this into the joblib artifact alongside the existing `_neurobridge_calibration` and `_neurobridge_fp_cols` so it survives save/load.

**Files:**
- Modify: `src/models/bbb_model.py`
- Modify: `tests/models/test_bbb_model.py`

### Step 1: Write the 2 failing tests (RED)

- [ ] Append a new `TestTrainStatsMetadata` class at the end of `/Users/mertgungor/Desktop/hackathon/tests/models/test_bbb_model.py` (after `TestCalibrationMetadata`):

```python
class TestTrainStatsMetadata:
    """Day 7 — T1A: train()-time confidence distribution stash."""

    def test_train_attaches_train_stats_attribute(self, trained_model_and_features):
        model, _ = trained_model_and_features
        assert hasattr(model, "_neurobridge_train_stats")
        stats = model._neurobridge_train_stats
        assert isinstance(stats, dict)
        for key in ("median", "std", "n_train"):
            assert key in stats, f"missing key {key!r} in train stats"
        assert 0.0 <= stats["median"] <= 1.0
        assert stats["std"] >= 0.0
        assert stats["n_train"] >= 1

    def test_train_stats_survives_save_load_roundtrip(
        self, trained_model_and_features, tmp_path: Path,
    ):
        from src.models import bbb_model
        model, _ = trained_model_and_features
        path = tmp_path / "m.joblib"
        bbb_model.save(model, path)
        reloaded = bbb_model.load(path)
        assert hasattr(reloaded, "_neurobridge_train_stats")
        assert reloaded._neurobridge_train_stats == model._neurobridge_train_stats
```

### Step 2: Run the new tests — verify RED

- [ ] Run only these tests:

```bash
pytest tests/models/test_bbb_model.py::TestTrainStatsMetadata -v
```
Expected: **2 failed** with `AssertionError: hasattr(model, '_neurobridge_train_stats')` (or similar). If they pass, STOP — the attribute already exists somewhere unexpected.

### Step 3: Implement `_compute_train_stats` and wire into `train()` (GREEN)

- [ ] Open `/Users/mertgungor/Desktop/hackathon/src/models/bbb_model.py`. Add this private helper immediately above `def train(`:

```python
def _compute_train_stats(
    model: RandomForestClassifier,
    X_train: np.ndarray,
) -> dict[str, float]:
    """Compute median + std of the model's own confidence on the training set.

    Used as the reference distribution for runtime drift detection. All values
    are floats so the dict is joblib-roundtrip-safe and JSON-serializable.
    """
    if len(X_train) == 0:
        return {"median": 0.0, "std": 0.0, "n_train": 0}
    proba = model.predict_proba(X_train)
    confidence = proba.max(axis=1)
    return {
        "median": float(np.median(confidence)),
        "std": float(np.std(confidence)),
        "n_train": int(len(X_train)),
    }
```

- [ ] In `train()`, immediately after the existing line `model._neurobridge_calibration = _compute_calibration_bins(model, X_test, y_test)`, add:

```python
    model._neurobridge_train_stats = _compute_train_stats(model, X_train)
```

- [ ] Update the existing `logger.info(...)` line at the end of `train()` to also surface the train-stats summary:

Replace:
```python
    logger.info(
        "Trained BBB classifier: n=%d, n_features=%d, classes=%s, "
        "calibration_bins=%d",
        len(y), X.shape[1], model.classes_.tolist(),
        len(model._neurobridge_calibration),
    )
```
With:
```python
    logger.info(
        "Trained BBB classifier: n=%d, n_features=%d, classes=%s, "
        "calibration_bins=%d, train_confidence_median=%.3f",
        len(y), X.shape[1], model.classes_.tolist(),
        len(model._neurobridge_calibration),
        model._neurobridge_train_stats["median"],
    )
```

### Step 4: Run the new tests — verify GREEN

- [ ] Run:

```bash
pytest tests/models/test_bbb_model.py::TestTrainStatsMetadata -v
```
Expected: **2 passed**.

### Step 5: Run the full suite — verify no regression

- [ ] Run:

```bash
pytest -q 2>&1 | tail -3
```
Expected: **167 passed** (165 + 2 new).

If any pre-existing test fails, the prime suspect is a model-equality assert that now fails because `_neurobridge_train_stats` was added. Read the failure; if it's a `model == reloaded_model` style check, update the assertion to `model._neurobridge_fp_cols == reloaded._neurobridge_fp_cols and model._neurobridge_calibration == reloaded._neurobridge_calibration and model._neurobridge_train_stats == reloaded._neurobridge_train_stats`. **Do not weaken assertions; expand them.**

### Step 6: Commit T1A

- [ ] Run:

```bash
git add src/models/bbb_model.py tests/models/test_bbb_model.py
git commit -m "$(cat <<'EOF'
feat(models): train-time confidence stats stashed on _neurobridge_train_stats

- _compute_train_stats() captures median, std, n_train of the model's
  own predict_proba on X_train. Joblib-roundtrip-safe.
- train() persists stats alongside _neurobridge_fp_cols and
  _neurobridge_calibration. INFO log line now surfaces the median.
- Foundation for Day-7 T1B drift z-score in /predict/bbb.
- 2 new tests (TestTrainStatsMetadata): attribute presence + roundtrip.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 1B — Drift z-score in /predict/bbb

**Why:** Surface "Adapt Over Time" to the jury. Each prediction's confidence is appended to a per-worker `deque(maxlen=100)`. When ≥10 samples are buffered, we compute a z-score against the train-time median. The number flows through the API response into the UI (T1C) and the LLM explainer (T3A).

**Files:**
- Modify: `src/api/schemas.py`
- Modify: `src/api/routes.py`
- Modify: `tests/api/test_routes.py`

### Step 1: Extend `BBBPredictResponse` schema

- [ ] Open `/Users/mertgungor/Desktop/hackathon/src/api/schemas.py`. Find the `BBBPredictResponse` class (it currently has `label`, `label_text`, `confidence`, `top_features`, `calibration`). Add two new optional fields:

```python
class BBBPredictResponse(BaseModel):
    """Decision-system payload: prediction + uncertainty + explanation + drift."""
    label: int
    label_text: str = Field(..., description="'permeable' or 'non-permeable'")
    confidence: float
    top_features: list[FeatureAttribution]
    calibration: CalibrationContext | None = None
    drift_z: float | None = Field(
        None,
        description=(
            "Z-score of the trailing-100 confidence median against the "
            "train-time median; None when warming up (<10 samples) or "
            "when the model lacks _neurobridge_train_stats."
        ),
    )
    rolling_n: int = Field(
        0,
        description=(
            "Number of confidence samples currently buffered in the worker's "
            "rolling window (max 100). Zero on a fresh worker."
        ),
    )
```

### Step 2: Write the 2 failing tests (RED)

- [ ] Open `/Users/mertgungor/Desktop/hackathon/tests/api/test_routes.py`. Find the `TestBBBPredictRoute` class. Add two NEW test methods inside that class (place them after the existing `test_returns_200_with_prediction_and_attributions`):

```python
    def test_predict_response_includes_drift_z_and_rolling_n(
        self, _set_bbb_model_path,
    ):
        """T1B: drift_z and rolling_n keys must always appear in the body."""
        # Reset deque before this test so rolling_n starts deterministic.
        from src.api import routes
        routes.WORKER_CONFIDENCE_DEQUE.clear()

        resp = client.post("/predict/bbb", json={"smiles": "CCO", "top_k": 5})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "drift_z" in body
        assert "rolling_n" in body
        # First request: buffer has 1 sample (just appended), so warming up.
        assert body["rolling_n"] == 1
        assert body["drift_z"] is None  # <10 samples = warming up

    def test_predict_deque_rolls_at_100(self, _set_bbb_model_path):
        """T1B: after 100 predictions, deque caps at maxlen=100 (rolls)."""
        from src.api import routes
        routes.WORKER_CONFIDENCE_DEQUE.clear()
        # Fire 105 calls; final rolling_n must be 100, not 105.
        last_body = None
        for _ in range(105):
            resp = client.post(
                "/predict/bbb", json={"smiles": "CCO", "top_k": 3},
            )
            assert resp.status_code == 200
            last_body = resp.json()
        assert last_body["rolling_n"] == 100
        # By call 105, drift_z is computable (≥10 samples) — assert numeric.
        assert isinstance(last_body["drift_z"], float)
```

### Step 3: Run the new tests — verify RED

- [ ] Run:

```bash
pytest tests/api/test_routes.py::TestBBBPredictRoute::test_predict_response_includes_drift_z_and_rolling_n -v
pytest tests/api/test_routes.py::TestBBBPredictRoute::test_predict_deque_rolls_at_100 -v
```
Expected: both **FAIL** — the deque doesn't exist yet (`AttributeError: module 'src.api.routes' has no attribute 'WORKER_CONFIDENCE_DEQUE'`).

### Step 4: Implement deque + drift helper + wire into `predict_bbb` (GREEN)

- [ ] Open `/Users/mertgungor/Desktop/hackathon/src/api/routes.py`. Add `from collections import deque` to the imports (alphabetical order):

```python
from collections import deque
```

- [ ] Just below the `_DEFAULT_BBB_MODEL_PATH = Path(...)` line (after the `_bbb_model_path()` helper), add the module-level deque + helper:

```python
# Per-worker rolling window of recent prediction confidences.
# Cleared on worker restart; multi-worker setups have independent windows.
WORKER_CONFIDENCE_DEQUE: deque[float] = deque(maxlen=100)
_DRIFT_MIN_SAMPLES = 10


def _compute_drift_z(model, confidence: float) -> tuple[float | None, int]:
    """Append `confidence` to the worker deque and compute the drift z-score.

    Returns (drift_z, rolling_n). drift_z is None until both:
      (1) the deque has at least `_DRIFT_MIN_SAMPLES` samples, AND
      (2) the model has `_neurobridge_train_stats` attached.

    z = (rolling_median - train_median) / max(train_std, 1e-9)
    """
    import statistics

    WORKER_CONFIDENCE_DEQUE.append(float(confidence))
    rolling_n = len(WORKER_CONFIDENCE_DEQUE)
    stats = getattr(model, "_neurobridge_train_stats", None)
    if rolling_n < _DRIFT_MIN_SAMPLES or stats is None:
        return None, rolling_n
    rolling_median = statistics.median(WORKER_CONFIDENCE_DEQUE)
    train_median = float(stats["median"])
    train_std = max(float(stats["std"]), 1e-9)
    drift_z = (rolling_median - train_median) / train_std
    return float(drift_z), rolling_n
```

- [ ] In `predict_bbb()`, immediately before the `return BBBPredictResponse(...)` block, compute drift:

```python
    drift_z, rolling_n = _compute_drift_z(model, pred["confidence"])
```

- [ ] Update the `return BBBPredictResponse(...)` to pass the new fields:

```python
    return BBBPredictResponse(
        label=pred["label"],
        label_text=label_text,
        confidence=pred["confidence"],
        top_features=[FeatureAttribution(**a) for a in attributions],
        calibration=calibration,
        drift_z=drift_z,
        rolling_n=rolling_n,
    )
```

### Step 5: Run the new tests — verify GREEN

- [ ] Run:

```bash
pytest tests/api/test_routes.py::TestBBBPredictRoute -v
```
Expected: **all TestBBBPredictRoute tests pass** (including the 2 new ones, totalling whatever was there before + 2 = currently 3 + 2 = 5 in this class).

### Step 6: Run the full suite — verify no regression

- [ ] Run:

```bash
pytest -q 2>&1 | tail -3
```
Expected: **169 passed** (167 + 2 new).

### Step 7: Commit T1B

- [ ] Run:

```bash
git add src/api/schemas.py src/api/routes.py tests/api/test_routes.py
git commit -m "$(cat <<'EOF'
feat(api): drift z-score in /predict/bbb response

- WORKER_CONFIDENCE_DEQUE: collections.deque(maxlen=100), per-worker
  rolling window of confidences; drift_z computed against train-time
  median when ≥10 samples buffered AND model has _neurobridge_train_stats.
- BBBPredictResponse gains drift_z (float | None) and rolling_n (int).
- 2 new tests: drift_z/rolling_n always present in body; deque rolls
  at 100 after 105 predictions.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 1C — Streamlit Drift Metric Line

**Why:** Without a UI surface, the drift signal is invisible to the jury. Render a one-line caption between the calibration caption and the SHAP section in `_render_prediction_card`.

**Files:**
- Modify: `src/frontend/app.py`

No new tests — UI wiring covered by the existing 2 import-smoke tests. Frontend test floor stays at 2.

### Step 1: Locate `_render_prediction_card`

- [ ] Open `/Users/mertgungor/Desktop/hackathon/src/frontend/app.py`. Find `_render_prediction_card(result)`. The function currently renders (in order): label badge → confidence progress → calibration caption → SHAP section. Drift goes between calibration and SHAP.

### Step 2: Add the drift line block

- [ ] Inside `_render_prediction_card`, immediately AFTER the existing calibration caption block (the `if calibration is not None:` / `elif calibration is not None:` block) and BEFORE the SHAP section header (the `st.markdown("**Top {n_features} SHAP attributions**" …)` or equivalent), insert:

```python
    drift_z = result.get("drift_z")
    rolling_n = result.get("rolling_n", 0)
    if drift_z is None and rolling_n < 10:
        st.caption(
            f"📈 Drift: warming up ({rolling_n}/10 predictions buffered)."
        )
    elif drift_z is None:
        st.caption(
            "📈 Drift: unavailable (model lacks train-time confidence stats)."
        )
    else:
        # Sign + magnitude: |z| < 1 in-band, 1–2 mild, >=2 significant.
        if abs(drift_z) < 1.0:
            tag = "within expected range"
        elif abs(drift_z) < 2.0:
            tag = "mild distribution shift"
        else:
            tag = "significant shift — retrain recommended"
        st.caption(
            f"📈 Drift: trailing-{rolling_n} confidence median is "
            f"**{drift_z:+.2f}σ** from train-time distribution ({tag})."
        )
```

### Step 3: Persist the last prediction in session state

- [ ] Inside `_render_prediction_card`, at the very TOP of the function body (before any other call), add:

```python
    st.session_state["last_bbb_prediction"] = result
```

This unlocks the AI Assistant tab in T3C — the tab can read `st.session_state["last_bbb_prediction"]` to populate its question form.

### Step 4: Smoke test

- [ ] Verify import + Streamlit boot:

```bash
pytest tests/frontend/ -v
```
Expected: **2 passed**.

```bash
streamlit run src/frontend/app.py --server.headless true --server.port 8530 &
STREAMLIT_PID=$!
sleep 6
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8530
kill $STREAMLIT_PID 2>/dev/null
sleep 1
```
Expected: HTTP `200`.

### Step 5: Full suite — verify no regression

- [ ] Run:

```bash
pytest -q 2>&1 | tail -3
```
Expected: **169 passed** (no count change from T1B; UI-only).

### Step 6: Commit T1C

- [ ] Run:

```bash
git add src/frontend/app.py
git commit -m "$(cat <<'EOF'
feat(frontend): drift metric line + last-prediction session state

- Renders one-line drift caption between the calibration caption and
  the SHAP section. Three states: warming up (<10 samples), unavailable
  (no train stats), drift z-score with magnitude tag (in-band / mild /
  significant).
- Stashes /predict/bbb response in st.session_state["last_bbb_prediction"]
  so the Day-7 T3C AI Assistant tab can pick it up.
- No backend / schema / test count changes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2 — MLflow Traceability Badge

**Why:** Spec §3.2. Jurors should be able to point at a decision card and ask "which exact training run produced this?". One smoke test on the API (the `provenance` field appears in the body), one badge in the UI.

**Files:**
- Modify: `src/api/schemas.py`
- Modify: `src/api/routes.py`
- Modify: `src/frontend/app.py`
- Modify: `tests/api/test_routes.py`

### Step 1: Add `ModelProvenance` schema

- [ ] Open `/Users/mertgungor/Desktop/hackathon/src/api/schemas.py`. Append (above `BBBPredictResponse` so the type is in scope when referenced):

Find the line `class BBBPredictResponse(BaseModel):` and add this class IMMEDIATELY ABOVE it:

```python
class ModelProvenance(BaseModel):
    """Auditable provenance of the BBB model that produced a prediction."""
    mlflow_run_id: str | None = Field(None, description="MLflow run id of the most recent training run, if any")
    model_version: str = Field("v1", description="Manually-bumped model version label")
    train_date: str | None = Field(None, description="ISO 8601 train timestamp from MLflow run start_time")
    n_examples: int | None = Field(None, description="Training set size (from model._neurobridge_train_stats[\"n_train\"])")
```

- [ ] Modify `BBBPredictResponse` to add a `provenance` field at the end:

```python
    provenance: ModelProvenance | None = Field(
        None,
        description="Auditing metadata (MLflow run id, train date, n_examples).",
    )
```

### Step 2: Write the failing test (RED)

- [ ] Open `/Users/mertgungor/Desktop/hackathon/tests/api/test_routes.py`. Inside `TestBBBPredictRoute`, append:

```python
    def test_predict_response_includes_provenance(self, _set_bbb_model_path):
        """T2: provenance field is present in body (fields may be None)."""
        from src.api import routes
        routes.WORKER_CONFIDENCE_DEQUE.clear()

        resp = client.post("/predict/bbb", json={"smiles": "CCO", "top_k": 3})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "provenance" in body
        assert body["provenance"] is not None, "provenance should be populated even when MLflow is empty"
        prov = body["provenance"]
        assert "mlflow_run_id" in prov
        assert "model_version" in prov
        assert prov["model_version"] == "v1"  # default until bumped manually
        assert "train_date" in prov
        assert "n_examples" in prov
        # n_examples comes from train_stats — must be a positive int for the test fixture
        assert isinstance(prov["n_examples"], int) and prov["n_examples"] >= 1
```

### Step 3: Run the test — verify RED

- [ ] Run:

```bash
pytest tests/api/test_routes.py::TestBBBPredictRoute::test_predict_response_includes_provenance -v
```
Expected: **FAIL** — `assert "provenance" in body` fails because no route populates it yet.

### Step 4: Implement provenance lookup + cache (GREEN)

- [ ] Open `/Users/mertgungor/Desktop/hackathon/src/api/routes.py`. Add the schema import:

```python
from src.api.schemas import (
    BBBPredictRequest,
    BBBPredictResponse,
    BBBRequest,
    CalibrationContext,
    EEGRequest,
    FeatureAttribution,
    HarmonizationRow,
    ModelProvenance,                 # NEW
    MRIDiagnosticsRequest,
    MRIDiagnosticsResponse,
    MRIRequest,
    PipelineResponse,
)
```

- [ ] Below the `_compute_drift_z` helper, add a provenance lookup helper. The cache is module-level so MLflow is queried once per worker:

```python
_PROVENANCE_CACHE: ModelProvenance | None = None
_MODEL_VERSION = "v1"  # bump manually per train cycle


def _build_provenance(model) -> ModelProvenance:
    """Look up the most recent BBB MLflow run; build a ModelProvenance.

    Cached at module level so we hit MLflow once per worker. Failures (no
    runs found, MLflow unreachable, NEUROBRIDGE_DISABLE_MLFLOW=1) all
    degrade to a partial ModelProvenance with mlflow_run_id=None — the
    badge still renders, just without a run id.
    """
    global _PROVENANCE_CACHE
    if _PROVENANCE_CACHE is not None:
        # Refresh n_examples each call from the model (cheap lookup).
        n_train = None
        stats = getattr(model, "_neurobridge_train_stats", None)
        if stats is not None:
            n_train = int(stats.get("n_train", 0)) or None
        return _PROVENANCE_CACHE.model_copy(update={"n_examples": n_train})

    run_id: str | None = None
    train_date: str | None = None
    if os.environ.get("NEUROBRIDGE_DISABLE_MLFLOW") != "1":
        try:
            runs = mlflow.search_runs(
                experiment_names=["bbb_pipeline"],
                max_results=1,
                order_by=["start_time DESC"],
            )
            if len(runs):
                row = runs.iloc[0]
                run_id = str(row["run_id"])
                ts = row.get("start_time")
                if ts is not None:
                    train_date = str(pd.Timestamp(ts).isoformat())
        except Exception as e:  # broad: MLflow store unreachable, schema mismatch, etc.
            logger.warning("MLflow provenance lookup failed: %s", e)

    n_train = None
    stats = getattr(model, "_neurobridge_train_stats", None)
    if stats is not None:
        n_train = int(stats.get("n_train", 0)) or None

    _PROVENANCE_CACHE = ModelProvenance(
        mlflow_run_id=run_id,
        model_version=_MODEL_VERSION,
        train_date=train_date,
        n_examples=n_train,
    )
    return _PROVENANCE_CACHE
```

- [ ] In `predict_bbb()`, immediately after `drift_z, rolling_n = _compute_drift_z(...)`, add:

```python
    provenance = _build_provenance(model)
```

- [ ] Update the `return BBBPredictResponse(...)` to pass `provenance=provenance`:

```python
    return BBBPredictResponse(
        label=pred["label"],
        label_text=label_text,
        confidence=pred["confidence"],
        top_features=[FeatureAttribution(**a) for a in attributions],
        calibration=calibration,
        drift_z=drift_z,
        rolling_n=rolling_n,
        provenance=provenance,
    )
```

### Step 5: Run the test — verify GREEN

- [ ] Run:

```bash
pytest tests/api/test_routes.py::TestBBBPredictRoute::test_predict_response_includes_provenance -v
```
Expected: **PASS**.

### Step 6: Render badge in Streamlit decision card

- [ ] Open `/Users/mertgungor/Desktop/hackathon/src/frontend/app.py`. In `_render_prediction_card`, immediately after the line `st.session_state["last_bbb_prediction"] = result` (added in T1C Step 3) and BEFORE the existing label badge, add:

```python
    provenance = result.get("provenance")
    if provenance is not None:
        run_id = provenance.get("mlflow_run_id")
        run_label = run_id[:8] if run_id else "—"
        train_date = provenance.get("train_date") or "—"
        n_examples = provenance.get("n_examples")
        n_label = f"n={n_examples}" if n_examples else "n=—"
        st.caption(
            f"🔎 MLflow run **{run_label}** · "
            f"Model **{provenance.get('model_version', 'v1')}** · "
            f"trained {train_date} · {n_label}"
        )
```

### Step 7: Full suite — verify no regression

- [ ] Run:

```bash
pytest -q 2>&1 | tail -3
```
Expected: **170 passed** (169 + 1 new).

### Step 8: Streamlit smoke

- [ ] Run:

```bash
streamlit run src/frontend/app.py --server.headless true --server.port 8531 &
STREAMLIT_PID=$!
sleep 6
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8531
kill $STREAMLIT_PID 2>/dev/null
sleep 1
```
Expected: HTTP `200`.

### Step 9: Commit T2

- [ ] Run:

```bash
git add src/api/schemas.py src/api/routes.py src/frontend/app.py tests/api/test_routes.py
git commit -m "$(cat <<'EOF'
feat(api+frontend): MLflow provenance badge in decision card

- ModelProvenance schema (mlflow_run_id, model_version, train_date,
  n_examples). BBBPredictResponse.provenance is always populated; failed
  MLflow lookup degrades to None fields without breaking the response.
- _build_provenance() module-level cache: one MLflow query per worker.
  NEUROBRIDGE_DISABLE_MLFLOW=1 short-circuits to None fields. n_examples
  pulled per-request from model._neurobridge_train_stats.
- Streamlit decision card renders a one-line audit badge above the
  label: run id (first 8 chars), model version, train date, n_examples.
- 1 new test: provenance field present in /predict/bbb body with the
  fixture model (n_examples ≥ 1 from train stats).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3A — LLM Explainer (template + OpenRouter)

**Why:** This is the heart of the Track-1 "AI Lab Agents" wink. A small, self-contained module that ALWAYS returns a usable rationale: deterministic template for reproducibility, OpenRouter llama-3.2-3b-instruct (free) for the "real agent" demo. Spec §3.3.

**Files:**
- Modify: `requirements.txt`
- Create: `src/llm/__init__.py`
- Create: `src/llm/explainer.py`
- Create: `tests/llm/__init__.py`
- Create: `tests/llm/test_explainer.py`

### Step 1: Add the new pip dep + install

- [ ] Open `/Users/mertgungor/Desktop/hackathon/requirements.txt`. Add `openai==1.51.0` in the appropriate alphabetical position (after `nibabel==…` if alphabetical, or at the end if grouped). To match existing style: scan the file with `head` first; if no clear ordering, append at the end with a comment:

Append:
```
openai==1.51.0  # OpenRouter SDK (Day-7 LLM explainer; deterministic-template fallback always available)
```

- [ ] Install:

```bash
pip install openai==1.51.0
pip check 2>&1 | tail -5
```
Expected: `pip check` reports no incompatibilities. If a conflict appears (e.g. with `httpx==0.27.2`), STOP and resolve before continuing — the spec sealed compatibility.

### Step 2: Create `src/llm/__init__.py`

- [ ] Run:

```bash
mkdir -p src/llm tests/llm
```

- [ ] Create `/Users/mertgungor/Desktop/hackathon/src/llm/__init__.py` with this exact content:

```python
"""LLM-backed natural-language explainers (Day 7).

`explain()` is the ONLY public entry point. It guarantees a non-empty
rationale every call: tries OpenRouter when available, falls back to a
deterministic template otherwise. The deterministic path is the source
of truth for tests; the LLM path is gated behind env config.
"""
from src.llm.explainer import ExplainPayload, ExplainResult, explain  # noqa: F401
```

### Step 3: Write the 4 failing tests (RED)

- [ ] Create `/Users/mertgungor/Desktop/hackathon/tests/llm/__init__.py` (empty):

```python
```

- [ ] Create `/Users/mertgungor/Desktop/hackathon/tests/llm/test_explainer.py` with this exact content:

```python
"""Tests for src.llm.explainer.

The deterministic template path is exhaustively tested here. The LLM
path is exercised only by env-gated integration tests in
test_explainer_integration.py (NOT run in CI by default).
"""
from __future__ import annotations

import os

import pytest

from src.llm.explainer import ExplainPayload, explain


def _payload(**overrides) -> ExplainPayload:
    """Build a representative ExplainPayload; overrides win."""
    base: ExplainPayload = {
        "smiles": "CCO",
        "label": 1,
        "label_text": "permeable",
        "confidence": 0.82,
        "top_features": [
            {"feature": "fp_341", "shap_value": 0.045},
            {"feature": "fp_902", "shap_value": -0.031},
            {"feature": "fp_77", "shap_value": 0.022},
        ],
        "calibration": {"threshold": 0.80, "precision": 0.92, "support": 18},
        "drift_z": 0.42,
        "user_question": "Why was this molecule predicted as permeable?",
    }
    base.update(overrides)
    return base


class TestTemplateExplain:
    """Day-7 T3A: deterministic-template path of the explainer."""

    def test_template_path_is_deterministic(self, monkeypatch):
        """Same input → byte-identical rationale string. No randomness."""
        monkeypatch.setenv("NEUROBRIDGE_DISABLE_LLM", "1")
        out_a = explain(_payload())
        out_b = explain(_payload())
        assert out_a["rationale"] == out_b["rationale"]
        assert out_a["source"] == "template"
        assert out_b["source"] == "template"
        assert out_a["model"] is None

    def test_template_includes_top_feature_names(self, monkeypatch):
        """Rationale must mention the SHAP features so jurors see attribution."""
        monkeypatch.setenv("NEUROBRIDGE_DISABLE_LLM", "1")
        result = explain(_payload())
        for feat in ("fp_341", "fp_902", "fp_77"):
            assert feat in result["rationale"], (
                f"expected feature {feat!r} in rationale, got {result['rationale']!r}"
            )

    def test_template_includes_label_text(self, monkeypatch):
        """The verdict word ('permeable' / 'non-permeable') must appear."""
        monkeypatch.setenv("NEUROBRIDGE_DISABLE_LLM", "1")
        result = explain(_payload(label=0, label_text="non-permeable"))
        assert "non-permeable" in result["rationale"]

    def test_disable_flag_forces_template_even_with_key_set(self, monkeypatch):
        """NEUROBRIDGE_DISABLE_LLM=1 wins over OPENROUTER_API_KEY presence."""
        monkeypatch.setenv("NEUROBRIDGE_DISABLE_LLM", "1")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-fake-not-used")
        result = explain(_payload())
        assert result["source"] == "template"
        assert result["model"] is None
```

### Step 4: Run the new tests — verify RED

- [ ] Run:

```bash
pytest tests/llm/ -v
```
Expected: 4 errors / fails — `ModuleNotFoundError: No module named 'src.llm.explainer'` (file doesn't exist yet). If by some accident the module exists, the tests will fail because `explain` is not implemented.

### Step 5: Implement `src/llm/explainer.py` (GREEN)

- [ ] Create `/Users/mertgungor/Desktop/hackathon/src/llm/explainer.py` with this exact content:

```python
"""Natural-language rationale for a single BBB prediction.

Public entry point: `explain(payload)`. Always returns a usable
ExplainResult — never raises. Tries OpenRouter first when a key is set
and the kill-switch is off; falls back to a deterministic template on
any failure (network, auth, rate limit, malformed response).

Test discipline: deterministic template path is the source of truth.
LLM path is env-gated and exercised by integration tests only.
"""
from __future__ import annotations

import os
from typing import Any, TypedDict

from src.core.logger import get_logger

logger = get_logger(__name__)


class FeatureRow(TypedDict):
    feature: str
    shap_value: float


class CalibrationDict(TypedDict):
    threshold: float
    precision: float
    support: int


class ExplainPayload(TypedDict, total=False):
    smiles: str
    label: int
    label_text: str
    confidence: float
    top_features: list[FeatureRow]
    calibration: CalibrationDict | None
    drift_z: float | None
    user_question: str


class ExplainResult(TypedDict):
    rationale: str
    source: str          # "llm" | "template"
    model: str | None    # llm model name when source="llm", else None


_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_DEFAULT_MODEL = "meta-llama/llama-3.2-3b-instruct:free"
_LLM_TIMEOUT_SECONDS = 8.0
_LLM_MAX_TOKENS = 256
_LLM_TEMPERATURE = 0.3


def _should_use_llm() -> bool:
    """Gate: env kill-switch off AND key present."""
    if os.environ.get("NEUROBRIDGE_DISABLE_LLM") == "1":
        return False
    if not os.environ.get("OPENROUTER_API_KEY"):
        return False
    return True


def _drift_interpretation(drift_z: float | None) -> str:
    if drift_z is None:
        return "drift unavailable"
    mag = abs(drift_z)
    if mag < 1.0:
        return "within expected range"
    if mag < 2.0:
        return "mild distribution shift"
    return "significant shift, retrain recommended"


def _template_explain(payload: ExplainPayload) -> str:
    """Deterministic, jury-friendly rationale. Never raises."""
    label_text = payload.get("label_text", "unknown")
    confidence = float(payload.get("confidence", 0.0))
    top_features = payload.get("top_features") or []

    # Sentence 1
    sentences = [
        f"Predicted **{label_text}** with {confidence * 100:.0f}% confidence."
    ]

    # Sentence 2 (calibration, optional)
    cal = payload.get("calibration")
    if cal is not None:
        thr_pct = float(cal["threshold"]) * 100
        prec_pct = float(cal["precision"]) * 100
        support = int(cal["support"])
        if support > 0:
            sentences.append(
                f"Calibration: predictions in the ≥{thr_pct:.0f}% bin are "
                f"correct {prec_pct:.0f}% of the time on held-out data "
                f"(n={support})."
            )

    # Sentence 3 (top-3 SHAP features)
    if top_features:
        feat_strs = [
            f"{row['feature']} (Δ{float(row['shap_value']):+.3f})"
            for row in top_features[:3]
        ]
        sentences.append(
            f"Top SHAP attributions toward this label: {', '.join(feat_strs)}."
        )

    # Sentence 4 (drift, optional)
    drift_z = payload.get("drift_z")
    if drift_z is not None:
        interp = _drift_interpretation(drift_z)
        sentences.append(
            f"Drift signal: trailing-100 confidence median is "
            f"{float(drift_z):+.2f}σ from training distribution ({interp})."
        )

    return " ".join(sentences)


def _build_llm_prompt(payload: ExplainPayload) -> str:
    """Format the payload + user question into a single LLM prompt."""
    top_features = payload.get("top_features") or []
    top_lines = "\n".join(
        f"  - {row['feature']}: Δ{float(row['shap_value']):+.3f}"
        for row in top_features[:5]
    ) or "  - (none)"
    drift_z = payload.get("drift_z")
    drift_str = "n/a" if drift_z is None else f"{float(drift_z):+.2f}"
    user_q = payload.get("user_question") or (
        "Explain the prediction in 2-4 sentences."
    )
    return (
        "You are a clinical-ML explainer for a B2B blood-brain-barrier "
        "permeability tool. Given the prediction details below, write a "
        "2-4 sentence rationale a researcher could paste into a paper. "
        "Use the SHAP attributions to justify the verdict. Mention drift "
        "if abnormal. Avoid hedging; be specific about the numbers.\n\n"
        f"Prediction:\n"
        f"- SMILES: {payload.get('smiles', '?')}\n"
        f"- Verdict: {payload.get('label_text', '?')} "
        f"({float(payload.get('confidence', 0.0)) * 100:.0f}% confident)\n"
        f"- Top SHAP features (positive = pushed toward verdict):\n"
        f"{top_lines}\n"
        f"- Drift z-score: {drift_str}\n"
        f"\nUser question: {user_q}\n"
        f"\nRespond with the rationale only, no preamble."
    )


def _llm_explain(payload: ExplainPayload) -> tuple[str, str] | None:
    """Try the OpenRouter chat completion. Return (rationale, model) or None."""
    try:
        # Local import — keeps this dep optional at module load time.
        from openai import OpenAI
    except ImportError as e:
        logger.warning("openai SDK not importable: %s", e)
        return None

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return None

    client = OpenAI(
        base_url=_OPENROUTER_BASE_URL,
        api_key=api_key,
        timeout=_LLM_TIMEOUT_SECONDS,
    )
    prompt = _build_llm_prompt(payload)
    try:
        completion = client.chat.completions.create(
            model=_DEFAULT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=_LLM_MAX_TOKENS,
            temperature=_LLM_TEMPERATURE,
        )
    except Exception as e:  # broad: APITimeoutError, APIConnectionError, RateLimitError, ...
        logger.warning("LLM call failed (%s); falling back to template.", type(e).__name__)
        return None

    try:
        text = completion.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as e:
        logger.warning("LLM response malformed (%s); falling back to template.", e)
        return None

    if not text or not text.strip():
        logger.warning("LLM returned empty rationale; falling back to template.")
        return None

    return text.strip(), _DEFAULT_MODEL


def explain(payload: ExplainPayload) -> ExplainResult:
    """Return a natural-language rationale for a BBB prediction.

    Tries the LLM first when env-permitted; falls back to a deterministic
    template on any failure. Never raises.
    """
    if _should_use_llm():
        llm_out: Any = _llm_explain(payload)
        if llm_out is not None:
            rationale, model = llm_out
            return ExplainResult(rationale=rationale, source="llm", model=model)
        # else: fall through to template
    return ExplainResult(
        rationale=_template_explain(payload),
        source="template",
        model=None,
    )
```

### Step 6: Run the new tests — verify GREEN

- [ ] Run:

```bash
pytest tests/llm/ -v
```
Expected: **4 passed**.

### Step 7: Full suite — verify no regression

- [ ] Run:

```bash
pytest -q 2>&1 | tail -3
```
Expected: **174 passed** (170 + 4 new).

### Step 8: UserWarning gate

- [ ] Verify the new `openai` import doesn't introduce sklearn-style UserWarnings:

```bash
pytest -W error::UserWarning tests/ 2>&1 | tail -3
```
Expected: same count (174), 0 UserWarning errors.

### Step 9: Commit T3A

- [ ] Run:

```bash
git add requirements.txt src/llm/ tests/llm/
git commit -m "$(cat <<'EOF'
feat(llm): explainer with deterministic template + OpenRouter fallback

- New module src/llm/explainer.py — single public entry point
  explain(payload). Returns {rationale, source, model}. Never raises.
- Deterministic template (4 sentences: verdict, calibration if any,
  top-3 SHAP, drift) is the source of truth for tests.
- LLM path: OpenRouter chat completions via openai==1.51.0 SDK,
  model meta-llama/llama-3.2-3b-instruct:free, 8s timeout, 256 max
  tokens, temperature 0.3. Gated by OPENROUTER_API_KEY presence and
  NEUROBRIDGE_DISABLE_LLM=1 kill-switch.
- Fallback chain: env-disabled → no key → SDK ImportError → API error
  → empty/malformed response → all degrade to template, log WARNING,
  source="template".
- 4 new tests: deterministic, top features included, label text
  included, kill-switch overrides key.
- New pip dep: openai==1.51.0 (~600KB, transitive deps already present).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3B — POST /explain/bbb Route

**Why:** Wire the explainer into the API surface so the Streamlit AI Assistant tab (T3C) can call it. Spec §3.4: new `explain_router` with `/explain` prefix.

**Files:**
- Modify: `src/api/schemas.py`
- Modify: `src/api/routes.py`
- Modify: `src/api/__init__.py` (or wherever the FastAPI app is assembled — verify in step 1)
- Modify: `tests/api/test_routes.py`

### Step 1: Locate the FastAPI app + router registration

- [ ] Find where `router` and `predict_router` are mounted on the FastAPI app:

```bash
grep -rn "include_router" /Users/mertgungor/Desktop/hackathon/src/
```
The output will point to a `main.py` or similar (likely `src/api/main.py`). Note the file path; we'll add `app.include_router(explain_router)` there.

### Step 2: Add `BBBExplainRequest` and `BBBExplainResponse` schemas

- [ ] Open `/Users/mertgungor/Desktop/hackathon/src/api/schemas.py`. Append at the bottom of the file:

```python
class BBBExplainRequest(BaseModel):
    """Day-7 T3B: payload for POST /explain/bbb (chat-style explainer)."""
    smiles: str = Field(..., description="SMILES string of the molecule")
    label: int = Field(..., description="Predicted label (0 = non-permeable, 1 = permeable)")
    label_text: str = Field(..., description="'permeable' or 'non-permeable'")
    confidence: float = Field(..., ge=0.0, le=1.0)
    top_features: list[FeatureAttribution] = Field(
        ..., min_length=1,
        description="Non-empty list of SHAP attributions; an empty list returns 400.",
    )
    calibration: CalibrationContext | None = None
    drift_z: float | None = None
    user_question: str | None = Field(
        None,
        description="Optional question from the user; passed to the LLM prompt only.",
    )


class BBBExplainResponse(BaseModel):
    """Day-7 T3B: response from POST /explain/bbb."""
    rationale: str = Field(..., description="2-4 sentence natural-language explanation")
    source: str = Field(..., description="'llm' or 'template'")
    model: str | None = Field(
        None,
        description="LLM model name when source='llm'; None when source='template'",
    )
```

### Step 3: Write the failing test (RED)

- [ ] In `/Users/mertgungor/Desktop/hackathon/tests/api/test_routes.py`, append at the very bottom (after `TestMRIDiagnosticsRoute`):

```python
class TestExplainBBBRoute:
    """Day-7 T3B: POST /explain/bbb."""

    def test_returns_200_with_template_source(self, monkeypatch):
        """Kill-switch on → /explain/bbb returns rationale with source=template."""
        monkeypatch.setenv("NEUROBRIDGE_DISABLE_LLM", "1")
        body = {
            "smiles": "CCO",
            "label": 1,
            "label_text": "permeable",
            "confidence": 0.82,
            "top_features": [
                {"feature": "fp_341", "shap_value": 0.045},
                {"feature": "fp_902", "shap_value": -0.031},
                {"feature": "fp_77", "shap_value": 0.022},
            ],
            "calibration": {"threshold": 0.80, "precision": 0.92, "support": 18},
            "drift_z": 0.42,
            "user_question": "Why permeable?",
        }
        resp = client.post("/explain/bbb", json=body)
        assert resp.status_code == 200, resp.text
        out = resp.json()
        assert out["source"] == "template"
        assert out["model"] is None
        # Template must mention all three features
        for feat in ("fp_341", "fp_902", "fp_77"):
            assert feat in out["rationale"]
        assert "permeable" in out["rationale"]
```

### Step 4: Run the test — verify RED

- [ ] Run:

```bash
pytest tests/api/test_routes.py::TestExplainBBBRoute -v
```
Expected: **FAIL with 404 Not Found** — `/explain/bbb` doesn't exist yet.

### Step 5: Add the route + schema imports + router registration (GREEN)

- [ ] Open `/Users/mertgungor/Desktop/hackathon/src/api/routes.py`. Add the new schemas to the import block (alphabetical):

```python
from src.api.schemas import (
    BBBExplainRequest,                # NEW
    BBBExplainResponse,               # NEW
    BBBPredictRequest,
    BBBPredictResponse,
    BBBRequest,
    CalibrationContext,
    EEGRequest,
    FeatureAttribution,
    HarmonizationRow,
    ModelProvenance,
    MRIDiagnosticsRequest,
    MRIDiagnosticsResponse,
    MRIRequest,
    PipelineResponse,
)
```

Add the explainer module import (alphabetical with other `src.*` imports):

```python
from src.llm import explainer as llm_explainer
```

Add a new router declaration immediately after the existing `predict_router` line (around line 38):

```python
explain_router = APIRouter(prefix="/explain")
```

Append the route at the end of the file:

```python
@explain_router.post("/bbb", response_model=BBBExplainResponse)
def explain_bbb(req: BBBExplainRequest) -> BBBExplainResponse:
    """Natural-language rationale for a single BBB prediction.

    Always returns 200 — the explainer is guaranteed to produce a
    rationale via deterministic-template fallback. Pydantic enforces
    a non-empty top_features list; an empty list returns 422 from
    FastAPI before this handler runs.
    """
    payload: llm_explainer.ExplainPayload = {
        "smiles": req.smiles,
        "label": req.label,
        "label_text": req.label_text,
        "confidence": req.confidence,
        "top_features": [
            {"feature": f.feature, "shap_value": f.shap_value}
            for f in req.top_features
        ],
        "calibration": (
            None
            if req.calibration is None
            else {
                "threshold": req.calibration.threshold,
                "precision": req.calibration.precision,
                "support": req.calibration.support,
            }
        ),
        "drift_z": req.drift_z,
        "user_question": req.user_question or "",
    }
    result = llm_explainer.explain(payload)
    return BBBExplainResponse(
        rationale=result["rationale"],
        source=result["source"],
        model=result["model"],
    )
```

- [ ] Open `src/api/main.py` (or whichever file Step 1 identified). Find where `app.include_router(predict_router)` is called. Immediately after that line, add:

```python
from src.api.routes import explain_router  # if not already imported
app.include_router(explain_router)
```

(If `predict_router` is imported as `from src.api.routes import predict_router`, add `explain_router` to that same import.)

### Step 6: Run the test — verify GREEN

- [ ] Run:

```bash
pytest tests/api/test_routes.py::TestExplainBBBRoute -v
```
Expected: **PASS**.

### Step 7: Full suite — verify no regression

- [ ] Run:

```bash
pytest -q 2>&1 | tail -3
```
Expected: **175 passed** (174 + 1 new).

### Step 8: Commit T3B

- [ ] Run:

```bash
git add src/api/schemas.py src/api/routes.py src/api/main.py tests/api/test_routes.py
git commit -m "$(cat <<'EOF'
feat(api): POST /explain/bbb — natural-language rationale endpoint

- New explain_router with /explain prefix; symmetric with /predict/bbb
  and reserves /explain/eeg, /explain/mri for future expansion.
- BBBExplainRequest carries the prediction snapshot + optional
  user_question. top_features is required and must be non-empty
  (Pydantic min_length=1 → 422 on empty).
- BBBExplainResponse: {rationale, source, model}. Always 200 because
  the explainer's template fallback never raises.
- 1 new test: 200 + source='template' under NEUROBRIDGE_DISABLE_LLM=1
  with full SHAP + calibration + drift payload.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3C — Streamlit "AI Assistant" Tab

**Why:** Spec §3.5. Lets the jury type / pick a question and watch the system reason in natural language. Pulls the last `/predict/bbb` result from `st.session_state` (populated in T1C Step 3) and POSTs to `/explain/bbb`.

**Files:**
- Modify: `src/frontend/app.py`

No new tests — covered by the 2 existing import-smoke tests.

### Step 1: Locate the tab assembly

- [ ] Open `/Users/mertgungor/Desktop/hackathon/src/frontend/app.py`. Find the `main()` function. The current tabs are likely created via something like:

```python
tab_bbb, tab_eeg, tab_mri = st.tabs(["BBB", "EEG", "MRI"])
```

Note the exact line so we can extend it.

### Step 2: Extend the tabs list

- [ ] Replace the existing tab declaration with:

```python
tab_bbb, tab_eeg, tab_mri, tab_assistant = st.tabs(
    ["BBB", "EEG", "MRI", "AI Assistant"]
)
```

- [ ] Wherever the existing 3 tabs are rendered (`with tab_bbb: _render_bbb_tab()` etc.), append:

```python
with tab_assistant:
    _render_ai_assistant_tab()
```

### Step 3: Add the helper function `_render_ai_assistant_tab`

- [ ] Add this new function above `main()` (near the other `_render_*_tab` helpers):

```python
def _render_ai_assistant_tab() -> None:
    """Day-7 T3C: chat-style explainer for the most recent BBB prediction."""
    _render_section(
        "AI Assistant",
        "Natural-language rationale (LLM or deterministic template)",
        "Pulls the most recent BBB prediction from this session and asks "
        "the explainer to justify it. Falls back to a deterministic, "
        "auditable template when no LLM is configured."
    )

    last = st.session_state.get("last_bbb_prediction")
    if last is None:
        st.info(
            "Run a BBB prediction first (BBB tab → Predict button), "
            "then come back here to ask the assistant about it."
        )
        return

    # Snapshot card so the user knows which prediction is being explained
    st.caption(
        f"Latest prediction: **{last['label_text']}** "
        f"({float(last['confidence']) * 100:.0f}% confident)  ·  "
        f"Top SHAP: {', '.join(f['feature'] for f in last.get('top_features', [])[:3])}"
    )

    PRESETS = [
        "Why was this molecule predicted as permeable?",
        "Which features pushed the verdict the most?",
        "Is this prediction trustworthy given the drift signal?",
    ]
    preset = st.selectbox("Preset question", options=PRESETS, key="ai_preset")
    custom = st.text_input(
        "Or type your own question (optional)",
        value="",
        key="ai_custom",
        help="Custom questions only affect the LLM path; the template gives a generic SHAP-driven rationale either way.",
    )
    question = custom.strip() or preset

    if st.button("Ask the AI Assistant", type="primary", key="ai_ask"):
        with st.spinner("Composing rationale…"):
            try:
                body = {
                    "smiles": last.get("smiles", ""),
                    "label": last["label"],
                    "label_text": last["label_text"],
                    "confidence": last["confidence"],
                    "top_features": last.get("top_features", []),
                    "calibration": last.get("calibration"),
                    "drift_z": last.get("drift_z"),
                    "user_question": question,
                }
                # The /predict/bbb response payload doesn't include the
                # user-supplied SMILES (only label/confidence/etc.), so
                # pull it from the input widget for paper-trail accuracy.
                # Streamlit text inputs persist via st.session_state.
                if not body["smiles"]:
                    body["smiles"] = st.session_state.get("bbb_smiles", "")
                resp = _post("/explain/bbb", body)
            except httpx.HTTPStatusError as e:
                st.error(
                    f"Explainer failed (HTTP {e.response.status_code}): "
                    f"{e.response.text}"
                )
                return
            except httpx.RequestError as e:
                st.error(f"Cannot reach FastAPI at {_API_URL}: {e!r}")
                return

        history = st.session_state.setdefault("explain_history", [])
        history.insert(0, (question, resp))

    # Render history (most recent first)
    history = st.session_state.get("explain_history", [])
    if history:
        st.markdown("### Conversation")
        for q, r in history[:10]:  # cap at 10 most recent
            with st.container():
                st.markdown(f"**Q:** {q}")
                st.markdown(f"**A:** {r['rationale']}")
                source = r.get("source", "?")
                model = r.get("model") or "—"
                st.caption(f"Source: `{source}`  ·  Model: `{model}`")
                st.divider()
```

### Step 4: Smoke test

- [ ] Run:

```bash
pytest tests/frontend/ -v
```
Expected: **2 passed**.

```bash
streamlit run src/frontend/app.py --server.headless true --server.port 8532 &
STREAMLIT_PID=$!
sleep 6
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8532
kill $STREAMLIT_PID 2>/dev/null
sleep 1
```
Expected: HTTP `200`.

### Step 5: Full suite — verify no regression

- [ ] Run:

```bash
pytest -q 2>&1 | tail -3
```
Expected: **175 passed** (no count change — UI only).

### Step 6: Commit T3C

- [ ] Run:

```bash
git add src/frontend/app.py
git commit -m "$(cat <<'EOF'
feat(frontend): AI Assistant tab — natural-language explainer

- New 4th tab in main(): BBB / EEG / MRI / AI Assistant.
- _render_ai_assistant_tab pulls last_bbb_prediction from session
  state, shows a snapshot caption, lets the user pick from 3 preset
  questions or type a custom one, POSTs to /explain/bbb, and renders
  a reverse-chronological history (capped at 10).
- Each history entry shows source (llm | template) and model so
  jurors can audit which path served each rationale.
- Empty state when no prediction yet: explicit prompt to run BBB tab
  first.
- No new tests; covered by 2 existing import-smoke tests.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4 — Close-out: AGENTS.md + README + DoD

**Why:** Anchor the new contracts in `AGENTS.md`, give the demo runner a `curl` recipe in `README.md`, run the full Day-7 DoD.

**Files:**
- Modify: `AGENTS.md`
- Modify: `README.md`

### Step 1: AGENTS.md — append §10 and §11

- [ ] Open `/Users/mertgungor/Desktop/hackathon/AGENTS.md`. Confirm the last section is currently §9 (Demo Features). Append at the end:

```markdown
## 10. Drift Surface (Day 7)

Each predict route maintains a per-worker rolling window of recent
prediction confidences (`collections.deque(maxlen=100)`). Train-time
median + std are stashed on `model._neurobridge_train_stats` (joblib
roundtrip-safe). The drift z-score is `(rolling_median − train_median) /
max(train_std, 1e-9)`, computed only when the buffer holds ≥10 samples
AND the model has the train-stats attribute. The `/predict/bbb`
response carries `drift_z: float | None` and `rolling_n: int`. The UI
renders a one-line caption with a magnitude tag (in-band, mild,
significant). Worker restart clears the deque; this is acceptable for
demo and removes the audit-trail concern.

## 11. LLM Explainer Surface (Day 7)

`src/llm/explainer.py` is the single entry point for natural-language
rationales. `explain(payload)` always returns `{rationale, source,
model}`. The deterministic template path is the source of truth for
tests; the LLM path is OpenRouter via the `openai==1.51.0` SDK using
`meta-llama/llama-3.2-3b-instruct:free`. Two env knobs control the
behavior:

- `OPENROUTER_API_KEY` — when absent, fallback to template.
- `NEUROBRIDGE_DISABLE_LLM=1` — hard kill-switch; force template even
  if a key is set. Use this for demo days when you want fully
  deterministic, reproducible rationales.

The `POST /explain/bbb` endpoint mirrors this contract. Pydantic
enforces a non-empty `top_features` list (422 on empty); every other
failure mode degrades to template + WARNING log + `source="template"`.
```

### Step 2: README.md — add Day 7 row + curl recipe

- [ ] Open `/Users/mertgungor/Desktop/hackathon/README.md`. Find the day-by-day status table from Day 6 (it should have a row like `| Day 6 — Final Polish & Demo Features ... | ✅ Shipped — 165 tests green |`). Append a new row immediately below it:

```markdown
| Day 7 — Final 5% (Drift, Traceability & Agents) | ✅ Shipped — 175 tests green |
```

- [ ] Find the "Where to Look" / pointers section (Day 6's close-out added entries here). Append:

- `docs/superpowers/specs/2026-05-05-day7-drift-traceability-agents-design.md` (Day-7 design spec)
- `docs/superpowers/plans/2026-05-05-day7-drift-traceability-agents.md` (Day-7 plan)
- New surface: `POST /explain/bbb` — natural-language rationale (LLM + deterministic fallback)
- New surface: `drift_z` / `rolling_n` / `provenance` fields in `POST /predict/bbb` response

- [ ] Find the existing "Demo Recipe" section if any; otherwise append a new section near the end:

```markdown
## Day 7 — Demo Recipe

Pre-flight (one terminal):

```bash
# Start API with deterministic explainer (no LLM key needed)
NEUROBRIDGE_DISABLE_LLM=1 BBB_MODEL_PATH=data/processed/bbb_model.joblib \
  uvicorn src.api.main:app --port 8000
```

Predict + explain (other terminal):

```bash
# 1) Predict — body now carries drift_z, rolling_n, provenance
curl -s -X POST http://localhost:8000/predict/bbb \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CCO", "top_k": 5}' | jq

# 2) Explain — feed the predict response back as the explain payload
curl -s -X POST http://localhost:8000/explain/bbb \
  -H "Content-Type: application/json" \
  -d '{
    "smiles": "CCO",
    "label": 1,
    "label_text": "permeable",
    "confidence": 0.82,
    "top_features": [
      {"feature": "fp_341", "shap_value": 0.045},
      {"feature": "fp_902", "shap_value": -0.031}
    ],
    "drift_z": 0.42,
    "user_question": "Why permeable?"
  }' | jq

# 3) Same call but with LLM enabled (set the key first)
unset NEUROBRIDGE_DISABLE_LLM
export OPENROUTER_API_KEY="sk-or-v1-…"
# Repeat the curl above; expect "source": "llm" and a model name.
```

Streamlit demo: `streamlit run src/frontend/app.py` → BBB tab → Predict → AI Assistant tab → ask a preset question.

Drift demo: refresh the BBB tab and predict 10+ times in a row — the drift caption transitions from "warming up" to a numeric z-score.
```

### Step 3: Run the full DoD verification

All of these must pass:

- [ ] **DoD-1: Full suite at 175.**

```bash
pytest -q 2>&1 | tail -3
```
Expected: **175 passed**.

- [ ] **DoD-2: UserWarning gate clean.**

```bash
pytest -W error::UserWarning tests/ 2>&1 | tail -3
```
Expected: 175 passed, 0 warnings escalated.

- [ ] **DoD-3: Streamlit boots.**

```bash
streamlit run src/frontend/app.py --server.headless true --server.port 8533 &
STREAMLIT_PID=$!
sleep 6
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8533
kill $STREAMLIT_PID 2>/dev/null
sleep 1
```
Expected: `200`.

- [ ] **DoD-4: Predict endpoint shape.**

Start the API in the background with the kill-switch on and a fresh deque:

```bash
NEUROBRIDGE_DISABLE_LLM=1 BBB_MODEL_PATH=data/processed/bbb_model.joblib \
  uvicorn src.api.main:app --port 8534 &
UVICORN_PID=$!
sleep 4
curl -s -X POST http://localhost:8534/predict/bbb \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CCO", "top_k": 3}' | python3 -c "
import json, sys
body = json.load(sys.stdin)
required = {'label','label_text','confidence','top_features','calibration','drift_z','rolling_n','provenance'}
missing = required - set(body.keys())
print('missing keys:', missing if missing else 'NONE')
"
kill $UVICORN_PID 2>/dev/null
sleep 1
```
Expected: `missing keys: NONE`.

- [ ] **DoD-5: Explain endpoint deterministic path.**

```bash
NEUROBRIDGE_DISABLE_LLM=1 BBB_MODEL_PATH=data/processed/bbb_model.joblib \
  uvicorn src.api.main:app --port 8535 &
UVICORN_PID=$!
sleep 4
curl -s -X POST http://localhost:8535/explain/bbb \
  -H "Content-Type: application/json" \
  -d '{
    "smiles": "CCO",
    "label": 1,
    "label_text": "permeable",
    "confidence": 0.82,
    "top_features": [{"feature":"fp_341","shap_value":0.045}],
    "drift_z": 0.42
  }' | python3 -c "
import json, sys
body = json.load(sys.stdin)
assert body['source'] == 'template', f\"expected source=template, got {body['source']}\"
assert body['model'] is None
assert 'fp_341' in body['rationale']
print('explain endpoint OK:', body['rationale'][:80], '…')
"
kill $UVICORN_PID 2>/dev/null
sleep 1
```
Expected: `explain endpoint OK: …` printed.

If ANY of DoD-1 through DoD-5 fails, STOP and report. Do NOT commit T4 with a failing DoD.

### Step 4: Commit T4

- [ ] Run:

```bash
git add AGENTS.md README.md
git commit -m "$(cat <<'EOF'
docs: Day-7 close-out — AGENTS §10 drift + §11 LLM explainer + README recipe

- AGENTS §10 documents the per-worker deque, train-stats stash, and
  z-score formula. §11 documents the explainer's two-path contract,
  env knobs (OPENROUTER_API_KEY, NEUROBRIDGE_DISABLE_LLM=1), and the
  /explain/bbb endpoint shape.
- README adds Day 7 to the status table (175 tests green), pointers
  to the Day-7 spec + plan + new surfaces, and a Demo Recipe section
  with curl invocations for both endpoints (template-only and LLM).
- DoD-1 through DoD-5 all green: pytest 175, UserWarning gate clean,
  Streamlit boot 200, predict body shape, explain template path.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Definition of Done (Day 7)

| Check | Pass criterion |
|---|---|
| Full suite green | `pytest -q` reports **175 passed** |
| UserWarning gate | `pytest -W error::UserWarning tests/` reports same count, 0 escalations |
| Streamlit boots | `streamlit run …` returns HTTP 200 |
| `/predict/bbb` body shape | Includes `drift_z`, `rolling_n`, `provenance` keys |
| `/explain/bbb` template path | Returns `source: "template"`, rationale contains top feature names |
| `_neurobridge_train_stats` persists | `TestTrainStatsMetadata.test_train_stats_survives_save_load_roundtrip` |
| Deque rolls at 100 | `TestBBBPredictRoute.test_predict_deque_rolls_at_100` |
| AI Assistant tab renders | Streamlit boot + manual click verify |
| MLflow badge appears in card | Streamlit boot + manual prediction verify |
| AGENTS §10 + §11 committed | yes |
| README has Day-7 row + curl recipe | yes |
| 9 commits in Day-7 ledger | T1A, T1B, T1C, T2, T3A, T3B, T3C, T4, plus the spec commit `09dd9c3` |

When all rows green: Day 7 mühürlü. Hackathon submission hazır.

---

## Self-Review (Plan Author)

**Spec coverage:**
- §1 Goal — covered by all 4 tasks.
- §2.1 Drift state location (deque + train_stats) — T1A + T1B.
- §2.2 LLM provider (OpenRouter, kill-switch) — T3A.
- §3.1 Drift layer (model, schemas, routes, frontend) — T1A + T1B + T1C.
- §3.2 MLflow traceability badge (schema, lookup, UI) — T2.
- §3.3 LLM explainer module (template + OpenRouter + fallback chain) — T3A.
- §3.4 `POST /explain/bbb` (explain_router, schemas, route) — T3B.
- §3.5 Streamlit AI Assistant tab (session state, presets, history) — T3C.
- §4 Test plan (+10 tests) — 2 (T1A) + 2 (T1B) + 1 (T2) + 4 (T3A) + 1 (T3B) = 10 ✅.
- §5 New dep — T3A Step 1.
- §6 Failure modes / lifelines — T2 Step 4 (`NEUROBRIDGE_DISABLE_MLFLOW`), T3A `_should_use_llm` + `_llm_explain` exception handler.
- §8 DoD — T4 Step 3 (DoD-1 through DoD-5).
- §9 Out of scope — explicitly NOT touched (no streaming, no retraining, no vector RAG, no provenance signing).

**Placeholder scan:** No `TBD`, `TODO`, `FIXME`, "implement later", "fill in details", or vague "add appropriate error handling" instructions remain. Every code step shows the actual code; every command shows the expected output.

**Type / name consistency:**
- `model._neurobridge_train_stats` keys: `median`, `std`, `n_train` — used identically in T1A (set), T1B (`stats["median"]`, `stats["std"]`), T2 (`stats.get("n_train", 0)`). ✅
- `WORKER_CONFIDENCE_DEQUE` — defined T1B Step 4, referenced in T1B tests Step 2. ✅
- `_compute_drift_z(model, confidence) -> tuple[float | None, int]` — return shape used in T1B Step 4 implementation matches the test assertions in Step 2. ✅
- `BBBPredictResponse` field additions: `drift_z` (T1B), `rolling_n` (T1B), `provenance` (T2). UI helper reads the same names in T1C / T2 Step 6. ✅
- `ExplainResult` keys: `rationale`, `source`, `model` — used in T3A tests, T3A implementation, T3B route handler, T3B test, T3C UI. ✅
- `explain_router` (prefix `/explain`) → `POST /explain/bbb` — declared T3B Step 5, mounted in same step, tested in T3B Step 3, called from UI in T3C Step 3. ✅

No issues found.
