# Day 6 — Final Polish & Demo Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pump the jury-day **Robustness, Interaction, and Creativity** sub-scores (slide 14) by adding 3 high-ROI demo features without touching the 158-test green floor.

**Architecture:** Day 5's `/predict/bbb` already returns `confidence` + SHAP top-k. Day 6 adds: (1) a frontend-only "Test Edge Cases" dropdown that picks from a curated catalog of robustness probes and visualizes how the system *gracefully* handles them; (2) a calibration metadata layer — `train()` does an 80/20 stratified split, computes precision-at-confidence-threshold bins, and stashes them on `model._neurobridge_calibration`; the API includes the matching bin in the response and the UI renders a one-line trust caption; (3) a new `POST /pipeline/mri/diagnostics` endpoint that runs the MRI pipeline twice (pre-ComBat features + post-ComBat features) and returns a long-format JSON for the Streamlit MRI tab to render as side-by-side altair KDE plots colored by site, headlined with the 3290× site-gap reduction KPI.

**Tech Stack:** No new pip deps — altair 5.5.0 ships with Streamlit; sklearn already has `train_test_split`. Existing brand tokens (navy `#0F172A`, sky `#0369A1`, slate `#475569`, Plus Jakarta Sans) are reused. New UX patterns conform to the Day-4 Trust & Authority style.

---

## File Structure

```
src/
├── models/
│   └── bbb_model.py                # MODIFY — Task 2: train() returns model with calibration metadata
├── api/
│   ├── schemas.py                  # MODIFY — Task 2/3: CalibrationContext, MRIDiagnosticsResponse
│   └── routes.py                   # MODIFY — Task 2/3: include calibration in /predict/bbb, new /pipeline/mri/diagnostics
├── pipelines/
│   └── mri_pipeline.py             # MODIFY — Task 3: expose compute_harmonization_diagnostics()
└── frontend/
    └── app.py                      # MODIFY — Tasks 1/2/3: edge-case dropdown, trust caption, MRI KDE viz

tests/
├── models/
│   └── test_bbb_model.py           # MODIFY — Task 2: TestCalibrationMetadata (3 tests)
├── api/
│   └── test_routes.py              # MODIFY — Task 2/3: calibration assertion + TestMRIDiagnosticsRoute (3 tests)
├── pipelines/
│   └── test_mri_pipeline.py        # MODIFY — Task 3: TestComputeHarmonizationDiagnostics (2 tests)
└── frontend/
    └── test_app_import.py          # (unchanged — 2 import-smoke tests still cover the module)

AGENTS.md                            # MODIFY — Task 4: §8 Calibration sub-section + §9 Demo Features
README.md                            # MODIFY — Task 4
```

**Test count target:** 158 + ~8 = **~166 tests green at end of Day 6**.

---

## Task 1: BBB Tab — "Test Edge Cases" dropdown (Robustness)

**Files:**
- Modify: `src/frontend/app.py`

**No backend changes** — this task purely curates inputs the existing `/predict/bbb` endpoint already handles correctly. The user value is that the dropdown turns implicit robustness into a visible demo artifact.

- [ ] **Step 1: Replace the bare `text_input` in `_render_bbb_tab` with a robustness-aware input flow**

In `/Users/mertgungor/Desktop/hackathon/src/frontend/app.py`, find `_render_bbb_tab()` (search for the existing `smiles = st.text_input("SMILES string", ...)` call) and replace the input section (the `st.text_input` + `st.slider` + `st.button` block) with this:

```python
    EDGE_CASES = {
        "Custom input (default)": {
            "smiles": "CCO",
            "label": "Ethanol — small, drug-like, BBB-permeable",
            "expectation": "High confidence, label = permeable",
        },
        "Invalid SMILES (parse-error path)": {
            "smiles": "this_is_not_a_valid_molecule_at_all_!!",
            "label": "Garbage string — should not parse",
            "expectation": "API returns HTTP 400 with parse error; UI shows recoverable warning",
        },
        "Empty string (boundary)": {
            "smiles": "",
            "label": "Empty input — boundary condition",
            "expectation": "Pydantic accepts empty; API returns 400 (RDKit cannot parse)",
        },
        "Massive OOD: cyclosporine-like macrocycle": {
            "smiles": (
                "CC[C@H](C)[C@@H]1NC(=O)[C@H](CC(C)C)N(C)C(=O)[C@H](CC(C)C)N(C)C(=O)"
                "[C@@H]2CCCN2C(=O)[C@H](C(C)C)NC(=O)[C@H]([C@@H](C)CC)N(C)C(=O)"
                "[C@H](C)NC(=O)[C@H](C)NC(=O)[C@H](CC(C)C)N(C)C(=O)[C@@H](NC(=O)"
                "[C@H](CC(C)C)N(C)C(=O)CN(C)C1=O)C(C)C"
            ),
            "label": "Cyclosporine — 11-residue macrocycle (~1.2 kDa)",
            "expectation": (
                "Far outside training distribution; model should hedge "
                "with low confidence (well-calibrated systems don't "
                "pretend to know)."
            ),
        },
        "OOD: heavy halogenated aromatic": {
            "smiles": "Fc1c(F)c(F)c(c(F)c1F)c2c(F)c(F)c(F)c(F)c2F",
            "label": "Decafluorobiphenyl — extreme halogen density",
            "expectation": "Rare scaffold; expect lowered confidence vs ethanol",
        },
    }

    case_name = st.selectbox(
        "Test Edge Cases",
        options=list(EDGE_CASES.keys()),
        index=0,
        key="bbb_case",
        help=(
            "Pick a robustness probe. Each case demonstrates how the "
            "system handles a real-world failure mode — invalid input, "
            "out-of-distribution molecules, or boundary conditions."
        ),
    )
    case = EDGE_CASES[case_name]
    st.caption(f"**Probe:** {case['label']}  ·  **Expected:** {case['expectation']}")

    smiles = st.text_input(
        "SMILES string",
        value=case["smiles"],
        key="bbb_smiles",
        help="Examples: CCO (ethanol), CC(=O)Nc1ccc(O)cc1 (paracetamol)",
    )
    top_k = st.slider(
        "SHAP features to display", min_value=3, max_value=10, value=5, key="bbb_topk",
    )

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
                elif e.response.status_code == 400:
                    # Robustness story: show the WARNING instead of an ERROR
                    # — invalid input is a recoverable path, not a crash.
                    st.warning(
                        f"Robustness check passed: API rejected the input "
                        f"with HTTP 400 (no crash). Detail: "
                        f"{e.response.json().get('detail', e.response.text)}"
                    )
                else:
                    st.error(
                        f"Prediction failed (HTTP {e.response.status_code}): "
                        f"{e.response.text}"
                    )
            except httpx.RequestError as e:
                st.error(f"Cannot reach FastAPI at {_API_URL}: {e!r}")
```

> **Note for the implementer**: read `_render_bbb_tab` first to confirm the current variable names (`smiles`, `top_k`, `bbb_predict`). Substitute as needed if the file diverged.

- [ ] **Step 2: Run the existing frontend smoke tests to confirm import still works**

```
cd /Users/mertgungor/Desktop/hackathon && source .venv312/bin/activate && pytest tests/frontend/ -v
```
Expected: 2 passed (the import-smoke tests still verify the module loads).

- [ ] **Step 3: Run the full suite to confirm 158 still green**

```
pytest -v 2>&1 | tail -3
```
Expected: 158 passed (no test count change — UI-only addition).

- [ ] **Step 4: Manual smoke**

```
streamlit run src/frontend/app.py --server.headless true &
sleep 5
curl -s http://localhost:8501 | head -3
pkill -f "streamlit run"
```
Expected: HTTP 200 with Streamlit HTML.

- [ ] **Step 5: Commit**

```bash
git add src/frontend/app.py
git commit -m "feat(frontend): edge-case dropdown for BBB robustness demo"
```

---

## Task 2: Decision Card calibration trust caption (Interaction & Trust)

**Files:**
- Modify: `src/models/bbb_model.py` — `train()` computes calibration bins on a held-out 20% test split, stashes them on `model._neurobridge_calibration`
- Modify: `src/api/schemas.py` — add `CalibrationContext` schema; extend `BBBPredictResponse` with `calibration: CalibrationContext | None`
- Modify: `src/api/routes.py` — `predict_bbb` looks up the matching calibration bin and includes it in the response
- Modify: `tests/models/test_bbb_model.py` — `TestCalibrationMetadata` (3 tests)
- Modify: `tests/api/test_routes.py` — extend `TestBBBPredictRoute.test_returns_200_*` with calibration assertion
- Modify: `src/frontend/app.py` — `_render_prediction_card` shows trust caption from `result["calibration"]`

### 2A — Calibration metadata in `train()`

- [ ] **Step 1: Append failing tests to `tests/models/test_bbb_model.py`**

Add a `TestCalibrationMetadata` class at the bottom of `tests/models/test_bbb_model.py`:

```python
class TestCalibrationMetadata:
    def test_train_attaches_calibration_attribute(self, trained_model_and_features):
        model, _ = trained_model_and_features
        assert hasattr(model, "_neurobridge_calibration")
        bins = model._neurobridge_calibration
        assert isinstance(bins, list)
        # Always at least one bin (the lowest-threshold one)
        assert len(bins) >= 1
        for b in bins:
            assert "threshold" in b
            assert "precision" in b
            assert "support" in b
            assert 0.0 <= b["threshold"] <= 1.0
            assert 0.0 <= b["precision"] <= 1.0
            assert b["support"] >= 0

    def test_calibration_thresholds_are_sorted_ascending(
        self, trained_model_and_features,
    ):
        model, _ = trained_model_and_features
        thresholds = [b["threshold"] for b in model._neurobridge_calibration]
        assert thresholds == sorted(thresholds)

    def test_calibration_survives_save_load_roundtrip(
        self, trained_model_and_features, tmp_path: Path,
    ):
        model, _ = trained_model_and_features
        artifact = tmp_path / "calibrated.joblib"
        bbb_model.save(model, artifact)
        reloaded = bbb_model.load(artifact)
        assert hasattr(reloaded, "_neurobridge_calibration")
        assert reloaded._neurobridge_calibration == model._neurobridge_calibration
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/models/test_bbb_model.py::TestCalibrationMetadata -v
```
Expected: 3 fails — `_neurobridge_calibration` attribute does not exist.

- [ ] **Step 3: Implement calibration in `train()`**

In `src/models/bbb_model.py`, add this near the top of the file (with other imports):

```python
from sklearn.model_selection import train_test_split
```

And add a private helper above `train()`:

```python
_CALIBRATION_THRESHOLDS: tuple[float, ...] = (0.50, 0.60, 0.70, 0.75, 0.80, 0.90)


def _compute_calibration_bins(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> list[dict[str, float]]:
    """Compute precision-at-confidence-threshold bins on a held-out test set.

    For each threshold T in `_CALIBRATION_THRESHOLDS`, picks the predictions
    whose max class probability >= T, computes precision and support, and
    returns one bin per threshold. Bins with zero support are still emitted
    (precision = 0.0, support = 0) so the API can always find a match.
    """
    if len(y_test) == 0:
        return [
            {"threshold": float(t), "precision": 0.0, "support": 0}
            for t in _CALIBRATION_THRESHOLDS
        ]
    proba = model.predict_proba(X_test)
    pred = model.predict(X_test)
    confidence = proba.max(axis=1)
    correct = (pred == y_test).astype(int)
    bins: list[dict[str, float]] = []
    for t in _CALIBRATION_THRESHOLDS:
        mask = confidence >= t
        support = int(mask.sum())
        if support == 0:
            precision = 0.0
        else:
            precision = float(correct[mask].mean())
        bins.append({
            "threshold": float(t), "precision": precision, "support": support,
        })
    return bins
```

Then modify the existing `train()` function. Find the line `model.fit(X, y)` and replace the body around it:

```python
    X, y, fp_cols = _split_features_and_label(df, label_col)
    # Stratified 80/20 split for honest calibration metrics. Falls back to
    # train-on-all if the dataset is too tiny for a stratified split (test
    # fixtures with 3-4 rows hit this branch).
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y,
        )
    except ValueError:
        # n_classes > n_samples_per_class for stratification: train on all,
        # leave the test set empty (calibration bins will be zero-support).
        X_train, X_test, y_train, y_test = X, np.empty((0, X.shape[1])), y, np.empty((0,))

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=1,
    )
    model.fit(X_train, y_train)
    # Stash the column names under a project-owned attribute so SHAP (Task 2)
    # can map values back to fp_<bit> indices. Sklearn's own feature_names_in_
    # is only set automatically when fit receives a DataFrame; setting it
    # manually fires UserWarning on every predict call.
    model._neurobridge_fp_cols = list(fp_cols)
    model._neurobridge_calibration = _compute_calibration_bins(model, X_test, y_test)
    logger.info(
        "Trained BBB classifier: n=%d, n_features=%d, classes=%s, "
        "calibration_bins=%d",
        len(y), X.shape[1], model.classes_.tolist(),
        len(model._neurobridge_calibration),
    )
    return model
```

- [ ] **Step 4: Run tests, expect 3 passed**

```
pytest tests/models/test_bbb_model.py::TestCalibrationMetadata -v
```

- [ ] **Step 5: Run full model + suite (the existing `TestTrain` tests now run on a slightly smaller training set; verify they still pass)**

```
pytest tests/models/ -v 2>&1 | tail -10
```
Expected: 16 passed (13 prior + 3 new calibration tests).

```
pytest -v 2>&1 | tail -3
```
Expected: 161 passed (158 prior + 3 new).

- [ ] **Step 6: Commit Task 2A**

```bash
git add src/models/bbb_model.py tests/models/test_bbb_model.py
git commit -m "feat(models): compute precision-at-confidence calibration bins"
```

### 2B — API + Pydantic schema

- [ ] **Step 7: Add `CalibrationContext` schema and extend `BBBPredictResponse`**

In `src/api/schemas.py`, append after `FeatureAttribution`:

```python
class CalibrationContext(BaseModel):
    """The calibration bin matching the prediction's confidence.

    Lets the UI show statements like 'predictions ≥75% confident are
    correct 92% of the time on the held-out test set'.
    """
    threshold: float = Field(..., description="Confidence threshold for this bin")
    precision: float = Field(..., description="Test-set precision at this threshold")
    support: int = Field(..., description="Test-set sample count at this threshold")
```

Modify `BBBPredictResponse`: add a new optional field `calibration: CalibrationContext | None = None`. The full updated class:

```python
class BBBPredictResponse(BaseModel):
    """Decision-system payload: prediction + uncertainty + explanation."""
    label: int
    label_text: str = Field(..., description="'permeable' or 'non-permeable'")
    confidence: float
    top_features: list[FeatureAttribution]
    calibration: CalibrationContext | None = Field(
        None,
        description=(
            "The calibration bin matching this prediction's confidence; "
            "None if the model lacks calibration metadata"
        ),
    )
```

- [ ] **Step 8: Extend `predict_bbb` route to look up the matching bin**

In `src/api/routes.py`, find the existing `predict_bbb` function. Add a helper above it:

```python
def _matching_calibration_bin(
    model: object, confidence: float,
) -> dict[str, float] | None:
    """Return the highest-threshold calibration bin still ≤ `confidence`.

    Returns None if the model has no calibration metadata.
    """
    bins = getattr(model, "_neurobridge_calibration", None)
    if not bins:
        return None
    eligible = [b for b in bins if b["threshold"] <= confidence]
    if not eligible:
        return None
    return max(eligible, key=lambda b: b["threshold"])
```

Modify the response construction at the end of `predict_bbb`:

```python
    label_text = "permeable" if pred["label"] == 1 else "non-permeable"
    cal_bin = _matching_calibration_bin(model, pred["confidence"])
    return BBBPredictResponse(
        label=pred["label"],
        label_text=label_text,
        confidence=pred["confidence"],
        top_features=[FeatureAttribution(**a) for a in attributions],
        calibration=(
            CalibrationContext(**cal_bin) if cal_bin is not None else None
        ),
    )
```

Update the imports in `src/api/routes.py` to include `CalibrationContext`:

```python
from src.api.schemas import (
    BBBPredictRequest,
    BBBPredictResponse,
    BBBRequest,
    CalibrationContext,
    EEGRequest,
    FeatureAttribution,
    MRIRequest,
    PipelineResponse,
)
```

- [ ] **Step 9: Extend the existing 200-path test in `tests/api/test_routes.py`**

Find `TestBBBPredictRoute.test_returns_200_with_prediction_and_attributions` and add these assertions before the closing of the `try:` block:

```python
        assert "calibration" in body
        if body["calibration"] is not None:
            cal = body["calibration"]
            assert "threshold" in cal and 0.0 <= cal["threshold"] <= 1.0
            assert "precision" in cal and 0.0 <= cal["precision"] <= 1.0
            assert "support" in cal and cal["support"] >= 0
```

- [ ] **Step 10: Run tests**

```
pytest tests/api/test_routes.py::TestBBBPredictRoute -v
```
Expected: 3 passed.

```
pytest -v 2>&1 | tail -3
```
Expected: 161 passed (no count change — extended existing test).

- [ ] **Step 11: Commit Task 2B**

```bash
git add src/api/schemas.py src/api/routes.py tests/api/test_routes.py
git commit -m "feat(api): include calibration context in /predict/bbb response"
```

### 2C — Streamlit trust caption

- [ ] **Step 12: Update `_render_prediction_card` in `src/frontend/app.py`**

Find `_render_prediction_card` and append the trust caption logic AFTER the existing `st.progress(float(result["confidence"]))` line and BEFORE the SHAP section (the `st.markdown` for "Top N SHAP attributions"):

```python
    cal = result.get("calibration")
    if cal is not None and cal["support"] > 0:
        threshold_pct = cal["threshold"] * 100
        precision_pct = cal["precision"] * 100
        st.caption(
            f"📊 **Calibration context:** on the held-out test set, "
            f"predictions with ≥{threshold_pct:.0f}% confidence are correct "
            f"**{precision_pct:.0f}%** of the time (n={cal['support']})."
        )
    elif cal is not None:
        # Bin matched but support=0 (tiny test fixtures). Still useful as honesty.
        threshold_pct = cal["threshold"] * 100
        st.caption(
            f"📊 **Calibration context:** test-set support at "
            f"≥{threshold_pct:.0f}% confidence is too small to estimate "
            f"precision (n=0). Train on a larger dataset to see precision."
        )
```

- [ ] **Step 13: Run frontend smoke**

```
pytest tests/frontend/ -v
```
Expected: 2 passed.

- [ ] **Step 14: Commit Task 2C**

```bash
git add src/frontend/app.py
git commit -m "feat(frontend): trust caption with calibration context in BBB card"
```

---

## Task 3: MRI ComBat histogram visualization (Creativity & Problem Depth)

**Files:**
- Modify: `src/pipelines/mri_pipeline.py` — add `compute_harmonization_diagnostics(input_dir, sites_csv, ...)` returning a long-format DataFrame
- Modify: `src/api/schemas.py` — `MRIDiagnosticsRequest`, `MRIDiagnosticsResponse`
- Modify: `src/api/routes.py` — `POST /pipeline/mri/diagnostics` endpoint
- Modify: `tests/pipelines/test_mri_pipeline.py` — `TestComputeHarmonizationDiagnostics` (2 tests)
- Modify: `tests/api/test_routes.py` — `TestMRIDiagnosticsRoute` (2 tests)
- Modify: `src/frontend/app.py` — replace bare MRI tab with diagnostics-driven KDE viz + site-gap KPI

### 3A — `compute_harmonization_diagnostics` function

- [ ] **Step 1: Append failing tests to `tests/pipelines/test_mri_pipeline.py`**

Append at the bottom:

```python
class TestComputeHarmonizationDiagnostics:
    def test_returns_long_format_with_pre_and_post_states(self, tmp_path: Path):
        from tests.fixtures.build_mri_fixture import build as build_mri
        from src.pipelines.mri_pipeline import compute_harmonization_diagnostics

        fixture_dir = build_mri(out_dir=tmp_path / "mri")
        diagnostics = compute_harmonization_diagnostics(
            input_dir=fixture_dir,
            sites_csv=fixture_dir / "sites.csv",
        )
        assert "feature_value" in diagnostics.columns
        assert "site" in diagnostics.columns
        assert "harmonization_state" in diagnostics.columns
        assert "feature" in diagnostics.columns
        states = set(diagnostics["harmonization_state"].unique())
        assert states == {"Pre-ComBat", "Post-ComBat"}

    def test_post_combat_site_gap_is_smaller_than_pre(self, tmp_path: Path):
        """Day-3 demonstrated 5.0 → 0.0015 gap reduction. This regression
        test pins the property: post-ComBat per-site means MUST be closer
        together than pre-ComBat per-site means."""
        from tests.fixtures.build_mri_fixture import build as build_mri
        from src.pipelines.mri_pipeline import compute_harmonization_diagnostics

        fixture_dir = build_mri(out_dir=tmp_path / "mri")
        diagnostics = compute_harmonization_diagnostics(
            input_dir=fixture_dir,
            sites_csv=fixture_dir / "sites.csv",
        )
        pre = diagnostics[diagnostics["harmonization_state"] == "Pre-ComBat"]
        post = diagnostics[diagnostics["harmonization_state"] == "Post-ComBat"]
        # Compute site-gap as range of per-site means on the first feature
        feat = diagnostics["feature"].iloc[0]
        pre_gap = pre[pre["feature"] == feat].groupby("site")["feature_value"].mean().agg(lambda s: s.max() - s.min())
        post_gap = post[post["feature"] == feat].groupby("site")["feature_value"].mean().agg(lambda s: s.max() - s.min())
        assert post_gap < pre_gap, (
            f"Expected post-gap < pre-gap, got pre={pre_gap}, post={post_gap}"
        )
```

- [ ] **Step 2: Run failing tests**

```
pytest tests/pipelines/test_mri_pipeline.py::TestComputeHarmonizationDiagnostics -v
```
Expected: ImportError — `compute_harmonization_diagnostics` does not exist.

- [ ] **Step 3: Implement `compute_harmonization_diagnostics`**

In `src/pipelines/mri_pipeline.py`, append at the end of the file (after `run_pipeline`):

```python
def compute_harmonization_diagnostics(
    input_dir: Path,
    sites_csv: Path | None = None,
    intensity_threshold: float | None = None,
    n_roi_axes: tuple[int, int, int] = DEFAULT_N_ROI_AXES,
) -> pd.DataFrame:
    """Run the MRI pipeline twice — pre-ComBat features and post-ComBat —
    and return a long-format DataFrame ready for visualization.

    Output columns: ``subject_id``, ``site``, ``feature``, ``feature_value``,
    ``harmonization_state`` ('Pre-ComBat' or 'Post-ComBat').

    Used by the FastAPI ``/pipeline/mri/diagnostics`` endpoint to feed the
    Streamlit MRI tab's KDE / histogram comparison plot.

    Raises:
        FileNotFoundError: if ``input_dir`` does not exist.
        KeyError: if any subject is missing a site assignment.
    """
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"MRI input directory not found: {input_dir}")
    sites_csv = Path(sites_csv) if sites_csv is not None else input_dir / "sites.csv"
    sites_df = pd.read_csv(sites_csv)

    feature_cols = [
        f"feat_roi{i}_{stat}"
        for i in range(int(np.prod(n_roi_axes)))
        for stat in ROI_STATS
    ]

    rows: list[dict[str, object]] = []
    for nifti_path in sorted(input_dir.glob("*.nii*")):
        subject_id = nifti_path.stem.replace(".nii", "")
        volume = nib.load(nifti_path).get_fdata()
        if not is_valid_volume(volume):
            continue
        mask = mask_brain(volume, intensity_threshold=intensity_threshold)
        feats = extract_features_from_volume(
            volume, mask, n_roi_axes=n_roi_axes,
        )
        row: dict[str, object] = {"subject_id": subject_id}
        row.update(feats)
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=[
            "subject_id", "site", "feature", "feature_value", "harmonization_state",
        ])

    raw_features = pd.DataFrame(rows).merge(sites_df, on="subject_id", how="left")
    if raw_features["site"].isna().any():
        missing = raw_features.loc[raw_features["site"].isna(), "subject_id"].tolist()
        raise KeyError(
            f"sites_csv missing site assignment for subjects: {missing}"
        )

    # Post-ComBat: variance-aware harmonization. Reuses the same logic as
    # run_pipeline so diagnostics reflect production behavior exactly.
    col_std = raw_features[feature_cols].std()
    var_feature_cols = [
        c for c in feature_cols if col_std[c] > _MIN_VAR_THRESHOLD
    ]
    zero_var_cols = [
        c for c in feature_cols if col_std[c] <= _MIN_VAR_THRESHOLD
    ]
    if not var_feature_cols:
        harmonized = raw_features[feature_cols].copy()
    else:
        harmonized = harmonize_combat(
            raw_features, raw_features["site"], var_feature_cols,
        )
        for c in zero_var_cols:
            harmonized[c] = raw_features[c].to_numpy()
        harmonized = harmonized[feature_cols]
    post_features = pd.concat(
        [raw_features[["subject_id", "site"]].reset_index(drop=True),
         harmonized.reset_index(drop=True)],
        axis=1,
    )

    long_pre = raw_features.melt(
        id_vars=["subject_id", "site"], value_vars=feature_cols,
        var_name="feature", value_name="feature_value",
    )
    long_pre["harmonization_state"] = "Pre-ComBat"
    long_post = post_features.melt(
        id_vars=["subject_id", "site"], value_vars=feature_cols,
        var_name="feature", value_name="feature_value",
    )
    long_post["harmonization_state"] = "Post-ComBat"
    return pd.concat([long_pre, long_post], ignore_index=True)
```

- [ ] **Step 4: Run tests**

```
pytest tests/pipelines/test_mri_pipeline.py::TestComputeHarmonizationDiagnostics -v
```
Expected: 2 passed.

- [ ] **Step 5: Run full MRI tests + suite**

```
pytest tests/pipelines/test_mri_pipeline.py -v 2>&1 | tail -5
```
Expected: 42 passed (40 prior + 2 new).

```
pytest -v 2>&1 | tail -3
```
Expected: 163 passed.

- [ ] **Step 6: Commit Task 3A**

```bash
git add src/pipelines/mri_pipeline.py tests/pipelines/test_mri_pipeline.py
git commit -m "feat(mri): compute_harmonization_diagnostics for pre/post comparison"
```

### 3B — API endpoint

- [ ] **Step 7: Add Pydantic schemas to `src/api/schemas.py`**

Append at the bottom:

```python
class MRIDiagnosticsRequest(BaseModel):
    """Request body for /pipeline/mri/diagnostics — same as MRIRequest minus output_path."""
    input_dir: str = Field(..., description="Directory of .nii.gz files")
    sites_csv: str = Field(..., description="CSV mapping subject_id → site")


class HarmonizationRow(BaseModel):
    subject_id: str
    site: str
    feature: str
    feature_value: float
    harmonization_state: str


class MRIDiagnosticsResponse(BaseModel):
    """Long-format pre/post ComBat data for visualization."""
    rows: list[HarmonizationRow]
    site_gap_pre: float = Field(..., description="Range of per-site means before ComBat")
    site_gap_post: float = Field(..., description="Range of per-site means after ComBat")
    reduction_factor: float = Field(..., description="site_gap_pre / max(site_gap_post, eps)")
```

- [ ] **Step 8: Add the route to `src/api/routes.py`**

Update the schema imports to include the 3 new types:

```python
from src.api.schemas import (
    BBBPredictRequest,
    BBBPredictResponse,
    BBBRequest,
    CalibrationContext,
    EEGRequest,
    FeatureAttribution,
    HarmonizationRow,
    MRIDiagnosticsRequest,
    MRIDiagnosticsResponse,
    MRIRequest,
    PipelineResponse,
)
```

Add the route at the end of the file:

```python
@router.post("/mri/diagnostics", response_model=MRIDiagnosticsResponse)
def mri_diagnostics(req: MRIDiagnosticsRequest) -> MRIDiagnosticsResponse:
    """Run the MRI pipeline twice and return pre/post ComBat data + site-gap KPIs."""
    input_dir = Path(req.input_dir)
    sites_csv = Path(req.sites_csv)
    try:
        df = mri_pipeline.compute_harmonization_diagnostics(
            input_dir=input_dir, sites_csv=sites_csv,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if df.empty:
        return MRIDiagnosticsResponse(
            rows=[], site_gap_pre=0.0, site_gap_post=0.0, reduction_factor=0.0,
        )

    # Site-gap KPI on the first feature, averaged per site
    feat = df["feature"].iloc[0]
    feat_df = df[df["feature"] == feat]
    pre_means = feat_df[feat_df["harmonization_state"] == "Pre-ComBat"].groupby(
        "site"
    )["feature_value"].mean()
    post_means = feat_df[feat_df["harmonization_state"] == "Post-ComBat"].groupby(
        "site"
    )["feature_value"].mean()
    site_gap_pre = float(pre_means.max() - pre_means.min())
    site_gap_post = float(post_means.max() - post_means.min())
    eps = 1e-9
    reduction_factor = site_gap_pre / max(site_gap_post, eps)

    rows = [
        HarmonizationRow(**rec) for rec in df.to_dict(orient="records")
    ]
    return MRIDiagnosticsResponse(
        rows=rows,
        site_gap_pre=site_gap_pre,
        site_gap_post=site_gap_post,
        reduction_factor=reduction_factor,
    )
```

> **Note**: this route belongs on `router` (the `/pipeline/...` router), NOT on `predict_router`. Verify the existing `router = APIRouter(prefix="/pipeline")` at the top of routes.py.

- [ ] **Step 9: Append failing route tests to `tests/api/test_routes.py`**

```python
class TestMRIDiagnosticsRoute:
    def test_returns_200_with_pre_and_post_data(self, tmp_path: Path):
        from tests.fixtures.build_mri_fixture import build as build_mri
        fixture_dir = build_mri(out_dir=tmp_path / "mri")
        resp = client.post(
            "/pipeline/mri/diagnostics",
            json={
                "input_dir": str(fixture_dir),
                "sites_csv": str(fixture_dir / "sites.csv"),
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["rows"]) > 0
        assert body["site_gap_pre"] >= 0.0
        assert body["site_gap_post"] >= 0.0
        # Reduction factor is the headline KPI
        assert body["reduction_factor"] >= 1.0  # ComBat must reduce, not amplify
        states = {r["harmonization_state"] for r in body["rows"]}
        assert states == {"Pre-ComBat", "Post-ComBat"}

    def test_returns_404_when_input_dir_missing(self, tmp_path: Path):
        resp = client.post(
            "/pipeline/mri/diagnostics",
            json={
                "input_dir": str(tmp_path / "does_not_exist"),
                "sites_csv": str(tmp_path / "sites.csv"),
            },
        )
        assert resp.status_code == 404
```

- [ ] **Step 10: Run tests**

```
pytest tests/api/test_routes.py::TestMRIDiagnosticsRoute -v
```
Expected: 2 passed.

```
pytest -v 2>&1 | tail -3
```
Expected: 165 passed (163 + 2).

- [ ] **Step 11: Commit Task 3B**

```bash
git add src/api/schemas.py src/api/routes.py tests/api/test_routes.py
git commit -m "feat(api): POST /pipeline/mri/diagnostics for pre/post ComBat KPIs"
```

### 3C — Streamlit MRI tab visualization

- [ ] **Step 12: Replace `_render_mri_tab` body in `src/frontend/app.py`**

Find the existing `_render_mri_tab` function (which currently does a basic POST to `/pipeline/mri`) and replace its body entirely with:

```python
def _render_mri_tab() -> None:
    _render_section(
        "IMAGE — MRI",
        "Multi-site harmonization via ComBat",
        "Loads NIfTI volumes, masks brain tissue, computes per-ROI summary "
        "statistics, then harmonizes across acquisition sites with neuroHarmonize "
        "to remove scanner-driven domain shift. The diagnostic plot below "
        "compares per-site feature distributions before and after harmonization."
    )
    mri_dir = st.text_input(
        "Input NIfTI directory", "tests/fixtures/mri_sample", key="mri_dir",
        help="Path to a directory of .nii(.gz) files + sites.csv",
    )
    sites_csv = st.text_input(
        "Sites CSV", "tests/fixtures/mri_sample/sites.csv", key="mri_sites",
    )

    if st.button("Run ComBat diagnostics", type="primary", key="mri_diag"):
        with st.spinner("Running pre + post ComBat (×2 the work)…"):
            try:
                result = _post(
                    "/pipeline/mri/diagnostics",
                    {"input_dir": mri_dir, "sites_csv": sites_csv},
                )
                _render_combat_diagnostics(result)
                st.toast("Diagnostics complete", icon="✅")
            except httpx.HTTPStatusError as e:
                st.error(
                    f"Diagnostics failed (HTTP {e.response.status_code}): "
                    f"{e.response.text}"
                )
            except httpx.RequestError as e:
                st.error(f"Cannot reach FastAPI at {_API_URL}: {e!r}")
```

- [ ] **Step 13: Add `_render_combat_diagnostics` helper above `main()` in `src/frontend/app.py`**

```python
def _render_combat_diagnostics(result: dict) -> None:
    """Render the Pre/Post-ComBat KDE comparison + site-gap KPI strip."""
    import altair as alt
    import pandas as pd

    rows = result.get("rows", [])
    if not rows:
        st.info(
            "No data returned. Check that the input directory contains "
            ".nii(.gz) files and a sites.csv with subject_id/site columns."
        )
        return

    cols = st.columns(3)
    cols[0].metric("Site-gap (Pre-ComBat)", f"{result['site_gap_pre']:.4f}")
    cols[1].metric("Site-gap (Post-ComBat)", f"{result['site_gap_post']:.4f}")
    cols[2].metric(
        "Reduction factor",
        f"{result['reduction_factor']:.0f}×",
        help=(
            "Pre-gap / Post-gap. A 100× reduction means ComBat "
            "removed two orders of magnitude of site-driven domain shift."
        ),
    )

    df = pd.DataFrame(rows)
    # Pin the chart to the first feature (most recognizable for the audience).
    feat = df["feature"].iloc[0]
    feat_df = df[df["feature"] == feat]

    # Layered KDE: x = feature_value, color = site, faceted by harmonization_state.
    chart = (
        alt.Chart(feat_df)
        .transform_density(
            density="feature_value",
            groupby=["site", "harmonization_state"],
            as_=["feature_value", "density"],
        )
        .mark_area(opacity=0.55)
        .encode(
            x=alt.X("feature_value:Q", title=f"{feat} (intensity)"),
            y=alt.Y("density:Q", title="Density"),
            color=alt.Color(
                "site:N",
                title="Site",
                scale=alt.Scale(scheme="tableau10"),
            ),
            tooltip=[
                alt.Tooltip("site:N"),
                alt.Tooltip("feature_value:Q", format=".4f"),
                alt.Tooltip("density:Q", format=".3f"),
            ],
        )
        .properties(width=380, height=260)
        .facet(
            column=alt.Column(
                "harmonization_state:N",
                title=None,
                sort=["Pre-ComBat", "Post-ComBat"],
                header=alt.Header(labelFontSize=13, labelFontWeight="bold"),
            )
        )
        .resolve_scale(x="shared", y="shared")
    )
    st.altair_chart(chart, use_container_width=True)

    st.caption(
        f"Per-site density of `{feat}` before and after ComBat. Each "
        f"colored region is one acquisition site. **Convergence of the "
        f"colored regions in the Post-ComBat panel is the visual proof "
        f"of harmonization** — the same property the {result['reduction_factor']:.0f}× "
        f"site-gap reduction quantifies."
    )
```

- [ ] **Step 14: Run frontend smoke**

```
pytest tests/frontend/ -v
```
Expected: 2 passed.

```
pytest -v 2>&1 | tail -3
```
Expected: 165 passed (no new tests added at this step).

- [ ] **Step 15: Manual smoke**

```
streamlit run src/frontend/app.py --server.headless true &
sleep 5
curl -s http://localhost:8501 | head -3
pkill -f "streamlit run"
```
Expected: HTTP 200 with Streamlit HTML.

- [ ] **Step 16: Commit Task 3C**

```bash
git add src/frontend/app.py
git commit -m "feat(frontend): MRI tab — Pre/Post ComBat KDE + site-gap KPI"
```

---

## Task 4: Final close-out — AGENTS.md + README + DoD

**Files:**
- Modify: `AGENTS.md` — §8 sub-section on calibration metadata; new §9 on Day-6 demo features
- Modify: `README.md` — Day 6 row in status table

- [ ] **Step 1: Update AGENTS.md**

Add a sub-section to §8 right after the uniform-surface bullets:

```markdown
**Calibration metadata** (Day 6): `train()` does an 80/20 stratified split,
computes precision-at-confidence-threshold bins on the held-out test set,
and stashes them on `model._neurobridge_calibration: list[dict]` (sorted
ascending by threshold). The API includes the bin matching each
prediction's confidence in `BBBPredictResponse.calibration`. UI uses this
to render an honest trust caption ("≥75% confident → 92% precision, n=18").
For tiny test fixtures where stratified split fails, calibration falls
back to zero-support bins so the API contract is always populated.
```

After §8, append a new §9:

```markdown
## 9. Demo Features (Day 6)

The frontend includes three jury-day demo amplifiers that don't change
the core contract:

- **Edge-case dropdown** (BBB tab): a curated catalog of 5 robustness
  probes — invalid SMILES, empty input, OOD macrocycle (cyclosporine-like),
  heavy halogenated aromatic. Each has a stated expectation; the UI
  visualizes graceful failure (HTTP 400 → recoverable warning, never
  a crash).
- **Calibration trust caption** (BBB decision card): renders the
  precision-at-confidence-threshold from `BBBPredictResponse.calibration`.
  Demonstrates that the system knows what it doesn't know.
- **MRI ComBat diagnostics** (MRI tab): `POST /pipeline/mri/diagnostics`
  runs the pipeline twice (pre + post ComBat) and returns long-format
  data + site-gap KPIs (Pre, Post, Reduction factor). The UI renders
  a faceted altair density plot — visual proof that ComBat removes
  site-driven domain shift.
```

- [ ] **Step 2: Update README.md**

Add Day 6 to the status table:

```markdown
| Day 6 — Final Polish & Demo Features (Edge cases + Calibration + ComBat viz) | ✅ Shipped — 165 tests green |
```

Add to "Where to Look":
- `docs/superpowers/plans/2026-05-04-day6-final-polish-demo-features.md` (Day-6 plan)
- New surface: `POST /pipeline/mri/diagnostics`

- [ ] **Step 3: DoD verification**

Run all 4 checks:

```
pytest -v 2>&1 | tail -3
```
Expected: **165 passed**.

```
pytest -W error::UserWarning tests/models/ tests/api/ tests/pipelines/ tests/frontend/ 2>&1 | tail -3
```
Expected: same count, zero UserWarning errors.

```
streamlit run src/frontend/app.py --server.headless true &
sleep 5
curl -s http://localhost:8501 | head -3
pkill -f "streamlit run"
```
Expected: HTML response.

If `data/raw/bbbp.csv` exists, do a full E2E retrain → predict → diagnostics smoke. Otherwise skip (covered by tests).

- [ ] **Step 4: Commit close-out**

```bash
git add AGENTS.md README.md
git commit -m "docs: Day-6 close-out — AGENTS §8 calibration + §9 demo features"
```

---

## Definition of Done (Day 6)

| Check | Pass criterion |
|---|---|
| Full suite green | `pytest -v` reports 165 passed |
| Edge-case dropdown lists 5 cases (incl. invalid + OOD) | manual / Streamlit run |
| Calibration metadata persists across save/load | `TestCalibrationMetadata.test_calibration_survives_save_load_roundtrip` |
| `BBBPredictResponse.calibration` is populated for predictions ≥0.5 confidence | `TestBBBPredictRoute.test_returns_200_*` calibration assertions |
| `POST /pipeline/mri/diagnostics` returns site-gap KPIs + Pre/Post rows | `TestMRIDiagnosticsRoute.test_returns_200_*` |
| `compute_harmonization_diagnostics` regression-tests post < pre | `TestComputeHarmonizationDiagnostics.test_post_combat_site_gap_is_smaller_than_pre` |
| Streamlit MRI tab renders altair faceted density plot | manual |
| 158 prior tests still green | yes |
| AGENTS.md §8 documents calibration; §9 documents demo features | yes |

When all rows green: Day 6 mühürlü. Jüri demosu hazır.
