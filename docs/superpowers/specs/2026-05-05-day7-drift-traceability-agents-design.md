# Day 7 — The Final 5% (Drift, Traceability & Agents) Design Spec

**Date:** 2026-05-05
**Status:** Approved by user; ready for `/superpowers:writing-plans`.
**Predecessor:** Day 6 (`docs/superpowers/plans/2026-05-04-day6-final-polish-demo-features.md`) — closed at SHA `d05fcf1`, 165 green tests.

---

## 1. Goal

Close the remaining 5% gap to a top-tier hackathon submission by hardening the two evaluation dimensions where Day-6 left visible weakness:

- **Adapt Over Time** (Slide 12 "Living Systems — Your Edge"): the system is currently static post-training. Add a drift-detection stub that compares trailing prediction confidence to the training distribution.
- **AI Lab Agents** (Slide 7, Track 1): we have no agent surface. Add a chat-style "Why?" endpoint that explains a BBB prediction in natural language using SHAP attributions and drift context.

Plus one System-Quality polish (MLflow traceability badge in the decision card) so jurors can audit *which* model produced a given prediction.

**Non-goals (YAGNI):**

- No retraining loop. Drift is *observed and reported*, not acted on.
- No conversational state, no multi-turn agent. One question → one rationale.
- No vector store or RAG corpus. The "context" is the prediction payload itself plus a small built-in literature primer.
- No new front-end framework. Stay on Streamlit; reuse Trust & Authority brand tokens.
- No new visualization library. Reuse altair (already shipped).

---

## 2. Sealed Architectural Decisions

These are locked. The implementation plan must follow them as-is.

### 2.1 Drift state location

**Decision:** in-process `collections.deque(maxlen=100)` per FastAPI worker, plus train-time median/std baked into `model._neurobridge_train_stats: dict` (joblib-roundtrip-safe).

| Field | Type | Source |
|---|---|---|
| `model._neurobridge_train_stats["median"]` | `float` | `np.median(model.predict_proba(X_train).max(axis=1))` |
| `model._neurobridge_train_stats["std"]` | `float` | `np.std(model.predict_proba(X_train).max(axis=1))` |
| `model._neurobridge_train_stats["n_train"]` | `int` | `len(y_train)` |
| `WORKER_CONFIDENCE_DEQUE` | `deque[float]` (maxlen=100) | module-level singleton in `src/api/routes.py` |

**Z-score formula:** `drift_z = (rolling_median − train_median) / max(train_std, 1e-9)`

**Edge cases:**
- `len(deque) < 10` → `drift_z = None`, UI shows "Warming up (n/100)".
- `_neurobridge_train_stats` missing (legacy joblib) → `drift_z = None`, UI shows "Drift unavailable".

**Rejected alternatives:**
- Joblib sidecar file (B): I/O race risk, slower.
- SQLite (C): production-grade but +30 min setup overhead, not worth it for the demo budget.
- Train-time stats only without rolling window (D): kills the "trailing 100" narrative.

### 2.2 LLM provider

**Decision:** OpenRouter via `openai==1.51.0` SDK, hybrid template fallback, kill-switch env gate.

| Setting | Value |
|---|---|
| Base URL | `https://openrouter.ai/api/v1` |
| Default model | `meta-llama/llama-3.2-3b-instruct:free` |
| Auth | `OPENROUTER_API_KEY` env var |
| Lifeline gate | `NEUROBRIDGE_DISABLE_LLM=1` → force template path |
| Timeout | 8 seconds (HTTP request) |
| Max tokens | 256 (response cap) |
| Temperature | 0.3 (deterministic-ish, jury demos must be predictable) |

**Fallback chain (in order):**

1. `NEUROBRIDGE_DISABLE_LLM=1` set → template
2. `OPENROUTER_API_KEY` not set → template
3. `openai` SDK raises `APIConnectionError` / `APITimeoutError` / `RateLimitError` → log warning, template
4. LLM returns empty / malformed response → log warning, template
5. Otherwise → LLM rationale

**Response contract (always populated):**

```python
{"rationale": str, "source": "llm" | "template", "model": str | None}
```

`source` makes the auditing story crisp: "this rationale came from the deterministic template" vs "this came from llama-3.2-3b". Jurors can verify reproducibility.

**Rejected alternatives:**
- Anthropic API (C): no key available.
- Local Ollama (B): demo-day install/load risk too high.
- Pure deterministic template (A): kills the "AI Lab Agents" narrative.
- Pure LLM with no fallback: demo-day network failure = total failure.

---

## 3. Component Design

### 3.1 Drift layer

**Files touched:**

- `src/models/bbb_model.py` — extend `train()` to compute and stash `_neurobridge_train_stats`.
- `src/api/schemas.py` — add `drift_z: float | None` and `rolling_n: int` to `BBBPredictResponse`.
- `src/api/routes.py` — module-level deque, helper `_compute_drift_z(model, confidence) -> tuple[float | None, int]`, wire into `predict_bbb`.
- `src/frontend/app.py` — add a drift line to `_render_prediction_card` between the calibration caption and the SHAP section.

**Boundary contracts:**

- `bbb_model._compute_train_stats(model, X_train, y_train) -> dict` (private helper, mirrors `_compute_calibration_bins` shape).
- `routes._compute_drift_z(model, confidence) -> tuple[float | None, int]` — returns `(drift_z, len_after_append)`. Side effect: appends to module-level deque.
- The deque is module-level so it survives across requests but resets per worker restart. This is acceptable: drift is a *demo-day signal*, not a production audit trail.

**Streamlit rendering rule:**

```
if drift_z is None and rolling_n < 10:
    show "Drift: warming up ({rolling_n}/10)"
elif drift_z is None:
    show "Drift: unavailable (no train-time stats)"
else:
    show "Drift: trailing-100 median is {drift_z:+.2f}σ from training distribution"
```

### 3.2 MLflow traceability badge

**Files touched:**

- `src/api/schemas.py` — add `provenance: ModelProvenance | None` (new schema) to `BBBPredictResponse`.
- `src/api/routes.py` — read MLflow run metadata once at module load (cached); attach to every `/predict/bbb` response.
- `src/frontend/app.py` — render badge in `_render_prediction_card` near the top of the card.

**`ModelProvenance` schema:**

```python
class ModelProvenance(BaseModel):
    mlflow_run_id: str | None = None
    model_version: str = "v1"  # bumped manually per train cycle
    train_date: str | None = None  # ISO 8601, from MLflow run start_time
    n_examples: int | None = None  # from model._neurobridge_train_stats["n_train"]
```

**Lookup logic (one-time per process startup, then cached):**

1. Try `mlflow.search_runs(experiment_names=["bbb_pipeline"], max_results=1, order_by=["start_time DESC"])`.
2. If found → populate `mlflow_run_id`, `train_date`.
3. If not found or `NEUROBRIDGE_DISABLE_MLFLOW=1` → all fields stay None except hardcoded `model_version="v1"`.
4. `n_examples` comes from `model._neurobridge_train_stats["n_train"]` (set in T1A).

The badge is purely informational — the API still works without MLflow, just shows "Provenance unavailable" in the UI.

### 3.3 LLM explainer

**New file:** `src/llm/explainer.py`

Public surface:

```python
def explain(payload: ExplainPayload) -> ExplainResult:
    """Return a natural-language rationale for a BBB prediction.

    Falls back to a deterministic template when LLM is unavailable.
    Never raises — always returns a usable rationale.
    """
```

Where `ExplainPayload` is a typed dict with: `smiles`, `label_text`, `confidence`, `top_features` (list of `{feature, shap_value}`), `calibration` (optional), `drift_z` (optional).

**Internal structure:**

```
explain(payload)
├── _should_use_llm() → bool                 # gates: env flag, key, etc.
├── _llm_explain(payload) → str | None       # OpenRouter call, returns None on any failure
├── _template_explain(payload) → str         # always-available deterministic path
└── compose ExplainResult with source/model fields
```

**Template (deterministic, jury-friendly):**

The template stitches together:
1. Sentence 1: "Predicted **{label_text}** with {confidence*100:.0f}% confidence."
2. Sentence 2 (if calibration): "Calibration: predictions in the ≥{threshold}% bin are correct {precision}% of the time on held-out data (n={support})."
3. Sentence 3: "Top SHAP attributions toward this label: {feat_1} (Δ{shap_1:+.3f}), {feat_2} (Δ{shap_2:+.3f}), {feat_3} (Δ{shap_3:+.3f})."
4. Sentence 4 (if drift_z): "Drift signal: trailing-100 confidence median is {drift_z:+.2f}σ from training distribution; {interpretation}."
   - interpretation: `|drift_z| < 1` → "within expected range"; `1 ≤ |drift_z| < 2` → "mild distribution shift"; `|drift_z| ≥ 2` → "significant shift, retrain recommended".

The template is auditable: every word is derived from numeric inputs. Useful for jurors who challenge "is this actually using the model output?".

**LLM prompt (single-shot, no system message clutter):**

```
You are a clinical-ML explainer for a B2B blood-brain-barrier permeability tool.
Given the prediction details below, write a 2–4 sentence rationale a researcher
could paste into a paper. Use the SHAP attributions to justify the verdict.
Mention drift if abnormal. Avoid hedging; be specific about the numbers.

Prediction:
- SMILES: {smiles}
- Verdict: {label_text} ({confidence*100:.0f}% confident)
- Top SHAP features (positive = pushed toward verdict):
{top_features_bulleted}
- Drift z-score: {drift_z}

Respond with the rationale only, no preamble.
```

### 3.4 `POST /explain/bbb` route

**Files touched:**

- `src/api/schemas.py` — `BBBExplainRequest`, `BBBExplainResponse`.
- `src/api/routes.py` — register on `predict_router` so URL is `/predict/bbb` (existing) + `/explain/bbb` (new); both under `/predict` prefix... actually wait, see below.

**Routing decision:** new `explain_router` with prefix `/explain` → final URL `POST /explain/bbb`. Mounted on the FastAPI app alongside the existing `router` (prefix `/pipeline`) and `predict_router` (prefix `/predict`). This mirrors the prediction surface symmetrically (`/predict/bbb` ↔ `/explain/bbb`) and leaves room for `/explain/eeg` and `/explain/mri` later without restructuring.

**Request:**

```python
class BBBExplainRequest(BaseModel):
    smiles: str
    label: int
    label_text: str
    confidence: float
    top_features: list[FeatureAttribution]
    calibration: CalibrationContext | None = None
    drift_z: float | None = None
```

**Response:**

```python
class BBBExplainResponse(BaseModel):
    rationale: str
    source: str  # "llm" | "template"
    model: str | None = None  # llm model name when source="llm"
```

**Error cases:**

- Empty `top_features` → 400 (a real prediction always has SHAP attributions).
- Otherwise → 200 always (the explainer never raises; template fallback ensures success).

### 3.5 Streamlit "AI Assistant" tab

**File touched:** `src/frontend/app.py`

**Layout:**

```
┌──────────────────────────────────────────────────────────────┐
│ AI Assistant — explain the last BBB prediction               │
├──────────────────────────────────────────────────────────────┤
│ [Last prediction card preview: label, confidence, top-3 SHAP]│
│                                                              │
│ Pre-canned questions (st.selectbox):                         │
│   • Why was this molecule predicted as permeable?            │
│   • Which features pushed the verdict the most?              │
│   • Is the prediction trustworthy given drift?               │
│                                                              │
│ [Custom question text_input — optional]                      │
│                                                              │
│ [Ask the AI Assistant — primary button]                      │
├──────────────────────────────────────────────────────────────┤
│ Response card:                                               │
│   "{rationale}"                                              │
│   Source: {llm | template}  ·  Model: {model or "—"}         │
└──────────────────────────────────────────────────────────────┘
```

**Question routing into the prompt:** the user's selected/typed question is **not** sent to the LLM as a separate field. It is *appended to the prompt as a "User question:" line* before the closing instruction. This keeps the response contract (`{rationale, source, model}`) identical regardless of question, and lets the deterministic template ignore the question entirely (template always answers the meta-question "explain this prediction" — which subsumes all three pre-canned questions). For a custom question that diverges far from the canned three, the LLM path will adapt; the template path will give the same generic SHAP-driven rationale. Acceptable trade-off for Day 7.

**Session state:**

- `st.session_state["last_bbb_prediction"]` — populated by `_render_prediction_card` after every successful BBB predict (stores the entire `/predict/bbb` response dict).
- `st.session_state["explain_history"]` — list of `(question, response)` tuples; rendered in reverse-chronological order.
- If `last_bbb_prediction` is None, show empty state: "Run a BBB prediction first to enable the AI Assistant."

**No multi-turn conversation.** Each question is independent; history is visible but not fed back into subsequent prompts.

---

## 4. Test Plan

| Suite | New Tests | What they cover |
|---|---|---|
| `tests/models/test_bbb_model.py` | +2 | `_neurobridge_train_stats` attribute presence, joblib roundtrip |
| `tests/api/test_routes.py` (T1B) | +2 | `drift_z` and `rolling_n` in `/predict/bbb` body; deque rolls (101st predict drops 1st) |
| `tests/api/test_routes.py` (T2) | +1 | `provenance` field in `/predict/bbb` response (smoke — fields can be None) |
| `tests/llm/test_explainer.py` (new dir) | +4 | (a) template path returns deterministic rationale; (b) template includes top feature names; (c) template includes label_text; (d) `NEUROBRIDGE_DISABLE_LLM=1` forces template even with key set |
| `tests/api/test_routes.py` (T3B) | +1 | `POST /explain/bbb` 200 happy path with template source |
| **Total** | **+10** | **165 → 175 green** |

**LLM integration tests (env-gated, NOT counted in 175):**

- `tests/llm/test_explainer_integration.py` — marked `@pytest.mark.llm_integration`, runs only when `RUN_LLM_TESTS=1` set. Verifies real OpenRouter round-trip. Default: skip.

**TDD discipline:** For T1A, T1B, T3A, T3B: write the new tests, watch them fail (RED), then implement. T1C, T2, T3C are UI-only or thin glue; covered by import-smoke and the existing assertion extensions.

---

## 5. New Dependency

`openai==1.51.0` — added to `requirements.txt`. ~600 KB, minimal transitive (`httpx`, `pydantic`, `typing_extensions` — all already present). Pinned to 1.51.0 because that's a known-stable version with the OpenRouter-compatible client interface.

No other new pip deps. Streamlit, altair, sklearn, RDKit, MNE, nibabel, MLflow stay at current pins.

---

## 6. Failure Modes & Lifelines

| Failure | Detection | Lifeline |
|---|---|---|
| OpenRouter rate-limit during demo | HTTP 429 from SDK | Auto-fallback to template; log warning |
| OpenRouter network outage | `APIConnectionError` | Auto-fallback to template |
| API key revoked / typo'd | HTTP 401 | Auto-fallback to template |
| Demo runner forgot key | `os.environ.get("OPENROUTER_API_KEY") is None` | Auto-fallback to template |
| User wants to force template (e.g., for reproducibility) | `NEUROBRIDGE_DISABLE_LLM=1` | Hard gate, never calls LLM |
| Drift deque accumulates noise across worker lifetime | n/a | Worker restart clears state; demo runner can `pkill -f uvicorn && uvicorn …` between dry-runs. Documented in README's "Day 7 — Demo Recipe". |
| `_neurobridge_train_stats` missing on legacy model | `getattr(model, ..., None) is None` | `drift_z=None`, UI hedge string |
| MLflow store unreachable | `mlflow.search_runs` raises | `provenance` fields all None, UI shows "Provenance unavailable" |

---

## 7. Risks & Mitigations

- **Risk:** Streamlit session-state quirks may lose `last_bbb_prediction` across reruns.
  **Mitigation:** Use `st.session_state` (persistent across reruns within a session). Test by clicking Predict → switching to AI Assistant tab → verifying last prediction is visible.

- **Risk:** OpenRouter free model returns garbage for chemistry questions.
  **Mitigation:** Tightly scoped prompt (2–4 sentences, no preamble). Worst case the rationale is verbose but harmless; the source label tells jurors it's the LLM not the deterministic path.

- **Risk:** New `openai` dep conflicts with existing `httpx`.
  **Mitigation:** `openai==1.51.0` uses `httpx>=0.23,<1.0`; we already pin `httpx==0.27.2`. Compatible. Verify with `pip check` after install.

- **Risk:** Reset-drift endpoint adds attack surface.
  **Mitigation:** It's a POST that clears in-process state on the API server; no auth needed for a hackathon demo, but documented as "demo only" in OpenAPI description.

- **Risk:** MLflow lookup at module load slows API startup.
  **Mitigation:** Wrap in try/except; on any error, set `_PROVENANCE_CACHE = None` and continue. Lazy-evaluate per-request only if cache is None. Time-bound the lookup to 2 seconds.

---

## 8. Definition of Done

- ✅ `pytest -q` reports **175 passed**.
- ✅ `pytest -W error::UserWarning tests/` reports zero warnings as errors.
- ✅ `pytest tests/llm/ -v` passes (template path, 4 tests).
- ✅ `streamlit run src/frontend/app.py` boots without ImportError; AI Assistant tab visible.
- ✅ `curl POST /predict/bbb {smiles: "CCO"}` returns body with `drift_z`, `rolling_n`, `provenance` keys.
- ✅ `curl POST /explain/bbb {…}` returns 200 with `{rationale, source: "template"}` when `NEUROBRIDGE_DISABLE_LLM=1`.
- ✅ With real `OPENROUTER_API_KEY` set + flag unset, `source: "llm"` and `model: "meta-llama/llama-3.2-3b-instruct:free"`.
- ✅ Streamlit BBB decision card shows: confidence progress + calibration caption + drift line + MLflow badge + SHAP bars (in that order).
- ✅ AI Assistant tab can ask "Why permeable?" and render a rationale (from either source).
- ✅ AGENTS.md §10 (Drift Surface) and §11 (LLM Explainer Surface) committed.
- ✅ README has a "Day 7 — Demo Recipe" section with two `curl` invocations.
- ✅ Final commit ledger has 5 commits: T1, T2, T3A, T3B+T3C, T4 (or finer granularity, but at least one commit per task boundary).

---

## 9. Out of Scope (deferred to "someday")

- Multi-turn conversation memory.
- Per-user drift profiles (currently shared across all clients of one worker).
- Retraining trigger when `|drift_z| > 2` for N consecutive predictions.
- Vector-store RAG over actual chemistry literature.
- LLM rationale streaming (Streamlit chat-style typewriter).
- Provenance signing / cryptographic audit trail.
- Drift state persistence across worker restarts.

These are recognized but explicitly not Day 7. Doing any of them would either blow the time budget or shift the demo focus away from the four sealed tasks.
