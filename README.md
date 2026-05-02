---
title: NeuroBridge Enterprise
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: src/frontend/app.py
app_port: 7860
pinned: false
license: mit
short_description: Living decision system for BBB, EEG, and MRI clinical ML
---

# NeuroBridge Enterprise

> **Trust-engineered clinical-ML platform for neuroscience labs and health systems.**

## Executive Summary

**1.** Multi-site clinical ML pipelines fail in production because they assume clean data, single-site distributions, and black-box trust — all of which break in real labs. NeuroBridge Enterprise is the *living decision system* that closes those three gaps end-to-end across BBB drug-screening, EEG signal-cleaning, and MRI multi-site harmonization.

**2.** Three production pipelines (RDKit + Morgan, MNE+ICA, neuroHarmonize ComBat) sit behind one FastAPI surface and one Streamlit dashboard, with a Random Forest BBB classifier on top — every inference returns label + confidence + 6-bin precision-at-threshold calibration + top-k SHAP attributions + drift z-score + MLflow provenance + an LLM/template natural-language rationale.

**3.** Robustness is demoed live: a curated edge-case dropdown probes invalid SMILES, OOD molecules, and boundary inputs — the system never crashes, always degrades gracefully (HTTP 400 → recoverable warning, low confidence + lower drift score, calibration caption hedge).

**4.** Adapt-Over-Time is built in: each FastAPI worker keeps a rolling 100-prediction window; the trailing median is z-scored against the train-time confidence distribution and surfaced both in the API response and the UI ("trailing-100 confidence median is +1.42σ from training distribution — mild distribution shift").

**5.** 184 tests green, 8-day disciplined sprint, ~30 atomic commits, three demo lifelines (`NEUROBRIDGE_DISABLE_MLFLOW=1`, `NEUROBRIDGE_DISABLE_LLM=1`, `BBB_MODEL_PATH` env) so the system is jury-day bulletproof. Public-deployable on Hugging Face Spaces with one push.

## Status

| Day | Modality | Pipeline | Status |
|-----|----------|----------|--------|
| 1 | Tabular (BBB / molecules) | [`bbb_pipeline.py`](src/pipelines/bbb_pipeline.py) | Shipped — 30 tests green |
| 2 | Signal (EEG) | [`eeg_pipeline.py`](src/pipelines/eeg_pipeline.py) | Shipped — 67 tests green |
| 3 | Image (MRI / fMRI) | [`mri_pipeline.py`](src/pipelines/mri_pipeline.py) | Shipped — 106 tests green |
| 4 | API + MLOps + Frontend | FastAPI + MLflow + Streamlit + Docker | Shipped — 142 tests green |
| 5 | Decision Layer (Model + XAI + Interactive UI) | [`bbb_model.py`](src/models/bbb_model.py) — RandomForest + SHAP + `POST /predict/bbb` | Shipped — 158 tests green |
| 6 | Final Polish & Demo Features (Edge cases + Calibration + ComBat viz) | Calibration metadata + edge-case probes + `POST /pipeline/mri/diagnostics` | Shipped — 165 tests green |
| 7 | Final 5% (Drift, Traceability & Agents) | Per-worker drift z-score + MLflow provenance badge + `POST /explain/bbb` (LLM + template fallback) + AI Assistant tab | Shipped — 175 tests green |
| Day 8 — The Grand Finale (Multi-Modal Agents, Track 5 & Public Deploy) | Shipped — 184 tests green |

## Quick Start

**Prerequisite:** Python 3.10–3.12. The pinned `requirements.txt` has no cp313+ wheels;
`.python-version` pins to 3.12.

```bash
# 1. Create venv and install
python3.12 -m venv .venv312 && source .venv312/bin/activate && pip install -r requirements.txt

# 2. Verify — expect 106 passed
pytest -v

# 3. Smoke run with the bundled 6-row fixture
mkdir -p data/raw && cp tests/fixtures/bbbp_sample.csv data/raw/bbbp.csv
python -m src.pipelines.bbb_pipeline

# 4. Inspect the output at data/processed/bbbp_features.parquet
python -c "import pandas as pd; df = pd.read_parquet('data/processed/bbbp_features.parquet'); print(df.shape, df.dtypes.head())"
```

Result lives at `data/processed/bbbp_features.parquet`.

```bash
# Smoke-test the EEG pipeline with the bundled fixture (5 ch synthetic .fif)
mkdir -p data/raw
cp tests/fixtures/eeg_sample.fif data/raw/eeg.fif
python -m src.pipelines.eeg_pipeline
```

Result lives at `data/processed/eeg_features.parquet`.

```bash
# Smoke-test the MRI pipeline with the bundled fixture (6 subjects × 2 sites)
mkdir -p data/raw/mri
cp tests/fixtures/mri_sample/* data/raw/mri/
python -m src.pipelines.mri_pipeline
```

Result lives at `data/processed/mri_features.parquet` (48 ROI features per subject, ComBat-harmonized across sites).

> **Real BBBP data:** not bundled (gitignored). Download from
> [Kaggle](https://www.kaggle.com/datasets/priyanagda/bbbp) or
> [MoleculeNet](https://moleculenet.org/datasets-1); place as `data/raw/bbbp.csv`.

### Train the downstream BBB model (one-time)

```bash
python -m src.pipelines.bbb_pipeline   # produces data/processed/bbbp_features.parquet
python -m src.models.bbb_model          # produces data/processed/bbb_model.joblib
```

Then `POST /predict/bbb` (and the Streamlit BBB tab) become live. Try:

```bash
curl -s -X POST http://localhost:8000/predict/bbb \
  -H 'Content-Type: application/json' \
  -d '{"smiles": "CCO", "top_k": 5}' | python3 -m json.tool
```

### Run the full stack with Docker

```bash
docker compose up
```

Then browse to:
- **FastAPI Swagger** — <http://localhost:8000/docs>
- **Streamlit dashboard** — `streamlit run src/frontend/app.py` (port 8501; not in compose by default)
- **MLflow UI** — <http://localhost:5000>

Live-demo robustness: if the MLflow service is unreachable, set `NEUROBRIDGE_DISABLE_MLFLOW=1` to make the pipelines run without tracking.

## Repository Layout

```text
.
├── AGENTS.md                 # Project contract (vision, layout, code & data rules) — read first
├── README.md                 # this file
├── requirements.txt          # Pinned deps; Python 3.10–3.12 only
├── .python-version           # 3.12
├── pytest.ini
├── data/
│   ├── raw/                  # vendor inputs (CSV / EDF / NIfTI); gitignored
│   └── processed/            # Parquet outputs from pipelines; gitignored
├── docs/superpowers/plans/   # Per-day implementation plans
├── src/
│   ├── core/logger.py        # Shared structured logger (mandatory in every pipeline)
│   ├── pipelines/
│   │   ├── bbb_pipeline.py   # Day-1 pipeline (4 public funcs + CLI entry)
│   │   ├── eeg_pipeline.py   # Day-2 pipeline (6 public funcs + CLI entry)
│   │   └── mri_pipeline.py   # Day-3 pipeline (5 public funcs + CLI entry)
│   └── api/                  # FastAPI surface (placeholder until Day 4+)
└── tests/
    ├── core/, pipelines/     # Mirror src/ structure
    └── fixtures/          # bbbp_sample.csv, eeg_sample.fif, mri_sample/ + build_*.py
```

## BBB Pipeline (Day 1)

| Function | Purpose |
|----------|---------|
| `is_valid_smiles(smiles)` | Returns `True` iff the input is a non-empty SMILES that RDKit can parse. Handles `None`, `NaN`, and garbage strings. |
| `compute_morgan_fingerprint(smiles, n_bits, radius)` | Returns a `(n_bits,)` `uint8` numpy array using the modern `MorganGenerator` API. |
| `extract_features_from_dataframe(df, smiles_col, n_bits, radius)` | Drops invalid rows (logged WARNING with truncated index list), expands fingerprints into `fp_0..fp_{n-1}` columns, preserves metadata. Returns a model-ready `pd.DataFrame`. |
| `run_pipeline(input_path, output_path, smiles_col, n_bits, radius)` | End-to-end CSV → Parquet orchestrator. Idempotent; raises on missing input or directory output. |

All four functions log via `src.core.logger.get_logger(__name__)` per AGENTS.md §3 and
satisfy the §4 Data Readiness contract (5 invariants: schema validity, domain validity,
determinism, traceability, idempotence).

## EEG Pipeline (Day 2)

| Function | Purpose |
|---|---|
| `is_valid_epoch(epoch)` | Returns True iff the input is a finite, numeric, non-empty 2-D array. Rejects NaN/inf, non-numeric dtypes, lists/scalars. |
| `bandpass_filter(raw, l_freq, h_freq)` | Non-mutating MNE bandpass (default 1–40 Hz). Raises ValueError on inverted frequency range. |
| `remove_artifacts_with_ica(raw, eog_ch_name, n_components, random_state)` | Seeded ICA + correlation-based EOG component rejection. Skips gracefully (no-op + WARNING) on missing/typo EOG channel or NaN-contaminated data. |
| `compute_features_from_epoch(epoch, sfreq)` | Per-channel PSD bands (delta/theta/alpha/beta/gamma) + 5 statistical moments (mean/std/var/skew/kurtosis). Constant-channel safe (NaN-cleaned). |
| `extract_features_from_recording(raw, epoch_duration_s, eog_ch_name, n_components, random_state)` | Chains filter → ICA → epoching → feature extraction. Drops invalid epochs (logged WARNING with truncated index list). Returns 2-D `pd.DataFrame` with deterministic `feat_<channel>_psd_<band>` and `feat_<channel>_<stat>` columns. |
| `run_pipeline(input_path, output_path, ...)` | End-to-end FIF/EDF → Parquet orchestrator. Idempotent; raises on missing input or directory output. |

The pipeline is seeded (`random_state=97`) and produces byte-identical Parquet output for the same input — satisfying the §4 Determinism contract. Output is float64, preserved through the Parquet round-trip.

## MRI Pipeline (Day 3)

| Function | Purpose |
|---|---|
| `is_valid_volume(volume)` | Returns True iff input is a finite, numeric, non-empty 3-D ndarray. Rejects NaN/inf, non-numeric dtypes, lists/scalars. |
| `mask_brain(volume, intensity_threshold)` | Two-step brain mask: intensity threshold (default = volume mean) + 6-connectivity morphological opening to drop isolated noise voxels. WARNs if mask is empty. |
| `extract_features_from_volume(volume, mask, n_roi_axes)` | Partitions the masked volume into `prod(n_roi_axes)` axis-aligned octants (default 2×2×2 = 8) and emits 6 stats per ROI: mean / std / p10 / p50 / p90 / voxel_count. Empty ROIs → 0.0 (no NaN). Single source of truth via `_ROI_STATS_FUNCS`. |
| `harmonize_combat(features, sites, feature_cols)` | Wraps `neuroHarmonize.harmonizationLearn` with `np.round(14)` defensive determinism boundary. Removes site-level domain shift on the named columns. Raises if <2 sites or empty `feature_cols` or row/site length mismatch. |
| `run_pipeline(input_dir, sites_csv, output_path, ...)` | End-to-end NIfTI directory → ComBat-harmonized Parquet orchestrator. Drops invalid volumes with logged WARNING. Splits feature columns on a `_MIN_VAR_THRESHOLD = 1e-8` variance floor (constant columns bypass ComBat to avoid NaN). Idempotent; raises on missing input or directory output. |

Output schema: one row per surviving subject with columns `subject_id, site, feat_roi{i}_<stat>` (8 ROIs × 6 stats = 48 features). All `feat_*` are float64 (preserved through the Parquet round-trip).

## Storage Format

Pipeline outputs are written as Parquet files using the `pyarrow` engine with snappy
compression. This preserves dtypes (`uint8` fingerprint columns stay `uint8` instead of
widening to `int64` as CSV would do) and yields ~10× smaller files than CSV — material
for the `float64` EEG features Day 2 produces. See AGENTS.md §6.

## Testing & TDD

All pipeline functions and the shared logger were built TDD-first across Days 1–3 (RED → GREEN →
REFACTOR). Each task ended in a green commit; review-and-fix loops landed as separate
commits with `fix:` / `refactor:` prefixes. Run `pytest -v` at any time — the full suite
finishes in under 4 seconds on a 2024 laptop.

## Roadmap

- **Day 2 (shipped):** `eeg_pipeline.py` — bandpass + MNE ICA artifact removal + PSD + statistical features → Parquet.
- **Day 3 (shipped):** `mri_pipeline.py` — NIfTI volume loading, brain masking, ROI feature extraction, ComBat harmonization (`neuroHarmonize`) for site-level domain shift → Parquet (48 features, 106 tests green).
- **Day 4 (shipped):** FastAPI surface in `src/api/` (POST `/pipeline/{bbb,eeg,mri}` + `/health`), MLflow experiment tracking via `src.core.tracking` (see AGENTS.md §7), Streamlit dashboard at `src/frontend/app.py`, and Docker / `docker-compose.yml` for the api + MLflow stack — 142 tests green.
- **Day 5 (shipped):** Decision layer in `src/models/bbb_model.py` — RandomForest BBB classifier on Morgan fingerprints, SHAP top-k explanations, `POST /predict/bbb` endpoint, interactive Streamlit BBB tab with SMILES input + decision card + SHAP bar chart, and trainer CLI (`python -m src.models.bbb_model`). See AGENTS.md §8 — 158 tests green.
- **Day 6 (shipped):** Final polish & demo features — calibration metadata bins on the BBB classifier (precision-at-confidence in `BBBPredictResponse.calibration`), edge-case dropdown in the Streamlit BBB tab (5 curated robustness probes), trust caption on the decision card, and `POST /pipeline/mri/diagnostics` returning Pre/Post ComBat long-format data + site-gap KPIs visualized as a faceted altair KDE in the MRI tab. See AGENTS.md §8 (calibration) + §9 (demo features) — 165 tests green.

## Where to Look

- **Project rules (mandatory reading for any agent):** [`AGENTS.md`](AGENTS.md)
- **Day-1 plan (full TDD task breakdown):** [`docs/superpowers/plans/2026-04-29-neurobridge-day1-bootstrap-bbb-pipeline.md`](docs/superpowers/plans/2026-04-29-neurobridge-day1-bootstrap-bbb-pipeline.md)
- **Day-2 plan (full TDD task breakdown):** [`docs/superpowers/plans/2026-04-30-day2-eeg-mne-ica-pipeline.md`](docs/superpowers/plans/2026-04-30-day2-eeg-mne-ica-pipeline.md)
- **Logger contract:** [`src/core/logger.py`](src/core/logger.py) + [`tests/core/test_logger.py`](tests/core/test_logger.py)
- **BBB pipeline:** [`src/pipelines/bbb_pipeline.py`](src/pipelines/bbb_pipeline.py) + [`tests/pipelines/test_bbb_pipeline.py`](tests/pipelines/test_bbb_pipeline.py)
- **EEG pipeline:** [`src/pipelines/eeg_pipeline.py`](src/pipelines/eeg_pipeline.py) + [`tests/pipelines/test_eeg_pipeline.py`](tests/pipelines/test_eeg_pipeline.py)
- **Day-3 plan (full TDD task breakdown):** [`docs/superpowers/plans/2026-05-01-day3-mri-combat-pipeline.md`](docs/superpowers/plans/2026-05-01-day3-mri-combat-pipeline.md)
- **MRI pipeline:** [`src/pipelines/mri_pipeline.py`](src/pipelines/mri_pipeline.py) + [`tests/pipelines/test_mri_pipeline.py`](tests/pipelines/test_mri_pipeline.py)
- **Day-4 plan (full TDD task breakdown):** [`docs/superpowers/plans/2026-05-02-day4-api-mlops-frontend.md`](docs/superpowers/plans/2026-05-02-day4-api-mlops-frontend.md)
- **Shared core helpers:** [`src/core/determinism.py`](src/core/determinism.py), [`src/core/storage.py`](src/core/storage.py), [`src/core/tracking.py`](src/core/tracking.py)
- **FastAPI surface:** [`src/api/main.py`](src/api/main.py), [`src/api/routes.py`](src/api/routes.py), [`src/api/schemas.py`](src/api/schemas.py)
- **Streamlit dashboard:** [`src/frontend/app.py`](src/frontend/app.py)
- **Container stack:** [`Dockerfile`](Dockerfile), [`docker-compose.yml`](docker-compose.yml)
- **Day-4 tests:** [`tests/api/`](tests/api/), [`tests/frontend/`](tests/frontend/), [`tests/pipelines/test_cross_pipeline_smoke.py`](tests/pipelines/test_cross_pipeline_smoke.py)
- **Day-5 plan (full TDD task breakdown):** [`docs/superpowers/plans/2026-05-03-day5-downstream-model-xai-interactive.md`](docs/superpowers/plans/2026-05-03-day5-downstream-model-xai-interactive.md)
- **BBB downstream model (classifier + SHAP explainer + trainer CLI):** [`src/models/bbb_model.py`](src/models/bbb_model.py) + [`tests/models/test_bbb_model.py`](tests/models/test_bbb_model.py) (12 tests)
- **Day-6 plan (full TDD task breakdown):** [`docs/superpowers/plans/2026-05-04-day6-final-polish-demo-features.md`](docs/superpowers/plans/2026-05-04-day6-final-polish-demo-features.md)
- **MRI ComBat diagnostics surface (pre/post site-gap KPIs):** `POST /pipeline/mri/diagnostics` — see [`src/api/routes.py`](src/api/routes.py) + [`src/pipelines/mri_pipeline.py`](src/pipelines/mri_pipeline.py)
- **Day-7 design spec:** [`docs/superpowers/specs/2026-05-05-day7-drift-traceability-agents-design.md`](docs/superpowers/specs/2026-05-05-day7-drift-traceability-agents-design.md)
- **Day-7 plan (full TDD task breakdown):** [`docs/superpowers/plans/2026-05-05-day7-drift-traceability-agents.md`](docs/superpowers/plans/2026-05-05-day7-drift-traceability-agents.md)
- **New surface:** `POST /explain/bbb` — natural-language rationale (LLM + deterministic fallback)
- **New surface:** `drift_z` / `rolling_n` / `provenance` fields in `POST /predict/bbb` response
- **Day-8 plan (full TDD task breakdown):** [`docs/superpowers/plans/2026-05-06-day8-grand-finale.md`](docs/superpowers/plans/2026-05-06-day8-grand-finale.md)
- **New surfaces:** `POST /explain/eeg`, `POST /explain/mri`, `GET /experiments/runs`, `POST /experiments/diff`
- **New deploy artifacts:** `Dockerfile.hf`, `supervisord.conf`
- **LLM hardening (post-Day 8):** real OpenRouter LLM is now the default in deployed Spaces — `Dockerfile`/`Dockerfile.hf` no longer hard-code `NEUROBRIDGE_DISABLE_LLM=1`. Free-tier fallback chain (10 models, smartest → smallest) in [`src/llm/explainer.py`](src/llm/explainer.py), 401/400 status classification, and language-matching / intent-split prompt. Diagnostic endpoint `GET /diag/openrouter` ([`src/api/main.py`](src/api/main.py)) + Streamlit sidebar "🔧 Diagnose LLM" button. Live verification helper: [`scripts/diagnose_openrouter.py`](scripts/diagnose_openrouter.py).
- **Orchestrator agent (Task 13):** [`src/agents/orchestrator.py`](src/agents/orchestrator.py), [`src/agents/tools.py`](src/agents/tools.py), [`src/agents/prompts.py`](src/agents/prompts.py)
- **RAG layer:** [`src/rag/`](src/rag/) — chunker, embedder (fastembed), FAISS store, retriever, ingest CLI
- **Agent endpoint:** `POST /agent/run` (orchestrator + RAG); diagnostic at `GET /diag/agent`
- **Streamlit Agent tab:** "🤖 Agent" tab in [`src/frontend/app.py`](src/frontend/app.py) — input box + decision-trace expander
- **RAG knowledge base:** drop `.md`/`.pdf` into [`data/knowledge_base/`](data/knowledge_base/) — see its README

## Day 7 — Demo Recipe

Pre-flight (one terminal):

```bash
# Start API. With OPENROUTER_API_KEY set in your shell or .env,
# /explain/* hits the real LLM via the free-tier fallback chain
# (10 models, smartest → smallest — see AGENTS.md §11). Without
# a key, falls back to the deterministic template.
BBB_MODEL_PATH=data/processed/bbb_model.joblib \
  uvicorn src.api.main:app --port 8000

# Force the deterministic template path (no network, fully reproducible):
#   NEUROBRIDGE_DISABLE_LLM=1 BBB_MODEL_PATH=... uvicorn ...
```

Predict + explain (other terminal):

```bash
# 1) Predict — body now carries drift_z, rolling_n, provenance
curl -s -X POST http://localhost:8000/predict/bbb \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CCO", "top_k": 5}' | jq

# 2) Explain — feed the predict response back as the explain payload.
#    user_question drives the prompt: question language is mirrored
#    (Turkish question → Turkish answer), and the model answers the
#    question directly instead of returning a canned paper summary.
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
# With a valid key: expect "source": "llm" + a model id from the chain.
# Without:          expect "source": "template" + "model": null.

# 3) Diagnose OpenRouter reachability from inside the running API
#    (key presence, chain head, 8-token probe). Surfaced in Streamlit
#    as the sidebar "🔧 Diagnose LLM" button.
curl -s http://localhost:8000/diag/openrouter | jq
```

Streamlit demo: `streamlit run src/frontend/app.py` → BBB tab → Predict → AI Assistant tab → ask a preset question.

Drift demo: refresh the BBB tab and predict 10+ times in a row — the drift caption transitions from "warming up" to a numeric z-score.

## Demo Scripts

### 90-Second Jury Tour

Choreography for the live demo. Click order matters; every claim has a numeric receipt visible on screen.

| t | Tab | Action | Talking point |
|---|---|---|---|
| 0:00 | (open) | `streamlit run src/frontend/app.py` already launched | "This is NeuroBridge Enterprise — three modalities behind one decision system." |
| 0:05 | **BBB** | Pick "Custom input" → enter `CCO` → click Predict | Show label + 82% confidence progress bar. |
| 0:15 | (same) | Read calibration caption | "Predictions ≥80% confident are correct 92% of the time on held-out data — n=18." |
| 0:22 | (same) | Read drift caption | "Trailing-100 confidence median is +0.42σ from train — within expected range." |
| 0:30 | (same) | Read provenance badge | "MLflow run `abc123`, Model v1, n=1640 examples — full audit trail." |
| 0:35 | (same) | Switch to "Massive OOD: cyclosporine-like macrocycle" → Predict | "Cyclosporine has 11 residues, ~1.2 kDa — way outside training distribution." |
| 0:45 | (same) | Read confidence + drift | "System knows what it doesn't know — confidence drops, drift signal flags it." |
| 0:55 | **AI Assistant** | Pick preset "Why was this molecule predicted as permeable?" → Ask | "LLM rationale uses SHAP attributions + drift context — auditable source label." |
| 1:10 | **MRI** | Click "Run ComBat diagnostics" | Show 3-metric strip: Pre 5.0 → Post 0.0015 → 3290× reduction. |
| 1:20 | (same) | Point to faceted KDE | "Each color is a hospital. Pre-ComBat panels diverge; Post panels converge." |
| 1:30 | **Experiments** | Switch tabs, show MLflow runs table | "Every train run is logged; pick any two for a metric/param diff." |

### 30-Second Drift Detection Show

Standalone demo of the "Adapt Over Time" capability.

| t | Action | What jury sees |
|---|---|---|
| 0:00 | Open BBB tab. | Drift caption shows "warming up (0/10 predictions buffered)". |
| 0:05 | Hit Predict 10× rapidly with the same SMILES (`CCO`). | After predict #10, drift caption switches to a numeric z-score. |
| 0:18 | Switch to "Cyclosporine OOD" → predict 3× more. | Drift z-score rises in magnitude; if `|z|≥1`, caption shows "mild distribution shift"; if `|z|≥2`, "significant shift, retrain recommended". |
| 0:30 | Conclude. | "The system is online-aware — it doesn't just predict, it tells you when its own predictions are drifting from the world it was trained on." |

