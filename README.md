# NeuroBridge Enterprise Pipeline

NeuroBridge Enterprise tackles the three chronic failure modes in clinical ML — data drift
across acquisition sites, missing modalities, and signal/image artifacts — by running
three specialist preprocessing pipelines (MRI ComBat harmonization, EEG MNE+ICA, and BBB
molecular featurization with RDKit) behind a single FastAPI surface with MLflow tracking
and Docker shipping.

## Status

| Day | Modality | Pipeline | Status |
|-----|----------|----------|--------|
| 1 | Tabular (BBB / molecules) | [`bbb_pipeline.py`](src/pipelines/bbb_pipeline.py) | Shipped — 30 tests green |
| 2 | Signal (EEG) | [`eeg_pipeline.py`](src/pipelines/eeg_pipeline.py) | Shipped — 67 tests green |
| 3 | Image (MRI / fMRI) | [`mri_pipeline.py`](src/pipelines/mri_pipeline.py) | Shipped — 106 tests green |
| 4 | API + MLOps + Frontend | FastAPI + MLflow + Streamlit + Docker | Shipped — 142 tests green |
| 5 | Decision Layer (Model + XAI + Interactive UI) | [`bbb_model.py`](src/models/bbb_model.py) — RandomForest + SHAP + `POST /predict/bbb` | Shipped — 158 tests green |

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
