# BBB ↔ MRI Bridge + Drug-Dose Adjuster — Implementation Notes

> Implemented inline on `feat/external-assets-integration`. This document is the design record.

## Why

Clinical fact: **DCE-MRI** (Dynamic Contrast-Enhanced MRI) measures BBB leakage by tracking gadolinium contrast washout — Ktrans, Kep, Ve maps. Compromised BBB lets molecules cross into the brain that normally wouldn't. Two consequences:

1. A "BBB Permeability Map" or scalar score from a patient's MRI is the **primary data point for the researcher persona**.
2. **If the BBB is leaky, drug concentrations need revising** — a CNS-permeable drug crosses too easily; even a non-CNS drug may now reach the brain at unsafe levels.

This plan adds both: the bridge (MRI → BBB permeability score) and the adjuster (drug dose revision based on the score).

## Independence

This is the **only legitimate place BBB and MRI couple** in the platform — and it sits in the **researcher** lane, not the doctor's diagnostic lane. The fusion engine's "BBB is NOT a fusion modality" rule is preserved: this bridge does not feed into Alzheimer's/Parkinson's confidence.

## Architecture (two modules + two routes + two agent tools)

```
src/models/bbb_permeability_map.py    pure-Python scorer with two modes
src/research/drug_dose_adjuster.py    pure-function dose revision logic

POST /predict/bbb_permeability_map     {input_path, mode} -> {score, interpretation, method}
POST /research/drug_dose_adjustment    {smiles?, baseline_dose_mg, bbb_score, drug_permeable?}
                                       -> {recommended_dose_mg, factor, risk_level, rationale}

agent tool: compute_bbb_leakage_score
agent tool: adjust_drug_dose
```

### Permeability scoring modes

- **`heuristic_proxy`** (default, demo-ready). Uses the existing 2D resnet18 4-class
  classifier's class probabilities. Score = `1 - P(NonDemented)` — published
  literature shows BBB breakdown correlates with disease severity. Auditable
  and demonstrable today.
- **`dce_onnx`** (real-DCE artifact, swap-in later). Loads an ONNX model from
  `data/processed/bbb_permeability_dce.onnx` (env: `BBB_PERMEABILITY_DCE_PATH`).
  Contract: input 4D NIfTI `(X, Y, Z, T)`, output a 3D Ktrans map normalized
  to `[0, 1]`. Stub for now; works when artifact lands.

### Dose-adjustment math

`adjust(baseline_mg, perm_score, drug_permeable, safety_factor=0.5)`:

| BBB intact (score < 0.2) | any drug | factor=1.0, risk=low |
| BBB leaky, drug permeable | factor = max(0.3, 1 - 0.7·score), risk = high if score > 0.6 else moderate |
| BBB leaky, drug NOT permeable | factor = max(0.6, 1 - 0.4·score), risk = moderate |
| drug_permeable=None (unknown) | treat as permeable (safer assumption) |

The adjuster is a **pure function**. The route can optionally call `bbb_model.predict_one(smiles)` to populate `drug_permeable` from the existing BBB classifier — closing the researcher loop end-to-end.

## Tests

- `tests/models/test_bbb_permeability_map.py` — heuristic-proxy correctness, mode dispatch, missing-artifact errors.
- `tests/research/test_drug_dose_adjuster.py` — all four (BBB × drug-permeability) quadrants, monotonicity, safety-floor invariants.
- `tests/api/test_bbb_research_routes.py` — both new routes end-to-end.

## What this does NOT do

- Does not retrain anything.
- Does not compute real Ktrans/Kep/Ve from DCE data — that requires a 4D pharmacokinetic model we don't ship. The `dce_onnx` mode is a contract for an external trainer.
- Does not feed into the fusion engine. Researcher-only flow.
- Does not prescribe — the rationale string explicitly says "research suggestion, not medical advice".

## Demo path

1. Researcher pastes a SMILES (e.g., `CCO`) and a baseline dose (e.g., `200 mg`).
2. Researcher uploads/picks an MRI image.
3. Streamlit calls `/predict/mri` (resnet18_2d if active, else volumetric ONNX) → BBB classifier on SMILES → `/research/drug_dose_adjustment`.
4. Card shows: BBB leakage gauge, drug BBB permeability badge, recommended dose with adjustment factor, plain-language rationale.
