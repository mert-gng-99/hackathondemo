# External Assets Integration — Roadmap

> **For agentic workers:** Index of three sub-plans for integrating the user's external assets. Each sub-plan executes via `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans`.

**Vision.** The user has supplied three external assets that should replace or extend our current placeholders:

| Asset | What it is | Replaces / extends |
|---|---|---|
| Pretrained MRI 2D classifier | PyTorch resnet18 trained on Kaggle's 4-class Alzheimer's MRI dataset (`MildDemented` / `ModerateDemented` / `NonDemented` / `VeryMildDemented`) | The dummy ONNX model in `tests/fixtures/build_dummy_mri_onnx.py`; the placeholder behaviour in `src/models/mri_model.py` |
| TF-IDF RAG corpus | 14 medical PDFs (Alzheimer + Parkinson + lifestyle/nutrition/exercise) with a pre-built TF-IDF index and Turkish query expansion | The existing FAISS+fastembed RAG in `src/rag/` (or runs alongside it) |
| OASIS tabular classifier (ipynb) | sklearn ensemble on OASIS longitudinal biomarkers (MMSE, eTIV, nWBV, ASF, …) | **Not an EEG model** — see sub-plan #3 for two routing options |

---

## Sub-projects

| # | Sub-plan file | Owner concern | Depends on | Demo on its own? |
|---|---|---|---|---|
| 1 | `2026-05-02-mri-dl-2d-integration.md` | Real MRI deep-learning model in production path | — (parallel to fusion) | yes (Streamlit + curl) |
| 2 | `2026-05-02-tfidf-rag-integration.md` | Lifestyle / clinical-paper RAG with Turkish support | — | yes (CLI + agent tool) |
| 3 | `2026-05-02-oasis-tabular-fusion-integration.md` | Tabular OASIS classifier as a fusion-engine feature **OR** wait for a real EEG model | fusion engine (#1 of clinical-platform-roadmap) | yes (POST /fusion/predict) |

---

## Build order

```
        ┌────────────────────────────┐      ┌────────────────────────────┐
        │ #1 MRI DL 2D integration   │      │ #2 TF-IDF RAG integration  │
        │ (independent)              │      │ (independent)              │
        └─────────────┬──────────────┘      └────────────┬───────────────┘
                      │                                  │
                      └──────────────┬───────────────────┘
                                     │
                              ┌──────▼─────────┐
                              │ #3 OASIS       │
                              │ classifier as  │
                              │ fusion feature │
                              └────────────────┘
```

#1 and #2 can be built in parallel (different files). #3 should follow once both are stable so the demo flows end-to-end.

---

## Open prerequisites (user must resolve)

These are **not** dev gaps — they are inputs we need from outside this codebase. Each sub-plan calls them out explicitly in its preamble, but listing here so they are in one place.

### A. MRI checkpoint file is not on this machine

The user said the artifact lives at `outputs\checkpoints\best_model.pt` (Windows-style path). `find /Users/mertgungor` returns no `best_model.pt`. Sub-plan #1 cannot start until the file is at `data/processed/mri_dl_2d/best_model.pt` (gitignored — never commit a model binary). Confirm class index order matches the trainer:

```python
CLASS_TO_IDX = {
    "MildDemented": 0,
    "ModerateDemented": 1,
    "NonDemented": 2,
    "VeryMildDemented": 3,
}
```

If the trainer used a different ordering (`ImageFolder` alphabetises by default), the labels we surface will be wrong. Sub-plan #1 ships a sanity test that catches this.

### B. The "EEG ipynb" is OASIS tabular, not EEG

`/Users/mertgungor/Downloads/rag/detecting-early-alzheimer-s (1).ipynb` trains an sklearn ensemble (LogReg / SVM / DT / RF / AdaBoost) on the OASIS longitudinal MRI **tabular** dataset (`oasis_longitudinal.csv` — MMSE, eTIV, nWBV, ASF, EDUC, SES, …). It contains zero EEG signal processing and saves no model artifact.

Sub-plan #3 has **two branches**:

- **Branch 3a (default).** Treat the OASIS biomarker model as a clinical-tests extension to the fusion engine (already accepts MMSE etc. as features — this just adds eTIV/nWBV/ASF and re-runs the trained sklearn model in-process).
- **Branch 3b.** If the user has a real EEG model elsewhere (a checkpoint file that consumes raw FIF / EDF data and emits class probabilities), the user must point us to it and we re-scope sub-plan #3 around that artifact.

The user must pick the branch before sub-plan #3 starts.

### C. RAG corpus location

The new RAG lives at `/Users/mertgungor/Downloads/rag/`. It must be copied into the repo at `data/external_rag/` (or a symlink — but symlinks break in Docker). The pre-built `index/rag_index.pkl` is 12.9 MB — gitignore the binary, commit only the source PDFs (or, for hackathon speed, gitignore both and document the manual copy step). Sub-plan #2 commits the wrapper code and a small fixture copy of one PDF for tests; the full corpus stays out of git.

---

## Decoupling guarantees (carry forward from clinical-platform-roadmap.md)

Independence rules from the existing roadmap apply unchanged. Specifically:

- **MRI DL 2D model is a swap-in for the existing 3D ONNX path.** The `src/models/mri_model.py` API surface stays stable; the new module sits alongside as `src/models/mri_dl_2d.py` and is selected via env var (`MRI_MODEL_KIND=resnet18_2d` vs. `volumetric_onnx`). Pipelines that don't load it must continue to work.
- **TF-IDF RAG and FAISS RAG run side-by-side.** The agent tool `retrieve_context` is widened to accept a `corpus` parameter (`"clinical"` for new TF-IDF, `"reference"` for existing FAISS). Existing tests stay green.
- **BBB stays decoupled.** Same rule from the fusion plan: no sub-plan here introduces a BBB↔MRI hard dependency.

---

## "When am I done?" gates (apply to every sub-plan)

1. All TDD tasks committed.
2. Full test suite passes (current baseline: 295 passed, 1 skipped).
3. Feature reachable end-to-end: Streamlit UI **OR** curl `/predict/mri` / `/agent/run` / `/fusion/predict` works.
4. README has a one-paragraph note describing the new asset and how to swap it out.
5. Final code-reviewer subagent verdict: "Ready to merge".
