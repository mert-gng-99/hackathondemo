# External Assets Integration — Roadmap

> **For agentic workers:** Index of three sub-plans for integrating the user's external assets. Each sub-plan executes via `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans`.

**Vision.** The user has supplied three external assets that should replace or extend our current placeholders:

| Asset | What it is | Replaces / extends |
|---|---|---|
| Pretrained MRI 2D classifier | PyTorch resnet18 trained on Kaggle's 4-class Alzheimer's MRI dataset (`MildDemented` / `ModerateDemented` / `NonDemented` / `VeryMildDemented`) | The dummy ONNX model in `tests/fixtures/build_dummy_mri_onnx.py`; the placeholder behaviour in `src/models/mri_model.py` |
| TF-IDF RAG corpus | 14 medical PDFs (Alzheimer + Parkinson + lifestyle/nutrition/exercise) with a pre-built TF-IDF index and Turkish query expansion | The existing FAISS+fastembed RAG in `src/rag/` (or runs alongside it) |
| EEG pretrained classifier (assumed for demo) | Any classifier with a `predict_proba` interface that emits Alzheimer's-related class probabilities. **The real artifact is not in the repo yet** — for the hackathon demo we ship a stub-able contract; swap the real `.joblib` (or `.onnx` / `.pt`) in later. | The current EEG path which is signal-processing only (no classifier yet in `src/models/`) |

---

## Sub-projects

| # | Sub-plan file | Owner concern | Depends on | Demo on its own? |
|---|---|---|---|---|
| 1 | `2026-05-02-mri-dl-2d-integration.md` | Real MRI deep-learning model in production path | — (parallel to fusion) | yes (Streamlit + curl) |
| 2 | `2026-05-02-tfidf-rag-integration.md` | Lifestyle / clinical-paper RAG with Turkish support | — | yes (CLI + agent tool) |
| 3 | `2026-05-02-eeg-stub-integration.md` | Stub-able EEG classifier contract that flows into fusion as the `eeg` modality. Real artifact swaps in later without code changes. | fusion engine (already shipped) | yes (POST /predict/eeg + fusion) |

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
                              │ #3 EEG stub    │
                              │ (real artifact │
                              │  drops in later)│
                              └────────────────┘
```

All three are independent on file boundaries — they can be built in parallel by different subagents. The diagram shows demo flow priority, not a build dependency.

---

## Open prerequisites (user must resolve)

These are **not** dev gaps — they are inputs we need from outside this codebase. Each sub-plan calls them out explicitly in its preamble, but listing here so they are in one place.

### A. MRI checkpoint drop-in

The artifact lives at `outputs\checkpoints\best_model.pt` on the trainer machine. Drop it at `data/processed/mri_dl_2d/best_model.pt` in this repo (gitignored — never commit a model binary). The user's BEST_PARAMS are final: `image_size=160`, `model_name=resnet18`, 4-class head with the index order below. The integration code does not retrain or second-guess; it loads and predicts.

```python
CLASS_TO_IDX = {
    "MildDemented": 0,
    "ModerateDemented": 1,
    "NonDemented": 2,
    "VeryMildDemented": 3,
}
```

Sub-plan #1 ships a real-artifact sanity test that runs only when the file is present (skipped otherwise) — catches any class-order or input-shape drift the trainer might surprise us with later.

### B. EEG artifact is intentionally a stub for the demo

Real EEG checkpoint will land later. For the hackathon, sub-plan #3 ships a stub artifact (`tests/fixtures/build_dummy_eeg_clf.py` produces a synthetic joblib-pickled `RandomForestClassifier`) and a clear contract: **input** = numpy array of shape `(n_features,)` matching the existing `eeg_pipeline.py` feature output; **output** = class probabilities for `("control", "alzheimers")`. Swapping in the real artifact later requires zero code changes — just drop the file at `data/processed/eeg_clf.joblib` and update labels in env if the real classes differ.

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
