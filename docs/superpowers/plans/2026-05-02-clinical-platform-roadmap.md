# Clinical Decision Platform — Roadmap

> **For agentic workers:** This is an INDEX, not an implementation plan. Each sub-plan listed below is itself a complete plan to be executed via `superpowers:subagent-driven-development` or `superpowers:executing-plans`.

**Vision.** A multi-modal Alzheimer's / Parkinson's decision platform with three personas — **Doctor, Patient, Researcher** — sharing one backend.

- **Doctor** uploads MRI and/or EEG, enters clinical-test scores (MMSE, MoCA, UPDRS, gait, age), and gets per-disease confidence with attribution.
- **Patient** sees a sanitised summary with lifestyle suggestions retrieved from peer-reviewed papers (RAG).
- **Researcher** sees a BBB-permeability map derived from MRI plus a drug-dosing adjustment hint when BBB leakage is elevated.

**Why decomposed.** Six subsystems with weak coupling. Building as one plan would produce an unreviewable mega-PR. Each sub-plan below ends in working software you could demo on its own.

---

## Sub-projects

| # | Sub-plan file | Owner concern | Depends on | Demo on its own? |
|---|---|---|---|---|
| 1 | `2026-05-02-fusion-engine.md` | Multi-modal disease confidence | — | yes (curl/JSON) |
| 2 | `2026-05-03-clinical-test-weighting.md` *(spec only)* | Doctor's clinical-test inputs + preset weights | 1 | yes (Streamlit form) |
| 3 | `2026-05-03-bbb-from-mri.md` *(spec only)* | DCE-MRI → BBB permeability map | parallel to 1 | yes (heatmap PNG) |
| 4 | `2026-05-04-persona-ui-gating.md` *(spec only)* | Doctor / Patient / Researcher views | 1, 2 | yes |
| 5 | `2026-05-04-lifestyle-rag.md` *(spec only)* | Patient lifestyle suggestions via RAG | 1 | yes |
| 6 | `2026-05-05-drug-dosing-adjustment.md` *(spec only)* | BBB leakage → drug concentration hint | 3 | yes |

---

## Sequencing

```
        ┌───────────────────────┐        ┌──────────────────────┐
        │ 1. Fusion Engine      │        │ 3. BBB-from-MRI      │
        │    (foundation)       │        │    (independent)     │
        └─────────┬─────────────┘        └──────────┬───────────┘
                  │                                 │
        ┌─────────▼──────────┐                      │
        │ 2. Clinical-test   │                      │
        │    weighting UI    │                      │
        └─────────┬──────────┘                      │
                  │                                 │
        ┌─────────▼──────────┐                      │
        │ 4. Persona UI      │                      │
        │    gating          │                      │
        └─────────┬──────────┘                      │
                  │                                 │
        ┌─────────▼──────────┐         ┌────────────▼────────────┐
        │ 5. Lifestyle RAG   │         │ 6. Drug-dosing          │
        │    (patient)       │         │    adjustment (researcher)│
        └────────────────────┘         └─────────────────────────┘
```

Build order: **1 → 2 → 4** (doctor demo) and in parallel **3 → 6** (researcher demo). Then **5** (patient demo).

---

## Independence guarantees (non-negotiable)

The pipelines must stay decoupled. Even though they share a backend, no sub-plan may introduce a hard dependency between BBB and MRI (or any other pair). Concretely:

- **`bbb_pipeline` runs on a SMILES CSV alone.** It must never require MRI input or DCE-MRI data. A drug researcher with no patient images can use BBB end-to-end.
- **`mri_pipeline` runs on a NIfTI directory + sites CSV alone.** It must never require SMILES, BBB output, or DCE-MRI. A doctor with structural T1/T2 MRI only can use MRI end-to-end.
- **`eeg_pipeline` runs on a FIF/EDF file alone.** No MRI / BBB / DCE coupling.
- **`fusion` consumes whichever modality predictions exist.** It treats absence as "no signal" (renormalises onto provided weights only — see fusion sub-plan §"Renormalisation rule"). It does **not** call BBB.
- **Sub-plan #3 (BBB-from-MRI) is the *only* place BBB and MRI touch.** That bridge requires a DCE-MRI sequence specifically. When DCE-MRI is absent, sub-plan #3 is a no-op — the standard MRI flow and the standard SMILES BBB flow both continue to work independently.

**Test discipline.** Every sub-plan that adds a new module ships at least one test that runs the touched pipeline with the *other* pipelines fully unavailable (e.g. uninstall-style: import only what's needed, assert the path completes). The roadmap-level smoke test in sub-plan #1 Task 8 already covers fusion-without-BBB; sub-plan #3 must add the symmetric "MRI without DCE" and "BBB without MRI" paths.

**Why this matters.** Real clinical reality: most patients will only have one modality. A platform that silently fails or produces nonsense when modalities are missing is unusable. Decoupling now also keeps the demo flexible — we can show any single persona without setting up data for all of them.

---

## Architectural conventions (apply to every sub-plan)

These are already in `AGENTS.md`. Stated here so each sub-plan can refer back.

- **Logging.** Use `src.core.logger.get_logger(__name__)`. All loggers have `propagate=False`, so tests must attach `caplog.handler` directly. See `tests/llm/test_explainer.py` for the canonical pattern.
- **Pydantic v2.** Any model with a `model_*` field needs `model_config = ConfigDict(protected_namespaces=())`. See `src/api/schemas.py:77`.
- **Schemas.** All API request/response models live in `src/api/schemas.py`. Keep them grouped by feature with a section comment.
- **Agent tools.** New tools register in `src/agents/tools.py`. Each tool has a pydantic input/output and a pure `execute` callable.
- **TDD.** Each task: failing test → minimal impl → passing test → commit.
- **Conventional commits.** `feat(fusion): …`, `fix(api): …`, `test(fusion): …`, `docs(plan): …`.
- **No silent failures.** When a piece of input is missing or malformed, log + exclude rather than fabricate.

---

## Out of scope (explicitly)

Do not let any sub-plan smuggle these in:

- HIPAA-grade auth or PHI storage
- Multi-tenant patient records / EMR integration
- Real DCE-MRI training pipeline (we use a stub-able ONNX contract — same pattern as `src/models/mri_model.py`)
- FDA / clinical validation framing
- Anything that requires real labelled patient data we do not already have

The platform is a hackathon decision-support **demo**, not a regulated medical device.

---

## "When am I done?" gates

A sub-plan is complete when:

1. All TDD tasks are committed.
2. Full test suite passes locally (`pytest -q`).
3. The feature is reachable end-to-end from the Streamlit UI **OR** documented in the plan as headless-only.
4. A short demo paragraph is added to `README.md` (or a feature-specific section) describing the persona path.
5. Final code-reviewer subagent verdict is "Ready to merge".
