# HF Space Live Audit Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix three production bugs found on the live HF Space (https://mekosotto-hackathon.hf.space, commit `84572d9`): EEG tab default path is unreachable, the Experiments tab is permanently empty, and the BBB decision card's MLflow provenance strip shows dashes — all because `NEUROBRIDGE_DISABLE_MLFLOW=1` is set in the Dockerfile and `data/raw/eeg.fif` is never seeded.

**Architecture:** Three sealed, sequential fixes. (1) Frontend default path swap — Signal tab points at the EEG fixture so a fresh user can click "Run" with zero ceremony. (2) Dockerfile hardening — drop the `NEUROBRIDGE_DISABLE_MLFLOW=1` env, seed `data/raw/eeg.fif` from the fixture, and run the EEG + MRI pipelines at build time so the file-store `mlruns/` is populated with one run per modality. (3) Smoke verification — a sealed `scripts/smoke_hf_space.sh` that probes the public URL plus a manual click-through checklist for the JS-only Streamlit interactions we cannot script.

**Tech Stack:** Python 3.12, pytest, FastAPI, Streamlit, MLflow (file-store), Docker (HF Spaces SDK), supervisord.

**Test growth:** 184 → 188 (one frontend default-path test, one Dockerfile env test, one Dockerfile pipeline-seeding test, plus an updated existing test).

---

## File Structure

| Path | Action | Responsibility |
|---|---|---|
| `src/frontend/app.py` | Modify line 1200 | Default EEG input becomes `tests/fixtures/eeg_sample.fif` so the Signal tab works on first click |
| `Dockerfile` | Modify env block + RUN step | Remove MLflow kill-switch; seed `data/raw/eeg.fif`; run EEG + MRI pipelines at build |
| `Dockerfile.hf` | Modify env block + RUN step | Same as Dockerfile (kept in sync; HF auto-discovers `Dockerfile`) |
| `tests/frontend/test_app_defaults.py` | Create | Asserts the EEG default path is a real fixture file, not the missing `data/raw/eeg.fif` |
| `tests/deploy/test_dockerfile_hf.py` | Modify lines 30-50 | Flip MLflow assertion (must be absent), add EEG-seed + multi-pipeline assertions |
| `scripts/smoke_hf_space.sh` | Create | Sealed health probe of the public Space URL — runs after deploy |
| `docs/superpowers/notes/2026-04-30-hf-smoke-checklist.md` | Create | Manual click-through checklist (browser-only verifications) |

**Reused utilities:** `src.pipelines.eeg_pipeline.run_pipeline`, `src.pipelines.mri_pipeline.run_pipeline` — both are kwargs-based with `Path` arguments, callable via `python -c` from the Dockerfile RUN step.

---

## Task 1: Frontend EEG default path

**Files:**
- Modify: `src/frontend/app.py:1200`
- Test: `tests/frontend/test_app_defaults.py`

**Why this task is first:** It's the smallest change with its own test, lets us validate the workflow before touching Docker.

- [ ] **Step 1: Write the failing test**

Create `tests/frontend/test_app_defaults.py`:

```python
"""Defaults shown in the Streamlit UI must point at files that actually exist
in the deployed image. The HF container does not seed `data/raw/eeg.fif`
unless we explicitly do so in the Dockerfile, so the EEG tab default must
either be a fixture file (always present) or be seeded server-side.
"""
from __future__ import annotations

from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_PY = REPO_ROOT / "src" / "frontend" / "app.py"


def _eeg_default_path() -> str:
    """Extract the default value passed to the EEG `input_path` text_input.

    The line looks like:
        eeg_in = st.text_input("Input FIF/EDF path", "tests/fixtures/eeg_sample.fif", key="eeg_in")
    We pull the second positional arg (the default value).
    """
    text = APP_PY.read_text()
    match = re.search(
        r'st\.text_input\(\s*"Input FIF/EDF path"\s*,\s*"([^"]+)"',
        text,
    )
    assert match is not None, "could not locate EEG text_input default in app.py"
    return match.group(1)


def test_eeg_default_path_points_at_existing_file() -> None:
    default = _eeg_default_path()
    abs_path = REPO_ROOT / default
    assert abs_path.exists(), (
        f"EEG default path {default!r} resolves to {abs_path} which does "
        f"not exist on disk. The HF container does not seed this path "
        f"either, so users get an HTTP 404 the moment they click Run."
    )


def test_eeg_default_path_is_eeg_sample_fixture() -> None:
    """Pin the exact fix: default must be the canonical fixture so both
    local dev and the deployed image agree."""
    assert _eeg_default_path() == "tests/fixtures/eeg_sample.fif"
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `pytest tests/frontend/test_app_defaults.py -v`
Expected: 2 failures — both assertions trip because the current default is `data/raw/eeg.fif` which neither exists locally (gitignored) nor in the HF image.

- [ ] **Step 3: Patch the default in `src/frontend/app.py:1200`**

Edit `src/frontend/app.py`. Find:

```python
    eeg_in = st.text_input("Input FIF/EDF path", "data/raw/eeg.fif", key="eeg_in")
```

Replace with:

```python
    eeg_in = st.text_input(
        "Input FIF/EDF path",
        "tests/fixtures/eeg_sample.fif",
        key="eeg_in",
        help=(
            "Defaults to the bundled EEG fixture so the demo runs out of "
            "the box. Replace with your own .fif/.edf path on a real run."
        ),
    )
```

- [ ] **Step 4: Run the test and confirm it passes**

Run: `pytest tests/frontend/test_app_defaults.py -v`
Expected: 2 passed.

- [ ] **Step 5: Run the full frontend test suite to confirm no regressions**

Run: `pytest tests/frontend/ -v`
Expected: 4 passed (2 existing + 2 new).

- [ ] **Step 6: Commit**

```bash
git add src/frontend/app.py tests/frontend/test_app_defaults.py
git commit -m "fix(frontend): EEG default path points at fixture (HF container has no data/raw/eeg.fif)"
```

---

## Task 2: Update existing Dockerfile MLflow assertion (RED for Task 3)

**Files:**
- Modify: `tests/deploy/test_dockerfile_hf.py:30-50`

**Why now:** The existing test asserts `NEUROBRIDGE_DISABLE_MLFLOW` is *present* in the Dockerfile. We need to flip it before Task 3 removes the env, otherwise Task 3's edit will fail this test (= a hostile RED that hides the real failure mode).

- [ ] **Step 1: Read the existing test to understand the structure**

Run: `cat tests/deploy/test_dockerfile_hf.py`
Confirm line 44-46 contains the `assert "neurobridge_disable_mlflow" in text` block.

- [ ] **Step 2: Replace the MLflow assertion + tighten the docstring**

Edit `tests/deploy/test_dockerfile_hf.py`. Replace the entire `test_dockerfile_contains_required_stages` method body with:

```python
    def test_dockerfile_contains_required_stages(self, dockerfile_text):
        """The HF Dockerfile must:
        - Start FROM a Python base
        - Install requirements.txt
        - Seed data/raw/bbbp.csv AND data/raw/eeg.fif from fixtures
        - Build the BBB model artifact at build time
        - Run all three pipelines (BBB / EEG / MRI) so mlruns/ has one
          run per modality available to /experiments/runs at startup
        - Expose port 7860 (HF Spaces convention)
        - Launch via supervisord
        """
        text = dockerfile_text.lower()
        assert "from python" in text, "must FROM a Python base image"
        assert "requirements.txt" in text, "must reference requirements.txt"
        assert "src.models.bbb_model" in dockerfile_text, (
            "must build the BBB model artifact at image-build time"
        )
        assert "src.pipelines.bbb_pipeline" in dockerfile_text, (
            "must run BBB pipeline at build so mlruns/ has a BBB run"
        )
        assert "src.pipelines.eeg_pipeline" in dockerfile_text, (
            "must run EEG pipeline at build so mlruns/ has an EEG run"
        )
        assert "src.pipelines.mri_pipeline" in dockerfile_text, (
            "must run MRI pipeline at build so mlruns/ has an MRI run"
        )
        assert "tests/fixtures/eeg_sample.fif" in dockerfile_text, (
            "must seed data/raw/eeg.fif from the bundled fixture so the "
            "Signal tab works without user file upload"
        )
        assert "7860" in text, "must expose port 7860 (HF Spaces convention)"
        assert "supervisord" in text, (
            "must launch FastAPI + Streamlit via supervisord"
        )

    def test_dockerfile_does_not_disable_mlflow(self, dockerfile_text):
        """The kill-switch was removed in 2026-04-30 — file-store mlruns/
        is built into the image and is safe to expose on the read-only
        demo. Re-introducing the kill-switch would silently kill the
        Experiments tab and the BBB provenance strip."""
        text = dockerfile_text.lower()
        assert "neurobridge_disable_mlflow=1" not in text, (
            "Dockerfile must NOT disable MLflow — that empties the "
            "Experiments tab and blanks the BBB provenance strip. "
            "If you need to disable MLflow at runtime, set the env "
            "manually on the Space, do not bake it into the image."
        )
```

- [ ] **Step 3: Run the test and confirm RED**

Run: `pytest tests/deploy/test_dockerfile_hf.py -v`
Expected: `test_dockerfile_contains_required_stages` FAILS on the new pipeline assertions; `test_dockerfile_does_not_disable_mlflow` FAILS because the current Dockerfile still has `NEUROBRIDGE_DISABLE_MLFLOW=1`. (Both correctly RED — they describe the desired post-fix state.)

- [ ] **Step 4: Commit (RED)**

```bash
git add tests/deploy/test_dockerfile_hf.py
git commit -m "test(deploy): assert Dockerfile seeds EEG fixture, runs all pipelines, drops MLflow kill-switch (RED)"
```

---

## Task 3: Dockerfile fixes — drop kill-switch, seed EEG fixture, run all pipelines

**Files:**
- Modify: `Dockerfile` (canonical, alias of Dockerfile.hf)
- Modify: `Dockerfile.hf` (kept in sync)

**Why this works:** Both `eeg_pipeline.run_pipeline` and `mri_pipeline.run_pipeline` accept Path arguments via kwargs, so we can drive them from `python -c`. They each call `track_pipeline_run(...)` which writes to `mlruns/` via the local file-store, so removing the kill-switch lets `/experiments/runs` find them.

- [ ] **Step 1: Read the current Dockerfile**

Run: `cat Dockerfile`
Confirm the env block contains `NEUROBRIDGE_DISABLE_MLFLOW=1` and the RUN step only seeds `bbbp.csv`.

- [ ] **Step 2: Patch `Dockerfile` — env block + RUN step**

Edit `Dockerfile`. Replace the env block (lines 7-13):

```dockerfile
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    DEPLOY_ENV=hf_spaces \
    NEUROBRIDGE_DISABLE_MLFLOW=1 \
    NEUROBRIDGE_DISABLE_LLM=1
```

with (drop only the MLflow kill-switch; LLM stays disabled because OpenRouter requires a key):

```dockerfile
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    DEPLOY_ENV=hf_spaces \
    NEUROBRIDGE_DISABLE_LLM=1
```

Then replace the build-time data RUN step (lines 39-42):

```dockerfile
RUN mkdir -p data/raw data/processed && \
    cp tests/fixtures/bbbp_sample.csv data/raw/bbbp.csv && \
    python -m src.pipelines.bbb_pipeline && \
    python -m src.models.bbb_model
```

with:

```dockerfile
# Seed raw data from fixtures so the deployed Signal/Image/Molecule tabs
# work on first click. Then run all three pipelines so mlruns/ contains
# one run per modality — feeds /experiments/runs and the BBB provenance
# strip. data/raw/* is gitignored locally so we cannot COPY it.
RUN mkdir -p data/raw data/processed && \
    cp tests/fixtures/bbbp_sample.csv data/raw/bbbp.csv && \
    cp tests/fixtures/eeg_sample.fif data/raw/eeg.fif && \
    python -m src.pipelines.bbb_pipeline && \
    python -m src.models.bbb_model && \
    python -c "from pathlib import Path; from src.pipelines.eeg_pipeline import run_pipeline; run_pipeline(input_path=Path('tests/fixtures/eeg_sample.fif'), output_path=Path('data/processed/eeg_features.parquet'))" && \
    python -c "from pathlib import Path; from src.pipelines.mri_pipeline import run_pipeline; run_pipeline(input_dir=Path('tests/fixtures/mri_sample'), sites_csv=Path('tests/fixtures/mri_sample/sites.csv'), output_path=Path('data/processed/mri_features.parquet'))"
```

- [ ] **Step 3: Mirror the same edit into `Dockerfile.hf`**

Edit `Dockerfile.hf` to be byte-identical to `Dockerfile` (same env block, same RUN step). Run:

```bash
diff Dockerfile Dockerfile.hf
```

Expected: no output (files identical).

- [ ] **Step 4: Run the deploy tests to confirm GREEN**

Run: `pytest tests/deploy/ -v`
Expected: 2 passed (`test_dockerfile_contains_required_stages` + `test_dockerfile_does_not_disable_mlflow`).

- [ ] **Step 5: Run the full test suite — must stay green**

Run: `pytest -q`
Expected: 188 passed (184 baseline + 2 new frontend defaults + 1 new deploy test + 1 widened existing test). Adjust expectation only if your local count differs by exactly the same delta.

- [ ] **Step 6: Smoke-build locally to catch syntax errors before pushing**

Run (under 5 min):

```bash
docker build -t neurobridge-hf-test -f Dockerfile .
```

Expected: build completes; the final two `python -c` invocations log INFO lines like:
```
INFO    | Wrote processed features to data/processed/eeg_features.parquet (rows=N, cols=M)
INFO    | Wrote processed features to data/processed/mri_features.parquet (rows=N, cols=M)
```

If `docker` is not available locally, skip this step — CI on HF Spaces will catch a broken Dockerfile within ~3 minutes of push.

- [ ] **Step 7: Commit**

```bash
git add Dockerfile Dockerfile.hf
git commit -m "fix(deploy): seed EEG fixture, run all pipelines at build, drop MLflow kill-switch

The Experiments tab was permanently empty and the BBB provenance strip
showed dashes because NEUROBRIDGE_DISABLE_MLFLOW=1 short-circuited the
MLflow file-store lookups even though mlruns/ IS produced inside the
image at build time. Drop the env, run EEG + MRI pipelines too so all
three experiments have at least one run, and seed data/raw/eeg.fif
from the fixture so the Signal tab works on first click."
```

---

## Task 4: Smoke probe script — what we CAN automate

**Files:**
- Create: `scripts/smoke_hf_space.sh`

**Why:** HF Spaces only exposes Streamlit on :7860 publicly. FastAPI on :8000 is container-internal, so we cannot probe `/predict/bbb` or `/experiments/runs` directly. We CAN probe Streamlit's own health and the Space's HTTP envelope.

- [ ] **Step 1: Create the script directory if missing**

Run:

```bash
mkdir -p scripts
```

- [ ] **Step 2: Write `scripts/smoke_hf_space.sh`**

Create `scripts/smoke_hf_space.sh`:

```bash
#!/usr/bin/env bash
# Sealed smoke probe for the live HF Space.
# Verifies what we can verify without a browser: HTTP envelope, Streamlit
# health, response headers. Returns 0 on success, 1 on any failure.
#
# Usage: scripts/smoke_hf_space.sh [base_url]
#   default base_url: https://mekosotto-hackathon.hf.space

set -euo pipefail

BASE="${1:-https://mekosotto-hackathon.hf.space}"
FAIL=0

probe() {
  local label="$1" url="$2" expect="$3"
  local actual
  actual="$(curl -sS -o /dev/null -w "%{http_code}" "$url")"
  if [[ "$actual" == "$expect" ]]; then
    printf "  OK   %-40s %s\n" "$label" "$actual"
  else
    printf "  FAIL %-40s expected=%s actual=%s\n" "$label" "$expect" "$actual"
    FAIL=1
  fi
}

echo "Probing $BASE"
probe "frontend root"             "$BASE/"                       "200"
probe "streamlit health"          "$BASE/_stcore/health"          "200"
probe "fastapi NOT publicly mounted (good)" "$BASE/health"        "403"

echo
if [[ "$FAIL" == "0" ]]; then
  echo "Smoke OK — proceed to manual click-through checklist:"
  echo "  docs/superpowers/notes/2026-04-30-hf-smoke-checklist.md"
  exit 0
else
  echo "Smoke FAILED — see HF Space logs at:"
  echo "  https://huggingface.co/spaces/mekosotto/hackathon/discussions"
  exit 1
fi
```

- [ ] **Step 3: Make it executable**

Run:

```bash
chmod +x scripts/smoke_hf_space.sh
```

- [ ] **Step 4: Test the script against the currently-deployed (broken) Space**

Run:

```bash
scripts/smoke_hf_space.sh
```

Expected output: all three probes return OK (frontend 200, streamlit 200, fastapi 403). The smoke envelope passes even on the broken deploy — it's checking infrastructure, not feature correctness. Feature correctness is the manual checklist (Task 5).

- [ ] **Step 5: Commit**

```bash
git add scripts/smoke_hf_space.sh
git commit -m "test(deploy): scripts/smoke_hf_space.sh — sealed HTTP envelope probe"
```

---

## Task 5: Manual click-through checklist (browser-only verifications)

**Files:**
- Create: `docs/superpowers/notes/2026-04-30-hf-smoke-checklist.md`

**Why:** Streamlit is JS-rendered; we cannot drive it from curl. Document the exact button-by-button verification so the user (or a reviewer) walks the demo and catches anything the unit tests missed.

- [ ] **Step 1: Create the notes directory if missing**

Run:

```bash
mkdir -p docs/superpowers/notes
```

- [ ] **Step 2: Write `docs/superpowers/notes/2026-04-30-hf-smoke-checklist.md`**

Create `docs/superpowers/notes/2026-04-30-hf-smoke-checklist.md`:

```markdown
# HF Space Manual Smoke Checklist — 2026-04-30

After the deploy completes (~3-5 min after `git push hf main`), open
https://mekosotto-hackathon.hf.space/ and walk this list. Anything
that fails is a regression vs the audit fix in
`docs/superpowers/plans/2026-04-30-hf-space-live-audit-fixes.md`.

## Hero strip (top of page)

- [ ] Hero title `NeuroBridge Enterprise` fades up smoothly (not instant)
- [ ] Status row shows three dots:
  - [ ] `api · operational` (green)
  - [ ] `mlflow · tracking` (green) — **was muted before this fix**
  - [ ] `explainer · template only` (muted, expected — LLM stays disabled)

## Molecule tab (BBB)

- [ ] Default edge case "Custom input (default)" is selected
- [ ] Input box shows `CCO`
- [ ] Click "Predict BBB permeability"
- [ ] Decision card animates in with spring scale-in on the verdict
- [ ] Provenance strip shows real values:
  - [ ] `mlflow · <8-char run id>` (NOT `—`) — **was `—` before this fix**
  - [ ] `model · v1`
  - [ ] `trained · <ISO timestamp>` (NOT `—`)
  - [ ] `n=<integer>` (NOT `n=—`)
- [ ] Verdict reads `permeable` with confidence ~80-100%
- [ ] SHAP bar chart renders with sand-colored bars
- [ ] Switch dropdown to "Invalid SMILES" → click Predict → see yellow warning, NOT red error

## Signal tab (EEG)

- [ ] Default input field shows `tests/fixtures/eeg_sample.fif` — **was `data/raw/eeg.fif` before this fix**
- [ ] Click "Run EEG pipeline"
- [ ] Result card shows rows / columns / duration_sec / mlflow_run_id
- [ ] Expand "Ask the AI Assistant about this EEG run"
- [ ] Click "Ask AI Assistant" → see deterministic-template rationale (no error)

## Image tab (MRI)

- [ ] Defaults: `tests/fixtures/mri_sample` and `tests/fixtures/mri_sample/sites.csv`
- [ ] Click "Run ComBat diagnostics"
- [ ] Three KPI cards render: Site-gap (Pre), Site-gap (Post), Reduction factor
- [ ] Pre/Post KDE altair chart renders
- [ ] Expand "Ask the AI Assistant about this ComBat run" → click → see rationale

## AI Assistant tab

- [ ] After running a BBB prediction in the Molecule tab, this tab shows
      "Latest prediction: ..." caption
- [ ] Click "Ask the AI Assistant" → conversation appears with source =
      `template` and model = `—` (LLM intentionally disabled on HF)

## Experiments tab

- [ ] Table loads with **at least 3 rows** — one each for `bbb_pipeline`,
      `eeg_pipeline`, `mri_pipeline` — **was empty before this fix**
- [ ] Compare-two-runs section is visible (≥2 rows are present)
- [ ] Pick two run IDs → click "Show diff" → diff table renders

## Sidebar

- [ ] Toggle "Dark mode" off → page rebuilds with cream paper theme
- [ ] Toggle back on → page rebuilds with editorial dark theme
- [ ] Both themes preserve the sand accent on the hero word-mark

## Reduced-motion respect

- [ ] (Optional) Open DevTools → Rendering → enable "prefers-reduced-motion"
- [ ] Reload — animations are near-instant (< 1ms duration), but layout
      and content are unchanged
```

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/notes/2026-04-30-hf-smoke-checklist.md
git commit -m "docs(notes): manual click-through checklist for HF Space smoke verification"
```

---

## Task 6: Final verification — full suite + smoke envelope

- [ ] **Step 1: Run the full pytest suite locally**

Run: `pytest -q`
Expected: `188 passed`. (184 baseline + 2 new frontend tests + 1 new deploy test + 1 widened existing test = 188.)

- [ ] **Step 2: Run pytest with UserWarning escalation to confirm no warnings**

Run: `pytest -W error::UserWarning tests/`
Expected: `188 passed, 0 escalations`.

- [ ] **Step 3: Run the smoke envelope against the still-broken deployed Space**

Run: `scripts/smoke_hf_space.sh`
Expected: 3 OK lines (we have not deployed yet — this confirms the script itself works).

- [ ] **Step 4: Inspect the local commit graph**

Run: `git log --oneline -10`
Expected: 5 new commits on top of `84572d9`:
```
<hash> docs(notes): manual click-through checklist for HF Space smoke verification
<hash> test(deploy): scripts/smoke_hf_space.sh — sealed HTTP envelope probe
<hash> fix(deploy): seed EEG fixture, run all pipelines at build, drop MLflow kill-switch
<hash> test(deploy): assert Dockerfile seeds EEG fixture, runs all pipelines, drops MLflow kill-switch (RED)
<hash> fix(frontend): EEG default path points at fixture (HF container has no data/raw/eeg.fif)
84572d9 feat(frontend): premium motion layer — Apple HIG / Netflix transitions
```

---

## Task 7: Hand-off to user — push + verify

The HF write token from the previous session was revoked (correct security move). The push must be done by the user with a fresh token.

- [ ] **Step 1: User generates a new HF write token**

Visit https://huggingface.co/settings/tokens → "New token" → role "Write" → copy.

- [ ] **Step 2: User pushes**

```bash
cd /Users/mertgungor/Desktop/hackathon
git push hf main
```

If prompted for username/password: username = HF username (e.g. `mekosotto`), password = the write token (NOT the account password). HF deprecated password auth.

- [ ] **Step 3: User waits ~3-5 min for HF rebuild**

Monitor at https://huggingface.co/spaces/mekosotto/hackathon — the "Building" badge flips to "Running" when ready. The build re-runs `bbb_pipeline + bbb_model + eeg_pipeline + mri_pipeline` so this rebuild is slightly slower than the last one (~+1 min).

- [ ] **Step 4: User runs the smoke probe**

```bash
scripts/smoke_hf_space.sh
```

Expected: 3 OK lines.

- [ ] **Step 5: User walks the manual checklist**

Open `docs/superpowers/notes/2026-04-30-hf-smoke-checklist.md` and tick boxes against the live UI at https://mekosotto-hackathon.hf.space/.

The two regression boxes that MUST flip from FAIL → OK are:
1. BBB provenance strip shows real `mlflow · <run id>` (not `—`)
2. Experiments tab shows ≥3 rows (not empty)
3. Signal tab default input is `tests/fixtures/eeg_sample.fif` and "Run EEG pipeline" succeeds on first click.

If any box still fails, the HF build logs at https://huggingface.co/spaces/mekosotto/hackathon/logs are the next stop — paste any error line back into the conversation.

---

## Self-Review

**Spec coverage:**
- Issue 1 (EEG default path) → Task 1 (frontend edit) + Task 3 (Dockerfile seeds the path too)
- Issue 2 (Experiments empty + provenance dashes) → Task 3 drops kill-switch + bakes EEG/MRI runs
- Issue 3 (Hero status dot) → resolves automatically via Issue 2 fix; called out in Task 5 checklist
- Test growth 184 → 188 → covered by Task 1 (+2), Task 2 (+1 new method), unchanged delta on widened existing test
- "User runs `git push hf main`" → Task 7

**Placeholder scan:** None. Every step has exact code or commands.

**Type consistency:** `run_pipeline` signatures match the verified source (`eeg_pipeline.py:411`, `mri_pipeline.py:282`). The `_eeg_default_path` regex matches the literal `st.text_input("Input FIF/EDF path", "...")` form. Dockerfile env block diff is exact line-for-line.

**Risks called out:**
- Local docker build (Task 3 Step 6) is optional — HF CI catches Dockerfile syntax errors anyway.
- Removing the kill-switch widens the path that hits MLflow's broad-`except` in `_build_provenance`; that's intentional fallback behavior, not a regression.
