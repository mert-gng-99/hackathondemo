# codex Branch Review-Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the 5 "Important" issues from the code review of commit `c0a7163` on the `codex` branch — silent label rename, frontend MRI shape mismatch, ambiguous BBB routing, parquet write race, and dropped out-of-stage tool calls.

**Architecture:** Surgical edits only. No refactors, no API changes, no new modules. Each task is one Important issue from the review, in order of demo risk (highest first). TDD where logic changes; direct edits where it's a comment, log line, or Streamlit widget.

**Tech Stack:** Python 3.11, pytest, pydantic v2, Streamlit (frontend), onnxruntime (MRI model), `src/core/logger.get_logger`.

**Branch policy:** Work on the existing `codex` branch. Each task = one commit. Run `pytest -q` after every task; do not proceed to the next task if anything regresses.

---

## File Map

| File | Change | Why |
|------|--------|-----|
| `src/models/mri_model.py` | Modify `predict_with_proba` (lines 96-99) | Add `logger.warning` on label/proba length mismatch |
| `tests/models/test_mri_model.py` | Add 1 test | Cover the warning path |
| `src/frontend/app.py` | Modify MRI Image Model panel (lines 1321-1361) | Replace hardcoded `[64,64,64]` with a number-input control |
| `src/agents/routing.py` | Modify `_looks_like_mri_input` (lines 65-70) | Treat existing local directory as MRI even without a slash |
| `tests/agents/test_routing.py` | Create | Cover the new MRI-dir-without-slash branch + regression for SMILES |
| `src/agents/tools.py` | Add comment near lines 105 / 132 | Document the parquet race; no behavior change |
| `src/agents/orchestrator.py` | Modify `_select_tool_calls` (lines 222-235) | `logger.info` when an out-of-stage tool call is dropped |
| `tests/agents/test_orchestrator.py` | Add 1 test | Verify the log emits |

---

## Task 1: MRI label-rename warning

**Files:**
- Modify: `src/models/mri_model.py:96-99`
- Test: `tests/models/test_mri_model.py` (add one method to `TestMRIDLModel`)

- [ ] **Step 1: Write the failing test**

Append this method inside `class TestMRIDLModel` in `tests/models/test_mri_model.py`:

```python
    def test_predict_warns_on_label_count_mismatch(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        artifact = build_dummy_mri_onnx(tmp_path / "mri_model.onnx")
        model = mri_model.load(artifact)

        with caplog.at_level("WARNING", logger="src.models.mri_model"):
            result = mri_model.predict_nifti(
                model,
                _FIXTURE_MRI,
                target_shape=(8, 8, 8),
                label_names=("control", "abnormal", "extra"),
            )

        assert result["label_text"] in {"class_0", "class_1"}
        assert any(
            "label_names length" in rec.message and "overriding" in rec.message
            for rec in caplog.records
        ), [rec.message for rec in caplog.records]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/models/test_mri_model.py::TestMRIDLModel::test_predict_warns_on_label_count_mismatch -v`
Expected: FAIL — no `WARNING` record is emitted (the current code overrides labels silently).

- [ ] **Step 3: Add the warning in `predict_with_proba`**

In `src/models/mri_model.py`, replace lines 97-98 (the silent override block) with:

```python
    if len(labels) != proba.shape[0]:
        logger.warning(
            "label_names length (%d) does not match model output dim (%d); "
            "overriding with class_0..class_N. Provided labels: %r",
            len(labels),
            proba.shape[0],
            list(labels),
        )
        labels = tuple(f"class_{i}" for i in range(proba.shape[0]))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/models/test_mri_model.py -v`
Expected: PASS for both the new test and the existing `test_predict_nifti_with_dummy_onnx` (which uses a 2-class dummy model with 2 labels — no warning expected).

- [ ] **Step 5: Run full suite to confirm no regression**

Run: `pytest -q`
Expected: 244 passed, 1 skipped (one more test than the 243-baseline).

- [ ] **Step 6: Commit**

```bash
git add src/models/mri_model.py tests/models/test_mri_model.py
git commit -m "fix(mri/model): warn when label_names length != model output dim (was silent override)"
```

---

## Task 2: Frontend MRI `target_shape` control

**Files:**
- Modify: `src/frontend/app.py:1321-1361` (the `#### MRI Image Model` block inside `_render_mri_panel`)

This is a Streamlit widget; we don't have UI tests for it. Verify visually after the edit by skimming the file.

- [ ] **Step 1: Replace the hardcoded shape with three number inputs**

In `src/frontend/app.py`, replace this block:

```python
    mri_labels = st.text_input(
        "Class labels",
        "control,abnormal",
        key="mri_predict_labels",
    )
    if st.button("Predict MRI image", key="mri_predict"):
        labels = [x.strip() for x in mri_labels.split(",") if x.strip()]
        payload: dict = {
            "input_path": mri_image,
            "target_shape": [64, 64, 64],
        }
```

with:

```python
    mri_labels = st.text_input(
        "Class labels",
        "control,abnormal",
        key="mri_predict_labels",
    )
    shape_cols = st.columns(3)
    target_d = shape_cols[0].number_input(
        "Resize D", min_value=1, max_value=256, value=64, step=1, key="mri_predict_d"
    )
    target_h = shape_cols[1].number_input(
        "Resize H", min_value=1, max_value=256, value=64, step=1, key="mri_predict_h"
    )
    target_w = shape_cols[2].number_input(
        "Resize W", min_value=1, max_value=256, value=64, step=1, key="mri_predict_w"
    )
    st.caption(
        "Defaults to 64³ for production exports. Use 8³ when testing with the "
        "dummy ONNX fixture from `tests/fixtures/build_dummy_mri_onnx.py`."
    )
    if st.button("Predict MRI image", key="mri_predict"):
        labels = [x.strip() for x in mri_labels.split(",") if x.strip()]
        payload: dict = {
            "input_path": mri_image,
            "target_shape": [int(target_d), int(target_h), int(target_w)],
        }
```

- [ ] **Step 2: Sanity-check by importing the module**

Run: `python -c "import src.frontend.app"`
Expected: no `SyntaxError`, no `ImportError`.

- [ ] **Step 3: Run full suite to confirm no regression**

Run: `pytest -q`
Expected: same pass count as after Task 1 (no new tests, no test removed).

- [ ] **Step 4: Commit**

```bash
git add src/frontend/app.py
git commit -m "feat(frontend/mri): expose target_shape as 3 number inputs (was hardcoded 64³)"
```

---

## Task 3: Routing — directory-without-slash heuristic

**Files:**
- Modify: `src/agents/routing.py:65-70` (`_looks_like_mri_input`)
- Create: `tests/agents/test_routing.py` (no existing test file for this module — see verification below)

Verify first that no test file for `routing.py` exists yet. Run `ls tests/agents/`. If `test_routing.py` already exists, append the new tests to its existing test class instead of creating the file.

- [ ] **Step 1: Write the failing tests**

Create `tests/agents/test_routing.py`:

```python
"""Tests for src.agents.routing — deterministic workflow-guard fallbacks."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.agents.routing import route_pipeline_input


class TestRoutePipelineInput:
    def test_smiles_routes_to_bbb(self) -> None:
        assert route_pipeline_input("CCO") == (
            "run_bbb_pipeline",
            {"smiles": "CCO", "top_k": 5},
        )

    def test_eeg_path_routes_to_eeg(self) -> None:
        name, args = route_pipeline_input("data/raw/sample.fif")
        assert name == "run_eeg_pipeline"
        assert args == {"input_path": "data/raw/sample.fif"}

    def test_nifti_path_routes_to_mri_with_parent_dir(self) -> None:
        name, args = route_pipeline_input("data/raw/subjects/subject_0.nii.gz")
        assert name == "run_mri_pipeline"
        assert args["input_dir"] == "data/raw/subjects"

    def test_existing_local_dir_without_slash_routes_to_mri(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "subject_dir").mkdir()
        name, args = route_pipeline_input("subject_dir")
        assert name == "run_mri_pipeline"
        assert args["input_dir"] == "subject_dir"

    def test_bare_string_with_no_matching_dir_still_routes_to_bbb(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        # Nothing on disk named "Aspirin" — should be treated as a SMILES-like token
        name, args = route_pipeline_input("Aspirin")
        assert name == "run_bbb_pipeline"
        assert args == {"smiles": "Aspirin", "top_k": 5}
```

- [ ] **Step 2: Run tests to verify the new MRI-dir test fails**

Run: `pytest tests/agents/test_routing.py -v`
Expected: 4 PASS, 1 FAIL (`test_existing_local_dir_without_slash_routes_to_mri`) because today `_looks_like_mri_input` only checks `path.exists() and path.is_dir()` *after* requiring `_looks_like_path` — and a bare `subject_dir` fails the path-shape check, so it falls through to BBB.

- [ ] **Step 3: Loosen the MRI heuristic**

In `src/agents/routing.py`, replace `_looks_like_mri_input` (lines 65-70) with:

```python
def _looks_like_mri_input(path: Path, lower: str) -> bool:
    if lower.endswith(".nii.gz") or path.suffix.lower() == ".nii":
        return True
    if path.exists() and path.is_dir():
        return True
    return not path.suffix and _looks_like_path(str(path))
```

Wait — that's the current code. The fix is: keep the existing `path.exists() and path.is_dir()` branch, but make sure we hit it even when the input has no slash. Audit shows the current code already does this: `_looks_like_mri_input` is called from `route_pipeline_input` for every non-EEG input, and the `path.exists() and path.is_dir()` branch does not require a slash. So the test should actually pass on the current code.

**Re-run Step 2 before changing code.** If the test passes as-is, this task collapses to "add the regression tests only" and we skip the code change. If it still fails, the actual gap is in `_primary_input` (which strips quotes) or in a path-resolution detail; debug by running:

```python
from pathlib import Path
print(Path("subject_dir").exists(), Path("subject_dir").is_dir())
```

…inside the `tmp_path` directory. The fix, if needed, is to also check `Path.cwd() / text` explicitly:

```python
def _looks_like_mri_input(path: Path, lower: str) -> bool:
    if lower.endswith(".nii.gz") or path.suffix.lower() == ".nii":
        return True
    if path.exists() and path.is_dir():
        return True
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists() and cwd_candidate.is_dir():
        return True
    return not path.suffix and _looks_like_path(str(path))
```

- [ ] **Step 4: Run tests to verify all 5 pass**

Run: `pytest tests/agents/test_routing.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Run full suite**

Run: `pytest -q`
Expected: 248 passed, 1 skipped (243 baseline + 1 from Task 1 + 4 from this task = 248).

- [ ] **Step 6: Commit**

```bash
git add src/agents/routing.py tests/agents/test_routing.py
git commit -m "fix(agents/routing): treat bare existing local directory as MRI input + tests"
```

---

## Task 4: Document the parquet write race in `tools.py`

**Files:**
- Modify: `src/agents/tools.py` near lines 99-105 and 126-133

No test — this is purely a comment that documents a known limitation. The reviewer flagged that concurrent `/agent/run` calls race on `data/processed/eeg_features.parquet` and `mri_features.parquet`; for the hackathon a TODO is sufficient. Do NOT change behavior in this task.

- [ ] **Step 1: Add the TODO comment to the EEG executor**

In `src/agents/tools.py`, replace this block:

```python
def _make_eeg_executor(processed_dir: Path) -> Callable[[EEGPipelineInput], EEGPipelineOutput]:
    """Closure factory: EEG pipeline, writes output under processed_dir."""
    def execute(inp: EEGPipelineInput) -> EEGPipelineOutput:
        from src.api.schemas import EEGRequest
        from src.api import routes as api_routes
        from fastapi import HTTPException
        out_path = processed_dir / "eeg_features.parquet"
```

with:

```python
def _make_eeg_executor(processed_dir: Path) -> Callable[[EEGPipelineInput], EEGPipelineOutput]:
    """Closure factory: EEG pipeline, writes output under processed_dir."""
    def execute(inp: EEGPipelineInput) -> EEGPipelineOutput:
        from src.api.schemas import EEGRequest
        from src.api import routes as api_routes
        from fastapi import HTTPException
        # TODO(post-hackathon): per-call output path. Concurrent /agent/run
        # invocations race on this file and clobber each other's MLflow runs.
        out_path = processed_dir / "eeg_features.parquet"
```

- [ ] **Step 2: Add the same TODO to the MRI executor**

In the same file, replace this block:

```python
def _make_mri_executor(processed_dir: Path) -> Callable[[MRIPipelineInput], MRIPipelineOutput]:
    """Closure factory: MRI pipeline, writes output under processed_dir."""
    def execute(inp: MRIPipelineInput) -> MRIPipelineOutput:
        from src.api.schemas import MRIRequest
        from src.api import routes as api_routes
        from fastapi import HTTPException
        out_path = processed_dir / "mri_features.parquet"
```

with:

```python
def _make_mri_executor(processed_dir: Path) -> Callable[[MRIPipelineInput], MRIPipelineOutput]:
    """Closure factory: MRI pipeline, writes output under processed_dir."""
    def execute(inp: MRIPipelineInput) -> MRIPipelineOutput:
        from src.api.schemas import MRIRequest
        from src.api import routes as api_routes
        from fastapi import HTTPException
        # TODO(post-hackathon): per-call output path. Concurrent /agent/run
        # invocations race on this file and clobber each other's MLflow runs.
        out_path = processed_dir / "mri_features.parquet"
```

- [ ] **Step 3: Run full suite**

Run: `pytest -q`
Expected: same pass count as after Task 3 (no behavior change, no test change).

- [ ] **Step 4: Commit**

```bash
git add src/agents/tools.py
git commit -m "docs(agents/tools): TODO race on shared parquet output for concurrent /agent/run"
```

---

## Task 5: Log dropped out-of-stage tool calls

**Files:**
- Modify: `src/agents/orchestrator.py:222-235` (`_select_tool_calls`)
- Test: `tests/agents/test_orchestrator.py` (add one method to `TestOrchestrator`)

- [ ] **Step 1: Write the failing test**

Append this method inside `class TestOrchestrator` in `tests/agents/test_orchestrator.py`:

```python
    def test_workflow_drops_out_of_stage_tool_call_with_log(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            # During the pipeline stage the model wrongly calls retrieve_context
            _fake_choice_with_tool_call("retrieve_context", {"query": "x", "k": 4}),
            # After the workflow guard runs the BBB pipeline, model produces text
            _fake_choice_with_text("Skipping retrieval."),
            # Then the guard runs retrieve_context, model finalizes
            _fake_choice_with_text("Final answer."),
        ]
        orch = Orchestrator(
            llm_client=client,
            tools=_make_workflow_tools(),
            system_prompt="sys",
            model="stub-model",
            max_steps=5,
            enforce_workflow=True,
            workflow_pipeline_tools={"run_bbb_pipeline"},
            workflow_retrieval_tool="retrieve_context",
            workflow_router=lambda user_input, context: (
                "run_bbb_pipeline",
                {"smiles": user_input},
            ),
            workflow_query_builder=lambda user_input, pipeline_trace, context: "q",
        )
        with caplog.at_level("INFO", logger="src.agents.orchestrator"):
            result = orch.run("CCO")
        assert result.finish_reason == "complete"
        assert any(
            "dropped out-of-stage tool call" in rec.message
            and "retrieve_context" in rec.message
            and "stage=pipeline" in rec.message
            for rec in caplog.records
        ), [rec.message for rec in caplog.records]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/agents/test_orchestrator.py::TestOrchestrator::test_workflow_drops_out_of_stage_tool_call_with_log -v`
Expected: FAIL — no INFO record matches because today the drop is silent.

- [ ] **Step 3: Add the log line in `_select_tool_calls`**

In `src/agents/orchestrator.py`, replace `_select_tool_calls` (lines 222-235) with:

```python
    def _select_tool_calls(self, tool_calls: list[Any], stage: str) -> list[Any]:
        if not self._enforce_workflow:
            return list(tool_calls)
        if stage == "pipeline":
            for tc in tool_calls:
                if tc.function.name in self._workflow_pipeline_tools:
                    return [tc]
            for tc in tool_calls:
                logger.info(
                    "dropped out-of-stage tool call: name=%s stage=%s",
                    tc.function.name,
                    stage,
                )
            return []
        if stage == "retrieve":
            for tc in tool_calls:
                if tc.function.name == self._workflow_retrieval_tool:
                    return [tc]
            for tc in tool_calls:
                logger.info(
                    "dropped out-of-stage tool call: name=%s stage=%s",
                    tc.function.name,
                    stage,
                )
            return []
        for tc in tool_calls:
            logger.info(
                "dropped out-of-stage tool call: name=%s stage=%s",
                tc.function.name,
                stage,
            )
        return []
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/agents/test_orchestrator.py -v`
Expected: PASS for the new test AND for `test_enforced_workflow_falls_back_when_model_skips_tool_calls` (the existing workflow test should still pass — the new code only adds logging, it does not change return values).

- [ ] **Step 5: Run full suite**

Run: `pytest -q`
Expected: 249 passed, 1 skipped (248 from after Task 4 + 1 from this task = 249).

- [ ] **Step 6: Commit**

```bash
git add src/agents/orchestrator.py tests/agents/test_orchestrator.py
git commit -m "fix(agents/orchestrator): log dropped out-of-stage tool calls (was silent)"
```

---

## Final Verification

After Task 5, run these checks before claiming the plan is complete:

- [ ] `pytest -q` → expect **249 passed, 1 skipped** (5 commits past `c0a7163`).
- [ ] `git log --oneline c0a7163..HEAD` → expect 5 commits in the order above.
- [ ] `git diff --stat c0a7163..HEAD` → expect changes only in:
  - `src/models/mri_model.py`
  - `src/frontend/app.py`
  - `src/agents/routing.py`
  - `src/agents/tools.py`
  - `src/agents/orchestrator.py`
  - `tests/models/test_mri_model.py`
  - `tests/agents/test_orchestrator.py`
  - `tests/agents/test_routing.py` (created)
- [ ] No changes outside that list (no doc edits, no schema edits, no API route edits).
- [ ] Run the frontend dev server (`streamlit run src/frontend/app.py` or whatever the project's dev command is — check `README.md`) and confirm the MRI Image Model panel shows three "Resize D/H/W" number inputs and the caption.

If all four checks pass, the codex branch is ready for the hackathon demo.
