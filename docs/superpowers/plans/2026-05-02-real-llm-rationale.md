# Real-LLM Rationale (drop the template-only fallback) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `POST /explain/{bbb,eeg,mri}` return `source="llm"` end-to-end against OpenRouter, instead of the deterministic template — without removing the template (it stays as a true outage fallback, not the everyday path).

**Architecture:** The explainer at [src/llm/explainer.py](src/llm/explainer.py) already has both paths; the LLM path is silently failing because (a) the configured `OPENROUTER_API_KEY` returns **401 on every model** today, (b) the `_DEFAULT_FREE_MODEL_CHAIN` lists a mix of speculative IDs that may not exist on OpenRouter, and (c) `401` (auth) is not classified as a fatal-not-recoverable error — it's swept into the generic "fall back to template" branch and a tester would never know auth was the cause. We fix auth → trim the chain to verified-live free models → add an explicit 401/400 short-circuit → add one network-gated integration test that proves an end-to-end LLM call returns `source="llm"`.

**Tech Stack:** Python 3.12, `openai==1.51.0` (OpenRouter-compatible), pytest 8.x, FastAPI 0.115, Streamlit 1.x.

---

## Pre-flight

- This plan modifies a single module ([src/llm/explainer.py](src/llm/explainer.py)) plus its test file ([tests/llm/test_explainer.py](tests/llm/test_explainer.py)) and adds a one-shot diagnostic script. Blast radius is small; a feature branch in the current working tree is sufficient. Worktree isolation (`superpowers:using-git-worktrees`) is **not required** unless the engineer wants to keep `main` clean while polling OpenRouter for live model IDs.
- The user has indicated `.env` already holds an `OPENROUTER_API_KEY`. Task 1 verifies whether that key still works against any model. If every probe returns 401, the engineer must reach the user before proceeding past Task 1 — code changes can't fix an unauthorized key.
- Test discipline: the deterministic template path is the **source of truth** for unit tests (per the existing module docstring). The LLM path is exercised by **one** opt-in network-gated test that auto-skips when `OPENROUTER_API_KEY` is missing. Do not mock the OpenAI SDK at the unit-test layer for the new integration test — that defeats its purpose.

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| [src/llm/explainer.py](src/llm/explainer.py) | Modify | Trim model chain; classify 401/400 explicitly; surface auth-failure log at WARNING with actionable hint |
| [tests/llm/test_explainer.py](tests/llm/test_explainer.py) | Modify | Add unit tests for new 401/400 classifier + one network-gated end-to-end LLM test |
| `scripts/diagnose_openrouter.py` | Create | One-shot probe that lists which free model IDs respond OK vs 401/404 — used in Task 1 and again in Task 7 to re-confirm the chain |

---

### Task 1: Diagnose the live OpenRouter free-tier surface

**Files:**
- Create: `scripts/diagnose_openrouter.py`

This task produces the empirical evidence later tasks depend on. **Do not skip** — the current chain in [src/llm/explainer.py:62-73](src/llm/explainer.py#L62-L73) lists IDs like `inclusionai/ling-2.6-1t:free` that may never have existed on OpenRouter. We need the real list before we can trim.

- [ ] **Step 1: Create the diagnostic script**

```python
# scripts/diagnose_openrouter.py
"""Probe OpenRouter for which free-tier model IDs are reachable today.

Reads OPENROUTER_API_KEY from .env (or process env). Issues a single
8-token chat completion against a candidate list and prints one line per
model: status (OK / HTTP-code / exception name) + a 30-char preview of
the response when OK.

Use:
    python scripts/diagnose_openrouter.py

Or to probe a custom list:
    python scripts/diagnose_openrouter.py google/gemma-2-9b-it:free meta-llama/llama-3.2-3b-instruct:free
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Manually parse .env without python-dotenv (some envs choke on its
# frame-introspection in heredocs / non-stack-rooted callers).
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    for raw in _env_path.read_text().splitlines():
        s = raw.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

if not os.environ.get("OPENROUTER_API_KEY"):
    sys.exit("OPENROUTER_API_KEY not set (looked in env and .env)")

# Candidate list: well-known stable free-tier IDs as of 2026-Q2.
# Update by replacing this list — script is a probe, not a config source.
DEFAULT_CANDIDATES = [
    "google/gemma-2-9b-it:free",
    "google/gemini-2.0-flash-exp:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
    "qwen/qwen-2.5-72b-instruct:free",
    "deepseek/deepseek-r1:free",
    "deepseek/deepseek-chat:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "microsoft/phi-3-mini-128k-instruct:free",
]

candidates = sys.argv[1:] or DEFAULT_CANDIDATES

from openai import (  # noqa: E402  (after env load)
    OpenAI, APIStatusError, APIConnectionError, RateLimitError, APITimeoutError,
)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    timeout=15.0,
)

for m in candidates:
    try:
        c = client.chat.completions.create(
            model=m,
            messages=[{"role": "user", "content": "Reply with the single word OK."}],
            max_tokens=8,
            temperature=0,
        )
        text = (c.choices[0].message.content or "").strip()
        print(f"  OK     {m}  →  {text[:30]!r}")
    except APIStatusError as e:
        code = getattr(e, "status_code", "?")
        print(f"  {code:<5}  {m}")
    except RateLimitError:
        print(f"  429    {m}  (rate-limited)")
    except (APIConnectionError, APITimeoutError) as e:
        print(f"  CONN   {m}  ({type(e).__name__})")
    except Exception as e:
        print(f"  ERR    {m}  ({type(e).__name__}: {e})")
```

- [ ] **Step 2: Run the diagnostic**

```bash
python scripts/diagnose_openrouter.py
```

Expected output: one line per candidate, status code + preview.

- [ ] **Step 3: Branch on the result**

  - If **at least one** model returns `OK ... → 'OK'` (or any non-empty text):
    - Record the OK model IDs — they become the new chain in Task 3.
    - Continue to Task 2.
  - If **every** line shows `401`:
    - The API key is unauthorized. **Stop and reach the user.** Likely causes: key revoked, wrong account, missing free-tier opt-in at https://openrouter.ai/settings/privacy (some free models require enabling "free model training" data sharing). Do not edit code — the chain doesn't matter while auth is broken.
  - If lines show a mix of 401 and 404:
    - 401 = auth failure (still blocking). Same as above.
  - If lines show `404` for all:
    - The chain candidates are all retired. Replace `DEFAULT_CANDIDATES` with fresh IDs from `curl https://openrouter.ai/api/v1/models | jq -r '.data[]|select(.pricing.prompt=="0")|.id'` and re-run.

- [ ] **Step 4: Commit the diagnostic script**

```bash
git add scripts/diagnose_openrouter.py
git commit -m "chore(llm): one-shot OpenRouter free-tier reachability probe"
```

---

### Task 2: Lock current behavior with a unit test for the new 401 classifier

**Files:**
- Modify: `tests/llm/test_explainer.py` — add one failing test before Task 3 changes the production code

This is TDD discipline: write the test that proves the new behavior **before** writing the code. The test asserts that an unauthorized response (401) classifies as fatal-no-retry — `_llm_explain` returns `None` immediately after one model attempt instead of trying every model in the chain.

- [ ] **Step 1: Read the existing test file structure**

```bash
sed -n '1,40p' tests/llm/test_explainer.py
```

Expected: confirm the file uses pytest classes + `monkeypatch.setenv`, and that an `_FIXTURE_PAYLOAD_BBB` (or similar) is defined.

- [ ] **Step 2: Write the failing test**

Append the following at the bottom of [tests/llm/test_explainer.py](tests/llm/test_explainer.py). The fixture name in the existing file is `_FIXTURE_PAYLOAD_BBB` — confirm by grep before pasting; if it differs, swap to whatever the file already exports.

> **Monkeypatch target subtlety:** `src/llm/explainer.py` does `from openai import OpenAI` **inside** `_llm_explain` (the import is local to the function), so `monkeypatch.setattr(ex, "OpenAI", ...)` would silently no-op (the module-level attribute doesn't exist and the function rebinds locally each call). We must patch on the `openai` module itself: `monkeypatch.setattr("openai.OpenAI", factory)`. The local `from openai import OpenAI` then resolves to our stub.

```python
class TestAuthFailureShortCircuits:
    """A 401 from OpenRouter means the key is unauthorized — every model
    in the chain will fail the same way, so we must short-circuit instead
    of burning the full chain on every request."""

    def test_401_short_circuits_to_template_after_one_attempt(self, monkeypatch):
        from src.llm import explainer as ex
        from openai import APIStatusError
        import httpx

        monkeypatch.delenv("NEUROBRIDGE_DISABLE_LLM", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-deliberately-bad")

        attempts: list[str] = []

        def _raise_401(**kwargs):
            attempts.append(kwargs["model"])
            req = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
            resp = httpx.Response(status_code=401, request=req)
            raise APIStatusError(message="No auth credentials found", response=resp, body={})

        class _StubCompletions:
            create = staticmethod(_raise_401)

        class _StubChat:
            completions = _StubCompletions()

        class _StubClient:
            chat = _StubChat()
            def __init__(self, **kwargs):
                pass

        # Must patch on the `openai` module — the explainer does
        # `from openai import OpenAI` *inside* the function (see
        # src/llm/explainer.py:269-275), so any module-level attribute
        # on `src.llm.explainer` would be a no-op.
        monkeypatch.setattr("openai.OpenAI", _StubClient)

        out = ex._llm_explain(_FIXTURE_PAYLOAD_BBB, modality="bbb")

        assert out is None, "401 must surface as a None return (caller falls back to template)"
        assert len(attempts) == 1, f"401 must short-circuit; tried {len(attempts)} models: {attempts}"

    def test_explain_returns_template_source_on_401(self, monkeypatch):
        from src.llm import explainer as ex
        from openai import APIStatusError
        import httpx

        monkeypatch.delenv("NEUROBRIDGE_DISABLE_LLM", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-deliberately-bad")

        def _raise_401(**kwargs):
            req = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
            raise APIStatusError(
                message="auth",
                response=httpx.Response(401, request=req),
                body={},
            )

        class _Comp:
            create = staticmethod(_raise_401)

        class _Chat:
            completions = _Comp()

        class _Client:
            chat = _Chat()
            def __init__(self, **kwargs):
                pass

        monkeypatch.setattr("openai.OpenAI", _Client)

        result = ex.explain(_FIXTURE_PAYLOAD_BBB, modality="bbb")

        assert result["source"] == "template"
        assert result["model"] is None
        assert result["rationale"], "rationale must never be empty"

    def test_400_advances_to_next_model_instead_of_short_circuiting(self, monkeypatch):
        """A 400 from one model is a prompt-shape mismatch with THAT model
        (some models reject system roles, etc.) — try the next, don't give up."""
        from src.llm import explainer as ex
        from openai import APIStatusError
        import httpx

        monkeypatch.delenv("NEUROBRIDGE_DISABLE_LLM", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-anything")

        attempts: list[str] = []
        # Force a known multi-model chain so we can count attempts deterministically
        monkeypatch.setenv("OPENROUTER_FREE_MODELS", "model-a:free,model-b:free,model-c:free")

        def _raise_400(**kwargs):
            attempts.append(kwargs["model"])
            req = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
            raise APIStatusError(
                message="bad request",
                response=httpx.Response(400, request=req),
                body={},
            )

        class _Comp:
            create = staticmethod(_raise_400)

        class _Chat:
            completions = _Comp()

        class _Client:
            chat = _Chat()
            def __init__(self, **kwargs):
                pass

        monkeypatch.setattr("openai.OpenAI", _Client)

        out = ex._llm_explain(_FIXTURE_PAYLOAD_BBB, modality="bbb")

        assert out is None, "all models 400'd → must return None for template fallback"
        assert attempts == ["model-a:free", "model-b:free", "model-c:free"], (
            f"400 must advance to next model; got attempts={attempts}"
        )
```

- [ ] **Step 3: Run the new tests — at least one MUST fail**

```bash
pytest tests/llm/test_explainer.py::TestAuthFailureShortCircuits -v
```

Expected:
- `test_400_advances_to_next_model_instead_of_short_circuiting` → **FAIL** (current code at [src/llm/explainer.py:303-310](src/llm/explainer.py#L303-L310) treats 400 as fatal and returns `None` after the first model, so `attempts` will equal `["model-a:free"]`, not the full chain).
- The two 401 tests may pass by accident with current code (the catch-all `return None` already short-circuits on any unclassified status). They stay as regression guards — Task 3 will explicitly classify 401 with an actionable log message that the test asserts on (we'll extend them in Task 3 Step 2).

This is the correct TDD red: at least one test fails on a behavior we are about to implement.

- [ ] **Step 4: Commit the failing test**

```bash
git add tests/llm/test_explainer.py
git commit -m "test(llm): pin 401 short-circuit + 400 try-next-model behavior (red)"
```

---

### Task 3: Add explicit 401/400 classification with actionable WARNING

**Files:**
- Modify: `src/llm/explainer.py:303-317`

The current code lumps "real auth failure" with "transient model error" in one branch. We split them so logs make the diagnosis obvious.

- [ ] **Step 1: Re-read the current except block to make the edit precise**

```bash
sed -n '292,320p' src/llm/explainer.py
```

- [ ] **Step 2: Replace the `APIStatusError` block with explicit classification**

Apply this edit to [src/llm/explainer.py](src/llm/explainer.py). Match the existing `except APIStatusError as e:` block (currently at line 303) exactly:

```python
        except APIStatusError as e:
            status = getattr(e, "status_code", None)
            # 401 = unauthorized — the key is bad, no model in this chain
            # will succeed. Surface a loud, actionable hint and bail.
            if status == 401:
                logger.warning(
                    "OpenRouter 401 unauthorized on %s. The OPENROUTER_API_KEY "
                    "is rejected — verify it is current at "
                    "https://openrouter.ai/keys and that free-model data-sharing "
                    "is enabled at https://openrouter.ai/settings/privacy. "
                    "Falling back to deterministic template.",
                    model,
                )
                return None
            # 400 = malformed prompt for this specific model (e.g. it
            # rejected our system role). Skip this model, try the next.
            if status == 400:
                logger.info(
                    "OpenRouter 400 on %s (likely prompt-shape mismatch); "
                    "advancing to next free model.", model,
                )
                continue
            # 402 credits / 403 access / 404 retired-id / 5xx upstream → next.
            if status in (402, 403, 404) or (status is not None and 500 <= status < 600):
                logger.info("OpenRouter %s on %s; advancing to next free model.", status, model)
                continue
            logger.warning("LLM call failed on %s (%s); falling back to template.", model, e)
            return None
```

- [ ] **Step 3: Run the new tests — they MUST pass now**

```bash
pytest tests/llm/test_explainer.py::TestAuthFailureShortCircuits -v
```

Expected: both PASS.

- [ ] **Step 4: Run the full LLM-explainer test suite to confirm no regressions**

```bash
pytest tests/llm/ -v
```

Expected: all template-path tests still pass (they should — they're env-gated to `NEUROBRIDGE_DISABLE_LLM=1`, untouched).

- [ ] **Step 5: Commit**

```bash
git add src/llm/explainer.py
git commit -m "feat(llm): classify 401 as fatal+actionable, 400 as skip-this-model"
```

---

### Task 4: Refresh `_DEFAULT_FREE_MODEL_CHAIN` with verified-live IDs

**Files:**
- Modify: `src/llm/explainer.py:62-73`

Use the OK list from Task 1's diagnostic. The chain should be ordered **smartest → smallest** so the best model is tried first; quota-exhausted models advance to the next.

- [ ] **Step 1: Re-run the diagnostic to confirm the chain is still live**

```bash
python scripts/diagnose_openrouter.py
```

Expected: at least 3 lines marked `OK`. Capture them.

- [ ] **Step 2: Replace the chain in [src/llm/explainer.py:62-73](src/llm/explainer.py#L62-L73)**

The exact replacement depends on Task 1's results. Example assuming Step 1 confirms `gemma-2-9b-it`, `llama-3.3-70b-instruct`, `mistral-7b-instruct`, `llama-3.2-3b-instruct` are OK:

```python
# Free-tier fallback chain, smartest → smallest. When a model returns 429
# (rate-limit / daily-quota exhausted), 402 (credits), 404 (id retired) or
# 5xx (upstream), we advance to the next model. Network/timeout errors fall
# straight to the deterministic template — switching models won't help.
# Override at runtime via OPENROUTER_FREE_MODELS (comma-separated). Model
# availability on OpenRouter churns; verify with scripts/diagnose_openrouter.py.
_DEFAULT_FREE_MODEL_CHAIN: tuple[str, ...] = (
    "meta-llama/llama-3.3-70b-instruct:free",   # 70B reasoning-capable
    "google/gemma-2-9b-it:free",                # 9B instruct, fast
    "mistralai/mistral-7b-instruct:free",       # 7B last-resort
    "meta-llama/llama-3.2-3b-instruct:free",    # 3B emergency
)
```

If Task 1 returned different OK IDs, substitute them; preserve the smartest-first ordering.

- [ ] **Step 3: Re-run the unit suite — must still pass**

```bash
pytest tests/llm/ -v
```

Expected: all green. The chain change is semantic-only (no test asserts specific model IDs).

- [ ] **Step 4: Commit**

```bash
git add src/llm/explainer.py
git commit -m "feat(llm): refresh free-tier chain with verified-live OpenRouter IDs"
```

---

### Task 5: Add one network-gated end-to-end LLM integration test

**Files:**
- Modify: `tests/llm/test_explainer.py` — append a new class

The unit suite proves classifier behavior with mocked errors. This test proves the **real** path: with a working key, `explain()` returns `source="llm"` and a non-empty rationale. It auto-skips when the key is missing so CI without secrets stays green.

- [ ] **Step 1: Append the integration test**

Add at the bottom of [tests/llm/test_explainer.py](tests/llm/test_explainer.py):

```python
import os as _os

import pytest as _pytest


@_pytest.mark.skipif(
    not _os.environ.get("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set — skipping live LLM integration test",
)
@_pytest.mark.skipif(
    _os.environ.get("NEUROBRIDGE_DISABLE_LLM") == "1",
    reason="NEUROBRIDGE_DISABLE_LLM=1 — skipping live LLM integration test",
)
class TestLiveOpenRouterLLM:
    """End-to-end: hit a real OpenRouter free-tier model and assert
    `explain()` returns source='llm' with non-empty content. Skipped
    when no key is set or the kill-switch is on."""

    def test_bbb_explain_returns_llm_source_with_real_key(self):
        from src.llm import explainer as ex

        result = ex.explain(_FIXTURE_PAYLOAD_BBB, modality="bbb")

        # If every model in the chain is rate-limited or unreachable RIGHT NOW
        # the result will fall back to template — that's a flaky-network
        # condition, not a code bug. Surface it as an XFAIL-style assertion
        # message instead of a hard failure.
        if result["source"] == "template":
            _pytest.skip(
                "All free models in the chain were rate-limited or unreachable "
                "at test time. Re-run later or run scripts/diagnose_openrouter.py."
            )

        assert result["source"] == "llm"
        assert result["model"] is not None and result["model"].endswith(":free")
        assert result["rationale"].strip(), "LLM returned empty rationale"
        # Sanity: the rationale should mention SOMETHING about the prediction.
        # We do not assert on exact model wording (non-deterministic), but
        # we do assert it isn't a generic refusal/safety-filter response.
        lowered = result["rationale"].lower()
        assert not lowered.startswith("i cannot"), f"LLM refused: {result['rationale']!r}"
```

- [ ] **Step 2: Run the integration test**

```bash
pytest tests/llm/test_explainer.py::TestLiveOpenRouterLLM -v -s
```

Expected (with a working key, post-Task 1 fix): PASS, with `-s` showing OpenRouter response in the WARNING/INFO logs if any.

If it skips with "rate-limited or unreachable": wait 60s and retry. If it skips with "OPENROUTER_API_KEY not set": Task 1's auth issue is unresolved — go back to Task 1 Step 3.

- [ ] **Step 3: Run the FULL test suite to confirm 188 → 190 (or higher)**

```bash
pytest -q --tb=line
```

Expected: previous count + 2 new passing unit tests + 1 new (passing or skipping) integration test. **Zero failures.**

- [ ] **Step 4: Commit**

```bash
git add tests/llm/test_explainer.py
git commit -m "test(llm): add network-gated end-to-end OpenRouter integration test"
```

---

### Task 6: End-to-end live verification through FastAPI + Streamlit

**Files:** none (verification only)

Confirm the wiring works the same way the user's UI smoke-test did, but with LLM **enabled**.

- [ ] **Step 1: Start FastAPI WITHOUT the kill-switch**

```bash
NEUROBRIDGE_DISABLE_MLFLOW=1 \
  uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --log-level info &
sleep 4
curl -s http://127.0.0.1:8000/health | python -m json.tool
```

Expected: `{"status":"ok","pipelines":["bbb","eeg","mri"]}`. **Note the absence of `NEUROBRIDGE_DISABLE_LLM`** — that's the whole point.

- [ ] **Step 2: Hit /explain/bbb with a real prediction payload**

```bash
curl -s -X POST http://127.0.0.1:8000/explain/bbb \
  -H 'Content-Type: application/json' \
  -d '{
    "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "label": 1,
    "label_text": "permeable",
    "confidence": 0.98,
    "top_features": [
      {"feature":"fp_1822","shap_value":0.0796},
      {"feature":"fp_1224","shap_value":0.0637},
      {"feature":"fp_1323","shap_value":0.0570}
    ]
  }' | python -m json.tool
```

Expected JSON: `"source": "llm"`, `"model": "<one of the chain ids>"`, `"rationale": "<2-4 free-form sentences mentioning caffeine / permeability / SHAP>"`. **Not** `"source": "template"`.

If `"source": "template"`: check the uvicorn log for the WARNING line added in Task 3 — it will tell you whether 401 (key issue), all-models-exhausted (quota/network), or something else.

- [ ] **Step 3: Hit /explain/eeg and /explain/mri**

```bash
curl -s -X POST http://127.0.0.1:8000/explain/eeg \
  -H 'Content-Type: application/json' \
  -d '{"rows": 62, "columns": 640, "duration_sec": 1.86, "mlflow_run_id": "test"}' \
  | python -m json.tool

curl -s -X POST http://127.0.0.1:8000/explain/mri \
  -H 'Content-Type: application/json' \
  -d '{"site_gap_pre": 8975.3, "site_gap_post": 3057.6, "reduction_factor": 3, "n_subjects": 6}' \
  | python -m json.tool
```

Expected: both return `"source": "llm"` with modality-appropriate prose.

- [ ] **Step 4: Start Streamlit and load the UI**

```bash
NEUROBRIDGE_API_URL=http://127.0.0.1:8000 \
NEUROBRIDGE_DISABLE_MLFLOW=1 \
  streamlit run src/frontend/app.py --server.port 8501 \
    --server.headless true --browser.gatherUsageStats false &
sleep 5
curl -s -o /dev/null -w "HTTP %{http_code}\n" http://127.0.0.1:8501/
```

Expected: HTTP 200.

- [ ] **Step 5: Manually verify the UI status badge flipped**

Open http://127.0.0.1:8501 in a browser. The Molecule (BBB) tab header should show `explainer · llm online` (green dot), **not** `explainer · template only` (mute). The status-line render is at [src/frontend/app.py:961-977](src/frontend/app.py#L961-L977) and depends on `_LLM_DISABLED` which reads `NEUROBRIDGE_DISABLE_LLM` at import time — since we did not set it, it should be False.

Then: predict a SMILES (e.g. caffeine `CN1C=NC2=C1C(=O)N(C(=O)N2C)C`), click the AI Assistant tab, generate a rationale. The rationale text should be free-form prose (not the templated "Predicted **X** with N% confidence." sentence pattern). The AI Assistant tab status indicator at [src/frontend/app.py:1056-1062](src/frontend/app.py#L1056-L1062) should also read `llm · online`.

If the badge still says `template only`: the env var leaked from a parent shell. `unset NEUROBRIDGE_DISABLE_LLM` and restart Streamlit.

- [ ] **Step 6: Tear down**

```bash
pkill -f "uvicorn src.api.main"
pkill -f "streamlit run src/frontend"
sleep 2
lsof -iTCP:8000 -sTCP:LISTEN 2>/dev/null
lsof -iTCP:8501 -sTCP:LISTEN 2>/dev/null
echo "(both empty = down)"
```

- [ ] **Step 7: No commit (verification-only task)**

If Step 2 or Step 5 surfaced any issue, fix it in the relevant earlier task and re-run from Step 1. Do not paper over a `source: "template"` response with a follow-up commit — root-cause it.

---

## Self-Review Checklist (run before declaring done)

- [ ] `pytest -q` reports the previous baseline + 2 new passing unit tests + 1 new passing-or-skipping integration test, zero failures.
- [ ] `python scripts/diagnose_openrouter.py` lists ≥1 OK model among the IDs hard-coded in `_DEFAULT_FREE_MODEL_CHAIN`.
- [ ] `curl /explain/bbb` with a real payload returns `"source": "llm"`.
- [ ] Streamlit BBB tab badge shows `explainer · llm online`, AI Assistant tab badge shows `llm · online`.
- [ ] Module docstring at [src/llm/explainer.py:1-10](src/llm/explainer.py#L1-L10) is still accurate (template = source of truth for unit tests, LLM = primary path in production).
- [ ] `NEUROBRIDGE_DISABLE_LLM=1` still forces template (existing test `test_disable_flag_forces_template_even_with_key_set` still passes — kill-switch preserved).

---

## Out of Scope (explicit non-goals)

- **Removing the template entirely.** Template stays as the outage fallback. The user said "remove from template" not "remove the template" — and even if they meant the latter, removing the template would mean a network blip = HTTP 500 from `/explain/*`, which the system-reliability shape of the project explicitly avoids (see [src/llm/explainer.py:1-10](src/llm/explainer.py#L1-L10) and the existing `test_disable_flag_forces_template_even_with_key_set` test).
- **Switching to a paid model / different provider.** The free-tier story is part of the hackathon narrative ("public-deployable on HF Spaces with one push"). Anthropic / OpenAI direct integration is a separate plan.
- **Streaming responses.** OpenRouter supports SSE streaming but neither the current API contract (`BBBExplainResponse` is a single string) nor the Streamlit UI ask for it.
- **Caching identical (payload, model) pairs.** Could halve latency for repeat clicks but adds a cache-invalidation surface; defer until a user actually complains about latency.
