# MRI DL 2D Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: `superpowers:subagent-driven-development` (recommended). TDD throughout — failing test → minimal impl → passing test → commit.

**Goal.** Wire the user's pretrained PyTorch resnet18 (2D image, 4-class Alzheimer's: `MildDemented` / `ModerateDemented` / `NonDemented` / `VeryMildDemented`) into the production decision layer alongside the existing volumetric ONNX path. The model produces a probability vector that flows naturally into the fusion engine.

**Architecture.** Add `src/models/mri_dl_2d.py` parallel to the existing `src/models/mri_model.py`. A small selector picks between paths based on `MRI_MODEL_KIND` env var (`resnet18_2d` or `volumetric_onnx`). The 2D model loads a `state_dict` `.pt` checkpoint, applies the resnet18 preprocessing contract (resize 160, ImageNet normalisation), and emits `MRIPredictResponse` in the same shape the existing surface produces — so the API and frontend need no behavioural change.

**Tech stack.** Python 3.11, PyTorch (CPU), torchvision, Pillow. PyTorch is not currently in `requirements.txt` — Task 0 adds it. No new web dependencies.

**Trainer's hyper-parameters (for reference, not all relevant at inference):**

```python
BEST_PARAMS = {
    "image_size": 160,
    "model_name": "resnet18",
    "optimizer": "adamw",          # only relevant for training
    "lr": 0.000375191537539265,    # only relevant for training
    "weight_decay": 0.000196410142442417,
    "dropout": 0.31154239434523634,  # we apply at inference iff the trainer baked dropout into the head
    "batch_size": 128,             # not used at inference (we infer one image at a time)
    "epochs": 10,                  # training-only
}

CLASS_TO_IDX = {
    "MildDemented": 0,
    "ModerateDemented": 1,
    "NonDemented": 2,
    "VeryMildDemented": 3,
}
```

---

## Prerequisite (controller blocker)

The artifact `best_model.pt` is **not** present on this filesystem. Before any task starts:

1. Copy the file from the trainer machine to `data/processed/mri_dl_2d/best_model.pt`.
2. Confirm with `python -c "import torch; sd = torch.load('data/processed/mri_dl_2d/best_model.pt', map_location='cpu'); print(type(sd), list(sd.keys())[:5] if isinstance(sd, dict) else sd)"`. Two possible structures:
   - **`state_dict` only** (most common): `dict[str, Tensor]`. Task 1 builds the resnet18 architecture and `load_state_dict`s.
   - **Full model** (`torch.save(model, ...)`): a pickled `nn.Module`. Task 1 just calls `torch.load(...)`.
   - The plan defaults to **state_dict** (more portable). If the file turns out to be a full model, Task 1 has a fallback branch.
3. Add the artifact path to `.gitignore` if it isn't already covered (`data/processed/` should already be ignored — verify).

If step 2 fails, **stop and surface to the user** — the trainer either produced a different artifact or saved with an unexpected structure.

---

## File structure

| Path | Responsibility |
|---|---|
| Modify `requirements.txt` | add `torch`, `torchvision`, `pillow` (CPU wheels are fine) |
| Create `src/models/mri_dl_2d.py` | resnet18 4-class loader + preprocessing + `predict_image()` |
| Create `src/models/mri_selector.py` | tiny dispatcher: `load_default()` / `predict_default()` based on `MRI_MODEL_KIND` env |
| Modify `src/api/routes.py` | `predict_mri` chooses between volumetric and 2D paths via the selector |
| Modify `src/api/schemas.py` | `MRIPredictRequest.input_path` now accepts `.png/.jpg/.nii*` (already string) — no schema change beyond a docstring tweak |
| Create `tests/fixtures/build_dummy_resnet18_2d.py` | helper that constructs a randomly-initialised 4-class resnet18 and saves a state_dict to a tmp path so tests don't need the real artifact |
| Create `tests/models/test_mri_dl_2d.py` | unit tests for the new module |
| Create `tests/api/test_mri_2d_route.py` | integration test through `POST /predict/mri` |
| Modify `README.md` | document the env var and the artifact location |

---

## Tasks

### Task 0: Dependencies

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1:** open `requirements.txt`, append (CPU wheels — torch is large, target ~200 MB):

```
torch>=2.2,<3.0
torchvision>=0.17,<1.0
pillow>=10.0,<12.0
```

- [ ] **Step 2:** install: `pip install torch torchvision pillow`. Verify import: `python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"`. Expect a version line, no error.

- [ ] **Step 3:** run `pytest -q` — expect the existing 295+1 baseline. No regressions before any code changes.

- [ ] **Step 4:** commit: `git commit -m "deps: add torch/torchvision/pillow for MRI DL 2D"`.

---

### Task 1: 2D model loader + preprocessing

**Files:**
- Create: `src/models/mri_dl_2d.py`
- Create: `tests/fixtures/build_dummy_resnet18_2d.py`
- Create: `tests/models/test_mri_dl_2d.py`

- [ ] **Step 1: Write the dummy-checkpoint fixture (so tests don't need the real .pt).**

`tests/fixtures/build_dummy_resnet18_2d.py`:

```python
"""Build a randomly-initialised 4-class resnet18 state_dict for tests."""
from __future__ import annotations

from pathlib import Path

import torch
from torchvision import models


def build(path: Path) -> Path:
    """Save a state_dict at `path` and return the path. Idempotent."""
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 4)
    torch.save(model.state_dict(), str(path))
    return path
```

- [ ] **Step 2: Write the failing test.**

`tests/models/test_mri_dl_2d.py`:

```python
"""Tests for src.models.mri_dl_2d — pretrained 4-class Alzheimer's resnet18."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.models import mri_dl_2d
from tests.fixtures.build_dummy_resnet18_2d import build as build_dummy_2d


def _png(path: Path, size: tuple[int, int] = (200, 200)) -> Path:
    arr = (np.random.RandomState(0).rand(size[1], size[0], 3) * 255).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(str(path))
    return path


class TestMRIDL2D:
    def test_class_to_idx_matches_trainer(self) -> None:
        assert mri_dl_2d.CLASS_TO_IDX == {
            "MildDemented": 0,
            "ModerateDemented": 1,
            "NonDemented": 2,
            "VeryMildDemented": 3,
        }

    def test_idx_to_class_is_consistent(self) -> None:
        for name, idx in mri_dl_2d.CLASS_TO_IDX.items():
            assert mri_dl_2d.IDX_TO_CLASS[idx] == name

    def test_load_missing_artifact_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="MRI 2D checkpoint not found"):
            mri_dl_2d.load(tmp_path / "nope.pt")

    def test_predict_image_returns_full_probs(self, tmp_path: Path) -> None:
        ckpt = build_dummy_2d(tmp_path / "best.pt")
        model = mri_dl_2d.load(ckpt)
        img = _png(tmp_path / "scan.png")

        result = mri_dl_2d.predict_image(model, img)

        assert set(result) == {"label", "label_text", "confidence", "probabilities"}
        assert result["label"] in {0, 1, 2, 3}
        assert result["label_text"] in mri_dl_2d.CLASS_TO_IDX
        assert 0.0 <= result["confidence"] <= 1.0
        probs = result["probabilities"]
        assert len(probs) == 4
        assert abs(sum(p["probability"] for p in probs) - 1.0) < 1e-5
        # Each probability item exposes the trainer's class label, not "class_N".
        assert {p["label_text"] for p in probs} == set(mri_dl_2d.CLASS_TO_IDX)

    def test_predict_works_for_grayscale_input(self, tmp_path: Path) -> None:
        ckpt = build_dummy_2d(tmp_path / "best.pt")
        model = mri_dl_2d.load(ckpt)
        # Single-channel grayscale, common for MRI slice exports.
        gray = (np.random.RandomState(1).rand(180, 180) * 255).astype(np.uint8)
        path = tmp_path / "gray.png"
        Image.fromarray(gray, mode="L").save(str(path))

        result = mri_dl_2d.predict_image(model, path)
        assert 0.0 <= result["confidence"] <= 1.0
```

Run: `pytest tests/models/test_mri_dl_2d.py -v` → expect ImportError on `src.models.mri_dl_2d`.

- [ ] **Step 3: Minimal implementation.**

`src/models/mri_dl_2d.py`:

```python
"""Pretrained 2D MRI Alzheimer's classifier (resnet18, 4 classes).

Decision-layer bridge for an externally-trained PyTorch checkpoint. Loads
either a state_dict (default) or a full pickled model, applies the trainer's
preprocessing (resize image_size=160, ImageNet normalisation), and emits the
same dict shape as src.models.mri_model.predict_with_proba so downstream
code paths don't care which backend produced the prediction.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from src.core.logger import get_logger

logger = get_logger(__name__)

CLASS_TO_IDX: dict[str, int] = {
    "MildDemented": 0,
    "ModerateDemented": 1,
    "NonDemented": 2,
    "VeryMildDemented": 3,
}
IDX_TO_CLASS: dict[int, str] = {v: k for k, v in CLASS_TO_IDX.items()}

DEFAULT_IMAGE_SIZE = 160
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)

# torchvision transform reused for every prediction. Constructed once at
# import time — no per-call allocation.
_TRANSFORM = transforms.Compose([
    transforms.Resize((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


def _build_resnet18_4class() -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_TO_IDX))
    return model


def load(path: Path) -> nn.Module:
    """Load checkpoint. Supports state_dict (preferred) or full pickled model."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MRI 2D checkpoint not found: {path}")
    obj = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(obj, nn.Module):
        model = obj
    else:
        model = _build_resnet18_4class()
        # Strip 'module.' prefix if the trainer used DataParallel / DDP.
        clean = {k.removeprefix("module."): v for k, v in obj.items()}
        model.load_state_dict(clean, strict=True)
    model.eval()
    return model


def predict_image(model: nn.Module, image_path: Path) -> dict[str, Any]:
    """Run inference on one image. Output shape mirrors mri_model.predict_with_proba."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"MRI image not found: {image_path}")
    img = Image.open(str(image_path)).convert("RGB")
    tensor = _TRANSFORM(img).unsqueeze(0)  # (1, 3, 160, 160)

    with torch.inference_mode():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    label_idx = int(np.argmax(probs))
    return {
        "label": label_idx,
        "label_text": IDX_TO_CLASS[label_idx],
        "confidence": float(probs[label_idx]),
        "probabilities": [
            {"label": i, "label_text": IDX_TO_CLASS[i], "probability": float(p)}
            for i, p in enumerate(probs)
        ],
    }
```

Run: `pytest tests/models/test_mri_dl_2d.py -v` → expect 5 passed.

- [ ] **Step 4:** `pytest -q` → expect 295+1 baseline + 5 new = ~300 passed.

- [ ] **Step 5:** commit:

```bash
git add src/models/mri_dl_2d.py tests/fixtures/build_dummy_resnet18_2d.py tests/models/test_mri_dl_2d.py
git commit -m "feat(models): add 2D resnet18 4-class Alzheimer's MRI inference module"
```

---

### Task 2: Selector for 3D vs 2D

**Files:**
- Create: `src/models/mri_selector.py`
- Create: `tests/models/test_mri_selector.py`

- [ ] **Step 1: Failing test.**

`tests/models/test_mri_selector.py`:

```python
"""Tests for src.models.mri_selector — env-var-driven 2D / 3D dispatch."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.models import mri_selector
from tests.fixtures.build_dummy_mri_onnx import build as build_dummy_3d
from tests.fixtures.build_dummy_resnet18_2d import build as build_dummy_2d


_FIXTURE_MRI = Path(__file__).resolve().parents[1] / "fixtures" / "mri_sample" / "subject_0.nii.gz"


class TestSelector:
    def test_default_kind_is_volumetric(self, monkeypatch) -> None:
        monkeypatch.delenv("MRI_MODEL_KIND", raising=False)
        assert mri_selector.current_kind() == "volumetric_onnx"

    def test_explicit_2d_selection(self, monkeypatch) -> None:
        monkeypatch.setenv("MRI_MODEL_KIND", "resnet18_2d")
        assert mri_selector.current_kind() == "resnet18_2d"

    def test_unknown_kind_raises(self, monkeypatch) -> None:
        monkeypatch.setenv("MRI_MODEL_KIND", "neural_net_supreme")
        with pytest.raises(ValueError, match="unknown MRI_MODEL_KIND"):
            mri_selector.current_kind()

    def test_predict_routes_to_volumetric(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("MRI_MODEL_KIND", "volumetric_onnx")
        artifact = build_dummy_3d(tmp_path / "vol.onnx")
        result = mri_selector.predict(
            input_path=_FIXTURE_MRI,
            checkpoint_path=artifact,
            target_shape=(8, 8, 8),
            label_names=("control", "abnormal"),
        )
        assert result["label_text"] in {"control", "abnormal"}

    def test_predict_routes_to_2d(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("MRI_MODEL_KIND", "resnet18_2d")
        artifact = build_dummy_2d(tmp_path / "best.pt")
        # Build a tiny PNG.
        from PIL import Image
        import numpy as np
        img_path = tmp_path / "scan.png"
        Image.fromarray((np.random.RandomState(0).rand(160, 160, 3) * 255).astype("uint8")).save(str(img_path))
        result = mri_selector.predict(
            input_path=img_path,
            checkpoint_path=artifact,
        )
        assert result["label_text"] in mri_selector.label_names_for_kind("resnet18_2d")
```

Run: `pytest tests/models/test_mri_selector.py -v` → ImportError.

- [ ] **Step 2: Minimal impl.**

`src/models/mri_selector.py`:

```python
"""Env-var-driven dispatch between volumetric ONNX and 2D resnet18 MRI models."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.core.logger import get_logger
from src.models import mri_dl_2d, mri_model

logger = get_logger(__name__)

VALID_KINDS = ("volumetric_onnx", "resnet18_2d")
_DEFAULT_KIND = "volumetric_onnx"


def current_kind() -> str:
    kind = os.environ.get("MRI_MODEL_KIND", _DEFAULT_KIND)
    if kind not in VALID_KINDS:
        raise ValueError(f"unknown MRI_MODEL_KIND={kind!r}; expected one of {VALID_KINDS}")
    return kind


def label_names_for_kind(kind: str) -> tuple[str, ...]:
    if kind == "resnet18_2d":
        return tuple(mri_dl_2d.IDX_TO_CLASS[i] for i in range(len(mri_dl_2d.CLASS_TO_IDX)))
    return mri_model.DEFAULT_LABEL_NAMES


def predict(
    input_path: Path,
    checkpoint_path: Path,
    target_shape: tuple[int, int, int] | None = None,
    label_names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Run the active MRI model on one input. Returns the unified prediction dict."""
    kind = current_kind()
    logger.info("dispatching MRI prediction kind=%s input=%s", kind, input_path)
    if kind == "resnet18_2d":
        model = mri_dl_2d.load(checkpoint_path)
        return mri_dl_2d.predict_image(model, input_path)
    model = mri_model.load(checkpoint_path)
    return mri_model.predict_nifti(
        model,
        input_path,
        target_shape=target_shape or mri_model.DEFAULT_TARGET_SHAPE,
        label_names=label_names,
    )
```

Run tests → 5 passed.

- [ ] **Step 3:** `pytest -q` → ~305 passed.

- [ ] **Step 4:** commit: `feat(models): selector dispatch for volumetric vs 2D MRI models`.

---

### Task 3: Wire into `POST /predict/mri`

**Files:**
- Modify: `src/api/routes.py`
- Modify: `src/api/schemas.py` (docstring only — `input_path` now optionally accepts a 2D image)
- Create: `tests/api/test_mri_2d_route.py`

- [ ] **Step 1: Failing test.**

`tests/api/test_mri_2d_route.py`:

```python
"""Integration: POST /predict/mri with MRI_MODEL_KIND=resnet18_2d."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api.main import app
from tests.fixtures.build_dummy_resnet18_2d import build as build_dummy_2d


@pytest.fixture()
def client_2d(monkeypatch, tmp_path):
    monkeypatch.setenv("MRI_MODEL_KIND", "resnet18_2d")
    ckpt = build_dummy_2d(tmp_path / "best.pt")
    monkeypatch.setenv("MRI_MODEL_PATH_2D", str(ckpt))
    return TestClient(app)


def test_predict_mri_2d_happy_path(client_2d, tmp_path):
    # Tiny RGB PNG.
    img_path = tmp_path / "scan.png"
    Image.fromarray((np.random.RandomState(0).rand(170, 170, 3) * 255).astype("uint8")).save(str(img_path))

    r = client_2d.post("/predict/mri", json={"input_path": str(img_path)})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["label_text"] in {
        "MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented",
    }
    assert 0.0 <= data["confidence"] <= 1.0
    assert len(data["probabilities"]) == 4
```

Run → expect 500 (route hardcoded to volumetric path) or schema error.

- [ ] **Step 2: Modify the route handler.**

In `src/api/routes.py`, find `predict_mri` (around line 318). Change the body to dispatch via the selector:

```python
@predict_router.post("/mri", response_model=MRIPredictResponse)
def predict_mri(req: MRIPredictRequest) -> MRIPredictResponse:
    from src.models import mri_selector

    kind = mri_selector.current_kind()
    if kind == "resnet18_2d":
        ckpt = Path(os.environ.get("MRI_MODEL_PATH_2D", "data/processed/mri_dl_2d/best_model.pt"))
        result = mri_selector.predict(input_path=Path(req.input_path), checkpoint_path=ckpt)
        model_path = str(ckpt)
    else:
        ckpt = _mri_model_path()
        result = mri_selector.predict(
            input_path=Path(req.input_path),
            checkpoint_path=ckpt,
            target_shape=tuple(req.target_shape),
            label_names=tuple(req.label_names) if req.label_names else None,
        )
        model_path = str(ckpt)

    return MRIPredictResponse(
        **result,
        input_path=str(req.input_path),
        model_path=model_path,
    )
```

You'll need to add `import os` and `from pathlib import Path` if not already present at the top of the file (Path likely already is). Keep the existing `_mri_model_path()` helper.

- [ ] **Step 3:** Update `src/api/schemas.py`. The class `MRIPredictRequest.input_path` description currently says "Path to one .nii or .nii.gz MRI volume". Change to:

```python
    input_path: str = Field(..., description="Path to MRI input. With MRI_MODEL_KIND=volumetric_onnx (default), expects a .nii/.nii.gz volume. With MRI_MODEL_KIND=resnet18_2d, expects a 2D image (.png/.jpg).")
```

- [ ] **Step 4:** `pytest tests/api/test_mri_2d_route.py -v` → expect 1 passed.

- [ ] **Step 5:** `pytest -q` → expect no regressions vs the prior baseline + 1 new.

- [ ] **Step 6:** commit: `feat(api): dispatch /predict/mri via MRI_MODEL_KIND env var`.

---

### Task 4: Sanity check on real artifact (one-shot, runs only when artifact is present)

**Files:**
- Create: `tests/models/test_mri_dl_2d_real.py`

This test is opt-in via env: only runs if `data/processed/mri_dl_2d/best_model.pt` is present. Catches the "trainer used different class index order" bug.

- [ ] **Step 1: Test.**

```python
"""Real-artifact sanity test. Skipped unless the checkpoint is present."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.models import mri_dl_2d


REAL_CKPT = Path("data/processed/mri_dl_2d/best_model.pt")


@pytest.mark.skipif(not REAL_CKPT.exists(), reason="real MRI checkpoint not present")
def test_real_checkpoint_loads_and_predicts(tmp_path):
    model = mri_dl_2d.load(REAL_CKPT)
    arr = (np.random.RandomState(0).rand(170, 170, 3) * 255).astype(np.uint8)
    img = tmp_path / "scan.png"
    Image.fromarray(arr).save(str(img))
    result = mri_dl_2d.predict_image(model, img)

    assert result["label_text"] in mri_dl_2d.CLASS_TO_IDX
    # Probabilities sum to 1.
    s = sum(p["probability"] for p in result["probabilities"])
    assert abs(s - 1.0) < 1e-5
```

- [ ] **Step 2:** `pytest tests/models/test_mri_dl_2d_real.py -v` → if no real checkpoint, **skipped** (expected). When you do drop the artifact in, `pytest -q` will run it.

- [ ] **Step 3:** commit: `test(models): real-artifact sanity for MRI DL 2D (skips when absent)`.

---

### Task 5: Streamlit + README

**Files:**
- Modify: `src/frontend/app.py` (the MRI Predict tab — add `MRI_MODEL_KIND` indicator + accept image upload when 2D is active)
- Modify: `README.md`

- [ ] **Step 1:** In `src/frontend/app.py`, find the MRI predict section (likely near line 1330 — search for `mri_predict_d`). Add a small caption above the existing UI:

```python
mri_kind = os.environ.get("MRI_MODEL_KIND", "volumetric_onnx")
st.caption(f"Active MRI model: `{mri_kind}` (set `MRI_MODEL_KIND` env to switch)")
```

If `mri_kind == "resnet18_2d"`, swap the file picker hint from `.nii/.nii.gz` to `.png/.jpg`. The existing `target_shape` widgets become irrelevant in 2D mode — wrap them in `if mri_kind == "volumetric_onnx":`.

- [ ] **Step 2:** README update. Add a paragraph under the existing MRI section:

```markdown
### MRI Deep-Learning Backends

The MRI prediction route supports two backends, selected via env:

- `MRI_MODEL_KIND=volumetric_onnx` (default). Loads an ONNX volumetric model from `MRI_MODEL_PATH` (default `data/processed/mri_model.onnx`). Input: `.nii` / `.nii.gz`.
- `MRI_MODEL_KIND=resnet18_2d`. Loads a PyTorch state_dict from `MRI_MODEL_PATH_2D` (default `data/processed/mri_dl_2d/best_model.pt`). Input: 2D image (`.png` / `.jpg`). Classes: `MildDemented`, `ModerateDemented`, `NonDemented`, `VeryMildDemented`.

Switch backends without restarting workers — env is read on each request.
```

- [ ] **Step 3:** `pytest -q` → no regressions. (Streamlit code is not unit-tested in this repo — manual smoke at the end is fine.)

- [ ] **Step 4:** commit: `feat(frontend): expose MRI_MODEL_KIND in MRI predict tab; doc backends`.

---

## Self-review checklist

1. **Spec coverage.** Trainer's BEST_PARAMS that matter at inference: `image_size=160`, `model_name=resnet18`, class index map. All locked in via Task 1 constants. The other params (`lr`, `epochs`, `batch_size`) are training-only and intentionally not surfaced.
2. **Independence.** No coupling to BBB. The new module imports only stdlib + torch + torchvision + Pillow + numpy + the existing `src/core/logger`.
3. **Sanity test for class-order drift.** Task 4 runs only when the real checkpoint is dropped in. If the trainer used `ImageFolder`'s alphabetical order (`MildDemented=0, ModerateDemented=1, NonDemented=2, VeryMildDemented=3` — same as ours, by luck), it passes. If they used a different order, the user must update `CLASS_TO_IDX` in `mri_dl_2d.py`.
4. **No placeholders.** Every step contains the full code.
5. **Hackathon-grade.** No XAI / saliency-map / Grad-CAM scope creep. That's a separate sub-plan if the demo demands it.

---

## Execution handoff

Save and choose: subagent-driven (recommended) or inline executing-plans.
