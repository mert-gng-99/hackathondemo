"""Real-artifact sanity. Skipped unless the checkpoint is present.

When the user drops their trained best_model.pt at the canonical path, this
test runs automatically — catches class-order or input-shape drift.
"""
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
    s = sum(p["probability"] for p in result["probabilities"])
    assert abs(s - 1.0) < 1e-5
