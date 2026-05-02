"""Real-artifact EEG sanity. Skipped unless data/processed/eeg_clf.joblib exists."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.models import eeg_model


REAL_CKPT = Path("data/processed/eeg_clf.joblib")


@pytest.mark.skipif(not REAL_CKPT.exists(), reason="real EEG checkpoint not present")
def test_real_eeg_checkpoint_loads_and_predicts():
    model = eeg_model.load(REAL_CKPT)
    n_features = int(getattr(model, "n_features_in_", 16))
    features = np.zeros((n_features,), dtype=np.float32)
    out = eeg_model.predict_features(model, features)
    s = sum(p["probability"] for p in out["probabilities"])
    assert abs(s - 1.0) < 1e-5
    assert out["label_text"]
