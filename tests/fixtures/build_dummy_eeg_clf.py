"""Build a stub EEG classifier (sklearn RF) for tests.

Demo-time placeholder — produces a 2-class probability output matching the
eeg_model.predict_features contract. Replace with the real artifact when
the user provides it; tests don't change.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def build(path: Path, n_features: int = 16, seed: int = 0) -> Path:
    """Save a fitted RandomForestClassifier at `path` and return the path."""
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    n = 200
    n_alz = n // 2
    X_ctrl = rng.normal(0.0, 1.0, size=(n - n_alz, n_features))
    X_alz = rng.normal(2.0, 1.0, size=(n_alz, n_features))
    X = np.vstack([X_ctrl, X_alz])
    y = np.array([0] * (n - n_alz) + [1] * n_alz)

    clf = RandomForestClassifier(n_estimators=12, max_depth=6, random_state=seed)
    clf.fit(X, y)
    joblib.dump(clf, str(path))
    return path
