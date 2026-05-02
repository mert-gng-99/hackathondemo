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
