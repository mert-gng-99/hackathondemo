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
_IMAGENET_STD = (0.229, 0.224, 0.225)

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
        clean = {k.removeprefix("module."): v for k, v in obj.items()}
        model.load_state_dict(clean, strict=True)
    model.eval()
    return model


def predict_image(model: nn.Module, image_path: Path) -> dict[str, Any]:
    """Run inference on one image. Output mirrors mri_model.predict_with_proba."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"MRI image not found: {image_path}")
    img = Image.open(str(image_path)).convert("RGB")
    tensor = _TRANSFORM(img).unsqueeze(0)

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
