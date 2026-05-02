"""Fastembed wrapper — ONNX-based, CPU-only, no torch dep.

Public entry: `Embedder().encode(texts) -> np.ndarray[N, D]`. Model is
loaded lazily on first call. Output is float32 to match FAISS's expected
input dtype.
"""
from __future__ import annotations

import numpy as np

from src.core.logger import get_logger

logger = get_logger(__name__)


# bge-small-en-v1.5: 384-dim, ~33MB ONNX, MTEB top-tier for size class.
_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384


class Embedder:
    """Lazy-loaded fastembed wrapper. One instance per process is enough."""

    def __init__(self, model_name: str = _MODEL_NAME) -> None:
        self._model_name = model_name
        self._model = None  # lazy-loaded on first encode()

    def _ensure_model(self) -> None:
        if self._model is None:
            from fastembed import TextEmbedding
            logger.info("Loading fastembed model %s (one-time)", self._model_name)
            self._model = TextEmbedding(model_name=self._model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        self._ensure_model()
        embeddings = list(self._model.embed(texts))
        return np.array(embeddings, dtype=np.float32)
