"""Load (or rebuild) the TF-IDF clinical RAG index.

The user's rag.py builds the index from `__main__.Chunk` (a frozen
dataclass). Picking up that pickle from a different module path requires
a custom Unpickler that re-routes the class — see _ChunkRoutingUnpickler.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from src.core.logger import get_logger
from src.rag.clinical.types import ClinicalChunk

logger = get_logger(__name__)


class _ChunkRoutingUnpickler(pickle.Unpickler):
    """Pickle's `find_class` hook lets us swap `__main__.Chunk` (and
    `rag.Chunk` if the user later runs the builder as a module) for our
    `ClinicalChunk` — both are frozen dataclasses with the same fields,
    so the swap is structurally safe.
    """

    def find_class(self, module: str, name: str):
        if name == "Chunk" and module in {"__main__", "rag", "rag.rag"}:
            return ClinicalChunk
        return super().find_class(module, name)


def load_index(path: Path) -> dict[str, Any]:
    """Unpickle a TF-IDF index produced by the user's rag.py."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"clinical RAG index not found: {path}")
    with path.open("rb") as f:
        payload = _ChunkRoutingUnpickler(f).load()
    expected = {"chunks", "vectorizer", "matrix"}
    if not expected <= set(payload):
        raise ValueError(
            f"clinical RAG index missing expected keys: have {sorted(payload)}, need {sorted(expected)}"
        )
    logger.info("loaded clinical RAG index: %d chunks from %s", len(payload["chunks"]), path)
    return payload
