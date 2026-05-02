"""FAISS vector store with parallel chunk metadata.

Public entry: `FAISSStore(dim)`. Vectors are L2-normalized on add and
search so inner-product == cosine similarity. Chunks are arbitrary dicts;
`text` and `source` keys are recommended but not enforced.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np


class FAISSStore:
    """Inner-product (cosine after L2-norm) FAISS store with chunk metadata."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._index: faiss.Index = faiss.IndexFlatIP(dim)
        self._chunks: list[dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, vectors: np.ndarray, chunks: list[dict[str, Any]]) -> None:
        if vectors.shape[0] != len(chunks):
            raise ValueError(
                f"size mismatch: {vectors.shape[0]} vectors vs {len(chunks)} chunks"
            )
        if vectors.shape[0] == 0:
            return
        v = np.array(vectors, dtype=np.float32, copy=True)
        faiss.normalize_L2(v)
        self._index.add(v)
        self._chunks.extend(chunks)

    def search(self, query: np.ndarray, k: int = 5) -> list[tuple[dict[str, Any], float]]:
        if len(self._chunks) == 0:
            return []
        q = np.array(query, dtype=np.float32, copy=True)
        if q.ndim == 1:
            q = q[np.newaxis, :]
        faiss.normalize_L2(q)
        k = min(k, len(self._chunks))
        scores, idx = self._index.search(q, k)
        out: list[tuple[dict[str, Any], float]] = []
        for i, s in zip(idx[0], scores[0]):
            if i == -1:
                continue
            out.append((self._chunks[int(i)], float(s)))
        return out

    def save(self, dir_path: Path) -> None:
        dir_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(dir_path / "index.bin"))
        (dir_path / "chunks.json").write_text(json.dumps(self._chunks, indent=2))

    @classmethod
    def load(cls, dir_path: Path, dim: int) -> "FAISSStore":
        store = cls(dim=dim)
        store._index = faiss.read_index(str(dir_path / "index.bin"))
        store._chunks = json.loads((dir_path / "chunks.json").read_text())
        return store
