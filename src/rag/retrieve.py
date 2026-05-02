"""Query → top-k chunks. Encapsulates the embedder + store pair so callers
don't have to assemble both. Loads from disk lazily.
"""
from __future__ import annotations

from pathlib import Path

from src.core.logger import get_logger
from src.rag.embed import EMBEDDING_DIM, Embedder
from src.rag.store import FAISSStore

logger = get_logger(__name__)


class RAGRetriever:
    """Bundle (embedder, store). Use `RAGRetriever.load(dir)` to construct."""

    def __init__(self, store: FAISSStore, embedder: Embedder) -> None:
        self._store = store
        self._embedder = embedder

    @classmethod
    def load(cls, index_dir: Path) -> "RAGRetriever":
        store = FAISSStore.load(Path(index_dir), dim=EMBEDDING_DIM)
        return cls(store=store, embedder=Embedder())

    def __len__(self) -> int:
        return len(self._store)

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Return up to `k` chunks most relevant to `query`, sorted by score desc.

        Each chunk dict carries `text`, `source`, `chunk_index`, `score`.
        Returns [] for empty query or empty store.
        """
        if not query.strip() or len(self._store) == 0:
            return []
        vec = self._embedder.encode([query])
        hits = self._store.search(vec[0], k=k)
        return [{**chunk, "score": score} for chunk, score in hits]
