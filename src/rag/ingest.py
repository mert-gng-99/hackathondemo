"""Walk a knowledge-base directory, chunk each file, embed, persist FAISS index.

CLI entry point: `python -m src.rag.ingest [<input_dir> [<output_dir>]]`.
Defaults: input=`data/knowledge_base/`, output=`data/processed/faiss_index/`.

Supported file types: `.md`, `.txt`, `.pdf`. Other extensions are ignored
with a logged WARNING.
"""
from __future__ import annotations

import sys
from pathlib import Path

from src.core.logger import get_logger
from src.rag.chunker import chunk_text
from src.rag.embed import EMBEDDING_DIM, Embedder
from src.rag.store import FAISSStore

logger = get_logger(__name__)


_DEFAULT_INPUT = Path("data/knowledge_base")
_DEFAULT_OUTPUT = Path("data/processed/faiss_index")
_SUPPORTED = {".md", ".txt", ".pdf"}


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def _read_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf(path)
    return path.read_text(encoding="utf-8", errors="replace")


def ingest_directory(input_dir: Path, output_dir: Path) -> int:
    """Ingest every supported file in `input_dir` into a FAISS index at `output_dir`.

    Returns the total number of chunks indexed.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    files = sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in _SUPPORTED)
    logger.info("Ingesting %d file(s) from %s", len(files), input_dir)

    all_chunks: list[dict] = []
    for path in files:
        try:
            text = _read_file(path)
        except Exception as e:
            logger.warning("Skipping %s (read failed: %s)", path, e)
            continue
        for i, ch in enumerate(chunk_text(text)):
            all_chunks.append({
                "text": ch,
                "source": str(path.relative_to(input_dir)),
                "chunk_index": i,
            })

    store = FAISSStore(dim=EMBEDDING_DIM)
    if all_chunks:
        embedder = Embedder()
        vectors = embedder.encode([c["text"] for c in all_chunks])
        store.add(vectors, all_chunks)

    store.save(output_dir)
    logger.info("Indexed %d chunk(s) → %s", len(all_chunks), output_dir)
    return len(all_chunks)


def main() -> None:
    args = sys.argv[1:]
    inp = Path(args[0]) if len(args) >= 1 else _DEFAULT_INPUT
    out = Path(args[1]) if len(args) >= 2 else _DEFAULT_OUTPUT
    n = ingest_directory(inp, out)
    print(f"Indexed {n} chunks into {out}")


if __name__ == "__main__":
    main()
