# Orchestrator Agent + RAG Feedback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wrap the three modality pipelines as function-calling tools, add an orchestrator agent that picks the right pipeline for each input, and feed the pipeline output through a RAG retrieval tool so the final response is grounded in user-curated reference documents.

**Architecture:** Single orchestrator agent (OpenAI-SDK function-calling loop, no framework) holds 4 tools — `run_bbb_pipeline`, `run_eeg_pipeline`, `run_mri_pipeline`, `retrieve_context`. Pipelines stay deterministic (already 184 tests green); only the wrapper layer is new. RAG uses `fastembed` for embeddings (lightweight ONNX, no torch) + `faiss-cpu` for vector search. Knowledge base is markdown / PDF files in `data/knowledge_base/` ingested at Docker build time. Streamlit gets a new "🤖 Agent" tab that surfaces the agent's tool-call trace as evidence.

**Tech Stack:** `openai==1.51.0` (existing — function calling), `fastembed==0.4.2` (embeddings, ~50MB), `faiss-cpu==1.8.0` (vector store), `pypdf==5.0.1` (PDF loader). Reuses the project's existing `get_logger`, Pydantic patterns, and `src/llm/explainer.py` model fallback discipline.

---

## File Structure

**New packages:**

```
src/agents/
├── __init__.py
├── schemas.py            # Pydantic I/O for each tool + AgentResult
├── tools.py              # Tool dataclass + registry + 4 tool implementations
├── orchestrator.py       # Orchestrator class (LLM loop + dispatch + trace)
└── prompts.py            # ORCHESTRATOR_SYSTEM_PROMPT + helpers

src/rag/
├── __init__.py
├── chunker.py            # Recursive character splitter
├── embed.py              # Embedder (fastembed wrapper)
├── store.py              # FAISSStore (load/save/add/search)
├── retrieve.py           # RAGRetriever (embed query → top-k chunks)
└── ingest.py             # CLI: walk data/knowledge_base/ → embed → persist

data/knowledge_base/      # NEW (gitignored, user drops .pdf / .md here)
├── README.md             # explains what to drop, format expectations
└── .gitkeep

data/processed/faiss_index/   # NEW (built at runtime / Dockerfile RUN)
├── index.bin
└── chunks.json

tests/agents/
├── test_schemas.py
├── test_tools.py
├── test_orchestrator.py
└── test_orchestrator_live.py    # network-gated, slow-marked

tests/rag/
├── test_chunker.py
├── test_embed.py
├── test_store.py
├── test_retrieve.py
└── test_ingest.py

tests/fixtures/kb_sample/        # NEW
├── lipinski_rule_of_five.md
├── combat_harmonization_primer.md
└── mne_ica_basics.md
```

**Modified:**

```
requirements.txt                 # +fastembed, +faiss-cpu, +pypdf
.gitignore                       # +data/knowledge_base/*.pdf, +data/processed/faiss_index/
src/api/routes.py                # +agent_router, POST /agent/run
src/api/schemas.py               # +AgentRunRequest, +AgentRunResponse, +ToolTraceItem
src/api/main.py                  # mount agent_router
src/frontend/app.py              # +"🤖 Agent" tab
Dockerfile                       # RUN python -m src.rag.ingest at build
Dockerfile.hf                    # same
AGENTS.md                        # +§15 Agent surface + §16 RAG surface
```

---

## Task 1: Add RAG dependencies

**Files:**
- Modify: `requirements.txt`
- Modify: `.gitignore`

- [ ] **Step 1: Add deps to requirements.txt**

Open `requirements.txt`, find the section after `# --- Tooling / tests ---` (around `httpx==0.27.2`) and insert before `# --- Frontend (B2B dashboard) ---`:

```
# --- RAG (knowledge retrieval for agent feedback loop) ---
fastembed==0.4.2          # ONNX-based embeddings, no torch dep
faiss-cpu==1.8.0          # vector store
pypdf==5.0.1              # PDF text extraction
```

- [ ] **Step 2: Update .gitignore**

Append to `.gitignore`:

```
# RAG knowledge base (user-supplied PDFs/MD; not source-controlled)
data/knowledge_base/*.pdf
data/knowledge_base/*.PDF

# RAG built artifacts
data/processed/faiss_index/
```

- [ ] **Step 3: Install deps + verify**

Run: `pip install fastembed==0.4.2 faiss-cpu==1.8.0 pypdf==5.0.1`

Expected: install succeeds. Then verify import:

```bash
python -c "from fastembed import TextEmbedding; import faiss; import pypdf; print('ok')"
```

Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add requirements.txt .gitignore
git commit -m "feat(rag): add fastembed/faiss-cpu/pypdf for retrieval layer"
```

---

## Task 2: RAG document chunker

**Files:**
- Create: `src/rag/__init__.py`
- Create: `src/rag/chunker.py`
- Create: `tests/rag/__init__.py`
- Create: `tests/rag/test_chunker.py`

- [ ] **Step 1: Create empty package markers**

```bash
mkdir -p src/rag tests/rag
touch src/rag/__init__.py tests/rag/__init__.py
```

- [ ] **Step 2: Write the failing test**

Create `tests/rag/test_chunker.py`:

```python
"""Tests for src.rag.chunker — paragraph-aware character splitter."""
from __future__ import annotations

import pytest

from src.rag.chunker import chunk_text


class TestChunkText:
    def test_short_text_returns_single_chunk(self) -> None:
        out = chunk_text("hello world", max_chars=100, overlap=10)
        assert out == ["hello world"]

    def test_empty_text_returns_empty_list(self) -> None:
        assert chunk_text("", max_chars=100, overlap=10) == []
        assert chunk_text("   \n\n  ", max_chars=100, overlap=10) == []

    def test_long_text_splits_into_multiple_chunks(self) -> None:
        text = "a" * 250
        out = chunk_text(text, max_chars=100, overlap=10)
        assert len(out) >= 3
        # every chunk respects max_chars
        for c in out:
            assert len(c) <= 100

    def test_overlap_between_chunks(self) -> None:
        text = "abcdefghij" * 30  # 300 chars, no natural break
        out = chunk_text(text, max_chars=100, overlap=20)
        # consecutive chunks share at least some characters
        for i in range(len(out) - 1):
            assert out[i][-10:] in out[i + 1] or out[i + 1][:10] in out[i]

    def test_paragraph_boundary_preferred(self) -> None:
        # First paragraph fits, second doesn't — split at \n\n
        para_a = "First paragraph content."
        para_b = "Second paragraph content " * 10
        text = f"{para_a}\n\n{para_b}"
        out = chunk_text(text, max_chars=100, overlap=10)
        # first chunk should end at the paragraph boundary, not mid-word
        assert para_a in out[0]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/rag/test_chunker.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'src.rag.chunker'`

- [ ] **Step 4: Implement the chunker**

Create `src/rag/chunker.py`:

```python
"""Paragraph-aware recursive character splitter for RAG ingestion.

Public entry: `chunk_text(text, max_chars, overlap)`. Splits on the first
of [paragraph break, sentence end, newline, space] that fits inside the
window. Empty / whitespace-only inputs return [].
"""
from __future__ import annotations


_SEPARATORS: tuple[str, ...] = ("\n\n", ". ", "\n", " ")


def chunk_text(text: str, max_chars: int = 600, overlap: int = 80) -> list[str]:
    """Split `text` into chunks of at most `max_chars`, with `overlap` carry-over."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        if end < n:
            # try to land on a clean boundary inside [start, end]
            for sep in _SEPARATORS:
                last = text.rfind(sep, start, end)
                if last > start:
                    end = last + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(start + 1, end - overlap)
    return chunks
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/rag/test_chunker.py -v`

Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
git add src/rag/__init__.py src/rag/chunker.py tests/rag/__init__.py tests/rag/test_chunker.py
git commit -m "feat(rag): paragraph-aware chunker (chunk_text)"
```

---

## Task 3: RAG embedder

**Files:**
- Create: `src/rag/embed.py`
- Create: `tests/rag/test_embed.py`

- [ ] **Step 1: Write the failing test**

Create `tests/rag/test_embed.py`:

```python
"""Tests for src.rag.embed — fastembed wrapper."""
from __future__ import annotations

import numpy as np
import pytest

from src.rag.embed import Embedder, EMBEDDING_DIM


class TestEmbedder:
    @pytest.fixture(scope="class")
    def embedder(self) -> Embedder:
        return Embedder()

    def test_dim_constant_matches_model(self, embedder: Embedder) -> None:
        out = embedder.encode(["hello"])
        assert out.shape == (1, EMBEDDING_DIM)

    def test_batch_encoding(self, embedder: Embedder) -> None:
        out = embedder.encode(["hello", "world", "blood-brain barrier"])
        assert out.shape == (3, EMBEDDING_DIM)
        assert out.dtype == np.float32

    def test_empty_list_returns_empty_array(self, embedder: Embedder) -> None:
        out = embedder.encode([])
        assert out.shape == (0, EMBEDDING_DIM)

    def test_similar_strings_have_higher_similarity_than_dissimilar(
        self, embedder: Embedder
    ) -> None:
        vecs = embedder.encode([
            "blood-brain barrier permeability",
            "BBB drug penetration",
            "MRI multi-site harmonization",
        ])
        # cosine similarity (vectors should be normalized for stable comparison)
        from numpy.linalg import norm
        def cos(a, b):
            return float(np.dot(a, b) / (norm(a) * norm(b)))
        sim_ab = cos(vecs[0], vecs[1])
        sim_ac = cos(vecs[0], vecs[2])
        assert sim_ab > sim_ac, f"Expected BBB-related strings closer; got {sim_ab=} vs {sim_ac=}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/rag/test_embed.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'src.rag.embed'`

- [ ] **Step 3: Implement the embedder**

Create `src/rag/embed.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/rag/test_embed.py -v`

Expected: 4 passed (first run downloads ~33MB model, ~30s; subsequent runs cached).

- [ ] **Step 5: Commit**

```bash
git add src/rag/embed.py tests/rag/test_embed.py
git commit -m "feat(rag): fastembed wrapper (Embedder, bge-small-en-v1.5, 384-dim)"
```

---

## Task 4: FAISS store

**Files:**
- Create: `src/rag/store.py`
- Create: `tests/rag/test_store.py`

- [ ] **Step 1: Write the failing test**

Create `tests/rag/test_store.py`:

```python
"""Tests for src.rag.store — FAISS vector store with metadata."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.rag.store import FAISSStore


def _rand_vecs(n: int, d: int = 4, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d), dtype=np.float32)


class TestFAISSStore:
    def test_add_then_search(self) -> None:
        store = FAISSStore(dim=4)
        vecs = _rand_vecs(3)
        chunks = [{"text": f"chunk-{i}", "source": "test.md"} for i in range(3)]
        store.add(vecs, chunks)
        results = store.search(vecs[0], k=2)
        assert len(results) == 2
        # the closest hit is the chunk we used as the query (cosine ~1.0)
        top_chunk, top_score = results[0]
        assert top_chunk["text"] == "chunk-0"
        assert top_score > 0.99

    def test_add_size_mismatch_raises(self) -> None:
        store = FAISSStore(dim=4)
        with pytest.raises(ValueError, match="size mismatch"):
            store.add(_rand_vecs(3), [{"text": "only-one"}])

    def test_search_k_larger_than_corpus(self) -> None:
        store = FAISSStore(dim=4)
        store.add(_rand_vecs(2), [{"text": f"c{i}"} for i in range(2)])
        results = store.search(_rand_vecs(1)[0], k=10)
        assert len(results) == 2

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        store = FAISSStore(dim=4)
        vecs = _rand_vecs(3)
        chunks = [{"text": f"chunk-{i}", "source": "test.md"} for i in range(3)]
        store.add(vecs, chunks)
        store.save(tmp_path / "idx")

        restored = FAISSStore.load(tmp_path / "idx", dim=4)
        results = restored.search(vecs[0], k=1)
        assert results[0][0]["text"] == "chunk-0"

    def test_search_on_empty_store_returns_empty(self) -> None:
        store = FAISSStore(dim=4)
        assert store.search(_rand_vecs(1)[0], k=5) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/rag/test_store.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'src.rag.store'`

- [ ] **Step 3: Implement the store**

Create `src/rag/store.py`:

```python
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
        v = np.asarray(vectors, dtype=np.float32)
        faiss.normalize_L2(v)
        self._index.add(v)
        self._chunks.extend(chunks)

    def search(self, query: np.ndarray, k: int = 5) -> list[tuple[dict[str, Any], float]]:
        if len(self._chunks) == 0:
            return []
        q = np.asarray(query, dtype=np.float32)
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/rag/test_store.py -v`

Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/rag/store.py tests/rag/test_store.py
git commit -m "feat(rag): FAISS inner-product store with chunk metadata + roundtrip"
```

---

## Task 5: RAG ingest CLI

**Files:**
- Create: `src/rag/ingest.py`
- Create: `tests/fixtures/kb_sample/lipinski_rule_of_five.md`
- Create: `tests/fixtures/kb_sample/combat_harmonization_primer.md`
- Create: `tests/fixtures/kb_sample/mne_ica_basics.md`
- Create: `tests/rag/test_ingest.py`

- [ ] **Step 1: Create the sample knowledge-base fixtures**

Create `tests/fixtures/kb_sample/lipinski_rule_of_five.md`:

```markdown
# Lipinski's Rule of Five — BBB Permeability Heuristic

Lipinski's Rule of Five (Lipinski 1997, 2001) is the foundational
medicinal-chemistry rule for predicting whether a small molecule will
cross the blood-brain barrier (BBB) by passive diffusion.

## The four criteria

A molecule is likely BBB-permeable if it satisfies all four:

1. Molecular weight (MW) <= 500 Daltons
2. Octanol-water partition coefficient (logP) <= 5
3. Hydrogen-bond donors <= 5
4. Hydrogen-bond acceptors <= 10

Molecules violating two or more criteria are typically poorly absorbed
or impermeant.

## Why ethanol crosses

Ethanol (CCO) has MW=46 Da, logP=-0.31, 1 H-bond donor, 1 H-bond
acceptor — well within all four thresholds. This explains its rapid
CNS penetration despite hydrophilicity.

## SHAP attribution interpretation

When a Random Forest BBB classifier flags Morgan fingerprint bits with
positive SHAP values toward a "permeable" label, the bit usually
corresponds to a small lipophilic substructure (CH3-, -OCH3-, aromatic
ring) consistent with Lipinski compliance.
```

Create `tests/fixtures/kb_sample/combat_harmonization_primer.md`:

```markdown
# ComBat Harmonization for Multi-Site Neuroimaging

ComBat (Johnson et al. 2007, adapted to MRI by Fortin et al. 2017, 2018)
is the de-facto standard for removing scanner / acquisition-site bias
from multi-center neuroimaging studies.

## How it works

ComBat models per-site location (mean) and scale (variance) parameters
using an empirical-Bayes hierarchical framework. It estimates these
parameters jointly across all sites and shrinks them toward a global
prior — small-N sites are pulled toward the global mean, preventing
overfitting.

## Site-gap reduction

A typical demonstration: the per-site mean of a hippocampus volume
feature can vary by 5+ standard deviations across hospitals. ComBat
typically collapses this gap to <0.005 — a 1000x+ reduction — while
preserving within-site biological variance (age, sex, diagnosis).

## When it fails

ComBat requires at least 2 sites with overlapping covariate
distributions. Single-site data, or sites with completely disjoint
populations (e.g., one site only-pediatric, another only-elderly),
produce unreliable harmonization.
```

Create `tests/fixtures/kb_sample/mne_ica_basics.md`:

```markdown
# MNE-Python ICA for EEG Artifact Removal

Independent Component Analysis (ICA, Hyvärinen 1999) decomposes a
multi-channel EEG recording into statistically independent source
components. It is the de-facto method for removing eye-blink and
heartbeat artifacts before downstream analysis.

## Why ICA, not PCA

PCA decomposes signals into orthogonal components — but neural sources
are not orthogonal in scalp space, they are statistically independent.
ICA's independence assumption matches the physics: the eye, the heart,
and cortical sources fire on uncorrelated schedules.

## The standard workflow

1. Bandpass the raw recording at 0.5-40 Hz to remove DC drift and line
   noise (50/60 Hz).
2. Fit ICA with N components (typically 15-30, less than channel count).
3. Identify artifact components by correlating each ICA source with the
   EOG (eye) channel; reject components with |correlation| > 0.5.
4. Reconstruct the cleaned signal by zeroing out the rejected
   components and inverse-transforming.

## Quality check

Post-ICA, the EOG channel should show minimal residual correlation
with frontal channels (Fp1/Fp2). If it doesn't, the ICA fit was likely
unstable — re-run with a different random seed or more components.
```

- [ ] **Step 2: Write the failing test**

Create `tests/rag/test_ingest.py`:

```python
"""Tests for src.rag.ingest — walk a directory, chunk, embed, persist."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src.rag.ingest import ingest_directory
from src.rag.store import FAISSStore


_FIXTURE_KB = Path(__file__).parent.parent / "fixtures" / "kb_sample"


class TestIngestDirectory:
    def test_ingests_markdown_files(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "idx"
        n = ingest_directory(_FIXTURE_KB, out_dir)
        assert n > 0  # at least one chunk per fixture file
        assert (out_dir / "index.bin").exists()
        assert (out_dir / "chunks.json").exists()

    def test_loaded_store_is_searchable(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "idx"
        ingest_directory(_FIXTURE_KB, out_dir)
        from src.rag.embed import EMBEDDING_DIM
        store = FAISSStore.load(out_dir, dim=EMBEDDING_DIM)
        assert len(store) > 0
        # chunks have source metadata
        assert all("source" in c for c in store._chunks)
        assert all("text" in c for c in store._chunks)

    def test_empty_directory_creates_empty_index(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty_kb"
        empty.mkdir()
        out_dir = tmp_path / "idx"
        n = ingest_directory(empty, out_dir)
        assert n == 0
        assert (out_dir / "index.bin").exists()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/rag/test_ingest.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'src.rag.ingest'`

- [ ] **Step 4: Implement the ingest CLI**

Create `src/rag/ingest.py`:

```python
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/rag/test_ingest.py -v`

Expected: 3 passed (first run may download embedding model if not cached from Task 3)

- [ ] **Step 6: Commit**

```bash
git add src/rag/ingest.py tests/rag/test_ingest.py tests/fixtures/kb_sample/
git commit -m "feat(rag): ingest CLI (markdown/PDF → chunks → FAISS) + sample KB fixtures"
```

---

## Task 6: RAG retriever

**Files:**
- Create: `src/rag/retrieve.py`
- Create: `tests/rag/test_retrieve.py`

- [ ] **Step 1: Write the failing test**

Create `tests/rag/test_retrieve.py`:

```python
"""Tests for src.rag.retrieve — query → top-k chunks."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.rag.ingest import ingest_directory
from src.rag.retrieve import RAGRetriever


_FIXTURE_KB = Path(__file__).parent.parent / "fixtures" / "kb_sample"


class TestRAGRetriever:
    @pytest.fixture(scope="class")
    def retriever(self, tmp_path_factory: pytest.TempPathFactory) -> RAGRetriever:
        idx_dir = tmp_path_factory.mktemp("rag_idx")
        ingest_directory(_FIXTURE_KB, idx_dir)
        return RAGRetriever.load(idx_dir)

    def test_bbb_query_returns_lipinski_chunk(self, retriever: RAGRetriever) -> None:
        hits = retriever.search("Why does ethanol cross the blood-brain barrier?", k=3)
        assert len(hits) == 3
        sources = [h["source"] for h in hits]
        assert "lipinski_rule_of_five.md" in sources
        # top hit should be from lipinski
        assert hits[0]["source"] == "lipinski_rule_of_five.md"

    def test_combat_query_returns_combat_chunk(self, retriever: RAGRetriever) -> None:
        hits = retriever.search("How does ComBat remove scanner bias from MRI data?", k=2)
        assert hits[0]["source"] == "combat_harmonization_primer.md"

    def test_eeg_query_returns_ica_chunk(self, retriever: RAGRetriever) -> None:
        hits = retriever.search("How do you remove eye blink artifacts from EEG?", k=2)
        assert hits[0]["source"] == "mne_ica_basics.md"

    def test_search_includes_score_and_text(self, retriever: RAGRetriever) -> None:
        hits = retriever.search("BBB permeability", k=1)
        h = hits[0]
        assert "text" in h
        assert "source" in h
        assert "score" in h
        assert isinstance(h["score"], float)
        assert 0.0 <= h["score"] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/rag/test_retrieve.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'src.rag.retrieve'`

- [ ] **Step 3: Implement the retriever**

Create `src/rag/retrieve.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/rag/test_retrieve.py -v`

Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/rag/retrieve.py tests/rag/test_retrieve.py
git commit -m "feat(rag): RAGRetriever (load + search → chunks with scores)"
```

---

## Task 7: Tool schemas + registry

**Files:**
- Create: `src/agents/__init__.py`
- Create: `src/agents/schemas.py`
- Create: `src/agents/tools.py`
- Create: `tests/agents/__init__.py`
- Create: `tests/agents/test_tools.py`

- [ ] **Step 1: Create empty package markers**

```bash
mkdir -p src/agents tests/agents
touch src/agents/__init__.py tests/agents/__init__.py
```

- [ ] **Step 2: Write the failing test**

Create `tests/agents/test_tools.py`:

```python
"""Tests for src.agents.tools — Tool dataclass + registry + 4 tool wrappers."""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from src.agents.tools import (
    Tool,
    build_default_tools,
    BBBPipelineInput,
    EEGPipelineInput,
    MRIPipelineInput,
    RetrieveContextInput,
)


class _DummyInput(BaseModel):
    x: int
    y: str = "default"


class _DummyOutput(BaseModel):
    result: int


class TestTool:
    def test_openai_schema_shape(self) -> None:
        tool = Tool(
            name="dummy",
            description="A dummy tool",
            input_model=_DummyInput,
            output_model=_DummyOutput,
            execute=lambda inp: _DummyOutput(result=inp.x * 2),
        )
        schema = tool.openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "dummy"
        assert schema["function"]["description"] == "A dummy tool"
        params = schema["function"]["parameters"]
        assert params["type"] == "object"
        assert "x" in params["properties"]
        assert "x" in params["required"]
        assert "y" not in params["required"]  # has default

    def test_invoke_validates_and_returns_dict(self) -> None:
        tool = Tool(
            name="dummy",
            description="d",
            input_model=_DummyInput,
            output_model=_DummyOutput,
            execute=lambda inp: _DummyOutput(result=inp.x * 2),
        )
        out = tool.invoke({"x": 5})
        assert out == {"result": 10}

    def test_invoke_invalid_input_raises(self) -> None:
        tool = Tool(
            name="dummy",
            description="d",
            input_model=_DummyInput,
            output_model=_DummyOutput,
            execute=lambda inp: _DummyOutput(result=inp.x * 2),
        )
        with pytest.raises(ValueError, match="invalid input"):
            tool.invoke({"y": "missing-x"})


class TestBuildDefaultTools:
    def test_default_set_has_four_tools(self, tmp_path: Path) -> None:
        # build with placeholder paths; tools won't be invoked here
        tools = build_default_tools(rag_index_dir=None)
        names = {t.name for t in tools}
        assert names == {
            "run_bbb_pipeline",
            "run_eeg_pipeline",
            "run_mri_pipeline",
            "retrieve_context",
        }

    def test_each_tool_has_pydantic_input_model(self) -> None:
        tools = build_default_tools(rag_index_dir=None)
        for t in tools:
            assert issubclass(t.input_model, BaseModel)
            assert issubclass(t.output_model, BaseModel)

    def test_input_models_have_smiles_paths(self) -> None:
        # verify the field names downstream system prompt depends on
        assert "smiles" in BBBPipelineInput.model_fields
        assert "input_path" in EEGPipelineInput.model_fields
        assert "input_dir" in MRIPipelineInput.model_fields
        assert "sites_csv" in MRIPipelineInput.model_fields
        assert "query" in RetrieveContextInput.model_fields
        assert "k" in RetrieveContextInput.model_fields
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/agents/test_tools.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'src.agents.tools'`

- [ ] **Step 4: Implement schemas**

Create `src/agents/schemas.py`:

```python
"""Pydantic input/output schemas for orchestrator tools and the agent result.

These schemas double as OpenAI function-calling parameter definitions
(via `model_json_schema()`) and as runtime validation gates. Keep field
names lowercase + snake_case so prompts and JSON outputs align.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# --- Pipeline tool inputs ---------------------------------------------------

class BBBPipelineInput(BaseModel):
    """Input for `run_bbb_pipeline` — a single SMILES string."""
    smiles: str = Field(..., description="A single molecular SMILES string, e.g. 'CCO'")
    top_k: int = Field(5, ge=1, le=20, description="Top-k SHAP attributions to return")


class EEGPipelineInput(BaseModel):
    """Input for `run_eeg_pipeline` — path to an EEG file (.fif or .edf)."""
    input_path: str = Field(..., description="Path to EEG recording file (.fif or .edf)")
    epoch_duration_s: float = Field(2.0, gt=0.1, le=60.0)


class MRIPipelineInput(BaseModel):
    """Input for `run_mri_pipeline` — directory of NIfTI files + sites CSV."""
    input_dir: str = Field(..., description="Directory containing .nii.gz volumes")
    sites_csv: str = Field(..., description="CSV mapping subject_id → site")


class RetrieveContextInput(BaseModel):
    """Input for `retrieve_context` — natural-language query into the KB."""
    query: str = Field(..., min_length=2, description="Search query for the knowledge base")
    k: int = Field(4, ge=1, le=10, description="Number of chunks to return")


# --- Pipeline tool outputs --------------------------------------------------

class BBBPipelineOutput(BaseModel):
    smiles: str
    label: int
    label_text: str
    confidence: float
    top_features: list[dict[str, Any]]
    drift_z: float | None = None


class EEGPipelineOutput(BaseModel):
    input_path: str
    output_path: str
    rows: int
    columns: int
    duration_sec: float


class MRIPipelineOutput(BaseModel):
    input_dir: str
    output_path: str
    rows: int
    columns: int
    duration_sec: float


class RetrieveContextOutput(BaseModel):
    query: str
    chunks: list[dict[str, Any]]


# --- Agent result -----------------------------------------------------------

class ToolTraceItem(BaseModel):
    """One step in the orchestrator's tool-call trace."""
    name: str
    args: dict[str, Any]
    result: dict[str, Any] | None = None
    error: str | None = None


class AgentResult(BaseModel):
    """Final orchestrator response: synthesized text + full trace."""
    text: str
    trace: list[ToolTraceItem] = Field(default_factory=list)
    model: str | None = None
    finish_reason: str = "complete"  # complete | max_steps | error
```

- [ ] **Step 5: Implement Tool dataclass + registry**

Create `src/agents/tools.py`:

```python
"""Tool dataclass + registry. Wraps each pipeline + the RAG retriever as a
function-callable tool the orchestrator can invoke.

Public entry: `build_default_tools(rag_index_dir)` returns the 4 tools.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from src.agents.schemas import (
    BBBPipelineInput,
    BBBPipelineOutput,
    EEGPipelineInput,
    EEGPipelineOutput,
    MRIPipelineInput,
    MRIPipelineOutput,
    RetrieveContextInput,
    RetrieveContextOutput,
)
from src.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Tool:
    """One callable tool exposed to the orchestrator.

    `execute(input_model_instance) -> output_model_instance` is the contract.
    `invoke(args_dict)` validates the dict, runs execute, returns a plain dict.
    """
    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    execute: Callable[[Any], BaseModel]

    def openai_schema(self) -> dict[str, Any]:
        """OpenAI/OpenRouter function-calling schema for this tool."""
        params = self.input_model.model_json_schema()
        # OpenAI doesn't accept top-level $defs / title in some clients —
        # strip the cosmetic ones; keep properties/required/type.
        cleaned = {
            "type": "object",
            "properties": params.get("properties", {}),
            "required": params.get("required", []),
        }
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": cleaned,
            },
        }

    def invoke(self, args: dict[str, Any]) -> dict[str, Any]:
        try:
            inp = self.input_model.model_validate(args)
        except ValidationError as e:
            raise ValueError(f"invalid input for {self.name}: {e}") from e
        out = self.execute(inp)
        return out.model_dump()


# ---------------------------------------------------------------------------
# Tool implementations — thin wrappers around existing pipelines + RAG.
# Heavy work stays in the underlying modules; these only adapt I/O.
# ---------------------------------------------------------------------------


def _execute_bbb(inp: BBBPipelineInput) -> BBBPipelineOutput:
    """Predict + SHAP for a single SMILES, reusing the existing model surface."""
    from src.api import routes as api_routes
    from src.api.schemas import BBBPredictRequest

    response = api_routes.predict_bbb(
        BBBPredictRequest(smiles=inp.smiles, top_k=inp.top_k)
    )
    return BBBPipelineOutput(
        smiles=inp.smiles,
        label=response.label,
        label_text=response.label_text,
        confidence=response.confidence,
        top_features=[f.model_dump() for f in response.top_features],
        drift_z=response.drift_z,
    )


def _execute_eeg(inp: EEGPipelineInput) -> EEGPipelineOutput:
    from src.api.schemas import EEGRequest
    from src.api import routes as api_routes

    out_path = Path("data/processed/eeg_features.parquet")
    response = api_routes.run_eeg_pipeline_route(
        EEGRequest(
            input_path=inp.input_path,
            output_path=str(out_path),
            epoch_duration_s=inp.epoch_duration_s,
        )
    )
    return EEGPipelineOutput(
        input_path=inp.input_path,
        output_path=response.output_path,
        rows=response.rows,
        columns=response.columns,
        duration_sec=response.duration_sec,
    )


def _execute_mri(inp: MRIPipelineInput) -> MRIPipelineOutput:
    from src.api.schemas import MRIRequest
    from src.api import routes as api_routes

    out_path = Path("data/processed/mri_features.parquet")
    response = api_routes.run_mri_pipeline_route(
        MRIRequest(
            input_dir=inp.input_dir,
            sites_csv=inp.sites_csv,
            output_path=str(out_path),
        )
    )
    return MRIPipelineOutput(
        input_dir=inp.input_dir,
        output_path=response.output_path,
        rows=response.rows,
        columns=response.columns,
        duration_sec=response.duration_sec,
    )


def _make_retrieve_executor(rag_index_dir: Path | None) -> Callable[[RetrieveContextInput], RetrieveContextOutput]:
    """Closure: capture the index dir; lazy-load the retriever on first call."""
    state: dict[str, Any] = {"retriever": None}

    def execute(inp: RetrieveContextInput) -> RetrieveContextOutput:
        if rag_index_dir is None or not (rag_index_dir / "index.bin").exists():
            return RetrieveContextOutput(query=inp.query, chunks=[])
        if state["retriever"] is None:
            from src.rag.retrieve import RAGRetriever
            state["retriever"] = RAGRetriever.load(rag_index_dir)
        hits = state["retriever"].search(inp.query, k=inp.k)
        return RetrieveContextOutput(query=inp.query, chunks=hits)

    return execute


def build_default_tools(rag_index_dir: Path | None) -> list[Tool]:
    """Return the 4 tools the orchestrator gets by default."""
    return [
        Tool(
            name="run_bbb_pipeline",
            description=(
                "Predict blood-brain-barrier permeability for a SINGLE SMILES "
                "string. Use this when the user input looks like a molecule "
                "(short alphanumeric string with no file extension, e.g. 'CCO', "
                "'c1ccccc1'). Returns label, confidence, top SHAP features, drift."
            ),
            input_model=BBBPipelineInput,
            output_model=BBBPipelineOutput,
            execute=_execute_bbb,
        ),
        Tool(
            name="run_eeg_pipeline",
            description=(
                "Run the EEG signal-processing pipeline (bandpass + ICA + "
                "epoching + feature extraction) on an EEG recording file. Use "
                "when input_path ends in .fif or .edf. Returns row/column "
                "counts + duration."
            ),
            input_model=EEGPipelineInput,
            output_model=EEGPipelineOutput,
            execute=_execute_eeg,
        ),
        Tool(
            name="run_mri_pipeline",
            description=(
                "Run the multi-site MRI ComBat-harmonization pipeline. Use "
                "when input is a directory containing .nii.gz volumes paired "
                "with a sites.csv. Returns row/column counts + duration."
            ),
            input_model=MRIPipelineInput,
            output_model=MRIPipelineOutput,
            execute=_execute_mri,
        ),
        Tool(
            name="retrieve_context",
            description=(
                "Retrieve up to k passages from the curated reference knowledge "
                "base. Use AFTER a pipeline tool returns, to ground your final "
                "synthesis in cited literature. Formulate a focused query "
                "based on the pipeline output (e.g., 'BBB permeability of "
                "small lipophilic molecules' or 'ComBat site harmonization')."
            ),
            input_model=RetrieveContextInput,
            output_model=RetrieveContextOutput,
            execute=_make_retrieve_executor(rag_index_dir),
        ),
    ]
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/agents/test_tools.py -v`

Expected: 6 passed

- [ ] **Step 7: Commit**

```bash
git add src/agents/__init__.py src/agents/schemas.py src/agents/tools.py tests/agents/__init__.py tests/agents/test_tools.py
git commit -m "feat(agents): Tool dataclass + registry + 4 tool wrappers (3 pipelines + RAG)"
```

---

## Task 8: Orchestrator agent loop

**Files:**
- Create: `src/agents/prompts.py`
- Create: `src/agents/orchestrator.py`
- Create: `tests/agents/test_orchestrator.py`

- [ ] **Step 1: Create the system prompt module**

Create `src/agents/prompts.py`:

```python
"""System prompts for the orchestrator agent.

Kept in a dedicated module so prompt edits are diff-readable and reviewable
in isolation from the orchestrator loop.
"""
from __future__ import annotations


ORCHESTRATOR_SYSTEM_PROMPT = """\
You are the NeuroBridge clinical-ML orchestrator. You have four tools:

- run_bbb_pipeline(smiles, top_k=5)         → for a SMILES molecular string
- run_eeg_pipeline(input_path)               → for a .fif or .edf EEG file path
- run_mri_pipeline(input_dir, sites_csv)     → for a directory of NIfTI MRI files
- retrieve_context(query, k=4)               → for grounding chunks from the knowledge base

Workflow — follow exactly:

1. Look at the user input. Decide which ONE pipeline tool fits:
   - SMILES (short, all-letters/digits, no slashes, no .ext)        → run_bbb_pipeline
   - Path ending in .fif or .edf                                    → run_eeg_pipeline
   - Path that is a directory (no file extension at the tail)       → run_mri_pipeline
   If ambiguous, prefer SMILES if it parses; otherwise return:
   "Cannot identify modality. Provide a SMILES, .fif/.edf path, or NIfTI directory."

2. Call the chosen pipeline tool exactly once with the user input.

3. After the pipeline returns, formulate ONE focused retrieval query that
   captures the scientific concept behind the prediction (NOT the raw input).
   Examples of good queries:
   - "BBB permeability of small lipophilic molecules" (after BBB predict)
   - "ICA artifact removal in multi-channel EEG" (after EEG run)
   - "ComBat scanner site harmonization in multi-center MRI" (after MRI run)
   Then call retrieve_context with that query.

4. Synthesize a final response in 3-5 sentences:
   - State the concrete pipeline result (label, confidence, key numbers).
   - Cite at least one specific fact from the retrieved chunks (mention the
     source file in parentheses, e.g. "(lipinski_rule_of_five.md)").
   - Match the user's question language: Turkish in → Turkish out, etc.
   - If retrieve_context returned 0 chunks, say so explicitly and answer
     using only the pipeline result.

Hard constraints:
- Call exactly ONE pipeline tool, then exactly ONE retrieve_context, then stop.
- Do NOT invent facts. Only use numbers from the pipeline tool output and
  text from the retrieved chunks.
- No preamble, no apologies, no meta-commentary about being an AI.
"""
```

- [ ] **Step 2: Write the failing test**

Create `tests/agents/test_orchestrator.py`:

```python
"""Tests for src.agents.orchestrator — agent loop with stubbed LLM client.

We do NOT hit OpenRouter here. We construct a fake client that returns
scripted tool-call responses, then verify the orchestrator dispatches
tools and assembles the trace correctly.
"""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from src.agents.orchestrator import Orchestrator
from src.agents.tools import Tool


# --- Helpers ----------------------------------------------------------------


def _fake_choice_with_tool_call(name: str, args: dict[str, Any], call_id: str = "c1") -> Any:
    msg = MagicMock()
    msg.content = None
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = json.dumps(args)
    tc.model_dump = MagicMock(return_value={"id": call_id, "type": "function",
                                            "function": {"name": name,
                                                         "arguments": json.dumps(args)}})
    msg.tool_calls = [tc]
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


def _fake_choice_with_text(text: str) -> Any:
    msg = MagicMock()
    msg.content = text
    msg.tool_calls = None
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


class _PingInput(BaseModel):
    msg: str


class _PingOutput(BaseModel):
    echo: str


def _make_ping_tool() -> Tool:
    return Tool(
        name="ping",
        description="Echo a string back.",
        input_model=_PingInput,
        output_model=_PingOutput,
        execute=lambda inp: _PingOutput(echo=f"pong:{inp.msg}"),
    )


# --- Tests ------------------------------------------------------------------


class TestOrchestrator:
    def test_single_tool_then_text_response(self) -> None:
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            _fake_choice_with_tool_call("ping", {"msg": "hello"}),
            _fake_choice_with_text("All done."),
        ]
        orch = Orchestrator(
            llm_client=client,
            tools=[_make_ping_tool()],
            system_prompt="sys",
            model="stub-model",
            max_steps=4,
        )
        result = orch.run("test input")
        assert result.text == "All done."
        assert result.finish_reason == "complete"
        assert len(result.trace) == 1
        assert result.trace[0].name == "ping"
        assert result.trace[0].args == {"msg": "hello"}
        assert result.trace[0].result == {"echo": "pong:hello"}

    def test_unknown_tool_recorded_as_error(self) -> None:
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            _fake_choice_with_tool_call("nonexistent_tool", {"x": 1}),
            _fake_choice_with_text("Done."),
        ]
        orch = Orchestrator(
            llm_client=client,
            tools=[_make_ping_tool()],
            system_prompt="sys",
            model="stub-model",
            max_steps=4,
        )
        result = orch.run("test")
        assert result.trace[0].error is not None
        assert "unknown tool" in result.trace[0].error
        assert result.text == "Done."

    def test_invalid_tool_args_recorded_as_error(self) -> None:
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            _fake_choice_with_tool_call("ping", {"wrong_field": "x"}),
            _fake_choice_with_text("Recovered."),
        ]
        orch = Orchestrator(
            llm_client=client,
            tools=[_make_ping_tool()],
            system_prompt="sys",
            model="stub-model",
            max_steps=4,
        )
        result = orch.run("test")
        assert result.trace[0].error is not None
        assert result.text == "Recovered."

    def test_max_steps_exhausted_returns_finish_reason(self) -> None:
        client = MagicMock()
        # Always return another tool call — never terminates with text
        client.chat.completions.create.side_effect = [
            _fake_choice_with_tool_call("ping", {"msg": f"{i}"}, call_id=f"c{i}")
            for i in range(10)
        ]
        orch = Orchestrator(
            llm_client=client,
            tools=[_make_ping_tool()],
            system_prompt="sys",
            model="stub-model",
            max_steps=3,
        )
        result = orch.run("test")
        assert result.finish_reason == "max_steps"
        assert len(result.trace) == 3

    def test_first_response_is_text_no_tools(self) -> None:
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            _fake_choice_with_text("Direct answer."),
        ]
        orch = Orchestrator(
            llm_client=client,
            tools=[_make_ping_tool()],
            system_prompt="sys",
            model="stub-model",
        )
        result = orch.run("trivial input")
        assert result.text == "Direct answer."
        assert result.trace == []
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/agents/test_orchestrator.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'src.agents.orchestrator'`

- [ ] **Step 4: Implement the orchestrator**

Create `src/agents/orchestrator.py`:

```python
"""Orchestrator agent: function-calling loop over a list of Tools.

No agent framework — uses the openai SDK's chat-completions function-calling
interface directly. This is the same SDK already used by src/llm/explainer.py,
keeping the dependency surface minimal.

Public entry: `Orchestrator(llm_client, tools, system_prompt, model).run(user_input)`.
Returns an `AgentResult` with synthesized text + full tool-call trace.
"""
from __future__ import annotations

import json
from typing import Any

from src.agents.schemas import AgentResult, ToolTraceItem
from src.agents.tools import Tool
from src.core.logger import get_logger

logger = get_logger(__name__)


class Orchestrator:
    """Single-agent function-calling loop. Stops on (a) text response, (b) max steps."""

    def __init__(
        self,
        llm_client: Any,
        tools: list[Tool],
        system_prompt: str,
        model: str,
        max_steps: int = 5,
        temperature: float = 0.0,
    ) -> None:
        self._client = llm_client
        self._tools_by_name = {t.name: t for t in tools}
        self._tool_schemas = [t.openai_schema() for t in tools]
        self._system_prompt = system_prompt
        self._model = model
        self._max_steps = max_steps
        self._temperature = temperature

    def run(self, user_input: str) -> AgentResult:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_input},
        ]
        trace: list[ToolTraceItem] = []

        for _step in range(self._max_steps):
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=self._tool_schemas,
                tool_choice="auto",
                temperature=self._temperature,
            )
            msg = response.choices[0].message

            if not getattr(msg, "tool_calls", None):
                return AgentResult(
                    text=(msg.content or "").strip(),
                    trace=trace,
                    model=self._model,
                    finish_reason="complete",
                )

            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
            })

            for tc in msg.tool_calls:
                name = tc.function.name
                tool = self._tools_by_name.get(name)
                if tool is None:
                    err = f"unknown tool: {name}"
                    trace.append(ToolTraceItem(name=name, args={}, error=err))
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({"error": err}),
                    })
                    continue
                try:
                    args = json.loads(tc.function.arguments or "{}")
                    result = tool.invoke(args)
                    trace.append(ToolTraceItem(name=name, args=args, result=result))
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({"result": result}, default=str),
                    })
                except Exception as e:
                    err = str(e)
                    trace.append(ToolTraceItem(name=name, args={}, error=err))
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({"error": err}),
                    })

        return AgentResult(
            text="Max steps reached without a final answer.",
            trace=trace,
            model=self._model,
            finish_reason="max_steps",
        )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/agents/test_orchestrator.py -v`

Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
git add src/agents/prompts.py src/agents/orchestrator.py tests/agents/test_orchestrator.py
git commit -m "feat(agents): orchestrator loop (function-calling + tool trace + max-steps gate)"
```

---

## Task 9: FastAPI /agent/run endpoint

**Files:**
- Modify: `src/api/schemas.py`
- Modify: `src/api/routes.py`
- Modify: `src/api/main.py`
- Create: `tests/agents/test_agent_route.py`

- [ ] **Step 1: Add request/response schemas**

Append to `src/api/schemas.py`:

```python


# --- Agent surface (orchestrator + RAG) ------------------------------------

class AgentRunRequest(BaseModel):
    """User input to the orchestrator."""
    user_input: str = Field(..., min_length=1, description="SMILES, file path, or directory path")
    user_question: str | None = Field(
        None, description="Optional natural-language question to language-match the response"
    )


class AgentToolTraceItem(BaseModel):
    name: str
    args: dict = Field(default_factory=dict)
    result: dict | None = None
    error: str | None = None


class AgentRunResponse(BaseModel):
    text: str
    trace: list[AgentToolTraceItem] = Field(default_factory=list)
    model: str | None = None
    finish_reason: str = "complete"
```

- [ ] **Step 2: Write the failing test**

Create `tests/agents/test_agent_route.py`:

```python
"""Tests for POST /agent/run — uses a stub orchestrator factory."""
from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.agents.schemas import AgentResult, ToolTraceItem
from src.api.main import app


client = TestClient(app)


class _FakeOrchestrator:
    """Returns a canned AgentResult; ignores input."""
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def run(self, user_input: str) -> AgentResult:
        return AgentResult(
            text=f"Synthesized answer for: {user_input}",
            trace=[
                ToolTraceItem(name="run_bbb_pipeline", args={"smiles": user_input},
                              result={"label": 1, "label_text": "permeable"}),
                ToolTraceItem(name="retrieve_context", args={"query": "BBB"},
                              result={"chunks": []}),
            ],
            model="stub-model",
            finish_reason="complete",
        )


class TestAgentRoute:
    def test_post_returns_synthesized_text_and_trace(self) -> None:
        with patch("src.api.routes._build_orchestrator", return_value=_FakeOrchestrator()):
            r = client.post("/agent/run", json={"user_input": "CCO"})
        assert r.status_code == 200
        body = r.json()
        assert "Synthesized answer for: CCO" in body["text"]
        assert len(body["trace"]) == 2
        assert body["trace"][0]["name"] == "run_bbb_pipeline"
        assert body["model"] == "stub-model"
        assert body["finish_reason"] == "complete"

    def test_empty_user_input_422(self) -> None:
        r = client.post("/agent/run", json={"user_input": ""})
        assert r.status_code == 422

    def test_missing_user_input_422(self) -> None:
        r = client.post("/agent/run", json={})
        assert r.status_code == 422
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/agents/test_agent_route.py -v`

Expected: FAIL with `404` or import error referencing `_build_orchestrator`.

- [ ] **Step 4: Wire up the route**

In `src/api/routes.py`, add to the imports block (alongside the existing `from src.api.schemas import ...`):

```python
from src.api.schemas import (
    AgentRunRequest,
    AgentRunResponse,
    AgentToolTraceItem,
    # ... existing imports continue ...
)
```

(Add `AgentRunRequest`, `AgentRunResponse`, `AgentToolTraceItem` to the alphabetized import block at the top.)

Append at the bottom of `src/api/routes.py`:

```python


# --- Agent router ----------------------------------------------------------

agent_router = APIRouter(prefix="/agent")


_DEFAULT_RAG_INDEX_DIR = Path("data/processed/faiss_index")
_AGENT_MODEL_ENV = "NEUROBRIDGE_AGENT_MODEL"
_AGENT_DEFAULT_MODEL = "google/gemini-2.0-flash-exp:free"


def _build_orchestrator():
    """Construct the default orchestrator. Patchable in tests."""
    from openai import OpenAI

    from src.agents.orchestrator import Orchestrator
    from src.agents.prompts import ORCHESTRATOR_SYSTEM_PROMPT
    from src.agents.tools import build_default_tools

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="OPENROUTER_API_KEY not set; agent surface unavailable.",
        )
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        timeout=30.0,
    )
    rag_dir = _DEFAULT_RAG_INDEX_DIR if _DEFAULT_RAG_INDEX_DIR.exists() else None
    tools = build_default_tools(rag_index_dir=rag_dir)
    model = os.environ.get(_AGENT_MODEL_ENV, _AGENT_DEFAULT_MODEL)
    return Orchestrator(
        llm_client=client,
        tools=tools,
        system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
        model=model,
        max_steps=5,
    )


@agent_router.post("/run", response_model=AgentRunResponse)
def run_agent(req: AgentRunRequest) -> AgentRunResponse:
    """Run the orchestrator on `user_input`. Picks a pipeline + grounds via RAG."""
    orch = _build_orchestrator()
    user_text = req.user_input
    if req.user_question:
        user_text = f"{req.user_input}\n\nUser question: {req.user_question}"
    result = orch.run(user_text)
    return AgentRunResponse(
        text=result.text,
        trace=[
            AgentToolTraceItem(name=t.name, args=t.args, result=t.result, error=t.error)
            for t in result.trace
        ],
        model=result.model,
        finish_reason=result.finish_reason,
    )
```

- [ ] **Step 5: Mount the router**

Modify `src/api/main.py`:

```python
from src.api.routes import (
    router as pipeline_router,
    predict_router,
    explain_router,
    experiments_router,
    agent_router,
)
```

And add the include line:

```python
app.include_router(experiments_router)
app.include_router(agent_router)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/agents/test_agent_route.py -v`

Expected: 3 passed

- [ ] **Step 7: Run the full test suite to verify no regressions**

Run: `pytest -q`

Expected: All previously-passing tests still pass.

- [ ] **Step 8: Commit**

```bash
git add src/api/schemas.py src/api/routes.py src/api/main.py tests/agents/test_agent_route.py
git commit -m "feat(api): POST /agent/run endpoint (orchestrator + RAG, stub-injectable)"
```

---

## Task 10: Streamlit Agent tab + decision trace UI

**Files:**
- Modify: `src/frontend/app.py`

- [ ] **Step 1: Locate the existing tabs declaration**

Open `src/frontend/app.py`, find the line containing `bbb_tab, eeg_tab, mri_tab, assistant_tab, experiments_tab = st.tabs([` (around line 1755).

- [ ] **Step 2: Add a new "🤖 Agent" tab**

Replace the tabs declaration:

```python
    bbb_tab, eeg_tab, mri_tab, assistant_tab, experiments_tab, agent_tab = st.tabs([
        "🧪 Molecule",
        "🌊 Signal",
        "🧠 Image",
        "🤝 AI Assistant",
        "🔬 Experiments",
        "🤖 Agent",
    ])
```

(Match the existing emoji + label style for the first five tabs — the exact strings may differ in your repo; only add the 6th tuple element and the 6th list element.)

- [ ] **Step 3: Implement the Agent tab body**

Find the end of `experiments_tab:` block. After it (still inside the same indentation as the other `with X_tab:` blocks), add:

```python
    with agent_tab:
        st.markdown("### Orchestrator Agent")
        st.caption(
            "Pick the pipeline automatically, run it, then ground the response "
            "in curated reference docs (RAG)."
        )

        with st.form("agent_form"):
            agent_input = st.text_input(
                "Input",
                value="CCO",
                help="SMILES (e.g., CCO), .fif/.edf path, or NIfTI directory path",
            )
            agent_question = st.text_input(
                "Question (optional)",
                value="",
                help="Ask in any language — the agent will mirror it in the response",
            )
            submitted = st.form_submit_button("Run agent")

        if submitted and agent_input:
            with st.spinner("Agent is reasoning..."):
                try:
                    payload: dict = {"user_input": agent_input}
                    if agent_question:
                        payload["user_question"] = agent_question
                    response = _post("/agent/run", payload, timeout=120.0)
                except Exception as e:
                    st.error(f"Agent run failed: {e}")
                else:
                    st.markdown("#### Response")
                    st.write(response.get("text", ""))
                    st.caption(
                        f"model: `{response.get('model', '?')}` · "
                        f"finish: `{response.get('finish_reason', '?')}`"
                    )
                    trace = response.get("trace", [])
                    with st.expander(f"🧠 Decision trace ({len(trace)} step{'s' if len(trace) != 1 else ''})", expanded=True):
                        if not trace:
                            st.write("_(no tool calls)_")
                        for i, step in enumerate(trace, start=1):
                            st.markdown(f"**{i}. `{step['name']}`**")
                            if step.get("error"):
                                st.error(step["error"])
                            else:
                                st.json(step.get("args", {}))
                                st.json(step.get("result", {}))
```

- [ ] **Step 4: Verify the file imports / `_post` helper**

`_post` is the existing helper used by other tabs. If your version doesn't accept a `timeout` kwarg, add it. Search for `def _post`:

```bash
grep -n "def _post" src/frontend/app.py
```

If `_post` lacks a timeout parameter, modify its signature. If it already accepts it, no change needed.

- [ ] **Step 5: Smoke-test the import**

Run: `python -c "import importlib.util; spec = importlib.util.spec_from_file_location('app', 'src/frontend/app.py'); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); print('imported ok')"`

Expected: `imported ok` (no syntax errors).

- [ ] **Step 6: Run the existing frontend smoke test**

Run: `pytest tests/frontend/ -v`

Expected: all green (existing import test still passes).

- [ ] **Step 7: Commit**

```bash
git add src/frontend/app.py
git commit -m "feat(frontend): Agent tab with decision-trace expander"
```

---

## Task 11: Knowledge base seed + Dockerfile RAG ingest

**Files:**
- Create: `data/knowledge_base/README.md`
- Create: `data/knowledge_base/.gitkeep`
- Modify: `Dockerfile`
- Modify: `Dockerfile.hf`

- [ ] **Step 1: Create the knowledge-base directory + README**

```bash
mkdir -p data/knowledge_base
touch data/knowledge_base/.gitkeep
```

Create `data/knowledge_base/README.md`:

```markdown
# RAG Knowledge Base

Drop reference documents here (`.md`, `.txt`, or `.pdf`). They will be
ingested by `python -m src.rag.ingest` at Docker build time and surfaced
to the orchestrator agent via the `retrieve_context` tool.

## Recommended seed set

For a clinical-ML / NeuroBridge demo:

- **BBB / molecules**: Lipinski's Rule of Five (1997, 2001), Pajouhesh & Lenz
  CNS multiparameter optimization (2005)
- **MRI / harmonization**: Fortin et al. ComBat for cortical thickness (2017),
  Fortin et al. ComBat for diffusion (2018), Johnson et al. original ComBat
  (2007, gene expression)
- **EEG / artifacts**: Hyvärinen ICA primer (1999), MNE-Python overview
  (Gramfort 2013)

## Format notes

- PDFs work via `pypdf`. OCR-only PDFs (scanned images) won't extract text;
  pre-OCR them first.
- Markdown is preferred — full text + headers chunk cleanly.
- Files are gitignored by default. Mount them via Docker volume in
  production, or COPY them in via a sub-path before the `RUN` ingest line.

## Re-indexing

After adding/removing files, re-run:

    python -m src.rag.ingest

This rewrites `data/processed/faiss_index/` from scratch (no incremental
update — the index is small enough to rebuild in seconds).
```

- [ ] **Step 2: Add the ingest step to Dockerfile**

Open `Dockerfile`. Find the existing big `RUN mkdir -p data/raw data/processed && ...` block (around line 38). At the END of that block (before the `EXPOSE` line), append a new RUN step:

```dockerfile
# --- RAG knowledge base ingest ---
# Build the FAISS index from any seed docs in tests/fixtures/kb_sample/
# (always present) plus data/knowledge_base/ (optional, user-supplied via
# additional COPY layer or volume mount). Empty KB → empty index, agent
# still functions, retrieve_context just returns no chunks.
COPY tests/fixtures/kb_sample/ ./data/knowledge_base/seed/
RUN python -m src.rag.ingest data/knowledge_base data/processed/faiss_index
```

(Place this after the existing pipeline-seed block and before `EXPOSE 7860`.)

- [ ] **Step 3: Mirror the change in Dockerfile.hf**

Apply the exact same edit to `Dockerfile.hf` (it's currently identical to `Dockerfile` per the readback).

- [ ] **Step 4: Verify Dockerfile parses**

Run: `docker build --no-cache -f Dockerfile -t neurobridge-test . 2>&1 | tail -30`

Expected: build succeeds; the `python -m src.rag.ingest` step logs `Indexed N chunks → data/processed/faiss_index`.

(If Docker isn't available locally, skip and verify on next HF push instead — note the assumption in the commit.)

- [ ] **Step 5: Commit**

```bash
git add data/knowledge_base/README.md data/knowledge_base/.gitkeep Dockerfile Dockerfile.hf
git commit -m "feat(deploy): build RAG index at Docker build time + KB seed dir"
```

---

## Task 12: Live OpenRouter integration test + diag endpoint

**Files:**
- Create: `tests/agents/test_orchestrator_live.py`
- Modify: `src/api/main.py`

- [ ] **Step 1: Write the network-gated live test**

Create `tests/agents/test_orchestrator_live.py`:

```python
"""Live integration test — hits real OpenRouter, picks pipeline, retrieves chunks.

Skipped unless OPENROUTER_API_KEY is set. Marked `slow` (network round-trips).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from openai import OpenAI

from src.agents.orchestrator import Orchestrator
from src.agents.prompts import ORCHESTRATOR_SYSTEM_PROMPT
from src.agents.tools import build_default_tools
from src.rag.ingest import ingest_directory


_FIXTURE_KB = Path(__file__).parent.parent / "fixtures" / "kb_sample"
_DEFAULT_MODEL = "google/gemini-2.0-flash-exp:free"
_FALLBACK_MODEL = "anthropic/claude-haiku-4-5"


@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set",
)
class TestOrchestratorLive:
    @pytest.fixture(scope="class")
    def rag_dir(self, tmp_path_factory: pytest.TempPathFactory) -> Path:
        d = tmp_path_factory.mktemp("rag_live")
        ingest_directory(_FIXTURE_KB, d)
        return d

    @pytest.fixture(scope="class")
    def client(self) -> OpenAI:
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
            timeout=30.0,
        )

    def test_smiles_input_picks_bbb_then_retrieves(self, client: OpenAI, rag_dir: Path) -> None:
        tools = build_default_tools(rag_index_dir=rag_dir)
        orch = Orchestrator(
            llm_client=client,
            tools=tools,
            system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
            model=os.environ.get("NEUROBRIDGE_AGENT_MODEL", _DEFAULT_MODEL),
            max_steps=5,
        )
        result = orch.run("CCO")
        # Soft assertions — model behavior varies but the workflow shape is fixed.
        assert result.finish_reason == "complete", f"got {result.finish_reason}, trace={result.trace}"
        tool_names = [t.name for t in result.trace]
        assert "run_bbb_pipeline" in tool_names, f"BBB pipeline not called; trace={tool_names}"
        assert "retrieve_context" in tool_names, f"RAG not called; trace={tool_names}"
        assert result.text, "empty final text"
```

- [ ] **Step 2: Run the test (live, requires key)**

Run: `OPENROUTER_API_KEY=$OPENROUTER_API_KEY pytest tests/agents/test_orchestrator_live.py -v -m slow`

Expected: 1 passed. If the BBB pipeline tool fails because the model artifact isn't present, that is a separate setup issue — the test still validates the orchestration shape.

- [ ] **Step 3: Add /diag/agent endpoint**

In `src/api/main.py`, after the existing `diag_openrouter` function, append:

```python


@app.get("/diag/agent")
def diag_agent() -> dict:
    """Reachability probe for the orchestrator agent surface.

    Reports key presence (length + 12-char prefix only — never the full
    secret), the configured agent model, knowledge-base index status,
    and the registered tool names.
    """
    import os as _os
    from pathlib import Path as _Path

    from src.agents.tools import build_default_tools

    key = _os.environ.get("OPENROUTER_API_KEY") or ""
    model = _os.environ.get("NEUROBRIDGE_AGENT_MODEL", "google/gemini-2.0-flash-exp:free")

    rag_dir = _Path("data/processed/faiss_index")
    rag_status: dict = {"index_dir": str(rag_dir), "exists": False, "chunk_count": 0}
    if (rag_dir / "index.bin").exists() and (rag_dir / "chunks.json").exists():
        rag_status["exists"] = True
        try:
            import json as _json
            rag_status["chunk_count"] = len(_json.loads((rag_dir / "chunks.json").read_text()))
        except Exception as e:
            rag_status["error"] = f"chunks.json unreadable: {e}"

    tools = build_default_tools(rag_index_dir=rag_dir if rag_status["exists"] else None)
    return {
        "has_key": bool(key),
        "key_len": len(key),
        "key_prefix": key[:12] if key else None,
        "agent_model": model,
        "rag": rag_status,
        "tool_names": [t.name for t in tools],
    }
```

- [ ] **Step 4: Smoke-test the diag endpoint**

Start the API in one shell:

```bash
uvicorn src.api.main:app --port 8000 &
sleep 3
curl -s http://localhost:8000/diag/agent | python3 -m json.tool
kill %1
```

Expected: JSON with `has_key`, `agent_model`, `rag.exists` (true if you ran the ingest CLI locally), and `tool_names: [...]` list of 4 tool names.

- [ ] **Step 5: Commit**

```bash
git add tests/agents/test_orchestrator_live.py src/api/main.py
git commit -m "feat(agents): live OpenRouter integration test (slow) + GET /diag/agent"
```

---

## Task 13: Documentation update

**Files:**
- Modify: `AGENTS.md`
- Modify: `README.md`

- [ ] **Step 1: Add §15 + §16 to AGENTS.md**

Append to `AGENTS.md`:

```markdown

## 15. Orchestrator Agent Surface

`src/agents/orchestrator.py` exposes a single-agent function-calling
loop over the openai SDK (no LangChain / framework dep). The agent
holds 4 tools, defined in `src/agents/tools.py`:

- `run_bbb_pipeline(smiles, top_k)` — wraps `POST /predict/bbb`
- `run_eeg_pipeline(input_path)` — wraps `POST /pipeline/eeg`
- `run_mri_pipeline(input_dir, sites_csv)` — wraps `POST /pipeline/mri`
- `retrieve_context(query, k)` — wraps `src/rag/retrieve.py`

The system prompt (`src/agents/prompts.py:ORCHESTRATOR_SYSTEM_PROMPT`)
locks the workflow: pick exactly one pipeline → run it → formulate a
focused retrieval query → call retrieve_context → synthesize a
3-5 sentence response that cites at least one chunk. Language of the
final response is mirrored from the user's question.

`POST /agent/run` is the public surface. Default model is
`google/gemini-2.0-flash-exp:free` on OpenRouter (function-calling
support verified). Override via `NEUROBRIDGE_AGENT_MODEL` env var.
Returns 503 when `OPENROUTER_API_KEY` is unset.

Diagnostics: `GET /diag/agent` returns key presence, configured model,
RAG index status (chunk count), and the registered tool names.

## 16. RAG Surface

`src/rag/` is the retrieval layer. Stack: `fastembed`
(`BAAI/bge-small-en-v1.5`, 384-dim, ONNX, no torch dep) for
embeddings + `faiss-cpu` (`IndexFlatIP` after L2-norm = cosine) for
vector search.

Knowledge base lives at `data/knowledge_base/` (gitignored;
user-supplied `.md` / `.txt` / `.pdf`). Build the FAISS index with:

    python -m src.rag.ingest [<input_dir> [<output_dir>]]

Defaults: input=`data/knowledge_base/`, output=`data/processed/faiss_index/`.
The Dockerfile runs this at build time so deployed Spaces start with
a populated index. Empty KB → empty index → `retrieve_context`
returns 0 chunks; the agent surfaces this and answers from the
pipeline result alone.

`tests/fixtures/kb_sample/` ships 3 seed markdown files (Lipinski,
ComBat, MNE+ICA) — these double as test fixtures and as the demo
seed if no user-supplied PDFs are added.
```

- [ ] **Step 2: Add agent + RAG bullets to README.md "Where to Look"**

In `README.md`, find the "Where to Look" list. Append:

```markdown
- **Orchestrator agent (Task 13):** [`src/agents/orchestrator.py`](src/agents/orchestrator.py), [`src/agents/tools.py`](src/agents/tools.py), [`src/agents/prompts.py`](src/agents/prompts.py)
- **RAG layer:** [`src/rag/`](src/rag/) — chunker, embedder (fastembed), FAISS store, retriever, ingest CLI
- **Agent endpoint:** `POST /agent/run` (orchestrator + RAG); diagnostic at `GET /diag/agent`
- **Streamlit Agent tab:** "🤖 Agent" tab in [`src/frontend/app.py`](src/frontend/app.py) — input box + decision-trace expander
- **RAG knowledge base:** drop `.md`/`.pdf` into [`data/knowledge_base/`](data/knowledge_base/) — see its README
```

- [ ] **Step 3: Commit**

```bash
git add AGENTS.md README.md
git commit -m "docs: §15 orchestrator agent + §16 RAG surface (AGENTS.md + README pointers)"
```

---

## Self-Review

**1. Spec coverage:** Walked through the user's spec — pipelines as tools (Tasks 6, 7), orchestrator at the front (Tasks 7, 8, 9), RAG feedback (Tasks 2-6, 7), modular (separate `src/agents/` and `src/rag/` packages with single-responsibility files), user-supplied KB files (Task 11). All covered.

**2. Placeholder scan:** No "TODO", "implement later", "fill in details", or "similar to Task N" in the body. Each step has full code.

**3. Type consistency:**
- `BBBPipelineInput.smiles` (str) used in Task 7 schemas, Task 7 tool `_execute_bbb`, Task 8 stub test, Task 12 live test ✓
- `RetrieveContextInput.query` + `k` used consistently in Task 7 schema, Task 7 tool, Task 8 prompt ✓
- `Tool.invoke(args: dict)` returns dict — used in Task 8 orchestrator ✓
- `AgentResult` / `ToolTraceItem` schemas used in Task 7 (define), Task 8 (build), Task 9 (route response model) ✓
- `Orchestrator.__init__(llm_client, tools, system_prompt, model, max_steps, temperature)` matches usage in Task 9 `_build_orchestrator` and Task 12 live test ✓
- Pipeline call paths: Task 7's `_execute_bbb` references `api_routes.predict_bbb` — verify this name matches `src/api/routes.py`. **Note for implementer:** If the actual function name differs (e.g., `predict_bbb_endpoint`), adapt the call site; the test in Task 8 uses a stub so it won't catch this. Same for `run_eeg_pipeline_route` / `run_mri_pipeline_route`.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-02-orchestrator-agent-rag.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
