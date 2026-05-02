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
