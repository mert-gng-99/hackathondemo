"""Deterministic fallbacks for the orchestrator workflow.

The LLM remains responsible for normal function-calling, but these helpers
keep the public agent route reliable when a model skips or mis-shapes a tool
call during a live demo.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.agents.schemas import ToolTraceItem


_EEG_SUFFIXES = {".fif", ".edf"}


def route_pipeline_input(
    user_input: str,
    context: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]] | None:
    """Map raw user input to exactly one pipeline tool and argument dict."""
    text = _primary_input(user_input)
    if not text:
        return None

    path = Path(text)
    lower = text.lower()
    if path.suffix.lower() in _EEG_SUFFIXES:
        return "run_eeg_pipeline", {"input_path": text}

    if _looks_like_mri_input(path, lower):
        input_dir = path.parent if lower.endswith(".nii.gz") or path.suffix.lower() == ".nii" else path
        sites_csv = _sites_csv_for(input_dir, context)
        return "run_mri_pipeline", {
            "input_dir": str(input_dir),
            "sites_csv": sites_csv,
        }

    if _looks_like_path(text):
        return None

    return "run_bbb_pipeline", {"smiles": text, "top_k": 5}


def build_retrieval_query(
    user_input: str,
    pipeline_trace: ToolTraceItem,
    context: dict[str, Any] | None = None,
) -> str:
    """Build the canonical scientific RAG query for a completed pipeline tool."""
    if pipeline_trace.name == "run_eeg_pipeline":
        return "ICA artifact removal in multi-channel EEG"
    if pipeline_trace.name == "run_mri_pipeline":
        return "ComBat scanner site harmonization in multi-center MRI"
    return "BBB permeability of small lipophilic molecules"


def _primary_input(user_input: str) -> str:
    """Return the first non-empty input line, excluding appended user questions."""
    before_question = user_input.split("\n\nUser question:", 1)[0]
    return before_question.strip().strip("\"'")


def _looks_like_mri_input(path: Path, lower: str) -> bool:
    if lower.endswith(".nii.gz") or path.suffix.lower() == ".nii":
        return True
    if path.exists() and path.is_dir():
        return True
    return not path.suffix and _looks_like_path(str(path))


def _looks_like_path(text: str) -> bool:
    return "/" in text or "\\" in text


def _sites_csv_for(input_dir: Path, context: dict[str, Any] | None) -> str:
    explicit = (context or {}).get("sites_csv")
    if explicit:
        return str(explicit)
    return str(input_dir / "sites.csv")
