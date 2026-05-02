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


def _make_bbb_executor() -> Callable[[BBBPipelineInput], BBBPipelineOutput]:
    """Closure factory: BBB permeability prediction + SHAP, translates HTTPException."""
    def execute(inp: BBBPipelineInput) -> BBBPipelineOutput:
        from src.api import routes as api_routes
        from src.api.schemas import BBBPredictRequest
        from fastapi import HTTPException
        try:
            response = api_routes.predict_bbb(
                BBBPredictRequest(smiles=inp.smiles, top_k=inp.top_k)
            )
        except HTTPException as e:
            raise ValueError(f"bbb tool failed: {e.detail}") from e
        return BBBPipelineOutput(
            smiles=inp.smiles,
            label=response.label,
            label_text=response.label_text,
            confidence=response.confidence,
            top_features=[f.model_dump() for f in response.top_features],
            drift_z=response.drift_z,
        )
    return execute


def _make_eeg_executor(processed_dir: Path) -> Callable[[EEGPipelineInput], EEGPipelineOutput]:
    """Closure factory: EEG pipeline, writes output under processed_dir."""
    def execute(inp: EEGPipelineInput) -> EEGPipelineOutput:
        from src.api.schemas import EEGRequest
        from src.api import routes as api_routes
        from fastapi import HTTPException
        out_path = processed_dir / "eeg_features.parquet"
        try:
            response = api_routes.run_eeg(
                EEGRequest(
                    input_path=inp.input_path,
                    output_path=str(out_path),
                    epoch_duration_s=inp.epoch_duration_s,
                )
            )
        except HTTPException as e:
            raise ValueError(f"eeg tool failed: {e.detail}") from e
        return EEGPipelineOutput(
            input_path=inp.input_path,
            output_path=response.output_path,
            rows=response.rows,
            columns=response.columns,
            duration_sec=response.duration_sec,
        )
    return execute


def _make_mri_executor(processed_dir: Path) -> Callable[[MRIPipelineInput], MRIPipelineOutput]:
    """Closure factory: MRI pipeline, writes output under processed_dir."""
    def execute(inp: MRIPipelineInput) -> MRIPipelineOutput:
        from src.api.schemas import MRIRequest
        from src.api import routes as api_routes
        from fastapi import HTTPException
        out_path = processed_dir / "mri_features.parquet"
        sites_csv = inp.sites_csv or str(Path(inp.input_dir) / "sites.csv")
        try:
            response = api_routes.run_mri(
                MRIRequest(
                    input_dir=inp.input_dir,
                    sites_csv=sites_csv,
                    output_path=str(out_path),
                )
            )
        except HTTPException as e:
            raise ValueError(f"mri tool failed: {e.detail}") from e
        return MRIPipelineOutput(
            input_dir=inp.input_dir,
            output_path=response.output_path,
            rows=response.rows,
            columns=response.columns,
            duration_sec=response.duration_sec,
        )
    return execute


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


def build_default_tools(
    rag_index_dir: Path | None,
    processed_dir: Path = Path("data/processed"),
) -> list[Tool]:
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
            execute=_make_bbb_executor(),
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
            execute=_make_eeg_executor(processed_dir),
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
            execute=_make_mri_executor(processed_dir),
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
