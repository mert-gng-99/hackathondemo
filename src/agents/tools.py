"""Tool dataclass + registry. Wraps each pipeline + the RAG retriever as a
function-callable tool the orchestrator can invoke.

Public entry: `build_default_tools(rag_index_dir)` returns the 7 tools.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from src.agents.schemas import (
    BBBPermeabilityMapInput,
    BBBPermeabilityMapOutput,
    BBBPipelineInput,
    BBBPipelineOutput,
    DrugDoseAdjustmentInput,
    DrugDoseAdjustmentOutput,
    EEGPipelineInput,
    EEGPipelineOutput,
    MRIPipelineInput,
    MRIPipelineOutput,
    RetrieveContextInput,
    RetrieveContextOutput,
)
from src.fusion.engine import fuse as fuse_engine
from src.fusion.types import FusionInput, FusionOutput
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
        # TODO(post-hackathon): per-call output path. Concurrent /agent/run
        # invocations race on this file and clobber each other's MLflow runs.
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
        # TODO(post-hackathon): per-call output path. Concurrent /agent/run
        # invocations race on this file and clobber each other's MLflow runs.
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


def _make_retrieve_executor(
    rag_index_dir: Path | None,
    clinical_rag_index_path: Path | None = None,
) -> Callable[[RetrieveContextInput], RetrieveContextOutput]:
    """Closure: capture both index sources; lazy-load each on first use."""
    state: dict[str, Any] = {"retriever": None, "clinical_payload": None}

    def execute(inp: RetrieveContextInput) -> RetrieveContextOutput:
        if inp.corpus == "clinical":
            if clinical_rag_index_path is None or not Path(clinical_rag_index_path).exists():
                logger.warning(
                    "retrieve_context corpus=clinical but no index path configured (path=%s)",
                    clinical_rag_index_path,
                )
                return RetrieveContextOutput(query=inp.query, chunks=[])
            if state["clinical_payload"] is None:
                from src.rag.clinical.loader import load_index
                state["clinical_payload"] = load_index(Path(clinical_rag_index_path))
            from src.rag.clinical.retrieve import retrieve_clinical
            result = retrieve_clinical(state["clinical_payload"], inp.query, top_k=inp.k)
            return RetrieveContextOutput(
                query=inp.query,
                chunks=[
                    {
                        "source": ev.source,
                        "page_start": ev.page_start,
                        "page_end": ev.page_end,
                        "text": ev.sentence,
                        "score": ev.score,
                    }
                    for ev in result.evidence
                ],
            )

        # corpus == "reference" — existing FAISS path.
        if rag_index_dir is None or not (rag_index_dir / "index.bin").exists():
            return RetrieveContextOutput(query=inp.query, chunks=[])
        if state["retriever"] is None:
            from src.rag.retrieve import RAGRetriever
            state["retriever"] = RAGRetriever.load(rag_index_dir)
        hits = state["retriever"].search(inp.query, k=inp.k)
        return RetrieveContextOutput(query=inp.query, chunks=hits)

    return execute


def _make_bbb_permeability_executor() -> Callable[[BBBPermeabilityMapInput], BBBPermeabilityMapOutput]:
    def execute(inp: BBBPermeabilityMapInput) -> BBBPermeabilityMapOutput:
        from src.models import bbb_permeability_map as bbb_perm
        result = bbb_perm.compute_permeability(
            input_path=Path(inp.input_path),
            mode=inp.mode,
        )
        return BBBPermeabilityMapOutput(
            permeability_score=float(result["permeability_score"]),
            interpretation=str(result["interpretation"]),
            method=str(result["method"]),
            voxel_map_available=bool(result.get("voxel_map_available", False)),
        )

    return execute


def _make_dose_adjuster_executor() -> Callable[[DrugDoseAdjustmentInput], DrugDoseAdjustmentOutput]:
    def execute(inp: DrugDoseAdjustmentInput) -> DrugDoseAdjustmentOutput:
        from src.research import drug_dose_adjuster

        drug_permeable = inp.drug_bbb_permeable
        if inp.smiles:
            try:
                from src.models import bbb_model
                import os as _os
                artifact = Path(_os.environ.get("BBB_MODEL_PATH", "data/processed/bbb_model.joblib"))
                if artifact.exists():
                    model = bbb_model.load(artifact)
                    pred = bbb_model.predict_with_proba(model, inp.smiles)
                    drug_permeable = bool(pred["label"] == 1)
            except (FileNotFoundError, ValueError, KeyError) as e:
                logger.warning(
                    "agent dose-adjuster could not auto-resolve BBB for smiles=%s: %s",
                    inp.smiles, e,
                )

        adj = drug_dose_adjuster.adjust(
            baseline_dose_mg=inp.baseline_dose_mg,
            bbb_permeability_score=inp.bbb_permeability_score,
            drug_bbb_permeable=drug_permeable,
        )
        return DrugDoseAdjustmentOutput(
            recommended_dose_mg=adj.recommended_dose_mg,
            adjustment_factor=adj.adjustment_factor,
            risk_level=adj.risk_level,
            rationale=adj.rationale,
            drug_bbb_permeable=drug_permeable,
        )

    return execute


def build_default_tools(
    rag_index_dir: Path | None,
    processed_dir: Path = Path("data/processed"),
    clinical_rag_index_path: Path | None = None,
) -> list[Tool]:
    """Return the 5 tools the orchestrator gets by default."""
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
                "Retrieve up to k passages from a knowledge base. corpus='clinical' "
                "queries the peer-reviewed Alzheimer's/Parkinson's papers (TF-IDF, "
                "supports Turkish keywords like 'egzersiz', 'beslenme', 'unutkanlik'); "
                "default corpus='reference' queries the curated FAISS index. Use "
                "AFTER a pipeline tool returns, to ground your final synthesis in "
                "cited literature."
            ),
            input_model=RetrieveContextInput,
            output_model=RetrieveContextOutput,
            execute=_make_retrieve_executor(rag_index_dir, clinical_rag_index_path),
        ),
        Tool(
            name="run_fusion",
            description=(
                "Combine MRI prediction, EEG prediction, and clinical-test "
                "scores (MMSE, MoCA, UPDRS, gait, age) into per-disease "
                "(Alzheimer's, Parkinson's, other) confidence with full "
                "attribution. Pass whichever modalities are available; "
                "missing ones are skipped, not imputed. Does NOT use BBB."
            ),
            input_model=FusionInput,
            output_model=FusionOutput,
            execute=lambda inp: fuse_engine(inp),
        ),
        Tool(
            name="compute_bbb_leakage_score",
            description=(
                "Researcher-only. Compute a BBB permeability score (0..1) "
                "from a patient MRI. Mode 'heuristic_proxy' (default) uses "
                "the 2D Alzheimer's classifier; 'dce_onnx' uses a real DCE "
                "ONNX model when available."
            ),
            input_model=BBBPermeabilityMapInput,
            output_model=BBBPermeabilityMapOutput,
            execute=_make_bbb_permeability_executor(),
        ),
        Tool(
            name="adjust_drug_dose",
            description=(
                "Researcher-only. Suggest a revised drug dose given the patient's "
                "BBB permeability score and the drug's BBB classification. If "
                "smiles is supplied, the BBB classifier auto-resolves whether "
                "the drug crosses the BBB. Output is a research suggestion, "
                "NOT medical advice."
            ),
            input_model=DrugDoseAdjustmentInput,
            output_model=DrugDoseAdjustmentOutput,
            execute=_make_dose_adjuster_executor(),
        ),
    ]
