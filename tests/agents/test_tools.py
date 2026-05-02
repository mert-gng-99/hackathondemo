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

    def test_retrieve_context_short_circuits_when_no_index(self) -> None:
        tools = build_default_tools(rag_index_dir=None)
        retrieve = next(t for t in tools if t.name == "retrieve_context")
        out = retrieve.invoke({"query": "anything", "k": 3})
        assert out == {"query": "anything", "chunks": []}

    def test_processed_dir_parameter_threads_to_executors(self, tmp_path: Path) -> None:
        # build_default_tools should accept processed_dir; executors should
        # eventually write under it (we don't invoke the pipelines here, just
        # verify the parameter is accepted and tools are built).
        tools = build_default_tools(rag_index_dir=None, processed_dir=tmp_path)
        names = {t.name for t in tools}
        assert "run_eeg_pipeline" in names
        assert "run_mri_pipeline" in names

    def test_default_processed_dir_when_omitted(self) -> None:
        # backwards-compat: omitting processed_dir keeps existing behavior
        tools = build_default_tools(rag_index_dir=None)
        # just ensure no exception and 4 tools returned
        assert len(tools) == 4

    def test_bbb_executor_translates_httpexception_to_valueerror(self) -> None:
        from unittest.mock import patch
        from fastapi import HTTPException

        tools = build_default_tools(rag_index_dir=None)
        bbb = next(t for t in tools if t.name == "run_bbb_pipeline")

        with patch("src.api.routes.predict_bbb",
                   side_effect=HTTPException(status_code=503, detail="model missing")):
            with pytest.raises(ValueError, match="bbb tool failed"):
                bbb.invoke({"smiles": "CCO"})
