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
