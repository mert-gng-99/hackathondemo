"""Tests for the run_fusion agent tool."""
from __future__ import annotations

from src.agents.tools import build_default_tools


class TestRunFusionTool:
    def test_fusion_tool_is_registered(self) -> None:
        tools = build_default_tools(rag_index_dir=None)
        names = [t.name for t in tools]
        assert "run_fusion" in names

    def test_fusion_tool_executes_with_only_clinical(self) -> None:
        tools = {t.name: t for t in build_default_tools(rag_index_dir=None)}
        tool = tools["run_fusion"]
        out = tool.execute(tool.input_model.model_validate({
            "clinical": {"mmse": 12.0, "age_years": 78.0},
        }))
        assert out.top_disease in {"alzheimers", "parkinsons", "other"}
        assert any(d.disease == "alzheimers" for d in out.diseases)
