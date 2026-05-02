"""End-to-end: agent calls run_fusion with realistic inputs, top disease is sane."""
from __future__ import annotations

import pytest

from src.agents.tools import build_default_tools


@pytest.mark.parametrize(
    "scenario,expected_top",
    [
        # Strong AD signal: low MMSE + MRI flags alzheimers.
        (
            {
                "mri": {
                    "label_text": "alzheimers", "label": 1, "confidence": 0.85,
                    "probabilities": [
                        {"label_text": "control", "probability": 0.15},
                        {"label_text": "alzheimers", "probability": 0.85},
                    ],
                },
                "clinical": {"mmse": 14.0, "age_years": 79.0},
            },
            "alzheimers",
        ),
        # Strong PD signal: high UPDRS + slow gait + EEG flags parkinsons.
        (
            {
                "eeg": {
                    "label_text": "parkinsons", "label": 1, "confidence": 0.78,
                    "probabilities": [
                        {"label_text": "control", "probability": 0.22},
                        {"label_text": "parkinsons", "probability": 0.78},
                    ],
                },
                "clinical": {"updrs": 80.0, "gait_speed_m_s": 0.4, "age_years": 70.0},
            },
            "parkinsons",
        ),
    ],
)
def test_realistic_scenarios_pick_correct_top_disease(scenario, expected_top) -> None:
    tools = {t.name: t for t in build_default_tools(rag_index_dir=None)}
    tool = tools["run_fusion"]
    out = tool.execute(tool.input_model.model_validate(scenario))
    assert out.top_disease == expected_top
