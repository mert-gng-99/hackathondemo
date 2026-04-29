"""Smoke-test that the Streamlit app module imports cleanly.

Streamlit UIs are hard to unit-test without `streamlit.testing` (which
spawns a headless app); for hackathon scope we settle for a clean import
+ presence of `main()`. Manual UX testing via `streamlit run`.
"""
from __future__ import annotations


def test_app_module_imports() -> None:
    from src.frontend import app  # noqa: F401


def test_app_module_defines_main() -> None:
    from src.frontend import app
    assert hasattr(app, "main")
    assert callable(app.main)
