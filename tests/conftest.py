"""Skip tests that need the gated conversation dataset when it has not been downloaded (see DATA_ACCESS.md)."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GATED = ROOT / "runs" / "v2" / "dialogues_both_arms.jsonl"
NEEDS_GATED = {
    "test_final_figs_a.py::test_all_figures_written",
    "test_evaluator_prompt_tuning.py::test_analyze_only_refuses_when_there_are_no_judgments",
}


def pytest_collection_modifyitems(config, items):
    if GATED.exists():
        return
    skip = pytest.mark.skip(reason="needs the gated dataset (see DATA_ACCESS.md)")
    for item in items:
        if f"{Path(str(item.fspath)).name}::{item.name}" in NEEDS_GATED:
            item.add_marker(skip)
