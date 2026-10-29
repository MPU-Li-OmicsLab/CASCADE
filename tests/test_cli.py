# tests/test_cli.py
import json
from cascade.cli import run_pilot


def test_run_pilot_returns_curves_and_writes_jsonl(tmp_path):
    out = tmp_path / "pilot.jsonl"
    result = run_pilot(max_turns=3, out_path=out)
    assert out.exists()
    assert len(result["a_curve"]) == 3
    assert len(result["b_curve"]) == 3
    assert 0.0 <= result["ausd_a"] <= 1.0
    assert 0.0 <= result["ausd_b"] <= 1.0
    # the JSONL is reloadable
    line = out.read_text(encoding="utf-8").splitlines()[0]
    assert json.loads(line)["target_model_id"] == "fake-target"
