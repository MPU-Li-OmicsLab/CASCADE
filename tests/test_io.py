# tests/test_io.py
from cascade.io import write_jsonl, read_jsonl, append_jsonl
from cascade.schema import (CaseProfile, CSSRSStage, Language, Arm,
                            DialogueRecord, TurnRecord, CAppraisal)


def _record(rid: str) -> DialogueRecord:
    appr = CAppraisal(empathy_detected=True, deescalation_attempt=False,
                      abandonment_felt=False, c_new=0.6)
    tr = TurnRecord(turn=1, target_response="r", appraisal=appr,
                    patient_utterance="u", attack_stage=CSSRSStage.ACTIVE_IDEATION, c=0.6)
    return DialogueRecord(case_profile=CaseProfile(
        id=rid, seed_text="s", persona_profile="p",
        trajectory_stage_target=CSSRSStage.PLAN, strategies=[],
        language=Language.ZH, explicit=False),
        arm=Arm.FIXED, turns=[tr], target_model_id="m", seed=1)


def test_write_then_read_roundtrips(tmp_path):
    path = tmp_path / "out.jsonl"
    records = [_record("a"), _record("b")]
    write_jsonl(records, path)
    loaded = read_jsonl(path)
    assert [r.case_profile.id for r in loaded] == ["a", "b"]
    assert loaded[0] == records[0]


def test_file_has_one_line_per_record(tmp_path):
    path = tmp_path / "out.jsonl"
    write_jsonl([_record("a"), _record("b")], path)
    assert path.read_text(encoding="utf-8").strip().count("\n") == 1  # 2 lines -> 1 newline between


def test_append_jsonl_accumulates(tmp_path):
    path = tmp_path / "out.jsonl"
    append_jsonl(_record("a"), path)
    append_jsonl(_record("b"), path)
    assert [r.case_profile.id for r in read_jsonl(path)] == ["a", "b"]


def test_append_jsonl_creates_file(tmp_path):
    path = tmp_path / "new.jsonl"
    append_jsonl(_record("a"), path)
    assert path.exists()
    assert read_jsonl(path)[0].case_profile.id == "a"


def test_read_jsonl_tolerates_utf8_bom(tmp_path):
    """Windows PowerShell 5.1 的 `Set-Content -Encoding utf8` 写 BOM；
    实测合并两臂 dialogues.jsonl 后首行带 ﻿ → pydantic json_invalid。"""
    rec = _record("bom-1")
    p = tmp_path / "bom.jsonl"
    p.write_text("﻿" + rec.model_dump_json() + "\n", encoding="utf-8")
    assert p.read_bytes()[:3] == b"\xef\xbb\xbf"        # 确认真写进了 BOM
    got = read_jsonl(p)
    assert len(got) == 1 and got[0].case_profile.id == rec.case_profile.id
