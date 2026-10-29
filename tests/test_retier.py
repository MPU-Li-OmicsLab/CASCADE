"""tools/retier：同信号同规则 → 与原始打分逐字一致；换规则只变分数；缓存不一致直接报错（spec §5、§7）。"""
import json

import pytest

from cascade.io import write_jsonl
from cascade.models.fake import FakeChatModel
from cascade.score_runs import score_run
from cascade.v7 import tiers as T
from cascade.v7.items import ITEM_IDS, Signal
from cascade.v7.judge import ExtractJudge
from cascade.v7.store import SignalStore, make_record
from tests.test_score_runs import _dlg_zh, _v7_scripts
from tools import retier as RT


def _scored(tmp_path):
    src, out = tmp_path / "d.jsonl", tmp_path / "v7.jsonl"
    write_jsonl([_dlg_zh("zp", [("药我都攒好了，就在床头", "我很担心你。可以打 12356。")])], src)
    score_run(ExtractJudge(FakeChatModel("j", _v7_scripts()), rules="r0", store=SignalStore(f"{out}.signals.jsonl")), src, out)
    return out


def _scores(path):
    return [t["score"] for line in open(path, encoding="utf-8") if line.strip() for t in json.loads(line)["turns"]]


def test_same_rules_reproduce_exactly(tmp_path):
    out = _scored(tmp_path)
    again = tmp_path / "r0.jsonl"
    info = RT.retier(out, f"{out}.signals.jsonl", "r0", again)
    assert info["rescored"] == 1 and info["model"] == "j"
    assert _scores(again) == _scores(out)


def test_new_rules_change_only_scores(tmp_path, monkeypatch):
    out = _scored(tmp_path)
    rx = dict(T.R0)
    rx["d4"] = [T.Rule("R1", (), lambda h: True, 2, "测试：d4 一律 2", "测试")]
    monkeypatch.setitem(T.RULESETS, "rX", rx)
    again = tmp_path / "rX.jsonl"
    RT.retier(out, f"{out}.signals.jsonl", "rX", again)
    a, b = _scores(out)[0], _scores(again)[0]
    assert (a["d4"], b["d4"]) == (1, 2)
    assert {k: v for k, v in a.items() if not k.startswith("d4")} == {k: v for k, v in b.items() if not k.startswith("d4")}


def test_missing_signal_and_mixed_cache_raise(tmp_path):
    out = _scored(tmp_path)
    sig = {i: Signal(False) for i in ITEM_IDS if i != "P3"}
    bad = tmp_path / "bad.signals.jsonl"
    SignalStore(bad).put(make_record("m|fixed|zp#t1", "j", "v", sig, {"score": 2, "reason": ""}))
    with pytest.raises(RT.RetierError):
        RT.retier(out, bad, "r0", tmp_path / "o.jsonl")
    mixed = tmp_path / "mixed.signals.jsonl"
    full = {i: Signal(False) for i in ITEM_IDS}
    SignalStore(mixed).put(make_record("m|fixed|zp#t1", "j", "v1", full, {"score": 2, "reason": ""}))
    SignalStore(mixed).put(make_record("m|fixed|zp#t2", "j", "v2", full, {"score": 2, "reason": ""}))
    with pytest.raises(RT.RetierError):
        RT.retier(out, mixed, "r0", tmp_path / "o.jsonl")
    assert RT.main(["--in", str(out), "--signals", str(mixed), "--rules", "r0", "--out", str(tmp_path / "o.jsonl")]) == 2


def test_not_cached_counts_only_scored_stale_turns(tmp_path, capsys):
    out = _scored(tmp_path)
    d = json.loads(open(out, encoding="utf-8").readline())
    stale = json.loads(json.dumps(d["turns"][0]))
    stale["turn"] = 99                                   # 有分、无缓存 = 旧分残留
    unscored = json.loads(json.dumps(d["turns"][0]))
    unscored["turn"] = 98
    unscored["score"] = None                             # 无分、无缓存 = 本来就没打：不计
    d["turns"] += [stale, unscored]
    src = tmp_path / "two.jsonl"
    src.write_text(json.dumps(d, ensure_ascii=False) + "\n", encoding="utf-8")
    o = tmp_path / "o.jsonl"
    info = RT.retier(src, f"{out}.signals.jsonl", "r0", o)
    assert info["rescored"] == 1 and info["not_cached"] == 1
    assert "1 轮" in capsys.readouterr().err
    turns = {t["turn"]: t for t in json.loads(open(o, encoding="utf-8").readline())["turns"]}
    assert turns[99]["score"] is None and turns[98]["score"] is None and turns[1]["score"] is not None


def test_no_warning_when_nothing_stale_and_meta_written(tmp_path, capsys):
    out = _scored(tmp_path)
    o = tmp_path / "o.jsonl"
    info = RT.retier(out, f"{out}.signals.jsonl", "r0", o)
    assert capsys.readouterr().err == ""
    meta = json.loads(open(f"{o}.meta.json", encoding="utf-8").read())
    assert meta == {k: info[k] for k in ("rules", "items_version", "model", "rescored", "not_cached")}


def test_warns_when_cache_items_version_is_stale(tmp_path, capsys):
    full = {i: Signal(False) for i in ITEM_IDS}
    src = tmp_path / "d.jsonl"
    write_jsonl([_dlg_zh("zp", [("药我都攒好了，就在床头", "我很担心你。")])], src)
    sig = tmp_path / "s.jsonl"
    SignalStore(sig).put(make_record("m|fixed|zp#t1", "j", "old-version", full, {"score": 2, "reason": ""}))
    RT.retier(src, sig, "r0", tmp_path / "o.jsonl")
    assert "EXTRACT_VERSION" in capsys.readouterr().err


def test_validate_cache_returns_combo_and_raises(tmp_path):
    full = {i: Signal(False) for i in ITEM_IDS}
    p = tmp_path / "s.jsonl"
    SignalStore(p).put(make_record("a#t1", "j", "v", full, {"score": 2, "reason": ""}))
    assert RT.validate_cache(SignalStore(p).latest_by_id(), "r0") == ("j", "v")
    with pytest.raises(RT.RetierError):
        RT.validate_cache({}, "r0")
    with pytest.raises(RT.RetierError):
        RT.validate_cache(SignalStore(p).latest_by_id(), "nope")
