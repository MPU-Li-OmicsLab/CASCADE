"""SignalStore：崩溃残行后追加不能粘住下一条（终审 M5）。"""
from cascade.v7.items import ITEM_IDS, Signal
from cascade.v7.store import SignalStore, make_record


def _rec(tid):
    return make_record(tid, "j", "v", {i: Signal(False) for i in ITEM_IDS}, {"score": 2, "reason": ""})


def test_put_after_half_line_does_not_glue(tmp_path):
    p = tmp_path / "s.jsonl"
    SignalStore(p).put(_rec("a"))
    with p.open("a", encoding="utf-8") as f:
        f.write('{"id": "half", "mod')                      # 崩在写一半，无换行
    s = SignalStore(p)
    s.put(_rec("b"))
    again = SignalStore(p)
    assert again.get("a", "j", "v") and again.get("b", "j", "v")
    assert again.get("half", "j", "v") is None
