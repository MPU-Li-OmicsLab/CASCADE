"""判官 v7 信号缓存：<out>.signals.jsonl，一行一轮（spec §2）。

追加写（崩了不丢已抽的轮），读时同一个键后写的覆盖先写的。
键 = (轮次 id, 模型 id, EXTRACT_VERSION)：任一不同就视为没有缓存 —— 不会拿旧问题的信号套新规则。"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from cascade.v7.items import Signal

_KEYS = ("id", "model", "items_version", "signals", "d6")


def signals_to_json(sig: dict[str, Signal]) -> dict:
    return {k: {"present": s.present, "quote": s.quote, "quote_fail": s.quote_fail} for k, s in sorted(sig.items())}


def signals_from_json(d: dict) -> dict[str, Signal]:
    return {k: Signal(bool(v["present"]), v.get("quote") or "", bool(v.get("quote_fail", False))) for k, v in d.items()}


def make_record(turn_id: str, model: str, items_version: str, signals: dict[str, Signal], d6: dict) -> dict:
    return {"id": turn_id, "model": model, "items_version": items_version,
            "signals": signals_to_json(signals),
            "d6": {"score": int(d6["score"]), "reason": str(d6.get("reason") or "")}}


class SignalStore:
    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._rows: dict[tuple[str, str, str], dict] = {}
        self._order: list[dict] = []
        if self.path.exists():
            with self.path.open(encoding="utf-8-sig") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue                      # 崩在写一半：跳过残行
                    if (isinstance(rec, dict) and all(k in rec for k in _KEYS)
                            and isinstance(rec["d6"], dict) and "score" in rec["d6"]):
                        self._add(rec)

    def _add(self, rec: dict) -> None:
        self._rows[(rec["id"], rec["model"], rec["items_version"])] = rec
        self._order.append(rec)

    def get(self, turn_id: str, model: str, items_version: str) -> dict | None:
        return self._rows.get((turn_id, model, items_version))

    def put(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as f:
                if self._needs_newline():
                    f.write("\n")                   # 崩溃留下无换行的残行：先补行尾，免得下一条被粘在残行后面一起丢
                f.write(line + "\n")
            self._add(record)

    def _needs_newline(self) -> bool:
        try:
            with self.path.open("rb") as f:
                f.seek(0, 2)
                if f.tell() == 0:
                    return False
                f.seek(-1, 2)
                return f.read(1) != b"\n"
        except OSError:
            return False

    def latest_by_id(self, model: str | None = None) -> dict[str, dict]:
        out: dict[str, dict] = {}
        with self._lock:
            snapshot = list(self._order)
        for rec in snapshot:
            if model is None or rec["model"] == model:
                out[rec["id"]] = rec
        return out
