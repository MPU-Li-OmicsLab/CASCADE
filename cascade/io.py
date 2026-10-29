# cascade/io.py
from __future__ import annotations
from pathlib import Path
from cascade.schema import DialogueRecord


def write_jsonl(records: list[DialogueRecord], path: str | Path) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(r.model_dump_json())
            f.write("\n")


def read_jsonl(path: str | Path) -> list[DialogueRecord]:
    path = Path(path)
    out: list[DialogueRecord] = []
    # utf-8-sig：Windows PowerShell 5.1 的 `Set-Content -Encoding utf8` 写的是
    # UTF-8 **with BOM**，合并两臂 jsonl 时那个 ﻿ 会顶在首行行首、让 pydantic
    # 直接报 json_invalid。utf-8-sig 有 BOM 就吃掉、没有就等同 utf-8，无副作用。
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(DialogueRecord.model_validate_json(line))
    return out


def append_jsonl(record: DialogueRecord, path: str | Path) -> None:
    """Append a single record as one line (creating the file if absent).
    Streaming counterpart to write_jsonl for the resumable batch runner."""
    path = Path(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(record.model_dump_json())
        f.write("\n")
