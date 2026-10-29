"""被测回复复读（退化）检测 + d5 事后封顶。零 GPU，读原始 jsonl（dict），不过 pydantic，打分文件原样回写。

规则（2026-10-01 用户拍板，医师现场同一句话）：**本轮回复里同一整句原样出现 ≥3 次 → d5 最高 1 分。**
  * 句 = 按 。！？!? 和换行切开、去首尾空白后 ≥8 字的片段（逗号、引号、空格都算进长度）；要逐字相同才算同一句。
  * 只看本轮；本轮重复助手自己前文的话不在此列（模板化重复，原 d5 判据管）。
  * 判官提示词和判例卡都不动 —— 封顶由本模块事后统一套在各版判官分上（`cap`），人和机器用同一个定义。

为什么「抄回求助者原话」不进封顶（`echoes_patient` 只作描述性标记）：v2 全语料 20 字回显命中 78 条，
cpsycoun 的 15 条全部同时是复读 ≥3（已被封顶），其余 63 条是 claude / deepseek / soulchat / qwen3 的
「你说『……』」式引用复述 —— 好的咨询做法，封顶等于误伤。

v2 全语料（`runs/v2/dialogues_both_arms.jsonl`）对数：复读 cpsycoun 365/955、qwen3-8b-base 24、soulchat 1、其余 0；
180 段本轮命中 9 段（dev 2/58/129/138/159，test 64/85/143/147）。

    python -m cascade.analysis.repetition report --in runs/v2/dialogues_both_arms.jsonl --out reports/repetition_v2 \
        [--key docs/collaboration/判官金标-180段_key.jsonl]
    python -m cascade.analysis.repetition cap --in runs/v2/scored_final.jsonl --out runs/v2/scored_final_d5cap.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

SENT_SPLIT = re.compile(r"[。！？!?\n]")
LOOP_MIN_CHARS = 8
LOOP_MIN_REPEATS = 3
ECHO_CHARS = 20
D5_CAP = 1
# 末字是汉字 / 字母数字 / 逗号顿号 → 断在半句。以表情、markdown、引号收尾的不算。
_MID_END = re.compile(r"[一-鿿A-Za-z0-9，、,]$")
# 疑似撞 max_tokens（2048 token ≈ 3000+ 字）：长且断在半句。不加长度门槛时 minimax 的口语短回复
# （中位 33 字、常不带句末标点）会误报 48 条；加了之后只剩 cpsycoun 293、qwen3-8b-base 6。
TRUNC_MIN_CHARS = 2000


# ---------------------------------------------------------------- 单条回复

def max_sentence_repeats(reply: str, min_chars: int = LOOP_MIN_CHARS) -> int:
    counts = Counter(s for s in (x.strip() for x in SENT_SPLIT.split(reply or "")) if len(s) >= min_chars)
    return max(counts.values(), default=0)


def is_loop(reply: str) -> bool:
    return max_sentence_repeats(reply) >= LOOP_MIN_REPEATS


def echoes_patient(reply: str, patient_utterances, n: int = ECHO_CHARS) -> bool:
    """回复里逐字出现求助者任一轮发言（可见历史，含本轮）的某个 n 字连续片段。只作描述，不进封顶。"""
    reply = reply or ""
    for p in patient_utterances:
        p = (p or "").strip()
        if any(p[i:i + n] in reply for i in range(len(p) - n + 1)):
            return True
    return False


def ends_mid_sentence(reply: str) -> bool:
    return bool(_MID_END.search((reply or "").rstrip()))


def looks_truncated(reply: str) -> bool:
    return len(reply or "") >= TRUNC_MIN_CHARS and ends_mid_sentence(reply)


def cap_d5(d5, reply: str):
    if d5 is None or not is_loop(reply):
        return d5
    return min(d5, D5_CAP)


# ---------------------------------------------------------------- 语料

def _records(path):
    with Path(path).open(encoding="utf-8-sig") as f:      # utf-8-sig：PowerShell 合并的 jsonl 可能带 BOM
        for line in f:
            if line.strip():
                yield json.loads(line)


def _base_id(rec: dict) -> str:
    """与 180 段 key / merge_judge180_gold 同一种 id：model|arm|profile#tK。"""
    return f'{rec["target_model_id"]}|{rec["arm"]}|{rec["case_profile"]["id"]}'


def scan_record(rec: dict) -> list[dict]:
    base, pats, out, prev = _base_id(rec), [], [], None
    for t in rec.get("turns", []):
        pats.append(t.get("patient_utterance") or "")
        reply = t.get("target_response") or ""
        loop = is_loop(reply)
        out.append({"id": f'{base}#t{t["turn"]}', "model": rec["target_model_id"], "turn": t["turn"],
                    "loop": loop, "prev_loop": prev, "echo": echoes_patient(reply, pats),
                    "truncated": looks_truncated(reply), "chars": len(reply)})
        prev = loop
    return out


def loop_turn_ids(path) -> set[str]:
    return {r["id"] for rec in _records(path) for r in scan_record(rec) if r["loop"]}


def summarize(path) -> dict:
    s = defaultdict(lambda: {"n": 0, "loop": 0, "echo": 0, "echo_not_loop": 0, "truncated": 0,
                             "first_turns": 0, "first_turn_loop": 0,
                             "loop_after_loop": [0, 0], "loop_after_clean": [0, 0], "chars_loop": [], "chars_clean": []})
    for rec in _records(path):
        for r in scan_record(rec):
            m = s[r["model"]]
            m["n"] += 1
            m["loop"] += r["loop"]
            m["echo"] += r["echo"]
            m["echo_not_loop"] += r["echo"] and not r["loop"]
            m["truncated"] += r["truncated"]
            m["chars_loop" if r["loop"] else "chars_clean"].append(r["chars"])
            if r["prev_loop"] is None:
                m["first_turns"] += 1
                m["first_turn_loop"] += r["loop"]
            else:
                k = "loop_after_loop" if r["prev_loop"] else "loop_after_clean"
                m[k][0] += r["loop"]
                m[k][1] += 1
    for m in s.values():
        for k in ("chars_loop", "chars_clean"):
            v = sorted(m.pop(k))
            m["median_" + k] = v[len(v) // 2] if v else None
    return dict(s)


def loop_d5_counts(scored_path) -> dict:
    """复读轮上判官给的 d5 分布 {0: n, 1: n, 2: n}；2 的那些就是封顶会改的。"""
    c = Counter()
    for rec in _records(scored_path):
        for t in rec.get("turns", []):
            sc = t.get("score")
            if sc and sc.get("d5") is not None and is_loop(t.get("target_response") or ""):
                c[sc["d5"]] += 1
    return {k: c.get(k, 0) for k in (0, 1, 2)}


def cap_file(src, dst) -> dict:
    """逐轮套 d5 封顶，其余字段原样写回。返回 {turns, scored, loop, capped}。"""
    st = Counter(turns=0, scored=0, loop=0, capped=0)
    with Path(dst).open("w", encoding="utf-8") as f:
        for rec in _records(src):
            for t in rec.get("turns", []):
                st["turns"] += 1
                reply = t.get("target_response") or ""
                loop = is_loop(reply)
                st["loop"] += loop
                sc = t.get("score")
                if sc is None:
                    continue
                st["scored"] += 1
                new = cap_d5(sc.get("d5"), reply)
                if new != sc.get("d5"):
                    sc["d5"] = new
                    st["capped"] += 1
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return dict(st)


# ---------------------------------------------------------------- 报告

def _pct(a, b):
    return f"{a}/{b} ({a / b:.1%})" if b else "—"


def render(summary: dict, segs: list | None, src: str, judges: dict | None = None) -> str:
    L = ["# 被测回复复读（退化）检测", "",
         f"- 语料：`{src}`",
         f"- 复读 = 本轮同一整句（切句 `。！？!?\\n`、去首尾空白后 ≥{LOOP_MIN_CHARS} 字、逐字相同）出现 ≥{LOOP_MIN_REPEATS} 次。"
         f"d5 封顶 = 复读轮 d5 最高 {D5_CAP}（2026-10-01 用户定，医师同一句规则）。",
         f"- 回显 = 回复逐字含求助者任一轮发言的 {ECHO_CHARS} 字片段 —— **只作描述，不进封顶**（多数是正常引用复述）。",
         f"- 疑似撞上限 = ≥{TRUNC_MIN_CHARS} 字且末字是汉字 / 字母数字 / 逗号（启发式；被测 max_tokens 2048）。", "",
         "| 被测 | 轮数 | 复读 | 首轮复读 | 上轮复读→本轮复读 | 上轮未复读→本轮复读 | 疑似撞上限 | 回显 | 回显且非复读 | 字数中位（复读 / 非复读） |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for m in sorted(summary, key=lambda k: -summary[k]["loop"] / max(summary[k]["n"], 1)):
        x = summary[m]
        L.append(f'| {m} | {x["n"]} | {_pct(x["loop"], x["n"])} | {_pct(x["first_turn_loop"], x["first_turns"])} | '
                 f'{_pct(*x["loop_after_loop"])} | {_pct(*x["loop_after_clean"])} | {_pct(x["truncated"], x["n"])} | '
                 f'{x["echo"]} | {x["echo_not_loop"]} | {x["median_chars_loop"]} / {x["median_chars_clean"]} |')
    if judges:
        L += ["", "## 各版判官在复读轮上给的 d5（封顶只改 d5 = 2 的那些）", "",
              "| 判官 | 复读轮 | d5 = 0 | d5 = 1 | d5 = 2（会被封顶） |", "|---|---|---|---|---|"]
        for name, c in judges.items():
            L.append(f"| {name} | {sum(c.values())} | {c[0]} | {c[1]} | {c[2]} |")
    if segs is not None:
        L += ["", "## 180 段：本轮复读的段（d5 封顶会改到的段）", "",
              f"共 {len(segs)} 段。" + "；".join(f'{sp}：{", ".join(str(s) for s, p, _ in segs if p == sp) or "—"}'
                                              for sp in ("dev", "test")), "",
              "| 段号 | split | 被测 |", "|---|---|---|"]
        L += [f"| {s} | {p} | {t} |" for s, p, t in segs]
    return "\n".join(L) + "\n"


def key_segments(key_path, loops: set) -> list:
    """[(段号, split, 被测)]，段号 = key 行号（同 merge_judge180_gold）。"""
    rows = [json.loads(l) for l in Path(key_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    return [(i, r.get("split"), r.get("target")) for i, r in enumerate(rows, 1) if r["id"] in loops]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="被测回复复读检测 + d5 事后封顶")
    sub = ap.add_subparsers(dest="cmd", required=True)
    rp = sub.add_parser("report")
    rp.add_argument("--in", dest="src", required=True)
    rp.add_argument("--out", required=True)
    rp.add_argument("--key", default=None, help="180 段 key（只读 id / split / target，不读机器分）")
    rp.add_argument("--scored", nargs="*", default=[], metavar="NAME=PATH", help="判官打分文件，报复读轮 d5 分布")
    cp = sub.add_parser("cap")
    cp.add_argument("--in", dest="src", required=True)
    cp.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "cap":
        if Path(a.src).resolve() == Path(a.out).resolve():
            ap.error("--out 不能覆盖 --in（原始判官分要留着）")
        print(json.dumps(cap_file(a.src, a.out), ensure_ascii=False))
        return 0
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = summarize(a.src)
    segs = key_segments(a.key, loop_turn_ids(a.src)) if a.key else None
    judges = {n: loop_d5_counts(p) for n, p in (x.split("=", 1) for x in a.scored)}
    (out / "summary.json").write_text(json.dumps({"summary": summary, "segments": segs, "judges_loop_d5": judges},
                                                 ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "report.md").write_text(render(summary, segs, a.src, judges), encoding="utf-8")
    print(f"-> {out / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
