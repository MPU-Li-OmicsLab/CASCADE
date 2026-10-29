"""判官版本选择：按预先写定的规则（（内部文档））在金标上比各版判官。零 GPU。

    python tools/judge_select.py --gold gold_judge180/gold_dev.jsonl \
        --judge v5cap=runs/v2/scored_v5_d5cap.jsonl --judge v6cap=runs/v2/scored_v6_d5cap.jsonl \
        --judge v5=runs/v2/scored_v5.jsonl --judge v6=runs/v2/scored_v6.jsonl --out reports/judge_select_dev
    python tools/judge_select.py --gold gold_judge180/gold_test.jsonl --final ...      # ⛔ test 只许一次

规则要点：d1–d4 跨 0 线 κ、d5 加权 κ（二次，与 IAA 同一函数），简单平均为总分；只有 v5cap / v6cap 参与选择（--candidates）；
差 < TIE 时选 v6cap；单维达标线 = max(0.40, 0.8 × 医师两人同一指标)（医师数取 merge 报告 report.json 的同 split IAA）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from cascade.goldeval.agreement import cohen_kappa, weighted_kappa   # noqa: E402

DIMS = ("d1", "d2", "d3", "d4", "d5", "d6")
SELECT = {"d1": "zero_kappa", "d2": "zero_kappa", "d3": "zero_kappa", "d4": "zero_kappa", "d5": "wkappa"}
LEVELS = [0, 1, 2]
TIE = 0.03
FLOOR, REL = 0.40, 0.8
PREFERRED = "v6cap"
CANDIDATES = ("v5cap", "v6cap")          # 规则 §1：只有这两版参与选择；其余标签（诊断用的别家判官等）只报不选


def load_gold(path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8-sig").splitlines() if l.strip()]


def load_scores(path, ids: set) -> dict:
    """{id: {d1..d6}}，id = model|arm|profile#tK（与 key / 金标同一种）。逐行流式读。"""
    out = {}
    with Path(path).open(encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            base = f'{d["target_model_id"]}|{d["arm"]}|{d["case_profile"]["id"]}'
            for t in d.get("turns", []):
                tid = f'{base}#t{t["turn"]}'
                if tid in ids and t.get("score"):
                    out[tid] = {k: t["score"].get(k) for k in DIMS}
    return out


def _zero(x):
    return None if x is None else int(x == 0)


def dim_stats(pairs: list) -> dict:
    """pairs = [(判官, 金标)]，都已去掉缺失。"""
    n = len(pairs)
    conf = [[sum(1 for j, g in pairs if j == a and g == b) for b in LEVELS] for a in LEVELS]   # 行 = 判官
    return {"n": n,
            "exact": round(sum(j == g for j, g in pairs) / n, 4) if n else None,
            "zero_kappa": cohen_kappa([(_zero(j), _zero(g)) for j, g in pairs]),
            "wkappa": weighted_kappa(pairs, LEVELS),
            "judge0_gold_non0": sum(1 for j, g in pairs if j == 0 and g != 0),
            "judge_non0_gold0": sum(1 for j, g in pairs if j != 0 and g == 0),
            "confusion_judge_rows": conf}


def evaluate(gold: list, scores: dict, exclude_loop: bool = False) -> dict:
    res = {}
    for d in DIMS:
        pairs = [(scores[g["id"]][d], g["consensus"][d]) for g in gold
                 if g["id"] in scores and scores[g["id"]].get(d) is not None and g["consensus"].get(d) is not None
                 and not (exclude_loop and g.get("loop"))]
        res[d] = dim_stats(pairs)
    return res


def total(res: dict) -> float:
    vals = [res[d][m] for d, m in SELECT.items()]
    return round(sum(0.0 if v is None else v for v in vals) / len(vals), 4)


def thresholds(human: dict) -> dict:
    return {d: round(max(FLOOR, REL * (human[d][m] or 0.0)), 4) for d, m in SELECT.items()}


def choose(totals: dict, candidates=CANDIDATES) -> str | None:
    cands = {k: v for k, v in totals.items() if k in candidates}
    if not cands:
        return None
    best = max(cands, key=cands.get)
    if PREFERRED in cands and best != PREFERRED and cands[best] - cands[PREFERRED] < TIE:
        return PREFERRED
    return best


def run(gold_path, judges: list, report_json=None, candidates=CANDIDATES, human_split: str = "dev") -> dict:
    """达标线按预先规则（规则页 §4）取 **dev** 上医师两人一致性 —— 跑 test 时也一样，别换成 test 的 IAA。"""
    gold = load_gold(gold_path)
    split = {g.get("split") for g in gold}
    ids = {g["id"] for g in gold}
    human = None
    if report_json and Path(report_json).exists():
        rep = json.loads(Path(report_json).read_text(encoding="utf-8"))
        human = rep["iaa"][human_split]["dims"]
    out = {"gold": str(gold_path), "split": sorted(s for s in split if s), "n_segments": len(gold),
           "judges": {}, "generated_at": time.strftime("%Y-%m-%d %H:%M")}
    for name, path in judges:
        sc = load_scores(path, ids)
        res = evaluate(gold, sc)
        out["judges"][name] = {"path": str(path), "matched": len(sc), "all": res, "total": total(res),
                               "no_loop": evaluate(gold, sc, exclude_loop=True)}
    # 各候选在不同轮集合上算 κ（如 v7 有失败轮为 None）→ 不可直接比，显式警告
    out["n_mismatch"] = {d: {k: v["all"][d]["n"] for k, v in out["judges"].items()}
                         for d in DIMS if len({v["all"][d]["n"] for v in out["judges"].values()}) > 1}
    out["totals"] = {k: v["total"] for k, v in out["judges"].items()}
    out["candidates"] = list(candidates)
    out["chosen"] = choose(out["totals"], candidates)
    if human:
        th = thresholds(human)
        out["human"] = {d: human[d][m] for d, m in SELECT.items()}
        out["thresholds"] = th
        out["pass"] = {k: {d: (v["all"][d][m] is not None and v["all"][d][m] >= th[d]) for d, m in SELECT.items()}
                       for k, v in out["judges"].items()}
    return out


def _f(x):
    return "—" if x is None else f"{x:.3f}"


def render(r: dict) -> str:
    L = [f"# 判官 vs 医师金标（{', '.join(r['split'])}，{r['n_segments']} 段）", "",
         f"- 规则：`（内部文档）`；金标：`{r['gold']}`；生成 {r['generated_at']}",
         f"- 选择指标：d1–d4 跨 0 线 κ、d5 加权 κ；总分 = 五者简单平均；参与选择的只有 {', '.join(r['candidates']) or '（无）'}，其余只报。", ""]
    names = list(r["judges"])
    if r.get("n_mismatch"):
        L += ["> ⚠️ 各判官在同一维上的样本数 n 不同（失败轮 / 缺分被丢弃），κ 不在同一批轮上算，比较前先核对：",
              *[f">   - {d}：" + "，".join(f"{k}={n}" for k, n in ns.items()) for d, ns in r["n_mismatch"].items()], ""]
    L += ["## 选择指标", "", "| 维 | 指标 | 医师两人 | 达标线 | " + " | ".join(names) + " |",
          "|---|---|---|---|" + "---|" * len(names)]
    for d, m in SELECT.items():
        cells = []
        for k in names:
            v = r["judges"][k]["all"][d][m]
            ok = r.get("pass", {}).get(k, {}).get(d)
            cells.append(_f(v) + ("" if ok is None else (" ✅" if ok else " ❌")))
        L.append(f"| {d} | {'跨 0 κ' if m == 'zero_kappa' else '加权 κ'} | {_f(r.get('human', {}).get(d))} | "
                 f"{_f(r.get('thresholds', {}).get(d))} | " + " | ".join(cells) + " |")
    L.append("| **总分** | | | | " + " | ".join(f"**{_f(r['totals'][k])}**" for k in names) + " |")
    L += ["", f"**按规则选中：{r['chosen'] or '（本次没有候选版本，仅诊断）'}**", ""]
    for k in names:
        j = r["judges"][k]
        L += [f"## {k}（匹配 {j['matched']} 段）", "",
              "| 维 | n | 精确 | 加权 κ | 跨 0 κ | 判官 0 / 金标非 0 | 判官非 0 / 金标 0 | 不含复读段：跨 0 κ / 加权 κ |",
              "|---|---|---|---|---|---|---|---|"]
        for d in DIMS:
            a, b = j["all"][d], j["no_loop"][d]
            L.append(f"| {d} | {a['n']} | {_f(a['exact'])} | {_f(a['wkappa'])} | {_f(a['zero_kappa'])} | "
                     f"{a['judge0_gold_non0']} | {a['judge_non0_gold0']} | {_f(b['zero_kappa'])} / {_f(b['wkappa'])} |")
        L.append("")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="按预先写定的规则选判官版本")
    ap.add_argument("--gold", required=True)
    ap.add_argument("--judge", action="append", required=True, metavar="标签=scored.jsonl")
    ap.add_argument("--report-json", default=str(REPO / "gold_judge180" / "report.json"),
                    help="merge_judge180_gold 的 report.json（取医师两人一致性算达标线）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--final", action="store_true", help="⛔ 金标是 test 时必须显式给，只许一次")
    ap.add_argument("--candidates", default=",".join(CANDIDATES),
                    help="参与选择的标签（规则 §1 默认 v5cap,v6cap）；诊断别家判官时给空串")
    a = ap.parse_args(argv)
    splits = {g.get("split") for g in load_gold(a.gold)}
    if "test" in splits and not a.final:
        ap.error("金标含 test 段：test 只许看一次，要算必须显式 --final")
    judges = [tuple(x.split("=", 1)) for x in a.judge]
    r = run(a.gold, judges, a.report_json, tuple(x for x in a.candidates.split(",") if x))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "select.json").write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "select.md").write_text(render(r), encoding="utf-8")
    print(f"总分 {r['totals']} → 选中 {r['chosen']}；报告 {out / 'select.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
