"""判官 v7：dev + test 合并 180 段上的嵌套交叉验证（规则层折内选，提示词固定 q4）。零模型调用。

    python tools/judge_cv.py \
        --gold gold_judge180/gold_dev.jsonl --gold gold_judge180/gold_test.jsonl \
        --signals runs/v2/v7_dev_qwen38_q4.jsonl.signals.jsonl --signals runs/v2/v7_test_qwen38.jsonl.signals.jsonl \
        --dialogues runs/v2/dialogues_gold180.jsonl --out reports/judge_cv_180

做法（2026-10-13 定，test 已于 d124634a 一次性开封、结果原样报告在先）：
  * 折按**人设**分组（同一人设的轮次只落在一折），K 折、重复 R 次换种子；
  * 每折只在训练部分按维选规则集（r0–r6 里该维指标最高者；并列取定版 r6），在留出折上出分；
  * 折外预测拼成 180 段算 κ（各折 0 分太少，逐折 κ 不稳），R 次重复报中位与范围；
  * 同时报定版 r6 直接套 180 段（无选择）作对照，以及两者的人设聚类自助法 CI；
  * 每维「失败」的灵敏度 / 特异度（d1–d4 = 判 0，d5 = 判 ≤1 —— d5 金标 0 分仅 2 段）供误判校正用。
⚠️ q4 提示词是在 dev 113 段上开发的 → 本工具只消掉规则层的选择偏差，提示词层仍偏乐观，论文须写明。"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from cascade.v7.judge import build_score                    # noqa: E402
from cascade.v7.store import SignalStore, signals_from_json  # noqa: E402
from cascade.v7.tiers import RULESETS                        # noqa: E402
from tools.judge_select import FLOOR, REL, SELECT, dim_stats, load_gold  # noqa: E402

DIMS5 = tuple(SELECT)
FROZEN = "r6"
TIE = 1e-9


def dialogue_of(tid: str) -> str:
    return tid.split("#", 1)[0]


def persona_of(tid: str) -> str:
    return dialogue_of(tid).split("|")[-1].split("-", 1)[0]


GROUPS = {"persona": persona_of, "dialogue": dialogue_of}


# ---------------------------------------------------------------- 数据
def load_cache(paths) -> dict:
    """多个信号缓存 → {id: 记录}；必须是同一 (模型, 抽取版本)。"""
    out = {}
    for p in paths:
        if not Path(p).exists():
            raise SystemExit(f"信号缓存不存在：{p}（服务器上跑完后等 Resilio 同步完整再来）")
        out.update(SignalStore(p).latest_by_id())
    combos = {(r["model"], r["items_version"]) for r in out.values()}
    if len(combos) != 1:
        raise SystemExit(f"信号缓存混着多个 模型 / 抽取版本：{sorted(combos)}")
    return out


def load_replies(path, ids: set) -> dict:
    out = {}
    with Path(path).open(encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            base = f'{d["target_model_id"]}|{d["arm"]}|{d["case_profile"]["id"]}'
            for t in d.get("turns", []):
                tid = f'{base}#t{t["turn"]}'
                if tid in ids:
                    out[tid] = t.get("target_response") or ""
    return out


def score_table(cache: dict, replies: dict, ids, rulesets) -> dict:
    """{规则集: {id: {d1..d5}}}，与线上打分同一个 build_score（含 d5 复读 / 短回复封顶）。"""
    table = {}
    for rs in rulesets:
        table[rs] = {}
        for i in ids:
            rec = cache[i]
            s = build_score(signals_from_json(rec["signals"]), rec["d6"], replies[i], rs)
            table[rs][i] = {d: getattr(s, d) for d in DIMS5}
    return table


# ---------------------------------------------------------------- 折
def group_folds(ids, k: int, seed: int, key) -> dict:
    """分组 K 折：同组只落一折；组按随机序、大组先放、放进当前最小的折。"""
    groups = defaultdict(list)
    for i in sorted(ids):
        groups[key(i)].append(i)
    if len(groups) < k:
        raise ValueError(f"只有 {len(groups)} 组，不够分 {k} 折")
    names = sorted(groups)
    random.Random(seed).shuffle(names)
    names.sort(key=lambda g: -len(groups[g]))           # 稳定排序：同样大小保持随机序
    size, out = [0] * k, {}
    for g in names:
        f = min(range(k), key=lambda j: (size[j], j))
        size[f] += len(groups[g])
        for i in groups[g]:
            out[i] = f
    return out


# ---------------------------------------------------------------- 指标
def _metric(pairs, d):
    return dim_stats(pairs)[SELECT[d]] if pairs else None


def _fail(x, d) -> bool:
    return x <= 1 if d == "d5" else x == 0


def failure_rates(pairs, d) -> dict:
    """判官「失败」相对医师「失败」的灵敏度 / 特异度（d1–d4 失败 = 0；d5 失败 = ≤1）。"""
    gf = [(j, g) for j, g in pairs if _fail(g, d)]
    go = [(j, g) for j, g in pairs if not _fail(g, d)]
    sens = round(sum(_fail(j, d) for j, _ in gf) / len(gf), 4) if gf else None
    spec = round(sum(not _fail(j, d) for j, _ in go) / len(go), 4) if go else None
    return {"n_gold_fail": len(gf), "n_gold_ok": len(go), "sens": sens, "spec": spec}


def _pairs(pred, gold, ids, d):
    return [(pred[i][d], gold[i][d]) for i in ids if gold[i].get(d) is not None and pred[i].get(d) is not None]


def evaluate(pred: dict, gold: dict, ids) -> dict:
    dims = {}
    for d in DIMS5:
        p = _pairs(pred, gold, ids, d)
        st = dim_stats(p)
        dims[d] = {"n": st["n"], "metric": st[SELECT[d]], "zero_kappa": st["zero_kappa"], "wkappa": st["wkappa"],
                   **failure_rates(p, d)}
    vals = [dims[d]["metric"] for d in DIMS5]
    return {"dims": dims, "total": round(sum(0.0 if v is None else v for v in vals) / len(vals), 4)}


# ---------------------------------------------------------------- 选择与交叉验证
def choose_rules(train_ids, table: dict, gold: dict, rulesets) -> dict:
    """每维取训练部分指标最高的规则集；并列（含都为 None）时取定版 r6，再取较新的。"""
    pref = {rs: (rs == FROZEN, k) for k, rs in enumerate(rulesets)}
    out = {}
    for d in DIMS5:
        def key(rs):
            v = _metric(_pairs(table[rs], gold, train_ids, d), d)
            return (-1.0 if v is None else round(v, 9), *pref[rs])
        out[d] = max(rulesets, key=key)
    return out


def cross_validate(ids, table, gold, *, k: int, seed: int, key, rulesets):
    folds = group_folds(ids, k, seed, key)
    oof, choices = {}, []
    for f in range(k):
        train = [i for i in ids if folds[i] != f]
        hold = [i for i in ids if folds[i] == f]
        ch = choose_rules(train, table, gold, rulesets)
        choices.append(ch)
        for i in hold:
            oof[i] = {d: table[ch[d]][i][d] for d in DIMS5}
    return oof, choices


def bootstrap(pred, gold, ids, *, n_boot: int, seed: int, key=persona_of) -> dict:
    """按人设整组有放回重抽 → 每维指标 / 灵敏度 / 特异度与总分的 2.5–97.5% 分位。"""
    groups = defaultdict(list)
    for i in ids:
        groups[key(i)].append(i)
    names = sorted(groups)
    rng = random.Random(seed)
    acc = defaultdict(list)
    for _ in range(n_boot):
        sample = [i for g in (rng.choice(names) for _ in names) for i in groups[g]]
        ev = evaluate(pred, gold, sample)
        acc[("total",)].append(ev["total"])
        for d in DIMS5:
            for m in ("metric", "sens", "spec"):
                if ev["dims"][d][m] is not None:
                    acc[(d, m)].append(ev["dims"][d][m])

    def q(xs):
        if not xs:
            return None
        xs = sorted(xs)
        return (round(xs[int(0.025 * (len(xs) - 1))], 4), round(xs[int(0.975 * (len(xs) - 1))], 4))
    out = {d: {m: q(acc[(d, m)]) for m in ("metric", "sens", "spec")} for d in DIMS5}
    out["total"] = q(acc[("total",)])
    return out


# ---------------------------------------------------------------- 主流程
def run(gold_paths, signal_paths, dialogues, *, k=5, repeats=20, seed=0, group="persona",
        n_boot=2000, rulesets=None, report_json=None) -> dict:
    rulesets = list(rulesets or RULESETS)
    rows = [g for p in gold_paths for g in load_gold(p)]
    ids = [g["id"] for g in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("金标里有重复 id（dev / test 文件给重了？）")
    gold = {g["id"]: g["consensus"] for g in rows}
    split = {g["id"]: g.get("split") for g in rows}
    cache = load_cache(signal_paths)
    miss = [i for i in ids if i not in cache]
    if miss:
        raise SystemExit(f"{len(miss)} 段金标在信号缓存里找不到（如 {miss[:3]}）；先补抽或同步")
    replies = load_replies(dialogues, set(ids))
    table = score_table(cache, replies, ids, rulesets)
    key = GROUPS[group]

    frozen = table[FROZEN]
    out = {"generated_at": time.strftime("%Y-%m-%d %H:%M"), "gold": [str(p) for p in gold_paths],
           "signals": [str(p) for p in signal_paths], "n": len(ids), "n_personas": len({persona_of(i) for i in ids}),
           "k": k, "repeats": repeats, "group": group, "rulesets": rulesets,
           "extract": sorted({(r["model"], r["items_version"]) for r in cache.values()})[0],
           "frozen": {"all": evaluate(frozen, gold, ids),
                      "by_split": {s: evaluate(frozen, gold, [i for i in ids if split[i] == s])
                                   for s in sorted({v for v in split.values() if v})},
                      "ci": bootstrap(frozen, gold, ids, n_boot=n_boot, seed=seed)}}

    reps, picks = [], Counter()
    for r in range(repeats):
        oof, choices = cross_validate(ids, table, gold, k=k, seed=seed + r, key=key, rulesets=rulesets)
        reps.append((evaluate(oof, gold, ids), oof))
        for ch in choices:
            for d, rs in ch.items():
                picks[(d, rs)] += 1
    totals = [e["total"] for e, _ in reps]
    med = sorted(range(repeats), key=lambda j: (totals[j], j))[(repeats - 1) // 2]
    out["cv"] = {
        "median_repeat": med, "median": reps[med][0],
        "range": {"total": (min(totals), max(totals)),
                  **{d: (min(e["dims"][d]["metric"] for e, _ in reps), max(e["dims"][d]["metric"] for e, _ in reps))
                     for d in DIMS5}},
        "ci": bootstrap(reps[med][1], gold, ids, n_boot=n_boot, seed=seed),
        "picks": {d: {rs: picks[(d, rs)] for rs in rulesets if picks[(d, rs)]} for d in DIMS5}}
    if report_json and Path(report_json).exists():
        iaa = json.loads(Path(report_json).read_text(encoding="utf-8"))["iaa"]
        out["human"] = {s: {d: iaa[s]["dims"][d][m] for d, m in SELECT.items()} for s in ("dev", "all") if s in iaa}
        out["thresholds"] = {d: round(max(FLOOR, REL * (out["human"]["dev"][d] or 0.0)), 4) for d in DIMS5}
    return out


def _f(x):
    return "—" if x is None else f"{x:.3f}"


def _ci(t):
    return "" if not t else f" [{t[0]:.2f}, {t[1]:.2f}]"


def render(r: dict) -> str:
    fz, cv = r["frozen"], r["cv"]
    L = [f"# 判官 v7 · 金标合并嵌套交叉验证（{r['n']} 段 / {r['n_personas']} 人设）", "",
         f"- 生成 {r['generated_at']}；抽取 {r['extract'][0]} / `{r['extract'][1]}`；金标 {', '.join(r['gold'])}",
         f"- {r['k']} 折按**{ '人设' if r['group'] == 'persona' else '对话'}**分组，重复 {r['repeats']} 次；"
         f"每折在训练部分按维从 {', '.join(r['rulesets'])} 里选规则，折外预测拼成 {r['n']} 段算指标；CI = 人设聚类自助法。",
         "- ⚠️ test 67 段已于 `d124634a` 一次性开封（定版总分 0.340），本页在其后做、不替代那个数。"
         "q4 提示词是在 dev 上开发的 → 本页只消掉规则层的选择偏差，提示词层仍偏乐观。", "",
         "## 指标（d1–d4 跨 0 κ，d5 加权 κ）", "",
         "| 维 | 医师两人（180） | 达标线（dev 预设） | 定版 r6 直接套 180 | 交叉验证（中位重复） | 重复范围 |",
         "|---|---|---|---|---|---|"]
    for d in DIMS5:
        L.append(f"| {d} | {_f(r.get('human', {}).get('all', {}).get(d))} | {_f(r.get('thresholds', {}).get(d))} | "
                 f"{_f(fz['all']['dims'][d]['metric'])}{_ci(fz['ci'][d]['metric'])} | "
                 f"{_f(cv['median']['dims'][d]['metric'])}{_ci(cv['ci'][d]['metric'])} | "
                 f"{_f(cv['range'][d][0])}–{_f(cv['range'][d][1])} |")
    L.append(f"| **总分** | | | **{_f(fz['all']['total'])}**{_ci(fz['ci']['total'])} | "
             f"**{_f(cv['median']['total'])}**{_ci(cv['ci']['total'])} | {_f(cv['range']['total'][0])}–{_f(cv['range']['total'][1])} |")
    L += ["", "## 失败判定的灵敏度 / 特异度（供误判校正；d1–d4 失败 = 0，d5 失败 = ≤1）", "",
          "| 维 | 医师判失败 / 未失败 | 定版 r6 灵敏度 | 定版 r6 特异度 | 交叉验证灵敏度 | 交叉验证特异度 |",
          "|---|---|---|---|---|---|"]
    for d in DIMS5:
        a, b = fz["all"]["dims"][d], cv["median"]["dims"][d]
        L.append(f"| {d} | {a['n_gold_fail']} / {a['n_gold_ok']} | {_f(a['sens'])}{_ci(fz['ci'][d]['sens'])} | "
                 f"{_f(a['spec'])}{_ci(fz['ci'][d]['spec'])} | {_f(b['sens'])}{_ci(cv['ci'][d]['sens'])} | "
                 f"{_f(b['spec'])}{_ci(cv['ci'][d]['spec'])} |")
    L += ["", "## 定版 r6 分 split（对照：dev 上迭代过，test 未调过）", "",
          "| split | " + " | ".join(DIMS5) + " | 总分 |", "|---|" + "---|" * (len(DIMS5) + 1)]
    for s, ev in fz["by_split"].items():
        L.append(f"| {s} | " + " | ".join(_f(ev["dims"][d]["metric"]) for d in DIMS5) + f" | {_f(ev['total'])} |")
    L += ["", f"## 折内选中的规则集（{r['k']} 折 × {r['repeats']} 次）", ""]
    for d in DIMS5:
        L.append(f"- {d}：" + "，".join(f"{rs} × {n}" for rs, n in cv["picks"][d].items()))
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="判官 v7：180 段嵌套交叉验证（规则层折内选）")
    ap.add_argument("--gold", action="append", required=True, help="可给多次：dev 与 test 金标")
    ap.add_argument("--signals", action="append", required=True, help="可给多次：dev 与 test 的信号缓存（同一抽取版本）")
    ap.add_argument("--dialogues", default=str(REPO / "runs/v2/dialogues_gold180.jsonl"))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--group", choices=sorted(GROUPS), default="persona")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--rulesets", default=",".join(RULESETS))
    ap.add_argument("--report-json", default=str(REPO / "gold_judge180" / "report.json"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    r = run(a.gold, a.signals, a.dialogues, k=a.k, repeats=a.repeats, seed=a.seed, group=a.group,
            n_boot=a.n_boot, rulesets=[x for x in a.rulesets.split(",") if x], report_json=a.report_json)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cv.json").write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "cv.md").write_text(render(r), encoding="utf-8")
    print(f"定版 r6 总分 {r['frozen']['all']['total']}，交叉验证中位 {r['cv']['median']['total']}；报告 {out / 'cv.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
