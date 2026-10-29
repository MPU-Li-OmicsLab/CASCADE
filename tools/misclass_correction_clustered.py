"""C 评估器误判校正：金标一侧也按人设整组重抽的 bootstrap CI（零 GPU，敏感性 / 订正用）。

原工具 tools/misclass_correction.py 的 contrast() 两层 bootstrap 里，语料一侧按人设重抽，
金标一侧却逐条重抽（rng.integers(0, n, n)）。240 条金标来自 8 个人设 × 24 个档案的对话，同一人设 /
同一段对话的轮次相关，逐条重抽低估了 se/sp 的抽样不确定性 → 校正差的 CI 偏窄。

本工具：点估计与原工具完全相同（同一误差模型、同一 Rogan–Gladen 分带），只改 CI：
  * unit    = 原做法（直接调用 misclass_correction.contrast，同 seed → 应逐位复现原 results.json）
  * persona = 金标按人设整组重抽（主口径；8 个人设）
  * case    = 金标按档案（= 一段对话的来源，如 zh_p05-c04）整组重抽（补充；24 个档案）
人设 = key.jsonl 里 real_id 的前缀（"zh_p08-c23#模型#臂#t8" → zh_p08）。金标对话来自旧语料，
人设 id 与语料 v2 同名但不是同一抽样（v2 的 zh_p08 改过），所以两侧的人设重抽相互独立。

    python tools/misclass_correction_clustered.py --n-boot 1000 --out reports/misclass_correction_clustered
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import misclass_correction as mc  # noqa: E402

GOLD_MODES = ("unit", "persona", "case")


# ------------------------------------------------------------------ 金标单元 → 人设 / 档案
def persona_of(real_id: str) -> str:
    return real_id.split("#")[0].split("-")[0]


def case_of(real_id: str) -> str:
    return real_id.split("#")[0]


def gold_unit_order(db, dataset, adjudicate, key_path, judg_csvs, variant):
    """复刻 mc.load_gold_table 的单元筛选与顺序（它不返回 unit_id），返回 unit_id 列表。"""
    import sqlite3
    from tools.tune_evaluator_prompt import MFIELDS, load_gold
    gold = load_gold(db, dataset, adjudicate)
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    text = {r[0] for r in con.execute("SELECT DISTINCT unit_id FROM units WHERE dataset=?", (dataset,))}
    con.close()
    key = {json.loads(l)["unit_id"] for l in Path(key_path).read_text(encoding="utf-8").splitlines() if l.strip()}
    machine = {}
    for p in judg_csvs:
        with open(p, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                if r["variant"] == variant:
                    machine[r["unit_id"]] = True
    return [u for u in machine if u in gold and u in text and u in key and all(gold[u][m] is not None for m in MFIELDS)]


def cluster_labels(units, key_path):
    key = {}
    for l in Path(key_path).read_text(encoding="utf-8").splitlines():
        if l.strip():
            d = json.loads(l); key[d["unit_id"]] = d
    persona = np.array([persona_of(key[u]["real_id"]) if key[u].get("real_id") else "" for u in units], dtype=object)
    case = np.array([case_of(key[u]["real_id"]) if key[u].get("real_id") else "" for u in units], dtype=object)
    return {"persona": persona, "case": case}


def cluster_index(labels):
    """标签数组 → (簇名列表, {簇名: 成员下标数组})。空标签（映射不到）各自成一簇。"""
    groups = {}
    for i, lab in enumerate(labels):
        groups.setdefault(lab if lab else f"__unit{i}", []).append(i)
    names = sorted(groups)
    return names, {k: np.array(v, dtype=int) for k, v in groups.items()}


def cluster_resample(names, members, rng):
    """有放回抽 len(names) 个簇，返回拼接后的单元下标（样本量随簇大小浮动）。"""
    pick = rng.choice(len(names), len(names))
    return np.concatenate([members[names[j]] for j in pick])


# ------------------------------------------------------------------ 对比（金标按簇重抽）
def contrast_clustered(tab, labels, tt_a, tt_b, metric, n_boot, seed):
    """与 mc.contrast 同一点估计与输出格式；金标一侧按 labels 整组重抽，语料一侧照旧按人设重抽。"""
    rng = np.random.default_rng(seed)
    g, mch = tab[metric]
    personas = sorted(set(tt_a) & set(tt_b))
    names, members = cluster_index(labels)

    def once(gi, ps):
        b = mc.fit_error_model(g[gi], mch[gi], tab["loglen"][gi], tab["w"][gi])
        out = {}
        for name, tt in (("a", tt_a), ("b", tt_b)):
            obs, ll = mc._pool(tt, ps, metric), mc._pool(tt, ps, "loglen")
            corr, skipped = mc.rogan_gladen(obs, ll, b)
            out[name] = (float(obs.mean()), corr, skipped, mc.low_youden_share(ll, b))
        return out

    full = once(np.arange(tab["n"]), personas)
    draws, dropped = [], 0
    for _ in range(n_boot):
        gi = cluster_resample(names, members, rng)
        if g[gi].min() == g[gi].max():
            dropped += 1
            continue
        ps = list(rng.choice(personas, len(personas)))
        o = once(gi, ps)
        draws.append((o["b"][0] - o["a"][0], o["b"][1] - o["a"][1], o["a"][1], o["b"][1]))
    d = np.array(draws)
    ci = lambda col: tuple(round(float(x), 3) for x in np.percentile(d[:, col], [2.5, 97.5]))
    return {"n_personas": len(personas), "n_gold_clusters": len(names), "dropped_draws": dropped,
            "observed": {"a": round(full["a"][0], 3), "b": round(full["b"][0], 3),
                         "diff": round(full["b"][0] - full["a"][0], 3), "ci": ci(0)},
            "corrected": {"a": round(full["a"][1], 3), "b": round(full["b"][1], 3),
                          "diff": round(full["b"][1] - full["a"][1], 3), "ci": ci(1),
                          "a_ci": ci(2), "b_ci": ci(3),
                          "uncorrectable_share": {"a": round(full["a"][2], 3), "b": round(full["b"][2], 3)},
                          "low_youden_share": {"a": round(full["a"][3], 3), "b": round(full["b"][3], 3)}}}


def crosses_zero(ci) -> bool:
    return ci[0] <= 0 <= ci[1]


def compare_modes(tab, labels_by_mode, ta, tb, metric, n_boot, seed, modes):
    out = {}
    for mode in modes:
        if mode == "unit":
            out[mode] = mc.contrast(tab, ta, tb, metric, n_boot, seed)
        else:
            out[mode] = contrast_clustered(tab, labels_by_mode[mode], ta, tb, metric, n_boot, seed)
    return out


# ------------------------------------------------------------------ 主流程
def run(a):
    from cascade.io import read_jsonl
    from cascade.schema import Arm
    tab, info = mc.load_gold_table(a.db, a.dataset, a.adjudicate, a.key, a.judgments, a.variant,
                                   not a.no_reweight, a.pool, a.pool_judgments)
    units = gold_unit_order(a.db, a.dataset, a.adjudicate, a.key, a.judgments, a.variant)
    if len(units) != tab["n"]:
        raise RuntimeError(f"金标单元数对不上：复刻 {len(units)} vs load_gold_table {tab['n']}")
    labels = cluster_labels(units, a.key)
    unmapped = int(sum(1 for p in labels["persona"] if not p))
    per_persona = {p: int((labels["persona"] == p).sum()) for p in sorted(set(labels["persona"]))}
    modes = tuple(a.modes)

    ctrl = [r for r in read_jsonl(a.control) if r.arm is Arm.FIXED and not r.risk_none and not r.intervention]
    intv = [r for p in a.intv for r in read_jsonl(p)]
    by = lambda recs, m: mc.turn_table([r for r in recs if r.target_model_id == m])
    res = {"meta": {"n_gold": tab["n"], **info, "n_bands": mc.N_BANDS, "min_denom": mc.MIN_DENOM,
                    "n_boot": a.n_boot, "seed": a.seed, "modes": list(modes),
                    "gold_mapping": {"n_unmapped": unmapped, "n_personas": len(per_persona),
                                     "n_cases": len(set(labels["case"])), "units_per_persona": per_persona}},
           "contrasts": {}, "dose": {}}
    specs = [(f"A6-S {m}: 干预 − 对照", by(ctrl, m), by(intv, m)) for m in a.intv_models]
    specs += [(f"配对 {ft} − {base}", by(ctrl, base), by(ctrl, ft)) for ft, base in
              (("cpsycoun", "internlm2-base"), ("mindchat", "qwen15-4b-base"), ("soulchat", "qwen3-8b-base"))]
    for name, ta, tb in specs:
        res["contrasts"][name] = {m: compare_modes(tab, labels, ta, tb, m, a.n_boot, a.seed, modes)
                                  for m in ("abandon", "empathy")}
        print(f"done {name}", flush=True)
    from cascade.analysis.groups import GROUPS
    lo, hi = mc.DOSE_STAGES
    for g, models in list(GROUPS.items()) + [("all", sorted({m for ms in GROUPS.values() for m in ms}))]:
        recs = [r for r in ctrl if r.target_model_id in models]
        ta = mc.turn_table(recs, keep=lambda t: t.attack_stage.value == lo)
        tb = mc.turn_table(recs, keep=lambda t: t.attack_stage.value == hi)
        res["dose"][g] = {m: compare_modes(tab, labels, ta, tb, m, a.n_boot, a.seed, modes) for m in mc.METRICS}
        print(f"done dose {g}", flush=True)
    if a.orig_results and Path(a.orig_results).exists() and "unit" in modes:
        res["meta"]["unit_mode_matches_original"] = check_against_original(res, json.loads(Path(a.orig_results).read_text(encoding="utf-8")))
    return res


def check_against_original(res, orig):
    """unit 口径的校正差与 CI 是否逐位等于原 results.json。返回不一致的条目列表（空 = 全部一致）。"""
    bad = []
    for sec in ("contrasts", "dose"):
        for name, dd in res.get(sec, {}).items():
            for m, d in dd.items():
                o = orig.get(sec, {}).get(name, {}).get(m)
                if o is None:
                    bad.append(f"{sec}/{name}/{m}: 原报告缺"); continue
                mine = d["unit"]["corrected"]
                if (mine["diff"], list(mine["ci"])) != (o["corrected"]["diff"], list(o["corrected"]["ci"])):
                    bad.append(f"{sec}/{name}/{m}: {mine['diff']} {list(mine['ci'])} vs 原 {o['corrected']['diff']} {o['corrected']['ci']}")
    return {"all_match": not bad, "mismatches": bad}


# ------------------------------------------------------------------ 报告
def _fmt(ci):
    return f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"


def _flag(old, new):
    oz, nz = crosses_zero(old), crosses_zero(new)
    if not oz and nz:
        return "⚠️ 新跨 0"
    if oz and nz:
        return "原已跨 0"
    if oz and not nz:
        return "原跨 0，新不跨"
    return "稳健"


def render(res) -> str:
    meta = res["meta"]; gm = meta["gold_mapping"]; modes = meta["modes"]
    extra = [m for m in modes if m not in ("unit", "persona")]
    L = ["# C 评估器误判校正：金标按人设聚类重抽的 CI", "",
         f"金标 n={meta['n_gold']}（{'按抽样层权重还原' if meta.get('reweighted') else '未加权'}），映射到 {gm['n_personas']} 个人设 / {gm['n_cases']} 个档案，"
         f"映射不到 {gm['n_unmapped']} 条。每人设单元数：" + "，".join(f"{k} {v}" for k, v in gm["units_per_persona"].items()) + "。", "",
         f"点估计与原报告（reports/misclass_correction/）完全相同；只改 CI。{meta['n_boot']} 次 bootstrap，seed {meta['seed']}。",
         "**原 CI** = 金标逐条重抽 × 语料按人设重抽（原工具口径）；**新 CI** = 金标按人设整组重抽 × 语料按人设重抽。"
         + ("**档案 CI** = 金标按档案（24 段对话来源）整组重抽，作补充。" if "case" in extra else ""), ""]
    chk = meta.get("unit_mode_matches_original")
    if chk is not None:
        L += [f"复现检查：原口径重跑与原 results.json {'逐位一致' if chk['all_match'] else '有 %d 处不一致（见 json）' % len(chk['mismatches'])}。", ""]
    L += ["⚠️ 金标只有 8 个人设，按人设重抽是很粗的簇 bootstrap（簇数少时百分位 CI 本身也不稳），新 CI 应读作「上界式」的保守估计。", ""]

    def table(rows, head_first):
        hdr = f"| {head_first} | 维 | 校正差 | 原 CI（逐条） | **新 CI（人设）** |" + (" 档案 CI |" if "case" in extra else "") + " 变化 |"
        out = [hdr, "|" + "---|" * (hdr.count("|") - 1)]
        for name, m, d in rows:
            c_u, c_p = d["unit"]["corrected"], d["persona"]["corrected"]
            line = f"| {name} | {m} | {c_u['diff']:+.3f} | {_fmt(c_u['ci'])} | **{_fmt(c_p['ci'])}** |"
            if "case" in extra:
                line += f" {_fmt(d['case']['corrected']['ci'])} |"
            out.append(line + f" {_flag(c_u['ci'], c_p['ci'])} |")
        return out

    mname = {"abandon": "抛弃", "empathy": "共情", "deesc": "降险"}
    L += ["## 配对消融与 A6-S 系统提示（b − a）", ""]
    L += table([(n, mname[m], d) for n, dd in res["contrasts"].items() for m, d in dd.items()], "对比")
    gname = {"A_flagship": "旗舰", "companion": "陪伴", "B2_psych": "心理微调", "C_base": "对照基座", "all": "全部 10 个模型"}
    L += ["", "## 剂量反应（fixed 臂，手段与时机档 − 被动意念档）", "",
          "> 主文只用抛弃维；共情 / 降险只作敏感性。", ""]
    L += table([(gname.get(g, g), mname[m], d) for g, dd in res["dose"].items() for m, d in dd.items()], "组")
    L += ["", "## 观测差（不经校正）的 CI 不受金标重抽方式影响", "",
          "观测差只用语料，两种口径下 CI 只因随机数序列不同而略有出入，此处不列。", "",
          "读法：「⚠️ 新跨 0」= 原报告说该校正差稳健，但金标按人设聚类后 CI 包含 0，论文里这条要收软。"]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="webapp/data/annotations.sqlite3")
    ap.add_argument("--dataset", default="gold_r2"); ap.add_argument("--adjudicate", default="adjudicate")
    ap.add_argument("--key", default="gold_ceval_r2/key.jsonl")
    ap.add_argument("--judgments", nargs="+", default=["tune_eval38/judgments_dev.csv", "tune_eval38/judgments_test.csv"])
    ap.add_argument("--variant", default="v10_deesc_v3")
    ap.add_argument("--no-reweight", action="store_true")
    ap.add_argument("--pool", nargs="+", default=["runs/valbatch_v2/dialogues.jsonl"])
    ap.add_argument("--pool-judgments", default="cmp_eval_v2/judgments.csv")
    ap.add_argument("--control", default="runs/v2/dialogues_both_arms.jsonl")
    ap.add_argument("--intv", nargs="+", default=["runs/v2_intv/dialogues.jsonl", "runs/v2_intv_api/dialogues.jsonl"])
    ap.add_argument("--intv-models", nargs="+", default=["deepseek/deepseek-v4-pro", "cpsycoun", "internlm2-base"])
    ap.add_argument("--modes", nargs="+", default=list(GOLD_MODES), choices=GOLD_MODES)
    ap.add_argument("--orig-results", default="reports/misclass_correction/results.json")
    ap.add_argument("--n-boot", type=int, default=1000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")     # Windows GBK 控制台打印 "−" 会崩
    except (AttributeError, ValueError):
        pass
    res = run(a)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "report.md").write_text(render(res), encoding="utf-8")
    print(f"wrote {out / 'report.md'}")


if __name__ == "__main__":
    main()
