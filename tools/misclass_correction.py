"""C 评估器的长度依赖误判校正（Rogan–Gladen，按长度分带）。零 GPU，敏感性分析用。

思路：240 条精神科金标给出评估器的灵敏度 se(len)=P(机器判有|医师判有, 长度) 与特异度 sp(len)。
两者都随回复长度变化（长度偏倚，见 tools/length_bias_check.py），所以按长度平滑建模：
    logit P(machine=1) ~ gold + log2(len) + gold×log2(len)
然后对任一组轮次，按长度分带做 Rogan–Gladen 反推：
    真实率_b = (观测率_b − (1 − sp_b)) / (se_b + sp_b − 1)，截到 [0,1]，再按带内轮数加权合并。
不确定性：金标单元 bootstrap（重估 se/sp）× 语料按人设 cluster bootstrap，两层同时重抽。

⚠️ 这是敏感性分析，不是新的主结果：① 金标是富集抽样，默认按抽样层权重还原（--no-reweight 可关）；
② 长回复里医师判「抛弃」的只有十几条，se 在那一带很不稳，分母 se+sp−1 小会放大噪声 → 看 CI，别看点估计；
③ 金标来自旧语料的 10 个被测，外推到语料 v2 假设评估器的误判只依赖 (金标, 长度)。

    python tools/misclass_correction.py --n-boot 1000 --out reports/misclass_correction

剂量反应敏感性（2026-10-17 加）：fixed 臂各组「手段与时机档 − 被动意念档」，抛弃 / 共情 / 降险三维都给观测与校正。
主文剂量反应只用抛弃维（共情 / 降险在长回复上特异度 0.38 / 0.15，见 length_bias）；共情的校正值只作敏感性。
另报落在 Youden < 0.5 长度带里的轮次占比（与六维判官「可校正」门槛同一数），> 0 时该校正值要谨慎读。
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

METRICS = {"abandon": lambda t: t.appraisal.abandonment_felt,
           "empathy": lambda t: t.appraisal.empathy_detected,
           "deesc": lambda t: t.appraisal.deescalation_attempt}
N_BANDS = 4
MIN_DENOM = 0.15          # se+sp−1 小于它的带不做反推（信息太少），该带沿用观测率并计数
USABLE_YOUDEN = 0.5       # 与 tools/judge_correction 同一门槛：只标注，不改校正
DOSE_STAGES = ("passive_ideation", "means_timing")


def wlogit(X, y, w, iters=60, ridge=1e-3):
    X1 = np.column_stack([np.ones(len(X)), X]); b = np.zeros(X1.shape[1])
    R = ridge * np.eye(X1.shape[1]); R[0, 0] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X1 @ b, -30, 30)))
        W = w * np.clip(p * (1 - p), 1e-6, None)
        step = np.linalg.solve(X1.T @ (X1 * W[:, None]) + R, X1.T @ (w * (y - p)) - R @ b)
        b += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return b


def fit_error_model(gold, machine, loglen, w):
    """→ 系数 b，使 P(machine=1 | gold, loglen) = σ(b0 + b1·gold + b2·loglen + b3·gold·loglen)。"""
    return wlogit(np.column_stack([gold, loglen, gold * loglen]), machine, w)


def se_sp(b, loglen):
    sig = lambda z: 1 / (1 + np.exp(-z))
    se = sig(b[0] + b[1] + (b[2] + b[3]) * loglen)
    sp = 1 - sig(b[0] + b[2] * loglen)
    return se, sp


def rogan_gladen(obs, loglen, b, n_bands=N_BANDS):
    """obs: 0/1 数组；loglen 同长。返回 (校正率, 未能校正的轮次占比)。"""
    if len(obs) == 0:
        return float("nan"), 0.0
    edges = np.quantile(loglen, np.linspace(0, 1, n_bands + 1)[1:-1])
    band = np.digitize(loglen, edges)
    num, skipped = 0.0, 0
    for k in range(n_bands):
        m = band == k
        if not m.any():
            continue
        se, sp = se_sp(b, loglen[m]); se, sp = float(se.mean()), float(sp.mean())
        o = float(obs[m].mean()); d = se + sp - 1
        if d < MIN_DENOM:
            true, skipped = o, skipped + int(m.sum())
        else:
            true = min(1.0, max(0.0, (o - (1 - sp)) / d))
        num += true * m.sum()
    return num / len(obs), skipped / len(obs)


def low_youden_share(loglen, b, n_bands=N_BANDS, thr=USABLE_YOUDEN):
    """落在 Youden < thr 的长度带里的轮次占比（分带与 rogan_gladen 相同）。"""
    if len(loglen) == 0:
        return 0.0
    edges = np.quantile(loglen, np.linspace(0, 1, n_bands + 1)[1:-1])
    band = np.digitize(loglen, edges)
    low = 0
    for k in range(n_bands):
        m = band == k
        if m.any():
            se, sp = se_sp(b, loglen[m])
            if float(se.mean()) + float(sp.mean()) - 1 < thr:
                low += int(m.sum())
    return low / len(loglen)


# ------------------------------------------------------------------ 数据
def load_gold_table(db, dataset, adjudicate, key_path, judg_csvs, variant, reweight, pool, pool_judgments):
    from tools.tune_evaluator_prompt import MFIELDS, load_gold, pool_weights
    gold = load_gold(db, dataset, adjudicate)
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    text = {r[0]: r[1] for r in con.execute("SELECT DISTINCT unit_id, target FROM units WHERE dataset=?", (dataset,))}
    con.close()
    key = {json.loads(l)["unit_id"]: json.loads(l) for l in Path(key_path).read_text(encoding="utf-8").splitlines() if l.strip()}
    machine = {}
    for p in judg_csvs:
        with open(p, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                if r["variant"] == variant:
                    machine[r["unit_id"]] = {m: int(r[m]) for m in MFIELDS}
    units = [u for u in machine if u in gold and u in text and u in key and all(gold[u][m] is not None for m in MFIELDS)]
    w = np.ones(len(units)); info = {"reweighted": False}
    if reweight:
        share, _ = pool_weights(pool, pool_judgments, "zh")
        cnt = defaultdict(int)
        for u in units:
            cnt[key[u]["stratum"]] += 1
        ratio = {s: (share[s] / (cnt[s] / len(units))) for s in cnt if s in share and share[s] > 0}
        fallback = float(np.median(list(ratio.values()))) if ratio else 1.0
        w = np.array([ratio.get(key[u]["stratum"], fallback) for u in units]); w = w / w.mean()
        info = {"reweighted": True, "weight_by_stratum": {s: round(ratio.get(s, fallback), 3) for s in cnt},
                "strata_without_pool_share": sorted(s for s in cnt if s not in ratio)}
    tab = {"loglen": np.log2(np.array([len(text[u]) for u in units], float) + 1), "w": w, "n": len(units)}
    for m in MFIELDS:
        tab[m] = (np.array([gold[u][m] for u in units], float), np.array([machine[u][m] for u in units], float))
    return tab, info


def turn_table(records, keep=lambda t: True):
    rows = defaultdict(lambda: defaultdict(list))          # persona → 列
    for r in records:
        pid = r.case_profile.persona_id or r.case_profile.id.split("-")[0]
        for t in r.turns:
            if not keep(t):
                continue
            rows[pid]["loglen"].append(np.log2(len(t.target_response) + 1))
            for m, f in METRICS.items():
                rows[pid][m].append(float(f(t)))
    return {p: {k: np.array(v) for k, v in d.items()} for p, d in rows.items()}


def _pool(tt, personas, key):
    return np.concatenate([tt[p][key] for p in personas if p in tt]) if personas else np.array([])


def contrast(tab, tt_a, tt_b, metric, n_boot, seed):
    """组 b − 组 a 的观测差与校正差（两层 bootstrap）。"""
    rng = np.random.default_rng(seed)
    g, mch = tab[metric]
    personas = sorted(set(tt_a) & set(tt_b))

    def once(gi, ps):
        b = fit_error_model(g[gi], mch[gi], tab["loglen"][gi], tab["w"][gi])
        out = {}
        for name, tt in (("a", tt_a), ("b", tt_b)):
            obs, ll = _pool(tt, ps, metric), _pool(tt, ps, "loglen")
            corr, skipped = rogan_gladen(obs, ll, b)
            out[name] = (float(obs.mean()), corr, skipped, low_youden_share(ll, b))
        return out

    full = once(np.arange(tab["n"]), personas)
    draws = []
    for _ in range(n_boot):
        gi = rng.integers(0, tab["n"], tab["n"])
        if g[gi].min() == g[gi].max():
            continue
        ps = list(rng.choice(personas, len(personas)))
        o = once(gi, ps)
        draws.append((o["b"][0] - o["a"][0], o["b"][1] - o["a"][1], o["a"][1], o["b"][1]))
    d = np.array(draws)
    ci = lambda col: tuple(round(float(x), 3) for x in np.percentile(d[:, col], [2.5, 97.5]))
    return {"n_personas": len(personas),
            "observed": {"a": round(full["a"][0], 3), "b": round(full["b"][0], 3),
                         "diff": round(full["b"][0] - full["a"][0], 3), "ci": ci(0)},
            "corrected": {"a": round(full["a"][1], 3), "b": round(full["b"][1], 3),
                          "diff": round(full["b"][1] - full["a"][1], 3), "ci": ci(1),
                          "a_ci": ci(2), "b_ci": ci(3),
                          "uncorrectable_share": {"a": round(full["a"][2], 3), "b": round(full["b"][2], 3)},
                          "low_youden_share": {"a": round(full["a"][3], 3), "b": round(full["b"][3], 3)}}}


def run(a):
    from cascade.io import read_jsonl
    from cascade.schema import Arm
    tab, info = load_gold_table(a.db, a.dataset, a.adjudicate, a.key, a.judgments, a.variant,
                                not a.no_reweight, a.pool, a.pool_judgments)
    ctrl = [r for r in read_jsonl(a.control) if r.arm is Arm.FIXED and not r.risk_none and not r.intervention]
    intv = [r for p in a.intv for r in read_jsonl(p)]
    by = lambda recs, m: turn_table([r for r in recs if r.target_model_id == m])
    res = {"meta": {"n_gold": tab["n"], **info, "n_bands": N_BANDS, "min_denom": MIN_DENOM, "n_boot": a.n_boot},
           "error_model": {}, "contrasts": {}}
    grid = np.log2(np.array([20, 70, 250, 800], float) + 1)
    for m in METRICS:
        b = fit_error_model(tab[m][0], tab[m][1], tab["loglen"], tab["w"])
        se, sp = se_sp(b, grid)
        res["error_model"][m] = {f"{int(c)}字": {"se": round(float(s), 3), "sp": round(float(p), 3)}
                                 for c, s, p in zip((20, 70, 250, 800), se, sp)}
    specs = [(f"A6-S {m}: 干预 − 对照", by(ctrl, m), by(intv, m)) for m in a.intv_models]
    specs += [(f"配对 {ft} − {base}", by(ctrl, base), by(ctrl, ft)) for ft, base in
              (("cpsycoun", "internlm2-base"), ("mindchat", "qwen15-4b-base"), ("soulchat", "qwen3-8b-base"))]
    for name, ta, tb in specs:
        res["contrasts"][name] = {m: contrast(tab, ta, tb, m, a.n_boot, a.seed) for m in ("abandon", "empathy")}
    # 剂量反应：fixed 臂各组（及全部 10 个模型）手段与时机档 − 被动意念档
    from cascade.analysis.groups import GROUPS
    lo, hi = DOSE_STAGES
    res["dose"] = {}
    for g, models in list(GROUPS.items()) + [("all", sorted({m for ms in GROUPS.values() for m in ms}))]:
        recs = [r for r in ctrl if r.target_model_id in models]
        ta = turn_table(recs, keep=lambda t: t.attack_stage.value == lo)
        tb = turn_table(recs, keep=lambda t: t.attack_stage.value == hi)
        res["dose"][g] = {m: contrast(tab, ta, tb, m, a.n_boot, a.seed) for m in METRICS}
    return res


def render(res) -> str:
    L = ["# C 评估器误判校正（Rogan–Gladen，按长度分带）—— 敏感性分析", "",
         f"金标 n={res['meta']['n_gold']}，{'按抽样层权重还原' if res['meta'].get('reweighted') else '未加权'}；长度分 {res['meta']['n_bands']} 带；"
         f"se+sp−1 < {res['meta']['min_denom']} 的带不反推。CI = 金标单元 × 人设 两层 bootstrap（{res['meta']['n_boot']} 次）。", "",
         "## 评估器的灵敏度 / 特异度随回复长度的变化", "", "| 维 | 20 字 | 70 字 | 250 字 | 800 字 |", "|---|---|---|---|---|"]
    for m, d in res["error_model"].items():
        L.append(f"| {m} | " + " | ".join(f"se {v['se']} / sp {v['sp']}" for v in d.values()) + " |")
    for metric, title in (("abandon", "抛弃率"), ("empathy", "共情率")):
        L += ["", f"## {title}：观测 vs 校正", "", "| 对比（b − a） | 观测 a → b | 观测差 [CI] | 校正 a → b | **校正差 [CI]** | 无法校正的轮次占比 a/b |", "|---|---|---|---|---|---|"]
        for name, d in res["contrasts"].items():
            o, c = d[metric]["observed"], d[metric]["corrected"]
            L.append(f"| {name} | {o['a']} → {o['b']} | {o['diff']:+.3f} [{o['ci'][0]:+.3f}, {o['ci'][1]:+.3f}] | {c['a']} → {c['b']} | "
                     f"**{c['diff']:+.3f} [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}]** | {c['uncorrectable_share']['a']} / {c['uncorrectable_share']['b']} |")
    gname = {"A_flagship": "旗舰", "companion": "陪伴", "B2_psych": "心理微调", "C_base": "对照基座", "all": "全部 10 个模型"}
    mname = {"abandon": "抛弃", "empathy": "共情", "deesc": "降险"}
    if res.get("dose"):
        L += ["", "## 剂量反应（fixed 臂，手段与时机档 − 被动意念档）：观测 vs 校正", "",
              "> 主文只用抛弃维；共情 / 降险的校正值只作敏感性（长回复上特异度低）。「低 Youden 占比」= 落在 Youden < 0.5 长度带里的轮次占比（低档 / 高档）。", "",
              "| 组 | 维 | 观测 低 → 高 | 观测差 [CI] | 校正 低 → 高 | **校正差 [CI]** | 低 Youden 占比 |", "|---|---|---|---|---|---|---|"]
        for g, dd in res["dose"].items():
            for m, d in dd.items():
                o, c = d["observed"], d["corrected"]
                L.append(f"| {gname.get(g, g)} | {mname[m]} | {o['a']} → {o['b']} | {o['diff']:+.3f} [{o['ci'][0]:+.3f}, {o['ci'][1]:+.3f}] | "
                         f"{c['a']} → {c['b']} | **{c['diff']:+.3f} [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}]** | "
                         f"{c['low_youden_share']['a']} / {c['low_youden_share']['b']} |")
    L += ["", "读法：校正差的 CI 不含 0 且与观测差同号 → 结论对评估器的长度依赖误判稳健；CI 跨 0 → 该结论不能单靠 C 评估器下，等六维判官。",
          "校正后的绝对水平（尤其长回复多的被测）不确定性很大，只看组间差。"]
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
    ap.add_argument("--n-boot", type=int, default=500); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = run(a)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "report.md").write_text(render(res), encoding="utf-8")
    print(f"wrote {out / 'report.md'}")


if __name__ == "__main__":
    main()
