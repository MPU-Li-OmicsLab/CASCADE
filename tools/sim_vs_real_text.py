"""模拟患者话 vs 真实中文求助文本的分布对比（零 GPU，纯 numpy，字符级）。

⚠️ 体裁差异先说清：手头唯一的真实中文语料 PsyQA 是**论坛发帖体**（中位 100+ 字、书面、一次说完），
模拟患者是**聊天体**（十几个字、口语、多轮）。所以除整帖外，另把帖子切成句子（REAL-sent）当聊天长度的对照单元，
并给两条参照线读数：
  * 地板 = 真实语料自己对半分（同分布时该是多少）
  * 天花板参照 = 被测 AI 的回复切句 vs 真实（「明显不是求助者写的文本」是多少）
模拟患者落在两者之间的哪里，就是它「像不像」的量化位置。

    python tools/sim_vs_real_text.py --out reports/sim_vs_real
PsyQA 原文只读、不复制进仓库；报告里只有聚合统计，不含原文。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SPLIT = re.compile(r"[。！？!?；;\n…]+|\.{3,}")
KW = ("抑郁", "绝望", "想死", "轻生", "自杀", "自残", "活不下去")      # 与 cascade/seeds.py 的 PSYQA_FILTER_KEYWORDS 一致
MARKERS = {
    "第一人称「我」": r"我",
    "否定（不/没/别）": r"[不没别]",
    "死亡/自杀词": r"死|自杀|轻生|自残|活不下去|不想活|结束生命",
    "犹疑词": r"可能|好像|也许|大概|似乎|是不是",
    "睡眠/身体": r"睡|失眠|吃不下|头疼|头痛|累|没力气",
    "家人": r"[爸妈]|父母|老公|老婆|孩子|儿子|女儿",
    "语气词": r"[嗯哦唉呵啊吧呢嘛]",
    "问号": r"[?？]",
    "省略号": r"…|\.{3,}",
    "感叹号": r"[!！]",
    "逗号": r"[,，]",
}


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SPLIT.split(text or "") if len(s.strip()) >= 2]


def load_real(paths: list[str]) -> list[str]:
    posts = []
    for p in paths:
        for rec in json.loads(Path(p).read_text(encoding="utf-8")):
            kw = rec.get("keywords", "")
            kw = " ".join(kw) if isinstance(kw, list) else str(kw)
            blob = f"{rec.get('question', '')} {rec.get('description', '')} {kw}"
            if any(k in blob for k in KW) and rec.get("description"):
                posts.append(rec["description"].strip())
    return posts


def load_real_tsv(paths: list[str], text_col: str = "comment") -> list[str]:
    """带表头的 TSV（如 SOS-1K：comment 	 myLabel）。只取文本列，去重。"""
    import csv
    seen, out = set(), []
    for p in paths:
        with open(p, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
                t = (row.get(text_col) or "").strip()
                if t and t not in seen:
                    seen.add(t); out.append(t)
    return out


def load_sim(path: str):
    from cascade.io import read_jsonl
    sim, asst, by_persona = [], [], defaultdict(list)
    for r in read_jsonl(path):
        if r.risk_none or r.intervention:
            continue
        pid = r.case_profile.persona_id or r.case_profile.id.split("-")[0]
        for t in r.turns:
            if t.turn >= 2 and t.patient_utterance.strip():       # 第 1 轮是种子开场白，不是模拟器生成的
                sim.append(t.patient_utterance.strip()); by_persona[pid].append(t.patient_utterance.strip())
            asst.extend(sentences(t.target_response))
    return sim, asst, by_persona


def length_stats(texts):
    ln = np.array([len(t) for t in texts])
    return {"n": int(len(ln)), "median": float(np.median(ln)), "p10": float(np.percentile(ln, 10)),
            "p90": float(np.percentile(ln, 90)), "mean": round(float(ln.mean()), 1)}


def marker_rates(texts):
    total = sum(len(t) for t in texts) or 1
    blob = "\n".join(texts)
    return {k: round(1000 * len(re.findall(rx, blob)) / total, 2) for k, rx in MARKERS.items()}


def bigram_counts(texts) -> Counter:
    c = Counter()
    for t in texts:
        t = re.sub(r"\s+", "", t)
        c.update(t[i:i + 2] for i in range(len(t) - 1))
    return c


def jsd(a: Counter, b: Counter) -> float:
    keys = list(set(a) | set(b))
    p = np.array([a.get(k, 0) for k in keys], float); q = np.array([b.get(k, 0) for k in keys], float)
    p /= p.sum(); q /= q.sum(); m = (p + q) / 2
    def kl(x, y):
        nz = x > 0
        return float(np.sum(x[nz] * np.log2(x[nz] / y[nz])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def sample_chars(texts, budget, rng):
    """抽到约 budget 个字符为止（让两边样本量可比，JSD 对样本量敏感）。"""
    idx = rng.permutation(len(texts)); out, n = [], 0
    for i in idx:
        out.append(texts[i]); n += len(texts[i])
        if n >= budget:
            break
    return out


def jsd_pair(a, b, budget, reps, rng, same=False):
    vals = []
    for _ in range(reps):
        if same:                                       # 同一语料对半分，得到「同分布」地板
            idx = rng.permutation(len(a)); h = len(a) // 2
            x = sample_chars([a[i] for i in idx[:h]], budget, rng); y = sample_chars([a[i] for i in idx[h:]], budget, rng)
        else:
            x, y = sample_chars(a, budget, rng), sample_chars(b, budget, rng)
        vals.append(jsd(bigram_counts(x), bigram_counts(y)))
    return {"mean": round(float(np.mean(vals)), 4), "sd": round(float(np.std(vals)), 4)}


def _hash_vec(text, dim):
    v = np.zeros(dim)
    t = re.sub(r"\s+", "", text)
    for i in range(len(t) - 1):
        v[zlib.crc32(t[i:i + 2].encode("utf-8")) % dim] += 1     # crc32：跨进程稳定（内置 hash 每次进程随机）
    return v


def auc(scores, labels) -> float:
    order = np.argsort(scores); ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


def nb_cv_auc(a, b, rng, n_each=2000, dim=1 << 15, folds=5, length_match=True):
    """字符二元组朴素贝叶斯、5 折交叉验证 AUC。0.5 = 分不出；1.0 = 一眼可辨。
    length_match：按长度分箱等量抽样，避免「光看长短就能分」的平凡线索。"""
    if length_match:
        bins = [0, 6, 10, 15, 20, 30, 45, 70, 10 ** 9]
        ga, gb = defaultdict(list), defaultdict(list)
        for t in a: ga[np.digitize(len(t), bins)].append(t)
        for t in b: gb[np.digitize(len(t), bins)].append(t)
        xa, xb = [], []
        for k in set(ga) & set(gb):
            m = min(len(ga[k]), len(gb[k]))
            xa += list(rng.choice(ga[k], m, replace=False)); xb += list(rng.choice(gb[k], m, replace=False))
        a, b = xa, xb
    if min(len(a), len(b)) < folds * 2:                # 长度配平后某一侧太少，算不了交叉验证
        return {"auc": None, "n_each": int(min(len(a), len(b)))}
    a = list(rng.choice(a, min(n_each, len(a)), replace=False)); b = list(rng.choice(b, min(n_each, len(b)), replace=False))
    X = np.array([_hash_vec(t, dim) for t in a + b]); y = np.array([1] * len(a) + [0] * len(b))
    idx = rng.permutation(len(y)); scores = np.zeros(len(y))
    for f in range(folds):
        te = idx[f::folds]; tr = np.setdiff1d(idx, te)
        lp = np.log((X[tr][y[tr] == 1].sum(0) + 1) / (X[tr][y[tr] == 1].sum() + dim))
        ln = np.log((X[tr][y[tr] == 0].sum(0) + 1) / (X[tr][y[tr] == 0].sum() + dim))
        scores[te] = X[te] @ (lp - ln) / np.maximum(X[te].sum(1), 1)      # 按长度归一，避免长文本分数发散
    return {"auc": round(auc(scores, y), 3), "n_each": int(min(len(a), len(b)))}


def run(real_paths, dialogues, seed=0, budget=40000, reps=20, real_format="psyqa", text_col="comment",
        unit="sentence", real_name="PsyQA"):
    rng = np.random.default_rng(seed)
    posts = load_real(real_paths) if real_format == "psyqa" else load_real_tsv(real_paths, text_col)
    real_sent = [s for p in posts for s in sentences(p)] if unit == "sentence" else list(posts)
    budget = min(budget, sum(len(x) for x in real_sent) // 2 - 1)      # 小语料：对半分后每侧都得够抽
    sim, asst, by_persona = load_sim(dialogues)
    res = {"meta": {"real_posts": len(posts), "real_sentences": len(real_sent), "sim_utterances": len(sim),
                    "assistant_sentences": len(asst), "char_budget_per_side": budget, "reps": reps,
                    "real_name": real_name, "unit": unit},
           "length": {"REAL-post": length_stats(posts), "REAL-sent": length_stats(real_sent),
                      "SIM": length_stats(sim), "ASSISTANT-sent": length_stats(asst)},
           "markers_per_1000_chars": {"REAL-post": marker_rates(posts), "SIM": marker_rates(sim),
                                      "ASSISTANT-sent": marker_rates(asst)},
           "jsd_char_bigram": {
               "floor REAL|REAL": jsd_pair(real_sent, None, budget, reps, rng, same=True),
               "floor SIM|SIM": jsd_pair(sim, None, budget, reps, rng, same=True),
               "SIM vs REAL": jsd_pair(sim, real_sent, budget, reps, rng),
               "ASSISTANT vs REAL": jsd_pair(asst, real_sent, budget, reps, rng),
               "SIM vs ASSISTANT": jsd_pair(sim, asst, budget, reps, rng)},
           "classifier_auc_length_matched": {
               "REAL vs REAL (应≈0.5)": nb_cv_auc(real_sent[::2], real_sent[1::2], rng),
               "SIM vs REAL": nb_cv_auc(sim, real_sent, rng),
               "ASSISTANT vs REAL": nb_cv_auc(asst, real_sent, rng),
               "SIM vs ASSISTANT": nb_cv_auc(sim, asst, rng)}}
    res["per_persona_jsd_vs_real"] = {
        p: jsd_pair(u, real_sent, min(budget, sum(len(x) for x in u)), 5, rng)["mean"] for p, u in sorted(by_persona.items())}
    res["per_persona_median_len"] = {p: float(np.median([len(x) for x in u])) for p, u in sorted(by_persona.items())}
    return res


def render(res) -> str:
    m = res["meta"]
    name, unit = m.get("real_name", "PsyQA"), m.get("unit", "sentence")
    L = [f"# 模拟患者 vs 真实求助文本（{name}）分布对比", "",
         f"真实：{name} {m['real_posts']} 条 → 对照单元（{'切句' if unit == 'sentence' else '整条'}）{m['real_sentences']} 个。模拟：语料 v2 模拟器生成的患者话 {m['sim_utterances']} 条（不含第 1 轮种子）。"
         f"参照：被测 AI 回复切句 {m['assistant_sentences']} 句。", "",
         "⚠️ 读数要对照体裁：论坛长帖 vs 聊天短句量到的主要是体裁差；树洞留言（短、口语、第一人称）才是体裁匹配的参照。", "",
         "## 1. 长度（字符）", "", "| 语料 | n | 中位 | P10 | P90 |", "|---|---|---|---|---|"]
    for k, v in res["length"].items():
        L.append(f"| {k} | {v['n']} | {v['median']:.0f} | {v['p10']:.0f} | {v['p90']:.0f} |")
    L += ["", "## 2. 语言标记（每千字出现次数）", "", "| 标记 | 真实 | 模拟患者 | 被测 AI |", "|---|---|---|---|"]
    mk = res["markers_per_1000_chars"]
    for k in MARKERS:
        L.append(f"| {k} | {mk['REAL-post'][k]} | {mk['SIM'][k]} | {mk['ASSISTANT-sent'][k]} |")
    L += ["", f"## 3. 字符二元组分布的 Jensen–Shannon 散度（每侧约 {m['char_budget_per_side']} 字，{m['reps']} 次重抽）", "",
          "| 对比 | JSD 均值 | sd |", "|---|---|---|"]
    for k, v in res["jsd_char_bigram"].items():
        L.append(f"| {k} | {v['mean']} | {v['sd']} |")
    j = res["jsd_char_bigram"]
    floor = j["floor REAL|REAL"]["mean"]; ceil = j["ASSISTANT vs REAL"]["mean"]; s = j["SIM vs REAL"]["mean"]
    pos = (s - floor) / (ceil - floor) if ceil > floor else float("nan")
    L += ["", f"位置 = (SIM−地板)/(AI−地板) = **{pos:.2f}**（0 = 与真实文本同分布，1 = 和 AI 回复一样远）。", "",
          "## 4. 可辨别性（字符二元组朴素贝叶斯，5 折 AUC，**长度配平**）", "", "| 对比 | AUC | 每类 n |", "|---|---|---|"]
    for k, v in res["classifier_auc_length_matched"].items():
        L.append(f"| {k} | {v['auc'] if v['auc'] is not None else '—'} | {v['n_each']} |")
    L += ["", "## 5. 分人设：与真实文本的 JSD（越小越近）", "", "| 人设 | JSD | 话长中位 |", "|---|---|---|"]
    for p, v in sorted(res["per_persona_jsd_vs_real"].items(), key=lambda kv: kv[1]):
        L.append(f"| {p} | {v} | {res['per_persona_median_len'][p]:.0f} |")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", nargs="+", default=["PsyQA/test.json", "PsyQA/valid.json"])
    ap.add_argument("--dialogues", default="runs/v2/dialogues_both_arms.jsonl")
    ap.add_argument("--real-format", choices=["psyqa", "tsv"], default="psyqa")
    ap.add_argument("--text-col", default="comment")
    ap.add_argument("--unit", choices=["sentence", "whole"], default="sentence", help="真实侧的对照单元：切句 / 整条")
    ap.add_argument("--real-name", default="PsyQA")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = run(a.real, a.dialogues, a.seed, real_format=a.real_format, text_col=a.text_col, unit=a.unit, real_name=a.real_name)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "report.md").write_text(render(res), encoding="utf-8")
    print(f"wrote {out / 'report.md'}")


if __name__ == "__main__":
    main()
