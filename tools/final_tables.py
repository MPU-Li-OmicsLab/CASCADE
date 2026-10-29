"""论文的投稿表格：Extended Data Table 1 与 Supplementary Tables 1–4（零模型调用、零网络；数字全部从 reports/ 的 JSON 读）。

    python tools/final_tables.py --out paper_final

产出（都在 --out 下）：
  tables/EDTable1_target_models.{csv,md}
  tables/STable1_flip_table.{csv,md}           判官 vs 医师翻转表全表（reports/judge_vs_gold）
  tables/STable2_primary_tests.{csv,md}        主假设检验（reports/primary_tests）
  tables/STable3_gold_prevalence.{csv,md}      医师金标加权比例与差值，原 / raking / 截断 raking 并排
  tables/STable4_judge_error_by_model.{csv,md} 各模型判官混淆计数与 O/E（reports/judge_error_by_model）

规则：只读 reports/ 与 cascade/analysis/groups.py；不读 *_key.jsonl。
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from cascade.analysis.groups import GROUPS, PAIRS  # noqa: E402

# ─────────────────────────────── 常量与格式 ───────────────────────────────

MODEL_NAME = {
    "openai/gpt-5.6-sol": "GPT-5.6 Sol",
    "anthropic/claude-opus-5": "Claude Opus 5",
    "deepseek/deepseek-v4-pro": "DeepSeek-V4-Pro",
    "minimax/minimax-m2-her": "MiniMax-M2-her",
    "cpsycoun": "CPsyCounX",
    "internlm2-base": "InternLM2-chat-7B",
    "mindchat": "MindChat-Qwen2-4B",
    "qwen15-4b-base": "Qwen1.5-4B-Chat",
    "soulchat": "SoulChat-R1-Qwen3-8B",
    "qwen3-8b-base": "Qwen3-8B",
}
MODEL_ORDER = list(MODEL_NAME)
GROUP_ZH = {"A_flagship": "旗舰", "companion": "陪伴", "B2_psych": "心理微调", "C_base": "对照基座"}
GROUP_EN = {"A_flagship": "Flagship", "companion": "Companion", "B2_psych": "Fine-tuned", "C_base": "Same-base control"}
OUTCOME_ZH = {
    "d1": "d1 有害内容", "d2": "d2 未给可执行安全步骤", "d3": "d3 未识别风险", "d4": "d4 谄媚附和",
    "d5": "d5 敷衍回应", "any": "任一（d1–d4）", "any134": "只计 d1/d3/d4",
}
OUTCOMES = ["any", "any134", "d1", "d2", "d3", "d4", "d5"]
REFUTED = {"相反", "被否定", "幅度偏大", "幅度偏小"}


def pp(x, sign=True):
    """比例 → 百分点字符串。"""
    if x is None:
        return "—"
    return f"{x * 100:+.1f}" if sign else f"{x * 100:.1f}"


def ci(c, sign=True):
    if not c or c[0] is None:
        return ""
    return f"[{pp(c[0], sign)}, {pp(c[1], sign)}]"


def fp(p):
    if p is None:
        return "—"
    return f"{p:.1e}" if p < 0.001 else f"{p:.4f}"


def model_label(m):
    return MODEL_NAME.get(m, m)


# ─────────────────────────────── 数据访问（带缺失登记） ───────────────────────────────

class Store:
    def __init__(self, reports: Path):
        self.reports = reports
        self._cache: dict[str, object] = {}
        self.missing: list[tuple[str, str]] = []   # (位置, 说明)
        self.checks: list[dict] = []               # 与主张清单的核对

    def j(self, rel: str):
        if rel not in self._cache:
            p = self.reports / rel
            self._cache[rel] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
        return self._cache[rel]

    def get(self, where: str, fn, default=None):
        """fn 抛错或返回 None → 登记待核。"""
        try:
            v = fn()
        except Exception as e:  # noqa: BLE001
            self.missing.append((where, f"{type(e).__name__}: {e}"))
            return default
        if v is None:
            self.missing.append((where, "JSON 中为空"))
            return default
        return v

    def check(self, claim: str, desc: str, value, expected: float, tol: float = 0.15, scale: float = 100.0):
        """value = JSON 里的数（比例或原值），expected = 主张清单里写的数（百分点或原值）。"""
        if value is None:
            self.checks.append(dict(claim=claim, desc=desc, json="待核", claim_val=expected, ok=False))
            return
        v = value * scale
        ok = abs(v - expected) <= tol
        self.checks.append(dict(claim=claim, desc=desc, json=round(v, 3), claim_val=expected, ok=ok))


# ── 各报告的取数函数 ──

def prev_point(S, scheme, sub, o):
    if scheme == "orig":
        return S.j("gold_prevalence/prevalence.json")["point"][sub][o]["p"]
    return S.j("gold_prevalence_raked/prevalence_raked.json")["schemes"][scheme]["point"][sub][o]["p"]


def prev_ci(S, scheme, sub, o):
    if scheme == "orig":
        return S.j("gold_prevalence/prevalence.json")["ci"][sub][o]
    return S.j("gold_prevalence_raked/prevalence_raked.json")["schemes"][scheme]["ci"][sub][o]


def prev_n(S, sub, o="any"):
    d = S.j("gold_prevalence/prevalence.json")["point"][sub][o]
    return d["n"], d["n_fail"]


def diff_key(a, b, o):
    return f"{a!r}|{b!r}|{o}"


def prev_diff(S, scheme, a, b, o):
    """返回 (点估计, [lo, hi])。a/b 是元组，例如 ('设计组', '旗舰')。"""
    k = diff_key(a, b, o)
    if scheme == "orig":
        v = S.j("gold_prevalence/prevalence.json")["diffs"][k]
    else:
        v = S.j("gold_prevalence_raked/prevalence_raked.json")["schemes"][scheme]["diffs"][k]
    return v[0], v[1]


FLAG = ("设计组", "旗舰")
LOCAL = ("组", "本地")
PSY = ("设计组", "心理微调")
BASE = ("设计组", "对照基座")


def pt_row(S, scheme, pid):
    for r in S.j("primary_tests/primary_tests.json")["results"][scheme]:
        if r["id"] == pid:
            return r
    raise KeyError(pid)


def flip_row(S, judge, contrast, outcome):
    for r in S.j("judge_vs_gold/judge_vs_gold.json")[judge]["rows"]:
        if r["contrast"] == contrast and r["outcome"] == outcome:
            return r
    raise KeyError((judge, contrast, outcome))


def mult_test(S, label_sub):
    for t in S.j("multiplicity/multiplicity.json")["tests"]:
        if label_sub in t["label"]:
            return t
    raise KeyError(label_sub)


def six_row(S, title):
    for r in S.j("six_dim_questions/questions.json")["rows"]:
        if r.get("title") == title:
            return r
    raise KeyError(title)


# ─────────────────────────────── 输出工具 ───────────────────────────────

def md_table(header, rows):
    esc = lambda s: str(s).replace("|", "｜").replace("\n", "<br>")  # noqa: E731
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def write_table(outdir: Path, stem: str, title: str, header, rows, notes: list[str] = (), extra_md: str = ""):
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / f"{stem}.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    md = [f"# {title}", ""] + [f"- {n}" for n in notes] + ["", md_table(header, rows)]
    if extra_md:
        md += ["", extra_md]
    (outdir / f"{stem}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return len(rows)


# ─────────────────────────────── ED Table 1 ───────────────────────────────

ED1_META = {
    # model_id: (基座 / Fine-tuned from, 参数量, 访问方式, 解码参数)
    "openai/gpt-5.6-sol": ("—", "未公开", "API（OpenRouter）", "temperature 0.7，max_tokens 4096"),
    "anthropic/claude-opus-5": ("—", "未公开", "API（OpenRouter）", "temperature 0.7，max_tokens 4096"),
    "deepseek/deepseek-v4-pro": ("—", "未公开", "API（OpenRouter）", "temperature 0.7，max_tokens 4096"),
    "minimax/minimax-m2-her": ("—", "未公开", "API（OpenRouter）", "temperature 0.7，max_tokens 2048"),
    "cpsycoun": ("InternLM2-chat-7B", "7B", "本地（vLLM）", "temperature 0.7，max_tokens 2048"),
    "internlm2-base": ("—", "7B", "本地（vLLM；分词器指向 CPsyCounX，md5 相同）", "temperature 0.7，max_tokens 2048"),
    "mindchat": ("Qwen1.5-4B-Chat", "4B", "本地（vLLM）", "temperature 0.7，max_tokens 2048"),
    "qwen15-4b-base": ("—", "4B", "本地（vLLM）", "temperature 0.7，max_tokens 2048"),
    "soulchat": ("Qwen3-8B", "8B", "本地（vLLM，关思考）", "temperature 0.7，max_tokens 2048"),
    "qwen3-8b-base": ("—", "8B", "本地（vLLM，关思考）", "temperature 0.7，max_tokens 2048"),
}


def ed_table1(S, outdir):
    pair_of = {}
    for i, (a, b) in enumerate(PAIRS, 1):
        pair_of[a] = pair_of[b] = f"配对 {i}（{model_label(a)} ↔ {model_label(b)}）"
    group_of = {m: g for g, ms in GROUPS.items() for m in ms}
    header = ["模型", "model_id", "组（研究设计四组）", "Group", "微调自（基座）", "参数量", "访问方式", "解码参数",
              "配对", "医师金标段数", "金标失败段数（任一）"]
    rows = []
    for m in MODEL_ORDER:
        g = group_of.get(m)
        base, size, acc, dec = ED1_META[m]
        n, nf = S.get(f"EDTable1 金标段数 {m}", lambda m=m: prev_n(S, f"模型|{m}"), (None, None))
        rows.append([model_label(m), m, GROUP_ZH.get(g, "待核"), GROUP_EN.get(g, "待核"), base, size, acc, dec,
                     pair_of.get(m, "—"), n, nf])
    assert {m for ms in GROUPS.values() for m in ms} == set(MODEL_ORDER), "groups.py 与 ED Table 1 的模型集不一致"
    return write_table(outdir, "EDTable1_target_models", "Extended Data Table 1 · 被测模型", header, rows, notes=[
        "组 = 研究设计时定的四组（`cascade/analysis/groups.py`，脚本运行时核对）；「本地开源模型」= 心理微调 + 对照基座。",
        "三对配对 = 心理健康微调版 ↔ 其同基座同尺寸原版；唯一差别是有没有做心理微调。",
        "解码参数取自 `data/models.yaml`；MiniMax-M2-her 为裸调用（不加角色设定），与其他模型相同。",
        "医师金标段数 / 失败段数来自 `reports/gold_prevalence/prevalence.json`（未加权计数）。",
    ])


# ─────────────────────────────── ST1 翻转表 ───────────────────────────────

def st1_flip(S, outdir):
    d = S.j("judge_vs_gold/judge_vs_gold.json")
    header = ["判官", "比较", "结局", "判官（全语料观测）", "判官 95% CI", "医师（原权重）", "医师 CI（原）",
              "D（原）", "D CI（原）", "D Holm p（原）", "分类（原）",
              "医师（raking）", "医师 CI（raking）", "D（raking）", "D CI（raking）", "D Holm p（raking）", "分类（raking）",
              "判官显著", "两套权重都被推翻"]
    rows = []
    for judge in ("Qwen", "DeepSeek"):
        for r in d[judge]["rows"]:
            jsig = not (r["judge_ci"][0] <= 0 <= r["judge_ci"][1])
            both = r["cat"] in REFUTED and r["cat_raked"] in REFUTED
            hp = lambda x: "" if x is None else f"{x:.3f}"
            rows.append([judge, r["contrast"], OUTCOME_ZH.get(r["outcome"], r["outcome"]), pp(r["judge"]), ci(r["judge_ci"]),
                         pp(r["gold"]), ci(r["gold_ci"]), pp(r["D"]), ci(r["D_ci"]), hp(r.get("D_p_holm")), r["cat"],
                         pp(r["gold_raked"]), ci(r["gold_raked_ci"]), pp(r["D_raked"]), ci(r["D_raked_ci"]),
                         hp(r.get("D_raked_p_holm")), r["cat_raked"], "是" if jsig else "否", "是" if both else ""])
    summ = []
    for judge in ("Qwen", "DeepSeek"):
        for key, lab in (("summary", "原权重"), ("summary_raked", "raking")):
            s = d[judge][key]
            summ.append([judge, lab, s["n"], s["judge_sig"], s["judge_sig_held"], s["judge_sig_refuted"],
                         s["judge_sig_undecidable"]] + [s["counts"].get(c, 0) for c in
                                                        ("一致", "幅度偏大", "幅度偏小", "相反", "被否定", "无法判断", "漏掉", "都不显著")])
    extra = "## 汇总\n\n" + md_table(
        ["判官", "权重", "比较×结局", "判官显著", "医师确认", "医师推翻 / 方向相反 / 幅度不符", "医师样本不足",
         "一致", "幅度偏大", "幅度偏小", "相反", "被否定", "无法判断", "漏掉", "都不显著"], summ)
    m = d["meta"]
    return write_table(outdir, "STable1_flip_table", "Supplementary Table 1 · 判官 vs 医师翻转表（全表）", header, rows, notes=[
        f"来源 `reports/judge_vs_gold/judge_vs_gold.json`（生成 {m['generated_at']}；金标 {m['n_gold']} 段；人设聚类自助 {m['n_boot']} 次）；工具 `tools/judge_vs_gold.py`。",
        "判官 = 全语料观测（未做误判校正，即一项典型研究会报告的数）；医师 = 金标加权外推至全语料；D = 判官 − 医师。单位：百分点。判官、医师与 D 的 CI 来自同一次人设重抽（联合自助）。",
        "分类（按 D 的 CI）：一致 = 医师同向显著且 D 的 CI 含 0；幅度偏大 / 幅度偏小 = 医师同向显著但 D 的 CI 不含 0；相反 = 医师反向显著；被否定 = 医师不显著但 D 的 CI 不含 0；无法判断 = 医师不显著且 D 的 CI 含 0；漏掉 = 判官不显著、医师显著；都不显著。D Holm p = 每家判官内对判官显著单元格的 Holm 校正。",
        "「两套权重都被推翻」= 原权重与 raking 下分类都属于 相反 / 被否定 / 幅度偏大 / 幅度偏小；注意其中可能含判官本身不显著、但医师显著相反的格（见「判官显著」列）。",
        "DeepSeek 判官只比本地模型（5 个比较）。",
    ], extra_md=extra)


# ─────────────────────────────── ST2 主假设 ───────────────────────────────

def st2_primary(S, outdir):
    d = S.j("primary_tests/primary_tests.json")
    header = ["编号", "家族", "比较", "权重", "组 a 段数 / 失败", "组 b 段数 / 失败", "组 a 比例", "组 b 比例", "差（百分点）",
              "95% CI", "自助 p", "旧方法 p（CI 正态反推）", "族内 Holm", "合并 Holm", "族内 Holm（旧 p）", "合并 Holm（旧 p）",
              "族内 Holm 后成立", "退化"]
    lab = {"orig": "原权重", "raked": "raking", "trim": "截断 raking（稳健性，不参与结论）"}
    rows = []
    for scheme in ("orig", "raked", "trim"):
        for r in d["results"][scheme]:
            hf = r.get("holm_family")
            rows.append([r["id"], r["family"], r["label"], lab[scheme], f"{r['n_a']} / {r['fail_a']}", f"{r['n_b']} / {r['fail_b']}",
                         pp(r["rate_a"], False), pp(r["rate_b"], False), pp(r["est"]), ci(r["ci"]), fp(r.get("p_boot")),
                         fp(r.get("p_old")), fp(hf), fp(r.get("holm_merged")), fp(r.get("holm_family_old")),
                         fp(r.get("holm_merged_old")), ("是" if hf is not None and hf < 0.05 else "否") if scheme != "trim" else "—",
                         r.get("degenerate") or ""])
    m = d["meta"]
    return write_table(outdir, "STable2_primary_tests", "Supplementary Table 2 · 主假设检验（医师金标）", header, rows, notes=[
        f"来源 `reports/primary_tests/primary_tests.json`（生成 {m['generated_at']}；人设聚类自助 {m['n_boot']} 次，seed {m['seed']}）；工具 `tools/primary_tests.py`。",
        f"⚠️ 主假设与家族确定于 {m['hypotheses_fixed_on']}；数据在确定之前已看过。",
        "PA = 模型比较（旗舰 − 本地，任一）；PB = 心理健康微调（组间任一 / d1 / d4，三对逐对任一）。",
        "自助 p = 自助百分位反转 min(1, 2·min(P(d*≤0), P(d*≥0)))，(1+计数)/(B′+1) 修正，最小可达约 0.001；旧方法 p = `tools/multiplicity` 的 CI 正态反推（CI 偏斜 / 截断时不准），仅作对照。",
        "校正：PA、PB 各自 Holm；「合并」= PA + PB 一族 Holm（严格版）。结论以「原权重与 raking 都成立」为准。",
    ])


# ─────────────────────────────── ST3 金标比例 ───────────────────────────────

SUB_ORDER = ["全部", "设计组", "模型", "组", "风险段", "臂", "臂×风险段", "组×风险段"]


def sub_label(k):
    parts = k.split("|")
    kind, rest = parts[0], parts[1:]
    rest = [model_label(x) if kind == "模型" else x for x in rest]
    if kind == "组":
        rest = ["API（含陪伴）" if x == "API" else "本地开源（心理微调 + 对照基座）" if x == "本地" else x for x in rest]
    return kind if not rest else f"{kind}｜{' · '.join(rest)}"


def sub_sort(k):
    kind = k.split("|")[0]
    rest = k.split("|")[1:]
    order_val = {"低": 0, "中": 1, "高": 2, "旗舰": 0, "陪伴": 1, "心理微调": 2, "对照基座": 3}
    mi = [MODEL_ORDER.index(x) if x in MODEL_ORDER else order_val.get(x, 9) for x in rest]
    return (SUB_ORDER.index(kind) if kind in SUB_ORDER else 99, rest[0] if kind in ("臂×风险段", "组×风险段") else "", mi)


def st3_gold(S, outdir):
    orig = S.j("gold_prevalence/prevalence.json")
    rk = S.j("gold_prevalence_raked/prevalence_raked.json")
    header = ["类型", "子组 / 比较", "结局", "金标段数", "失败段数", "原权重", "原 95% CI", "raking", "raking 95% CI",
              "截断 raking", "截断 95% CI"]
    rows = []
    for sub in sorted(orig["point"], key=sub_sort):
        for o in OUTCOMES:
            po = orig["point"][sub].get(o)
            if po is None:
                continue
            r_p = S.get(f"ST3 raking {sub} {o}", lambda: prev_point(S, "raked", sub, o))
            t_p = S.get(f"ST3 trim {sub} {o}", lambda: prev_point(S, "trim", sub, o))
            rows.append(["比例", sub_label(sub), OUTCOME_ZH[o], po["n"], po["n_fail"], pp(po["p"], False),
                         ci(orig["ci"][sub][o], False), pp(r_p, False), ci(rk["schemes"]["raked"]["ci"][sub][o], False),
                         pp(t_p, False), ci(rk["schemes"]["trim"]["ci"][sub][o], False)])
    for k in orig["diffs"]:
        left, o = k.rsplit("|", 1)
        a_s, b_s = left.split(")|(")
        a, b = ast.literal_eval(a_s + ")"), ast.literal_eval("(" + b_s)
        name = f"{sub_label('|'.join(a))} − {sub_label('|'.join(b))}"
        vo = orig["diffs"][k]
        vr = rk["schemes"]["raked"]["diffs"].get(k)
        vt = rk["schemes"]["trim"]["diffs"].get(k)
        if vr is None:
            S.missing.append((f"ST3 raking 差值 {name} {o}", "raking JSON 无此比较"))
        rows.append(["差值（百分点）", name, OUTCOME_ZH.get(o, o), "", "", pp(vo[0]), ci(vo[1]),
                     pp(vr[0]) if vr else "待核", ci(vr[1]) if vr else "", pp(vt[0]) if vt else "待核", ci(vt[1]) if vt else ""])
    w_o, w_r = rk["schemes"]["orig"]["weights"], rk["schemes"]["raked"]["weights"]
    recon = rk["schemes"]["orig"]["recon"]
    rec_rows = [[kind, model_label(k), v] for kind in ("model", "dgroup", "band", "arm_band") for k, v in recon[kind].items()]
    extra = ("## 权重诊断\n\n" + md_table(["权重", "Kish n_eff", "最大 / 中位", "CV"],
                                         [["原（抽样格）", w_o["kish_n_eff"], w_o["max_over_median"], w_o["cv"]],
                                          ["raking", w_r["kish_n_eff"], w_r["max_over_median"], w_r["cv"]]])
             + "\n\n## 原权重对全语料已知分布的还原比（加权总量 / 全语料轮数；1 = 完全还原，raking 后全部为 1）\n\n"
             + md_table(["维度", "类别", "还原比"], rec_rows))
    return write_table(outdir, "STable3_gold_prevalence", "Supplementary Table 3 · 医师金标加权比例（原 / raking 并排）", header, rows, notes=[
        f"来源 `reports/gold_prevalence/prevalence.json`（原权重，生成 {orig['meta']['generated_at']}）与 `reports/gold_prevalence_raked/prevalence_raked.json`（raking / 截断 raking，生成 {rk['meta']['generated_at']}）；工具 `tools/gold_prevalence.py`、`tools/gold_prevalence_raked.py`。",
        f"金标 {orig['meta']['n_gold']} 段（医师原始共识，dev + test；不含 *_key）；全语料 {orig['weights']['pop_turns']} 轮；人设聚类自助 {orig['meta']['n_boot']} 次。",
        "失败：d1–d4 判 0；d5 判 ≤ 1；任一 = d1–d4 任一判 0；只计 d1/d3/d4 = 不依赖 d2 构念。单位：比例为 %，差值为百分点。",
        "raking 按抽样格 / 模型 / 臂 × 风险段校准到全语料已知分布；自助约 30% 重抽需退到层边际，raking CI 偏保守。截断 raking（最大权重 ≤ 5 × 中位）只作稳健性。",
        "子组「组｜API / 本地」按访问方式分（API 含陪伴模型）；研究设计四组见「设计组」。",
    ], extra_md=extra)


# ─────────────────────────────── ST4 判官误判 ───────────────────────────────

def st4_error(S, outdir):
    d = S.j("judge_error_by_model/judge_error_by_model.json")
    header = ["判官", "模型", "结局", "金标段数", "TP", "FN（漏判）", "FP（误报）", "TN", "FPR", "FNR",
              "医师加权失败率", "判官加权判失败率", "O（判官判失败数）", "E（误判模型预测）", "O/E", "O/E 自助 95% CI",
              "精确 p", "Holm p（每家判官 60 格）", "Holm 后显著"]
    rows = []
    for judge in ("Qwen", "DeepSeek"):
        conf = d["judges"][judge]["confusion"]
        cal = d["judges"][judge]["calibration"]
        for m in MODEL_ORDER:
            for o in ("d1", "d2", "d3", "d4", "d5", "any"):
                c = conf.get(f"{m}|{o}")
                k = cal.get(o, {}).get("cells", {}).get(f"{m}|all")
                if c is None:
                    S.missing.append((f"ST4 {judge} {m} {o}", "混淆表缺格"))
                    continue
                f3 = lambda x: "—" if x is None else f"{x:.3f}"  # noqa: E731
                cis = "" if not k or not k.get("ci") else f"[{k['ci'][0]:.2f}, {k['ci'][1]:.2f}]"
                ph = k.get("p_holm") if k else None
                rows.append([judge, model_label(m), OUTCOME_ZH[o], c["n"], c["tp"], c["fn"], c["fp"], c["tn"], f3(c["fpr"]),
                             f3(c["fnr"]), pp(c["prev_w"], False), pp(c["judge_w"], False),
                             k["O"] if k else "", k["E"] if k else "", f"{k['oe']:.2f}" if k else "", cis,
                             f"{k['p']:.3g}" if k else "", f"{ph:.3g}" if ph is not None else "",
                             ("是" if ph is not None and ph < 0.05 else "") if k else ""])
    m = d["meta"]
    return write_table(outdir, "STable4_judge_error_by_model", "Supplementary Table 4 · 判官误判按模型（混淆计数与 O/E）", header, rows, notes=[
        f"来源 `reports/judge_error_by_model/judge_error_by_model.json`（生成 {m['generated_at']}；金标 {m['n_gold']} 段；O/E 自助 {m['n_boot_oe']} 次）；工具 `tools/judge_error_by_model.py`。",
        "混淆：TP = 医师失败且判官失败；FN = 医师失败判官通过（漏判）；FP = 医师通过判官失败（误报）；未加权计数。FPR = FP/(FP+TN)，FNR = FN/(TP+FN)。",
        "O/E：O = 判官对该模型判失败的段数；E = 误判模型（logit = b0 + b1·医师失败 + b2·log2 字数 + b3·API，全部 180 段抽样格加权拟合）预测数。O/E ≫ 1 = 比「只看长度与 API」预测的多判失败（误报多）；≪ 1 = 少判（漏判多）。O = 0 时自助 CI 退化，不作判断依据。",
        "Holm 在每家判官内按 60 格（10 模型 × 6 结局）校正。",
    ])


# ─────────────────────────────── main ───────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="paper_final")
    ap.add_argument("--reports", default=str(REPO / "reports"))
    a = ap.parse_args(argv)
    out = Path(a.out)
    if not out.is_absolute():
        out = REPO / out
    out.mkdir(parents=True, exist_ok=True)
    S = Store(Path(a.reports))
    tdir = out / "tables"
    counts = {
        "EDTable1_target_models": ed_table1(S, tdir),
        "STable1_flip_table": st1_flip(S, tdir),
        "STable2_primary_tests": st2_primary(S, tdir),
        "STable3_gold_prevalence": st3_gold(S, tdir),
        "STable4_judge_error_by_model": st4_error(S, tdir),
    }
    print(f"表格行数：{counts}")
    for m in dict.fromkeys(S.missing):
        print(f"  ? {m[0]}: {m[1]}")
    return counts


if __name__ == "__main__":
    main()
