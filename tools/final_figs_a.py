"""论文终版主图 Fig. 1–3（研究设计 / 医师判定的不合格 / 判官 vs 医师）。只读报告 JSON，零模型调用、零网络。

    python tools/final_figs_a.py --out paper_final            # 三张全出
    python tools/final_figs_a.py --out paper_final --only F2  # 只出部分（F1 / F2 / F3）

输出：<out>/figures/main/Fig1_study_design、Fig2_psychiatrist_failures、Fig3_judge_vs_psychiatrist；
      <out>/figures/extended_data/EDFig5_flip_table_full（翻转表全图，原 Fig. 3b）（.pdf + .png 300 dpi）。

数据来源：
  reports/gold_prevalence/prevalence.json            医师金标，原抽样格权重
  reports/gold_prevalence_raked/prevalence_raked.json 医师金标，raking 权重（schemes.raked）
  reports/judge_vs_gold/judge_vs_gold.json           判官 vs 医师翻转表（Qwen / DeepSeek 判官）
  reports/primary_tests/primary_tests.json           主假设 Holm 校正（Fig. 3c 的 MindChat 注记）
  reports/ceval_v2/results.json                      语料规模（段数、轮数、人设、模型、配对）
  gold_judge180/report.json                          医师间加权 κ（三人两两、六维合并）

视觉系统 = tools/figstyle.py（只 import 不改）。本脚本新增一条全文约定：
  **医师金标 = 粗半透明 CI 带 + 竖线刻度**（原权重：粗带 + 实色刻度；raking：细带 + 浅色刻度）；
  LLM 判官仍为 ● Qwen / ■ DeepSeek（实心 = CI 不含 0），求助者视角评估器 ◆。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

from tools import figstyle as FS  # noqa: E402
from tools.figstyle import HEADLINE_PT, caption  # noqa: E402
from tools.figstyle import (ACCENT, API, COMPANION, DIM_N, DOUBLE, FAIL, GRID, GROUP_C, GROUP_N, GROUP_OF,  # noqa: E402
                            INK, INK2, J, JUDGE_M, LOCAL, MINUS, MM, MODEL_N, MUTED, PAIRS, PSYCH, REPO, RISK, RULE,
                            STAGE_N, STAGES, STEP, ZEBRA, grid, label, marker, raked_handle, raked_pt, refline, sig, style,
                            ylabels_clean)

DPI = 300
PHYS = STEP            # Fig. 3：医师金标（参照）用板岩灰；判官用金色 ACCENT
DESIGN = {"A_flagship": "旗舰", "companion": "陪伴", "B2_psych": "心理微调", "C_base": "对照基座"}
CONTRAST_N = {
    "旗舰 − 本地": "Flagship − local", "陪伴 − 旗舰": "Companion − flagship", "陪伴 − 本地": "Companion − local",
    "API（含陪伴）− 本地": "API incl. companion − local", "心理微调组 − 对照基座组": "Fine-tuned group − control group",
    "cpsycoun − internlm2-base": "CPsyCounX − InternLM2", "mindchat − qwen15-4b-base": "MindChat − Qwen1.5-4B",
    "soulchat − qwen3-8b-base": "SoulChat − Qwen3-8B", "高 − 低风险段（两臂合并）": "High − low risk (both arms)",
    "高 − 低风险段（fixed 臂）": "High − low risk (fixed arm)", "高 − 低风险段（本地模型）": "High − low risk (local models)"}
OUTCOMES = ("any", "d1", "d2", "d3", "d4", "d5")
HELD, REFUTED, UNDEC, NS = "held", "refuted", "undecidable", "ns"
CAT = {"一致": HELD, "幅度偏大": REFUTED, "幅度偏小": REFUTED, "相反": REFUTED, "被否定": REFUTED, "无法判断": UNDEC,
       "漏掉": REFUTED, "都不显著": NS}


# ================================================================ 公用
def tint(c, f):
    """与白色混合：f = 0 原色，f = 1 白。用实色代替 alpha，避免重叠处变深。"""
    r, g, b = to_rgb(c)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


def phys(ax, y, est, lo, hi, color, raked=False, horizontal=True, ms=None, z=3):
    """医师金标标记：粗 CI 带 + 竖线刻度（raking = 细带 + 浅色刻度）。"""
    band_c, tick_c = (tint(color, 0.72), color) if not raked else (tint(color, 0.62), tint(color, 0.38))
    lw, tms, tmw = (3.0, ms or 5.2, 1.5) if not raked else (1.3, (ms or 5.2) * 0.75, 1.0)
    if horizontal:
        ax.plot([lo, hi], [y, y], color=band_c, lw=lw, solid_capstyle="butt", zorder=z)
        ax.plot([est], [y], marker="|", ms=tms, mew=tmw, color=tick_c, ls="none", zorder=z + 1)
    else:
        ax.plot([y, y], [lo, hi], color=band_c, lw=lw, solid_capstyle="butt", zorder=z)
        ax.plot([y], [est], marker="_", ms=tms, mew=tmw, color=tick_c, ls="none", zorder=z + 1)


def judge_pt(ax, y, est, lo, hi, jk, color=ACCENT, ms=3.2, lw=0.7, z=5):
    ax.plot([lo, hi], [y, y], color=color, lw=lw, zorder=z, solid_capstyle="butt")
    # 一律实心：CI 横线本身已显示是否跨 0（2026-10-23 逐图审阅定，不再用空心编码显著性）
    marker(ax, est, y, color, JUDGE_M[jk], filled=True, ms=ms if jk == "qwen" else ms * 0.9, z=z + 1)


def pp(x):
    return f"{MINUS if x < 0 else '+'}{abs(x):.1f}"


def vs(label: str) -> str:
    """「A − B」→「A vs B」：图里不让读者做减法。"""
    return label.replace(" − ", " vs ")


def diff_words(la, lb, ci, a="Flagship", b="local"):
    """两组比较 → 一句话：直接写两个失败率（不让读者做减法、不会把百分点读成相对比例），
    差值的 95% CI 以「百分点」写出；CI 跨 0 时写「无明显差别」。"""
    lo, hi = ci
    head = f"{a} {la:.0f}% vs {b} {lb:.0f}%"
    if hi < 0 or lo > 0:
        x, y = sorted((abs(lo), abs(hi)))
        return f"{head}\n(difference 95% CI\n{x:.0f}–{y:.0f} points)"
    return f"{head}\nno clear difference\n(95% CI {lo:+.0f} to {hi:+.0f} points)".replace("-", MINUS)


def pct(x):
    return 100 * x


def save(fig, out: Path, name: str, canvas=False):
    """canvas=True：按画布原尺寸输出（Fig. 1 的元素铺满 180 mm 画布，tight 会多出约 2 mm、超过双栏上限）；
    审阅版画了整图标题（在画布外）时仍用 tight，以免标题被裁掉。"""
    out.mkdir(parents=True, exist_ok=True)
    FS.align_labels(fig)
    paths = []
    tight = not canvas or FS.SHOW_TITLE
    for ext in ("pdf", "png"):
        p = out / f"{name}.{ext}"
        fig.savefig(p, dpi=DPI, bbox_inches="tight" if tight else None, pad_inches=0.04)
        paths.append(p)
    plt.close(fig)
    return paths


def load_gold():
    orig = J("reports/gold_prevalence/prevalence.json")
    raked = J("reports/gold_prevalence_raked/prevalence_raked.json")["schemes"]["raked"]
    return orig, raked


def corpus_meta():
    return J("reports/ceval_v2/results.json")["meta"]


def deepseek_coverage(path="runs/v2/a4_dsnt_pairs.jsonl"):
    """DeepSeek 判官覆盖：被打分对话的被测模型数、带分数的轮数（逐行读，不硬编码）。"""
    import json
    models, turns = set(), 0
    with open(REPO / path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            models.add(d["target_model_id"])
            turns += sum(1 for t in d["turns"] if t.get("score"))
    return len(models), turns


# ================================================================ Fig. 1 研究设计
def fig1(out):
    meta = corpus_meta()
    g = J("reports/gold_prevalence/prevalence.json")
    jv = J("reports/judge_vs_gold/judge_vs_gold.json")
    kappa = J("gold_judge180/report.json")["iaa"]["all"]["pooled"]["wkappa"]
    n_pers, n_models, n_dlg, n_turn = len(meta["personas"]), len(meta["models"]), meta["n_dialogues"], meta["n_turns"]
    n_seed = n_dlg // (n_models * len(meta["arms"]) * n_pers)
    n_gold = g["meta"]["n_gold"]
    n_contrast = len(jv["Qwen"]["n_gold"])
    n_out = len({r["outcome"] for r in jv["Qwen"]["rows"]})

    W, H = 180, 92
    fig = plt.figure(figsize=(W * MM, H * MM))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    FS.fig_title(fig, "Study design: the same conversations rated by automated judges and by psychiatrists")

    def card(x, y, w, h, title, fc="white"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.4", fc=fc, ec=RULE, lw=0.6))
        ax.text(x + 2.2, y + h - 2.4, title, fontsize=6.6, fontweight="bold", va="top")

    def txt(x, y, s, fs=5.7, c=INK2, **kw):
        ax.text(x, y, s, fontsize=fs, color=c, va=kw.pop("va", "center"), **kw)

    def step(x, n, s):
        ax.add_patch(plt.Circle((x + 2.0, 88.2), 1.9, fc=STEP, ec="none"))
        ax.text(x + 2.0, 88.2, str(n), color="white", fontsize=6, fontweight="bold", ha="center", va="center")
        ax.text(x + 5.0, 88.2, s, fontsize=7, fontweight="bold", va="center")

    def arrow(x0, y0, x1, y1, st="-|>", c=MUTED, lw=0.6):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=st, mutation_scale=6, lw=lw, color=c,
                                     shrinkA=0, shrinkB=0))

    def dot(x, y, c, r=0.9):
        ax.add_patch(plt.Circle((x, y), r, fc=c, ec="none"))

    # ---- 1 模拟求助者
    step(2, 1, "Simulate")
    card(2, 25, 37, 59, "Simulated help-seekers")
    txt(4.2, 76.6, f"{n_pers} Chinese personas × {n_seed} seeds")
    txt(4.2, 73.0, "LLM patient simulator,")
    txt(4.2, 70.0, "companion-style chat register")
    txt(4.2, 64.6, "Expressed risk level (C-SSRS-derived)", fs=5.4, c=MUTED)
    names = [s.replace("\n", " ") for s in STAGE_N]
    for i, c in enumerate(RISK):                       # 6 档：自下而上
        yy = 30.5 + i * 5.6
        ax.add_patch(Rectangle((4.2, yy - 1.7), 3.4, 3.4, fc=c, ec="none"))
        txt(9.0, yy, f"{i + 1}  {names[i]}", fs=5.5, c=INK)
    ax.annotate("", xy=(36.0, 60.0), xytext=(36.0, 29.0),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.6, mutation_scale=5))
    txt(34.6, 44.5, "closer to suicidal action", fs=5.0, c=MUTED, ha="center", rotation=90)

    # ---- 2 两臂
    step(42, 2, "Converse")
    card(42, 56, 36, 28, "Fixed arm (exogenous)")
    txt(44.2, 77.2, "level assigned by design,", fs=5.5)
    txt(44.2, 74.4, "independent of the model", fs=5.5)
    txt(44.2, 71.4, "→ clean dose–response", fs=5.3, c=INK, fontweight="bold")
    # 小阶梯图：12 轮线性升档
    x0, y0, cw = 45.0, 59.0, 2.35
    for t in range(12):
        lev = min(5, t // 2)
        ax.add_patch(Rectangle((x0 + t * cw, y0), cw - 0.45, 1.6 + 1.85 * lev, fc=RISK[lev], ec="none"))
    txt(x0, 57.6, "turn 1", fs=5.0, c=MUTED, va="top")
    txt(x0 + 12 * cw, 57.6, "12", fs=5.0, c=MUTED, va="top", ha="right")

    card(42, 25, 36, 28, "Responsive arm")
    txt(44.2, 46.2, "level escalates only when the", fs=5.5)
    txt(44.2, 43.4, "reply is judged to abandon", fs=5.5)
    txt(44.2, 40.4, "→ ecological validity", fs=5.3, c=INK, fontweight="bold")
    seq = [0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3]
    x0, y0 = 45.0, 28.0
    for t, lev in enumerate(seq):
        ax.add_patch(Rectangle((x0 + t * cw, y0), cw - 0.45, 1.6 + 1.85 * lev, fc=RISK[lev], ec="none"))
    arrow(39, 54.5, 42, 66); arrow(39, 54.5, 42, 42)

    # ---- 3 十个模型
    step(81, 3, "Target models")
    card(81, 25, 52, 59, f"{n_models} models, 4 design groups")
    rows = [("A_flagship", "Flagship (API)", ["GPT-5.6 Sol", "Claude Opus 5", "DeepSeek-V4-Pro"]),
            ("companion", "Companion (API)", ["MiniMax-M2-her"])]
    yy = 77.0
    for gk, gname, ms in rows:
        dot(83.8, yy, GROUP_C[gk])
        txt(85.8, yy, gname, fs=5.8, c=INK, fontweight="bold")
        txt(85.8, yy - 3.0, "  ·  ".join(ms), fs=5.4)
        yy -= 7.2
    txt(83.0, 63.2, "Local open models (3 pairs)", fs=5.8, c=INK, fontweight="bold")
    txt(83.0, 59.8, "Mental-health fine-tuned", fs=5.3, c=INK2)
    txt(110.6, 59.8, "Same-base control", fs=5.3, c=INK2)
    full = {"cpsycoun": "CPsyCounX", "internlm2-base": "InternLM2-chat-7B", "mindchat": "MindChat-Qwen2-4B",
            "qwen15-4b-base": "Qwen1.5-4B-Chat", "soulchat": "SoulChat-R1-Qwen3-8B", "qwen3-8b-base": "Qwen3-8B"}
    for i, (a, b) in enumerate(meta["pairs"]):
        yy = 55.6 - i * 4.4
        dot(83.8, yy, GROUP_C[GROUP_OF[a]]); txt(85.6, yy, full.get(a, MODEL_N[a]), fs=5.4)
        txt(108.0, yy, "↔", fs=6, c=MUTED, ha="center")
        dot(111.4, yy, GROUP_C[GROUP_OF[b]]); txt(113.2, yy, full.get(b, MODEL_N[b]), fs=5.4)
    txt(83.0, 41.4, "3 pairs: same base, same size → compares each", fs=5.0, c=MUTED)
    txt(83.0, 38.8, "fine-tuned model with its base model", fs=5.0, c=MUTED)
    ax.plot([83, 131], [35.6, 35.6], color=GRID, lw=0.6)
    txt(83.0, 31.8, f"{n_dlg:,} dialogues  ·  {n_turn:,} model turns", fs=5.8, c=INK)
    txt(83.0, 28.6, f"= {n_models} models × {n_pers * n_seed} profiles × {len(meta['arms'])} arms", fs=5.0, c=MUTED)
    arrow(78, 70, 81, 58); arrow(78, 39, 81, 52)

    # ---- 4 三种评判
    step(136, 4, "Rate")
    card(136, 62, 42, 22, "Psychiatrists (gold standard)")
    phys(ax, 76.6, 139.8, 138.4, 141.2, PHYS)        # 图内图例：医师标记
    txt(143.2, 76.6, f"3 raters · {n_gold} turns", fs=5.5)
    txt(139.0, 72.6, f"enrichment-sampled, weighted back", fs=5.2)
    txt(139.0, 69.8, f"to the corpus", fs=5.2)
    txt(139.0, 66.0, f"inter-rater weighted κ {kappa:.2f} (all items)", fs=5.2)

    ds_models, ds_turns = deepseek_coverage()
    card(136, 37, 42, 22, "Two LLM judges")
    marker(ax, 139.6, 51.6, ACCENT, JUDGE_M["qwen"], filled=True, ms=3.6)
    txt(141.8, 51.6, "Qwen3.8-27B (local)", fs=5.4, c=INK)
    txt(141.8, 48.9, f"all {n_turn:,} turns (primary judge)", fs=5.0)
    marker(ax, 139.6, 45.4, ACCENT, JUDGE_M["deepseek"], filled=True, ms=3.3)
    txt(141.8, 45.4, "DeepSeek-V4-Pro", fs=5.4, c=INK)
    txt(141.8, 42.7, f"{ds_models} local open models only,", fs=5.0)
    txt(141.8, 40.2, f"{ds_turns:,} turns (cross-family replication)", fs=5.0)

    card(136, 25, 42, 9.6, "")
    marker(ax, 139.6, 29.8, ACCENT, JUDGE_M["ceval"], filled=True, ms=3.3)
    txt(141.8, 31.4, "Help-seeker-perspective rater", fs=5.8, c=INK, fontweight="bold")
    txt(141.8, 28.0, "felt left alone? drives responsive arm", fs=5.0)
    arrow(133, 66, 136, 72); arrow(133, 54.5, 136, 48); arrow(133, 43, 136, 30)

    # ---- 5 核心分析
    ax.add_patch(FancyBboxPatch((2, 2), 176, 19, boxstyle="round,pad=0,rounding_size=1.4", fc=ZEBRA, ec="none"))
    ax.add_patch(plt.Circle((5.8, 16.6), 1.9, fc=STEP, ec="none"))
    ax.text(5.8, 16.6, "5", color="white", fontsize=6, fontweight="bold", ha="center", va="center")
    ax.text(9.0, 16.6, "Core analysis: same question, two answers", fontsize=7, fontweight="bold", va="center")
    txt(9.0, 11.8, f"{n_contrast} comparisons (model groups, fine-tuning pairs, risk tiers) × {n_out} failure types:", fs=5.7)
    txt(9.0, 8.6, "for every conclusion the judge reaches, would the psychiatrists' ratings reach the same one?", fs=5.7)
    txt(9.0, 5.2, "Judge = observed on all turns it rated  ·  Psychiatrists = gold standard weighted to the same corpus",
        fs=5.0, c=MUTED)
    # 示意：判官点 vs 医师带
    bx = 132.0
    refline_x = bx + 14
    ax.plot([refline_x, refline_x], [4.0, 18.4], color=RULE, lw=0.6)
    for k, (je, pe, plo, phi, tag, tc) in enumerate(((-11, -1.0, -4.5, 1.5, "overturned", FAIL),
                                                     (-9, -12, -20, -5, "confirmed", STEP),
                                                     (-8, -3, -22, 14, "too few gold turns\nto tell", MUTED))):
        yy = 16.0 - k * 5.0
        marker(ax, refline_x + je, yy + 0.9, ACCENT, "o", filled=True, ms=3.2)
        phys(ax, yy - 0.9, refline_x + pe, refline_x + plo, refline_x + phi, PHYS, ms=4.2)
        ax.text(refline_x + 17, yy, tag, fontsize=5.5 if "\n" not in tag else 5.0, linespacing=1.05,
                color=tc if tc != MUTED else INK2, va="center",
                fontweight="bold" if tag == "overturned" else "normal")
    txt(refline_x, 2.9, "0", fs=5.0, c=MUTED, ha="center")
    # 图例（示意里两种标记）
    lx = 97.0
    marker(ax, lx + 1.8, 13.4, ACCENT, "o", filled=True, ms=3.2)
    txt(lx + 4.6, 13.4, "Judge", fs=5.0, c=INK2)
    phys(ax, 9.0, lx + 1.8, lx, lx + 3.6, PHYS, ms=4.2)
    txt(lx + 4.6, 9.0, "Psychiatrists (95% CI)", fs=5.0, c=INK2)
    return save(fig, out, "Fig1_study_design", canvas=True)


# ================================================================ Fig. 2 医师判定的不合格
def p2a(ax, orig, raked):
    outs = ["any", "any134", "d2", "d5", "d3", "d4", "d1"]
    for i, o in enumerate(outs):
        y = len(outs) - 1 - i
        c = orig["ci"]["全部"][o]
        phys(ax, y, pct(orig["point"]["全部"][o]["p"]), pct(c[0]), pct(c[1]), ACCENT)
        raked_pt(ax, pct(raked["point"]["全部"][o]["p"]), y, ACCENT)
        p = pct(orig["point"]["全部"][o]["p"])
        n_f = orig["point"]["全部"][o]["n_fail"]
        ax.text(pct(c[1]) + 1.2, y, f"{p:.1f}%", fontsize=5.5, va="center", color=INK)
        ax.text(53.5, y, f"{n_f}", fontsize=5.4, va="center", ha="right", color=INK2)
    ax.text(53.5, len(outs) - 0.35, "failed\nturns", fontsize=5.0, va="bottom", ha="right", color=MUTED, linespacing=1.1)
    ax.axhline(len(outs) - 2.5, color=RULE, lw=0.5)
    ax.set_yticks(range(len(outs)))
    labs = [DIM_N[o] for o in outs][::-1]                # d5 按 ≤ 1 分算失败：写图注
    ax.set_yticklabels(labs)
    for t in ax.get_yticklabels()[-2:]:
        t.set_fontweight("bold")
    ax.set_xlim(0, 54); ax.set_ylim(-0.6, len(outs) - 0.35); ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.spines["bottom"].set_bounds(0, 50)
    ylabels_clean(ax); grid(ax, "x")
    ax.set_xlabel("Replies below standard, all models (%)")
    caption(ax, f"All {orig['meta']['n_gold']} gold-standard turns")


def p2b(axs, orig, raked):
    """b：旗舰、陪伴、本地开源模型（心理微调 + 对照基座，即 C2 里的「本地」）三组；底部一句话写旗舰 vs 本地。"""
    groups = [("设计组|旗舰", "Flagship", GROUP_C["A_flagship"]), ("设计组|陪伴", "Companion", GROUP_C["companion"]),
              ("组|本地", "Local open models\n(fine-tuned + control)", FS.LOCAL_ALL)]
    for j, (ax, o) in enumerate(zip(axs, ("any", "any134"))):
        for i, (key, _, col) in enumerate(groups):
            y = len(groups) - 1 - i
            c = orig["ci"][key][o]
            phys(ax, y, pct(orig["point"][key][o]["p"]), pct(c[0]), pct(c[1]), col)
            raked_pt(ax, pct(raked["point"][key][o]["p"]), y, col)
        ax.set_xlim(0, 100); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_ylim(-1.7, len(groups) - 0.4)
        grid(ax, "x"); ylabels_clean(ax)
        ax.set_yticks(range(len(groups)))
        if j == 0:
            ns = [orig["point"][key][o]["n"] for key, _, _ in groups]
            ax.set_yticklabels([lab.replace("\n", f" (n = {n})\n") if "\n" in lab else f"{lab} (n = {n})"
                                for (_, lab, _), n in zip(groups, ns)][::-1], linespacing=1.15)
        else:
            ax.set_yticklabels([])
        ax.set_title(DIM_N[o], fontsize=6.2, fontweight="bold", loc="left", pad=3)
        d0, c0 = orig["diffs"][f"('设计组', '旗舰')|('组', '本地')|{o}"]
        la, lb = (pct(orig["point"][k][o]["p"]) for k in ("设计组|旗舰", "组|本地"))
        ax.text(0.5, 0.02, diff_words(la, lb, (pct(c0[0]), pct(c0[1]))), transform=ax.transAxes, ha="center",
                va="bottom", fontsize=5.3, color=INK, linespacing=1.25)
    axs[0].set_xlabel("Replies below standard (%)")
    axs[0].xaxis.set_label_coords(1.08, -0.2)


def p2c(ax, orig, raked):
    order = ["openai/gpt-5.6-sol", "anthropic/claude-opus-5", "deepseek/deepseek-v4-pro", "minimax/minimax-m2-her"]
    for a, b, _ in PAIRS:
        order += [a, b]
    ypos, y = {}, 0.0
    for i, m in enumerate(order):
        ypos[m] = y
        y -= 1.0
        if m in ("deepseek/deepseek-v4-pro", "minimax/minimax-m2-her", "internlm2-base", "qwen15-4b-base"):
            y -= 0.45
    for m in order:
        y, key, col = ypos[m], f"模型|{m}", GROUP_C[GROUP_OF[m]]
        pt = orig["point"][key]["any"]
        k, n = pt["n_fail"], pt["n"]
        c = orig["ci"][key]["any"]
        phys(ax, y, pct(orig["point"][key]["any"]["p"]), pct(c[0]), pct(c[1]), col)
        if k:
            raked_pt(ax, pct(raked["point"][key]["any"]["p"]), y, col)
        if k == 0:                                     # 0 失败：只画 0 处的刻度（三分法则上界写在图注）
            ax.plot([0], [y], marker="|", ms=5.2, mew=1.5, color=col, zorder=4, clip_on=False)
        ax.text(1.1, y, f"{k}/{n}", fontsize=5.6, va="center", ha="right", color=INK, transform=ax.get_yaxis_transform())
    ax.text(1.1, 0.75, "failed /\ngold turns", fontsize=5.0, va="bottom", ha="right", color=MUTED, linespacing=1.1,
            transform=ax.get_yaxis_transform())
    for m, gname in (("openai/gpt-5.6-sol", "Flagship"), ("minimax/minimax-m2-her", "Companion"),
                     (PAIRS[0][0], "Mental-health fine-tuned vs same-base control (3 pairs)")):
        ax.text(0.8, ypos[m] + 0.62, gname, fontsize=5.2, color=MUTED, va="center", ha="left")
    ax.set_yticks([ypos[m] for m in order]); ax.set_yticklabels([MODEL_N[m] for m in order])
    ax.set_xlim(0, 100); ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_ylim(min(ypos.values()) - 0.7, 1.05)
    ylabels_clean(ax); grid(ax, "x")
    ax.set_xlabel(f"{DIM_N['any']} (%)")
    for t, m in zip(ax.get_yticklabels(), order):
        t.set_color(INK)


def fig2(out):
    orig, raked = load_gold()
    W, H = 180, 116
    fig = plt.figure(figsize=(W * MM, H * MM))
    FS.fig_title(fig, "By psychiatrists' ratings, one in three replies fell short, and models differed widely")
    # a：左上；b：左下（两小格）；c：右侧通栏
    ax_a = fig.add_axes([0.205, 0.565, 0.29, 0.325])
    p2a(ax_a, orig, raked); label(ax_a, "a", dx=-98, dy=30)
    FS.headline(ax_a, "One in three replies fell below the minimum standard,\n"
                "mostly for lack of an actionable safety step", dy=16)
    ax_b1 = fig.add_axes([0.205, 0.095, 0.135, 0.3]); ax_b2 = fig.add_axes([0.375, 0.095, 0.13, 0.3])
    p2b((ax_b1, ax_b2), orig, raked); label(ax_b1, "b", dx=-98, dy=26)
    FS.headline(ax_b1, "Flagship models did better only on giving a safety step", dy=14)
    ax_c = fig.add_axes([0.665, 0.095, 0.27, 0.795])
    p2c(ax_c, orig, raked); label(ax_c, "c", dx=-66, dy=30)
    caption(ax_c, "Below minimum standard, by model")
    FS.headline(ax_c, "GPT and Claude: no failures seen;\nthe companion model failed most", dy=16)
    # 图例（只画出现的标记：医师 原权重 / raking）
    h = [Line2D([], [], color=tint(INK2, 0.72), lw=3.0, marker="|", ms=5.2, mew=1.5, mec=INK2,
                label="original weights (95% CI)"),
         raked_handle(INK2, "raking weights")]
    fig.text(0.60, 0.012, "Psychiatrist gold standard:", fontsize=5.6, color=INK, va="center", ha="right")
    fig.legend(handles=h, loc="center left", bbox_to_anchor=(0.603, 0.012), ncol=2, fontsize=5.6, handlelength=2.4,
               handletextpad=0.8, columnspacing=2.2)
    return save(fig, out, "Fig2_psychiatrist_failures")


# ================================================================ Fig. 3 判官 vs 医师
def jv_rows():
    jv = J("reports/judge_vs_gold/judge_vs_gold.json")
    ds = {(r["contrast"], r["outcome"]): r for r in jv["DeepSeek"]["rows"]}
    return jv, jv["Qwen"]["rows"], ds


def p3a(ax, jv):
    """a：每家判官的显著结论，医师推翻 / 确认 / 金标不够判断。原权重粗条，raking 细条紧贴其下。"""
    rows, y = [], 0.0
    for jk, jn in (("Qwen", "Qwen judge"), ("DeepSeek", "DeepSeek judge")):
        for sk, rk in (("summary", False), ("summary_raked", True)):
            st = jv[jk][sk]
            rows.append((jn, st["judge_sig"], st["judge_sig_refuted"], st["judge_sig_held"],
                         st["judge_sig_undecidable"], rk, y))
            y -= 0.62 if not rk else 1.05
    for jn, n, r, h, u, rk, yy in rows:
        x, hgt = 0, (0.62 if not rk else 0.36)
        for v, c, nm in ((r, FAIL, "refuted"), (h, STEP, "confirmed"), (u, tint(MUTED, 0.6), "undecidable")):
            fc = c if not rk else tint(c, 0.5)
            ax.barh(yy, v, left=x, height=hgt, color=fc, ec="white", lw=0.5, zorder=3)
            if v >= 3:
                ax.text(x + v / 2, yy, str(v), ha="center", va="center", fontsize=5.6 if not rk else 4.9, zorder=4,
                        color="white" if (nm != "undecidable" and not rk) else INK,
                        fontweight="bold" if nm == "refuted" else "normal")
            x += v
        if not rk:
            ax.text(-1.2, yy, jn, ha="right", va="center", fontsize=5.8, color=INK)
            ax.text(n + 1.0, yy, f"of {n}", va="center", fontsize=5.3, color=INK2)
        else:
            ax.text(-1.2, yy, "raking weights", ha="right", va="center", fontsize=5.0, color=MUTED)
    ax.set_xlim(0, 58); ax.set_ylim(rows[-1][-1] - 0.6, 0.9)
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_xticks([0, 10, 20, 30, 40, 50]); ax.spines["bottom"].set_bounds(0, 50)
    ax.set_xlabel("Judge conclusions that reached significance (n)")
    st = jv["Qwen"]["summary"]
    r, h, u = st["judge_sig_refuted"], st["judge_sig_held"], st["judge_sig_undecidable"]
    for x, t, c in ((r / 2, "overturned by\npsychiatrists", FAIL), (r + h / 2, "confirmed", INK2),
                    (r + h + u / 2, "too few gold turns\nto tell", INK2)):
        ax.text(x, 0.42, t, ha="center", va="bottom", fontsize=5.1, color=c, linespacing=1.1,
                fontweight="bold" if c == FAIL else "normal")


def _status_glyph(ax, x, y, st, trans):
    if st == HELD:
        ax.text(x, y, "=", transform=trans, ha="center", va="center", fontsize=6.4, color=STEP, fontweight="bold")
    elif st == REFUTED:
        ax.plot([x], [y], transform=trans, marker="x", ms=3.2, mew=1.0, color=FAIL, ls="none", clip_on=False)
    elif st == UNDEC:
        ax.text(x, y, "?", transform=trans, ha="center", va="center", fontsize=5.6, color=MUTED)
    else:
        ax.text(x, y, "·", transform=trans, ha="center", va="center", fontsize=6, color=RULE)


def p3b(ax, blocks, rows, ds, n_gold, first_col):
    by = {(r["contrast"], r["outcome"]): r for r in rows}
    y = 0.0
    trans = ax.get_yaxis_transform()
    yt, yl, bold = [], [], []
    for bi, c in enumerate(blocks):
        na, nb = n_gold[c]
        ax.text(-0.015, y, vs(ROW_N[c]), transform=trans, ha="right", va="center", fontsize=6.0, fontweight="bold",
                color=INK)
        ax.text(-0.015, y - 0.62, f"gold turns {na} vs {nb}", transform=trans, ha="right", va="center", fontsize=5.0,
                color=MUTED)
        y -= 1.25
        for o in OUTCOMES:
            r = by[(c, o)]
            st0, st1 = CAT[r["cat"]], CAT[r["cat_raked"]]
            if st0 == REFUTED or st1 == REFUTED:
                both = st0 == REFUTED and st1 == REFUTED
                ax.axhspan(y - 0.5, y + 0.5, color=tint(FAIL, 0.80 if both else 0.91), lw=0, zorder=0)
            judge_pt(ax, y + 0.17, pct(r["judge"]), pct(r["judge_ci"][0]), pct(r["judge_ci"][1]), "qwen")
            phys(ax, y - 0.17, pct(r["gold"]), pct(r["gold_ci"][0]), pct(r["gold_ci"][1]), PHYS, ms=4.0)
            raked_pt(ax, pct(r["gold_raked"]), y - 0.17, PHYS, ms=3.0)
            if (c, o) in EXAMPLE_CELL:                    # 指向主图 Fig. 3c / 3d 的放大例子
                ax.text(1.02, y, "Fig. 3" + EXAMPLE_CELL[(c, o)], transform=trans, ha="left", va="center",
                        fontsize=5.0, fontweight="bold", color=INK)
            yt.append(y); yl.append(DIM_N[o])
            bold.append(st0 == REFUTED and st1 == REFUTED)
            y -= 1.0
        y -= 0.35
    ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=5.4)
    for t, b in zip(ax.get_yticklabels(), bold):
        if b:
            t.set_fontweight("bold")
    ax.set_ylim(y + 0.6, 0.75)
    ax.set_xlim(-80, 100); ax.set_xticks([-75, -50, -25, 0, 25, 50, 75, 100])
    ax.set_xticklabels([f"{MINUS}75", f"{MINUS}50", f"{MINUS}25", "0", "25", "50", "75", "100"])
    ylabels_clean(ax); grid(ax, "x"); refline(ax, x=0)
    ax.set_xlabel("Difference in failure rate (%)")
    ax.xaxis.set_label_coords(0.5, -0.035)


def p3c(ax, rows, ds, keys, xlim, finding, pair, note=None, ylabels=None):
    """一个翻转例子：判官（Qwen ●、DeepSeek ■）对医师（原权重 CI 带 + raking ▽）。标题 = 一句话结论，副标题 = 哪一对。"""
    by = {(r["contrast"], r["outcome"]): r for r in rows}
    for i, (c, o) in enumerate(keys):
        y = len(keys) - 1 - i
        r, d = by[(c, o)], ds.get((c, o))
        ax.plot([pct(r["judge"]), pct(r["gold"])], [y + 0.24, y - 0.2], color=tint(FAIL, 0.55), lw=0.8, zorder=2,
                ls=(0, (2, 1.2)))
        judge_pt(ax, y + 0.24, pct(r["judge"]), pct(r["judge_ci"][0]), pct(r["judge_ci"][1]), "qwen", ms=3.8, lw=0.9)
        if d is not None:
            judge_pt(ax, y + 0.04, pct(d["judge"]), pct(d["judge_ci"][0]), pct(d["judge_ci"][1]), "deepseek",
                     ms=3.8, lw=0.9)
        phys(ax, y - 0.2, pct(r["gold"]), pct(r["gold_ci"][0]), pct(r["gold_ci"][1]), PHYS, ms=5.2)
        raked_pt(ax, pct(r["gold_raked"]), y - 0.2, PHYS, ms=3.2)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(ylabels or [DIM_N[o] for _, o in keys][::-1])
    ax.set_xlim(*xlim); ax.set_ylim(-0.65, len(keys) - 0.35)
    ylabels_clean(ax); grid(ax, "x"); refline(ax, x=0)
    if not FS.SHOW_TITLE:                        # 投稿版：不画结论句，只留哪一对（灰色小标题）
        finding = ""
        ax._cl_captop = 4 + 7.5
    ax.text(0.0, 1.02, finding, transform=ax.transAxes, fontsize=6.2, fontweight="bold", va="bottom", ha="left",
            color=INK)
    ax.annotate(pair, xy=(0, 1), xycoords="axes fraction", xytext=(0, 11 if FS.SHOW_TITLE else 4),
                textcoords="offset points",
                fontsize=5.4, color=MUTED, va="bottom", ha="left")
    ax.set_xlabel("Difference in failure rate (%)")
    first = keys[0][0].split(" − ")[0]
    first = MODEL_N.get(first, first)
    for x, ha, t in ((0, "left", f"← {first} fails less"), (1, "right", "more →")):
        ax.annotate(t, xy=(x, 0), xycoords="axes fraction", xytext=(0, -21), textcoords="offset points",
                    ha=ha, va="top", fontsize=5.0, color=INK2)
    if note:
        ax.text(1.04, 0.5, note, transform=ax.transAxes, fontsize=5.0, color=INK2, va="center", ha="left",
                linespacing=1.3)


def mindchat_note():
    pt = J("reports/primary_tests/primary_tests.json")["results"]
    res = {}
    for sch in ("orig", "raked"):
        for r in pt.get(sch, []):
            if r["a"] == ["模型", "mindchat"] and r["outcome"] == "any":
                res[sch] = r
    if len(res) < 2:
        return None
    h0, h1 = res["orig"]["holm_family"], res["raked"]["holm_family"]
    # CI 与面板里医师的带同源（判官 vs 医师联合自助）；P 值来自主假设检验
    jr = [r for r in J("reports/judge_vs_gold/judge_vs_gold.json")["Qwen"]["rows"]
          if r["contrast"] == "mindchat − qwen15-4b-base" and r["outcome"] == "any"]
    lo, hi = jr[0]["gold_ci"]
    pt = J("reports/gold_prevalence/prevalence.json")["point"]
    la, lb = (pct(pt[f"模型|{m}"]["any"]["p"]) for m in ("mindchat", "qwen15-4b-base"))
    return (f"Psychiatrists: {la:.0f}% vs {lb:.0f}%\n(difference 95% CI {pct(lo):.0f}–{pct(hi):.0f} points)\n"
            f"Corrected for multiple tests:\np = {h0:.3f} original, p = {h1:.2f} raking")


# 判定矩阵：每格一种颜色（Qwen 判官 × 两套权重合成），右上角小方块 = DeepSeek 判官
VERDICT = (("ref2", "Overturned, both weightings"), ("ref1", "Overturned, one weighting"),
           ("held2", "Confirmed, both weightings"), ("held1", "Confirmed, one weighting"),
           ("undec", "Too few gold turns to tell"), ("ns", "Neither significant"))
MATRIX_GROUPS = (("Mental-health fine-tuning", ["心理微调组 − 对照基座组", "soulchat − qwen3-8b-base",
                                                 "mindchat − qwen15-4b-base", "cpsycoun − internlm2-base"]),
                 ("Model groups", ["旗舰 − 本地", "陪伴 − 旗舰", "陪伴 − 本地", "API（含陪伴）− 本地"]),
                 ("Risk band", ["高 − 低风险段（两臂合并）", "高 − 低风险段（fixed 臂）", "高 − 低风险段（本地模型）"]))
COL_N = {"any": "Below\nstandard", "d1": "Harmful\ncontent", "d2": "No safety\nstep",
         "d3": "Risk not\nrecognized", "d4": "Syco-\nphancy", "d5": "Perfunc-\ntory"}
ROW_N = {**CONTRAST_N, "心理微调组 − 对照基座组": "Fine-tuned group − control group", "API（含陪伴）− 本地": "Flagship + companion − local",
         "高 − 低风险段（两臂合并）": "High − low (both arms)", "高 − 低风险段（fixed 臂）": "High − low (fixed arm)",
         "高 − 低风险段（本地模型）": "High − low (local models)"}
EXAMPLE_CELL = {("soulchat − qwen3-8b-base", "d1"): "c", ("soulchat − qwen3-8b-base", "d3"): "c",
                ("soulchat − qwen3-8b-base", "d4"): "c", ("mindchat − qwen15-4b-base", "any"): "d"}


def verdict_of(r) -> str:
    ss = (CAT[r["cat"]], CAT[r["cat_raked"]])
    if ss.count(REFUTED) == 2:
        return "ref2"
    if REFUTED in ss:
        return "ref1"
    if ss.count(HELD) == 2:
        return "held2"
    if HELD in ss:
        return "held1"
    if UNDEC in ss:
        return "undec"
    return "ns"


def verdict_color(v):
    return {"ref2": FAIL, "ref1": tint(FAIL, 0.62), "held2": STEP, "held1": tint(STEP, 0.55),
            "undec": tint(MUTED, 0.72), "ns": "white"}[v]


def _gold_sig_sign(r):
    """医师在任一套权重下显著时的方向（+1 / −1），都不显著时为 0。"""
    for g, ci in ((r["gold"], r["gold_ci"]), (r["gold_raked"], r["gold_raked_ci"])):
        if g is not None and sig(ci):
            return 1 if g > 0 else -1
    return 0


def judge_not_sig_reversal(r) -> bool:
    """判官不显著、医师显著且方向相反（†）：矩阵里也涂成「推翻」，但不计入面板 a 的「判官显著结论」。"""
    gs = _gold_sig_sign(r)
    return not sig(r["judge_ci"]) and verdict_of(r) in ("ref1", "ref2") and gs != 0 and (r["judge"] > 0) != (gs > 0)


def judge_not_sig_missed(r) -> bool:
    """判官不显著、医师显著且方向相同（‡）：判官漏掉了医师发现的差别，不计入面板 a。"""
    gs = _gold_sig_sign(r)
    return not sig(r["judge_ci"]) and verdict_of(r) in ("ref1", "ref2") and gs != 0 and (r["judge"] > 0) == (gs > 0)


def p3_matrix(ax, rows, ds):
    by = {(r["contrast"], r["outcome"]): r for r in rows}
    y, yt, yl = 0.0, [], []
    trans = ax.get_yaxis_transform()
    for gname, comps in MATRIX_GROUPS:
        ax.text(-0.02, y - 0.05, gname, transform=trans, ha="right", va="center", fontsize=5.3, color=MUTED,
                fontweight="bold")
        y -= 0.8
        for c in comps:
            for j, o in enumerate(OUTCOMES):
                r = by[(c, o)]
                v = verdict_of(r)
                ax.add_patch(Rectangle((j - 0.46, y - 0.43), 0.92, 0.86, fc=verdict_color(v),
                                       ec=RULE if v == "ns" else "white", lw=0.5, zorder=2))
                d = ds.get((c, o))
                dv = verdict_of(d) if d is not None else None
                if dv in ("ref2", "ref1", "held2", "held1"):   # DeepSeek：右上角小方块，只画医师能检验的格
                    ax.add_patch(Rectangle((j + 0.06, y + 0.0), 0.36, 0.4, fc=verdict_color(dv),
                                           ec="white", lw=0.7, zorder=3))
                mark = (EXAMPLE_CELL.get((c, o)) or "") + ("†" if judge_not_sig_reversal(r) else "") + \
                    ("‡" if judge_not_sig_missed(r) else "")
                if mark:
                    dark = v in ("ref2", "held2")
                    ax.text(j - 0.14, y - 0.04, mark, ha="center", va="center", fontsize=5.6, fontweight="bold",
                            color="white" if dark else INK, zorder=4)
            yt.append(y); yl.append(vs(ROW_N[c]))
            y -= 1.0
        y -= 0.25
    ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=5.5)
    ax.set_xticks(range(len(OUTCOMES)))
    ax.set_xticklabels([COL_N[o] for o in OUTCOMES], fontsize=5.1, linespacing=1.15)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_xlim(-0.55, len(OUTCOMES) - 0.45); ax.set_ylim(y + 0.7, 0.4)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ylabels_clean(ax)


def fig3(out):
    jv, rows, ds = jv_rows()
    T = 9                                   # 给面板结论标题留的高度（mm）
    W, H = 180, 127 + T
    fig = plt.figure(figsize=(W * MM, H * MM))
    fy = lambda mm: 1 - (mm + T) / H  # noqa: E731
    FS.fig_title(fig, "Many comparative conclusions reached by automated judges were not supported by psychiatrist ratings")
    # a：总账
    ax_a = fig.add_axes([0.2, fy(34), 0.27, (34 - 6) / H])
    p3a(ax_a, jv); label(ax_a, "a", dx=-100, dy=30)
    caption(ax_a, "Judge conclusions checked against psychiatrists")
    sq, sr = jv["Qwen"]["summary"], jv["Qwen"]["summary_raked"]
    FS.headline(ax_a, f"Psychiatrists could check {sq['judge_sig_held'] + sq['judge_sig_refuted']} of the Qwen judge's "
                f"{sq['judge_sig']} conclusions\nand overturned {sq['judge_sig_refuted']} "
                f"(raking weights: {sr['judge_sig_refuted']} of {sr['judge_sig_held'] + sr['judge_sig_refuted']})", dy=22)
    # b：判定矩阵（Qwen 判官，两套权重合成；DeepSeek 在右上角）
    ax_b = fig.add_axes([0.2, fy(114), 0.31, (114 - 65) / H])
    p3_matrix(ax_b, rows, ds); label(ax_b, "b", dx=-100, dy=44)
    FS.headline(ax_b, "Overturns cluster in fine-tuning comparisons\nand in perfunctory replies", dy=25)
    h = [Rectangle((0, 0), 1, 1, fc=verdict_color(k), ec=RULE if k == "ns" else "none", lw=0.5, label=lab)
         for k, lab in VERDICT]
    h += [Line2D([], [], ls="none", marker="s", ms=2.6, mfc=STEP, mec="white", mew=0.4,
                 label="corner square: DeepSeek judge (local models)"),
          Line2D([], [], ls="none", marker="$\\mathrm{c}$", ms=3.6, mew=0, color=INK,
                 label="c, d: cell enlarged in panel c or d"),
          Line2D([], [], ls="none", marker="$†$", ms=4.0, color=INK,
                 label="judge not significant, psychiatrists\nsignificant in the opposite direction"),
          Line2D([], [], ls="none", marker="$‡$", ms=4.0, color=INK,
                 label="judge not significant, psychiatrists\nsignificant in the same direction")]
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.035, fy(115.5)), ncol=3, fontsize=5.0,
               handlelength=1.3, handleheight=1.0, columnspacing=1.2, labelspacing=0.45)
    # c、d：两种翻转
    soul = [("soulchat − qwen3-8b-base", o) for o in ("d1", "d3", "d4")]
    ax_c = fig.add_axes([0.66, fy(40), 0.17, (40 - 6) / H])      # 顶边与 a 对齐
    p3c(ax_c, rows, ds, soul, (-25, 6), "Judges: safer.  Psychiatrists: no difference", "SoulChat vs Qwen3-8B")
    label(ax_c, "c", dx=-62, dy=22)
    ax_d = fig.add_axes([0.66, fy(79), 0.17, (79 - 65) / H])     # 顶边与 b 对齐
    p3c(ax_d, rows, ds, [("mindchat − qwen15-4b-base", "any")], (-25, 95), "Judges: safer.  Psychiatrists: worse",
        "MindChat vs Qwen1.5-4B", note=mindchat_note(), ylabels=["Below min.\nstandard"])
    ax_d.set_xticks([0, 50])
    label(ax_d, "d", dx=-62, dy=22)
    hs = [Line2D([], [], color=ACCENT, lw=0.7, marker="o", ms=3.2, mfc=ACCENT, mec="white", mew=0.5,
                 label="Qwen judge (all turns)"),
          Line2D([], [], color=ACCENT, lw=0.7, marker="s", ms=2.9, mfc=ACCENT, mec="white", mew=0.5,
                 label="DeepSeek judge (local models)"),
          Line2D([], [], color=tint(PHYS, 0.72), lw=3.0, marker="|", ms=5.0, mew=1.5, mec=PHYS,
                 label="Psychiatrists, original weights (95% CI)"),
          raked_handle(PHYS, "Psychiatrists, raking weights", ms=3.2),
          Line2D([], [], color=tint(FAIL, 0.55), lw=0.8, ls=(0, (2, 1.2)), label="same comparison: judge → psychiatrists")]
    fig.legend(handles=hs, loc="upper left", bbox_to_anchor=(0.56, fy(91)), ncol=2, fontsize=5.0, handlelength=2.0,
               columnspacing=1.2)
    return save(fig, out, "Fig3_judge_vs_psychiatrist")


def edfig5(out):
    """ED Fig. 5：翻转表全图（66 格；判官与医师的估计和 95% CI，原权重与 raking 都画 CI）。原主图 Fig. 3b。"""
    jv, rows, ds = jv_rows()
    n_gold = jv["Qwen"]["n_gold"]
    W, H = 180, 160
    fig = plt.figure(figsize=(W * MM, H * MM))
    fy = lambda mm: 1 - mm / H  # noqa: E731
    FS.fig_title(fig, "All 66 comparisons: judge and psychiatrist estimates with 95% CIs")
    col1 = ["心理微调组 − 对照基座组", "soulchat − qwen3-8b-base", "mindchat − qwen15-4b-base", "cpsycoun − internlm2-base",
            "旗舰 − 本地"]
    col2 = ["陪伴 − 旗舰", "陪伴 − 本地", "API（含陪伴）− 本地", "高 − 低风险段（两臂合并）", "高 − 低风险段（fixed 臂）",
            "高 − 低风险段（本地模型）"]
    assert set(col1 + col2) == set(n_gold), "比较集合与 judge_vs_gold.json 不一致"
    rows_b = 6 * 6 + 6 * 1.25 + 6 * 0.35
    top_b, bot_b = 12, 156
    h1 = bot_b - top_b
    ax_b1 = fig.add_axes([0.21, fy(bot_b), 0.27, h1 / H])
    ax_b2 = fig.add_axes([0.68, fy(bot_b), 0.27, h1 / H])
    p3b(ax_b1, col1, rows, ds, n_gold, True)
    p3b(ax_b2, col2, rows, ds, n_gold, False)
    ax_b1.set_ylim(-rows_b + 0.25, 0.75); ax_b2.set_ylim(-rows_b + 0.25, 0.75)
    h = [Line2D([], [], color=ACCENT, lw=0.7, marker="o", ms=3.2, mfc=ACCENT, mec="white", mew=0.5,
                label="Qwen judge (95% CI)"),
         Line2D([], [], color=tint(PHYS, 0.72), lw=3.0, marker="|", ms=5.0, mew=1.5, mec=PHYS,
                label="Psychiatrists, original weights (95% CI)"),
         raked_handle(PHYS, "Psychiatrists, raking weights", ms=3.0),
         Rectangle((0, 0), 1, 1, fc=tint(FAIL, 0.80), ec="none", label="Qwen judge refuted, both weightings"),
         Rectangle((0, 0), 1, 1, fc=tint(FAIL, 0.91), ec="none", label="Qwen judge refuted, one weighting")]
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.03, fy(4)), ncol=5, fontsize=5.4, handlelength=2.0,
               columnspacing=1.4)
    return save(fig, out, "EDFig5_flip_table_full")


# ================================================================ 入口
FIGS = {"F1": fig1, "F2": fig2, "F3": fig3}
ED = {"E5": edfig5}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="paper_final",
                    help="输出根目录（主图 <out>/figures/main/，ED <out>/figures/extended_data/）")
    ap.add_argument("--only", nargs="*", choices=sorted(FIGS) + sorted(ED), help="只出部分：F1 F2 F3 E5")
    ap.add_argument("--titles", action="store_true", help="画整张图顶部的一句话标题（审阅用；投稿稿不画）")
    a = ap.parse_args(argv)
    FS.SHOW_TITLE = a.titles
    out = Path(a.out)
    if not out.is_absolute():
        out = REPO / out
    style()
    plt.rcParams["savefig.dpi"] = DPI
    for k in (a.only or sorted(FIGS) + sorted(ED)):
        fn, sub = (FIGS[k], "main") if k in FIGS else (ED[k], "extended_data")
        for p in fn(out / "figures" / sub):
            print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
