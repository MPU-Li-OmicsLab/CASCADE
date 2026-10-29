"""论文终稿图（B 组）：主图 Fig. 4–6 + Extended Data Fig. 1–4。

    python tools/final_figs_b.py --out paper_final [--only F4 F5 F6 E1 E2 E3 E4]

数据一律从 reports/ 下的 JSON（及一处手工诊断 md 的表格行）读取，不硬编码可读取的数字。
视觉系统只 import tools/figstyle.py：颜色只编码模型组，非模型组数据用金色 ACCENT；
形状编码测量工具（Qwen 判官 ●、DeepSeek 判官 ■、求助者视角评估器 ◆）；
医师金标 = 粗半透明 CI 带 + 竖线刻度（原权重）；主图里 raking 只画点估计（空心倒三角 ▽，CI 在 ED Fig. 3 / ST3）；
与判官并排时用板岩灰；实心 = CI 不含 0。Fig. 4a 为 O/E 热图（红 = 误报、蓝 = 漏判，CI 含 1 冲淡，逐格 CI 在 ST4）。
"""
from __future__ import annotations

import argparse
import collections
import math
import re
import statistics
import sys
from pathlib import Path

try:  # 作为包导入（测试）
    from tools import figstyle as F
except ImportError:  # 作为脚本运行：tools/ 在 sys.path[0]
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import figstyle as F  # type: ignore

import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

MM, DOUBLE, MINUS = F.MM, F.DOUBLE, F.MINUS
REPO = F.REPO
SIM_XLSX = REPO / "docs/collaboration/sim_realism_v2_all16_患者模拟器真实性已标注完.xlsx"

MODELS = ["openai/gpt-5.6-sol", "anthropic/claude-opus-5", "deepseek/deepseek-v4-pro", "minimax/minimax-m2-her",
          "cpsycoun", "mindchat", "soulchat", "internlm2-base", "qwen15-4b-base", "qwen3-8b-base"]
GZH = {"旗舰": "A_flagship", "陪伴": "companion", "心理微调": "B2_psych", "对照基座": "C_base"}
ZHG = {v: k for k, v in GZH.items()}
JUDGES = (("Qwen", "o", "Qwen judge"), ("DeepSeek", "s", "DeepSeek judge"))
BAND_C = {"低": F.RISK[1], "中": F.RISK[3], "高": F.RISK[5]}
BAND_N = {"低": "Low", "中": "Mid", "高": "High"}


# ---------------------------------------------------------------- 通用小工具
def pp(x):
    return None if x is None else 100.0 * x


def darker(c, k=0.35):
    r, g, b = mcolors.to_rgb(c)
    i = mcolors.to_rgb(F.INK)
    return (r * (1 - k) + i[0] * k, g * (1 - k) + i[1] * k, b * (1 - k) + i[2] * k)


def tint(c, f):
    """与白色混合：f = 0 原色，f = 1 白（实色代替 alpha，与 final_figs_a 一致）。"""
    r, g, b = mcolors.to_rgb(c)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


PHYS = F.STEP  # 判官 vs 医师并排时，医师金标用板岩灰作参照色


def phys(ax, y, est, lo, hi, color, scheme="orig", ms=None, z=3):
    """医师金标（全文统一）：粗半透明 CI 带 + 竖线刻度。原权重 = 粗带深刻度；raking = 细带浅刻度；截断 raking = 更细更浅。"""
    if scheme == "orig":
        band_c, tick_c, lw, tms, tmw = tint(color, 0.72), color, 3.0, ms or 5.2, 1.5
    elif scheme == "raked":
        band_c, tick_c, lw, tms, tmw = tint(color, 0.62), tint(color, 0.38), 1.3, (ms or 5.2) * 0.75, 1.0
    else:
        band_c, tick_c, lw, tms, tmw = tint(color, 0.75), tint(color, 0.5), 0.8, (ms or 5.2) * 0.6, 0.9
    ax.plot([lo, hi], [y, y], color=band_c, lw=lw, solid_capstyle="butt", zorder=z)
    ax.plot([est], [y], marker="|", ms=tms, mew=tmw, color=tick_c, ls="none", zorder=z + 1)


def phys_handle(color, scheme, label):
    from matplotlib.lines import Line2D
    if scheme == "orig":
        return Line2D([], [], color=tint(color, 0.72), lw=3.0, marker="|", ms=5.0, mew=1.5, mec=color, label=label)
    return Line2D([], [], color=tint(color, 0.62), lw=1.3, marker="|", ms=3.8, mew=1.0, mec=tint(color, 0.38), label=label)


def save300(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    F.align_labels(fig)
    paths = []
    for ext in ("pdf", "png"):
        p = out / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight", pad_inches=0.03, dpi=300)
        paths.append(p)
    plt.close(fig)
    return paths


def fmt_p(p):
    if p is None:
        return "—"
    if p < 0.001 + 1e-9:
        return "≤0.001"
    return f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


def signed(x, d=1):
    return f"{'+' if x >= 0 else MINUS}{abs(x):.{d}f}"


def ci_txt(lo, hi, d=1):
    return f"[{signed(lo, d)}, {signed(hi, d)}]"


def ci_words(lo, hi):
    """95% CI 写成「95% CI a–b」（两端同号时去掉符号，跨 0 时写 a to b）。"""
    if lo >= 0 or hi <= 0:
        a, b = sorted((abs(lo), abs(hi)))
        return f"95% CI {a:.0f}–{b:.0f}"
    return f"95% CI {signed(lo, 0)} to {signed(hi, 0)}"


def mname(m):
    return F.MODEL_N.get(m, m)


def note(ax, x, y, text, fs=5.6, color=None, ha="left", va="center", transform=None, **kw):
    ax.text(x, y, text, fontsize=fs, color=color or F.INK, ha=ha, va=va,
            transform=transform or ax.transData, **kw)


# ================================================================ Fig. 4 判官为什么会推翻结论
OE_DIMS = ["any", "d1", "d2", "d3", "d4"]   # 与 Fig. 2、3 同序：「低于最低标准」在前
OE_COL = {"d1": "Harmful\ncontent", "d2": "No safety\nstep", "d3": "Risk not\nrecognized",
          "d4": "Syco-\nphancy", "any": "Below\nstandard"}
OE_CLIP = 2.0          # log2 刻度截到 ±2（¼ … 4）


def _mix(c1, c2, f):
    a, b = mcolors.to_rgb(c1), mcolors.to_rgb(c2)
    return tuple(x + (y - x) * f for x, y in zip(a, b))


def oe_color(v, significant):
    """v = log2(O/E)。误报（>1）→ 红，漏判（<1）→ 板岩蓝；CI 含 1 的格冲淡。"""
    base = F.FAIL if v > 0 else F.STEP
    c = _mix("white", base, min(abs(v), OE_CLIP) / OE_CLIP)
    return tint(c, 0.6) if not significant else c


def _oe_heat(fig, gs_row):
    """面板 a：两张热图（Qwen、DeepSeek）。行 = 模型，列 = 维度，格色 = log2(O/E)；数字 = O/E；金框 = Holm p < 0.05。
    CI 不画（逐格 CI 在补充表 ST4）；CI 含 1 的格冲淡。"""
    J = F.J("reports/judge_error_by_model/judge_error_by_model.json")
    sub = gs_row.subgridspec(1, 2, wspace=0.06)
    axes = []
    for k, (jn, mk, jlab) in enumerate(JUDGES):
        ax = fig.add_subplot(sub[k])
        axes.append(ax)
        for i, m in enumerate(MODELS):
            for j, dim in enumerate(OE_DIMS):
                c = J["judges"][jn]["calibration"][dim]["cells"].get(f"{m}|all")
                fc, txt, tc, edge = tint(F.RULE, 0.6), "", F.INK, None
                if c and c["O"] > 0:
                    # O = 0（判官一次都没判失败）的格自助 CI 退化为 [0, 0]、信息量小 → 与「预期太少」同为浅灰空格
                    v = max(-OE_CLIP, min(OE_CLIP, math.log2(c["oe"])))
                    ci = c.get("ci")
                    significant = bool(ci) and (ci[0] > 1 or ci[1] < 1)
                    fc = oe_color(v, significant)
                    if significant:                   # 只给 CI 不含 1 的格写数字；其余只看颜色（数值在 ST4）
                        txt = f"×{c['oe']:.1f}" if c["oe"] >= 0.95 else f"×{c['oe']:.2f}"
                    tc = "white" if significant and abs(v) >= 1.1 else F.INK
                    if c.get("p_holm") is not None and c["p_holm"] < 0.05:
                        edge = F.ACCENT
                ax.add_patch(Rectangle((j - 0.47, i - 0.45), 0.94, 0.9, fc=fc, ec="white", lw=0.5, zorder=2))
                if edge:
                    ax.add_patch(Rectangle((j - 0.42, i - 0.40), 0.84, 0.8, fc="none", ec=edge, lw=1.3, zorder=3))
                if txt:
                    ax.text(j, i, txt, ha="center", va="center", fontsize=5.0, color=tc, zorder=4)
        ax.set_xlim(-0.55, len(OE_DIMS) - 0.45)
        ax.set_ylim(len(MODELS) - 0.5, -0.5)
        ax.set_xticks(range(len(OE_DIMS)))
        ax.set_xticklabels([OE_COL[d] for d in OE_DIMS], fontsize=5.0, linespacing=1.1)
        ax.xaxis.tick_top()
        ax.tick_params(axis="x", length=0, pad=2)
        ax.set_yticks(range(len(MODELS)))
        if k == 0:
            ax.set_yticklabels([mname(m) for m in MODELS], fontsize=5.6)
        else:
            ax.set_yticklabels([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        F.ylabels_clean(ax)
        ax.annotate(f"{jlab}  {'●' if mk == 'o' else '■'}", xy=(0.5, 1), xycoords="axes fraction", xytext=(0, 24),
                    textcoords="offset points", ha="center", va="bottom", fontsize=6.0, fontweight="bold", color=F.INK)
    return axes


def _oe_scale(ax):
    """O/E 色标：¼ … 4（对数），左 = 漏判，右 = 误报。"""
    vals = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    for i, v in enumerate(vals):
        ax.add_patch(Rectangle((i, 0), 1, 1, fc=oe_color(v, True), ec="white", lw=0.4))
    for i, lab in ((0.5, "×¼"), (2.5, "×½"), (4.5, "×1"), (6.5, "×2"), (8.5, "×4")):
        ax.text(i, -0.35, lab, ha="center", va="top", fontsize=5.0, color=F.INK2)
    ax.text(-0.3, 0.5, "← fewer than expected: missed failures,\nmodel looks safer", ha="right", va="center",
            fontsize=5.0, color=F.INK2, linespacing=1.15)
    ax.text(9.3, 0.5, "more than expected: false alarms, →\nmodel looks worse", ha="left", va="center",
            fontsize=5.0, color=F.INK2, linespacing=1.15)
    ax.set_xlim(0, 9); ax.set_ylim(-2.6, 1)
    ax.axis("off")


def _sens_panel(ax):
    """面板 b：按组的判官灵敏度。不画 CI（分母多为 1–9 段）；医师判失败 < 5 段的格画空心，只作描述。"""
    S = F.J("reports/judge_sensitivity_groups/sensitivity_groups.json")
    dims = ["d1", "d3", "d4"]          # d2 各组灵敏度 0.8–1.0，只作对照背景 → 写进图注，不占面板
    rows, ylabels, y = [], [], 0.0
    ypos = {}
    for dim in dims:
        for g in F.GROUPS:
            ypos[(dim, g)] = y
            y -= 1
        y -= 0.9
    for (dim, g), yy in ypos.items():
        if g == "companion" and dim in ("d1", "d3", "d4"):
            ax.add_patch(Rectangle((-3, yy - 0.5), 106, 1.0, color=F.ZEBRA, lw=0, zorder=0))
        col = F.GROUP_C[g]
        txt = []
        for k, (jn, mk, _) in enumerate(JUDGES):
            c = S[jn]["groups"].get(f"{ZHG[g]}|{dim}")
            if not c or not c["n_fail"]:
                txt.append("—")
                continue
            dy = 0.22 if k == 0 else -0.22
            F.marker(ax, 100 * c["n_hit"] / c["n_fail"], yy + dy, col, mk, filled=True, ms=3.0, z=5)
            txt.append(f"{c['n_hit']}/{c['n_fail']}")
        note(ax, 108, yy, "   ".join(txt), fs=5.4, color=F.INK2)
        rows.append(yy)
        ylabels.append(F.GROUP_N[g])
    for dim in dims:
        ytop = ypos[(dim, F.GROUPS[0])]
        note(ax, -2, ytop + 0.78, F.DIM_N[dim], fs=5.9, ha="left", fontweight="bold")
    ax.set_yticks(rows)
    ax.set_yticklabels(ylabels, fontsize=5.6)
    F.ylabels_clean(ax)
    ax.set_xlim(-3, 103)
    ax.set_ylim(min(rows) - 0.7, 1.2)
    ax.set_xticks([0, 50, 100])
    F.grid(ax, "x")
    ax.set_xlabel("Share of failures the judge caught (%)")
    note(ax, 108, 1.05, "Qwen  DeepSeek", fs=5.2, color=F.MUTED)
    tot = {}
    for jn, _, _ in JUDGES:
        h = sum(S[jn]["groups"][f"陪伴|{d}"]["n_hit"] for d in ("d1", "d3", "d4"))
        n = sum(S[jn]["groups"][f"陪伴|{d}"]["n_fail"] for d in ("d1", "d3", "d4"))
        tot[jn] = (h, n)
    return tot


def fig4(out):
    fig = plt.figure(figsize=(DOUBLE, 150 * MM))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 0.1, 0.85], hspace=0.42 if F.SHOW_TITLE else 0.3, left=0.13, right=0.97, top=0.85,
                          bottom=0.06)
    F.fig_title(fig, "Automated judges err differently for different models")
    axes = _oe_heat(fig, gs[0, 0])
    F.label(axes[0], "a", dx=-68, dy=58)
    F.headline(axes[0], "Judges over-call some models and miss failures in others", dy=57)
    F.caption(axes[0], "Each cell: failures the judge flagged ÷ failures expected (×1 = as expected)", dy=34)
    sub = gs[1, 0].subgridspec(1, 3, width_ratios=[0.22, 0.56, 0.22])
    axs = fig.add_subplot(sub[1])
    _oe_scale(axs)
    _p = axs.get_position()
    axs.set_position([_p.x0, _p.y0 + 0.03, _p.width, _p.height])   # 色标贴近热图，与 b 拉开
    hk = [Rectangle((0, 0), 1, 1, fc="white", ec=F.ACCENT, lw=1.3, label="still significant after multiple-test correction"),
          Rectangle((0, 0), 1, 1, fc=tint(F.RULE, 0.6), ec="none", label="too few failures to tell")]
    axs.legend(handles=hk, loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=5.0, frameon=False,
               handlelength=1.4, handleheight=0.9, columnspacing=2.5)
    row = gs[2, 0].subgridspec(1, 2, width_ratios=[0.84, 0.16], wspace=0.05)
    axb = fig.add_subplot(row[0])
    tot = _sens_panel(axb)
    F.label(axb, "b", dx=-50, dy=28)
    (hq, nq), (hd, _) = tot["Qwen"], tot["DeepSeek"]
    F.headline(axb, f"Of the companion model's {nq} dangerous failures, the Qwen judge caught {hq} "
               f"(DeepSeek judge: {hd})", dy=20)
    F.caption(axb, "Share of psychiatrist-rated failures the judge caught (counts: caught / failures)", dy=8)
    return save300(fig, out, "Fig4_why_judges_err")


# ================================================================ Fig. 5 心理健康微调与安全
def fig5_rows():
    PT = F.J("reports/primary_tests/primary_tests.json")["results"]
    JG = F.J("reports/judge_vs_gold/judge_vs_gold.json")
    PR = F.J("reports/gold_prevalence_raked/prevalence_raked.json")["schemes"]
    pt = {s: {r["id"]: r for r in PT[s]} for s in ("orig", "raked")}

    def judge(contrast, outcome):
        res = {}
        for jn, _, _ in JUDGES:
            for r in JG[jn]["rows"]:
                if r["contrast"] == contrast and r["outcome"] == outcome:
                    res[jn] = (r["judge"], r["judge_ci"])
        return res

    def prev_diff(outcome):
        key = f"('设计组', '心理微调')|('设计组', '对照基座')|{outcome}"
        return {s: PR[s]["diffs"][key] for s in ("orig", "raked")}

    G = "心理微调组 − 对照基座组"
    rows = []
    spec = [("PB1", "any", None), (None, "any134", None), ("PB2", "d1", None), (None, "d3", None), ("PB3", "d4", None),
            (None, "d2", None)]     # d3 不在主假设族（未检验），但它是 Fig. 3b 中两套权重都被推翻的格
    for pid, outc, _ in spec:
        if pid:
            g = {s: (pt[s][pid]["est"], pt[s][pid]["ci"]) for s in ("orig", "raked")}
            tests = {s: (pt[s][pid]["p_boot"], pt[s][pid]["holm_family"]) for s in ("orig", "raked")}
            cnt = (pt["orig"][pid]["fail_a"], pt["orig"][pid]["n_a"], pt["orig"][pid]["fail_b"], pt["orig"][pid]["n_b"])
        else:
            d = prev_diff(outc)
            g = {s: (d[s][0], d[s][1]) for s in ("orig", "raked")}
            tests, cnt = None, None
        rows.append(dict(section="group", label=F.DIM_N[outc], pid=pid, gold=g, tests=tests, cnt=cnt,
                         judge=judge(G, outc) if outc != "any134" else {}))
    for (a, b, lab), pid in zip(F.PAIRS, ("PB4", "PB5", "PB6")):
        g = {s: (pt[s][pid]["est"], pt[s][pid]["ci"]) for s in ("orig", "raked")}
        tests = {s: (pt[s][pid]["p_boot"], pt[s][pid]["holm_family"]) for s in ("orig", "raked")}
        cnt = (pt["orig"][pid]["fail_a"], pt["orig"][pid]["n_a"], pt["orig"][pid]["fail_b"], pt["orig"][pid]["n_b"])
        rows.append(dict(section="pair", label=lab, pid=pid, gold=g, tests=tests, cnt=cnt,
                         judge=judge(f"{a} − {b}", "any")))
    return rows


def fig5(out):
    """结论式标题 + 方向箭头 + 第一行直接标注（不另设图例）+ 右侧只留 Holm 校正 p（原始 p 在 ST2）。"""
    rows = fig5_rows()
    fig = plt.figure(figsize=(DOUBLE, 100 * MM))
    ax = fig.add_axes([0.25, 0.13, 0.46, 0.80])
    n_group = sum(r["section"] == "group" for r in rows)
    ypos, y = [], 0.0
    for i, r in enumerate(rows):
        if i == n_group:
            y -= 1.4
        ypos.append(y)
        y -= 1
    off = {"orig": 0.22, "Qwen": 0.0, "DeepSeek": -0.22}
    for k, (r, yy) in enumerate(zip(rows, ypos)):
        go = r["gold"]["orig"]
        phys(ax, yy + off["orig"], pp(go[0]), pp(go[1][0]), pp(go[1][1]), PHYS)
        F.raked_pt(ax, pp(r["gold"]["raked"][0]), yy + off["orig"], PHYS, ms=3.0)
        for jn, mk, _ in JUDGES:
            yj = yy + off[jn]
            if jn in r["judge"]:
                est, ci = r["judge"][jn]
                ax.plot([pp(ci[0]), pp(ci[1])], [yj, yj], color=F.ACCENT, lw=0.7, zorder=3, solid_capstyle="butt")
                F.marker(ax, pp(est), yj, F.ACCENT, mk, filled=True, ms=3.3, z=5)
        if not r["judge"]:
            note(ax, pp(r["gold"]["orig"][1][1]) + 2.5, yy, "no judge estimate", fs=5.0, color=F.MUTED)
    F.refline(ax, x=0)
    ax.set_yticks(ypos)
    ax.set_yticklabels([r["label"] for r in rows])
    F.ylabels_clean(ax)
    F.grid(ax, "x")
    ax.set_ylim(ypos[-1] - 0.6, 0.85)
    ax.set_xlim(-75, 95)
    ax.set_xlabel("Difference in failure rate, fine-tuned vs control (%)")
    # 方向提示：负 = 微调版失败更少
    ax.annotate("← fine-tuned fails less", xy=(0, 0), xycoords="axes fraction", xytext=(0, -17.5),
                textcoords="offset points", ha="left", va="top", fontsize=5.2, color=F.INK2)
    ax.annotate("fine-tuned fails more →", xy=(1, 0), xycoords="axes fraction", xytext=(0, -17.5),
                textcoords="offset points", ha="right", va="top", fontsize=5.2, color=F.INK2)
    ytr = ax.get_yaxis_transform()
    for yh, t in ((ypos[0] + 0.62, "Three fine-tuned models vs\ntheir controls, pooled"),
                  (ypos[n_group] + 0.62, "Each pair\n(below minimum standard)")):
        ax.text(-0.02, yh, t, transform=ytr, ha="right", va="bottom", fontsize=5.6, fontweight="bold",
                color=F.MUTED, linespacing=1.1)
    ax.axhline(ypos[n_group - 1] - 0.72, color=F.RULE, lw=0.5, xmin=-0.5, clip_on=False)
    # 右侧：四列 —— 多重比较校正后的 p（原权重 / raking），两组各自的失败段 / 金标段
    tx = fig.add_axes([0.74, 0.13, 0.25, 0.80], sharey=ax)
    tx.axis("off")
    tx.set_xlim(0, 1)
    cols = ((0.0, "Corrected p\n(original)"), (0.24, "Corrected p\n(raking)"), (0.5, "Fine-tuned\nfailed / rated"),
            (0.77, "Control\nfailed / rated"))
    for x0, t in cols:
        note(tx, x0, 0.72, t, fs=5.2, color=F.INK2, va="bottom", linespacing=1.15)
    for r, yy in zip(rows, ypos):
        if r["tests"]:
            for (x0, _), sch in zip(cols[:2], ("orig", "raked")):
                h = r["tests"][sch][1]
                note(tx, x0, yy, fmt_p(h), fs=5.6, color=F.INK if h < 0.05 else F.INK2,
                     fontweight="bold" if h < 0.05 else "normal")
            fa, na, fb, nb = r["cnt"]
            note(tx, cols[2][0], yy, f"{fa}/{na}", fs=5.5, color=F.INK2)
            note(tx, cols[3][0], yy, f"{fb}/{nb}", fs=5.5, color=F.INK2)
        else:
            note(tx, 0.0, yy, "—", fs=5.6, color=F.MUTED)
    F.fig_title(fig, "No evidence that mental-health fine-tuned models were safer overall; the judges reversed the result for missed risk")
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=tint(PHYS, 0.72), lw=3.0, marker="|", ms=5.0, mew=1.5, mec=PHYS,
                 label="Psychiatrists (95% CI)"),
          F.raked_handle(PHYS, "Psychiatrists, raking weights", ms=3.0),
          Line2D([], [], color=F.ACCENT, lw=0.7, marker="o", ms=3.2, mfc=F.ACCENT, mec="white", mew=0.5,
                 label="Qwen judge"),
          Line2D([], [], color=F.ACCENT, lw=0.7, marker="s", ms=2.9, mfc=F.ACCENT, mec="white", mew=0.5,
                 label="DeepSeek judge")]
    fig.legend(handles=hs, loc="lower left", bbox_to_anchor=(0.12, -0.01), ncol=4, fontsize=5.0, frameon=False,
               handlelength=2.0, handletextpad=0.7, columnspacing=1.6)
    return save300(fig, out, "Fig5_psych_finetuning")


# ================================================================ Fig. 6 评估设计的三个陷阱
def arm_rates(outcome="any"):
    """按臂直接计算的观测比例与「高 − 低」差（tools/judge_dose_by_arm，人设聚类自助 CI）。"""
    A = F.J("reports/judge_dose_by_arm/judge_dose_by_arm.json")["cells"]
    return A[f"fixed|{outcome}"], A[f"responsive|{outcome}"]


def _fig6a(ax):
    fxd, rsd = arm_rates()
    bands = ["低", "中", "高"]
    fx = [pp(fxd["point"][b]) for b in bands]
    rv = [pp(rsd["point"][b]) for b in bands]
    x = [0, 1, 2]
    ax.plot(x, fx, color=F.LOCAL_ALL, lw=1.1, zorder=3)
    ax.plot(x, rv, color=F.LOCAL_ALL, lw=1.1, ls=(0, (3, 1.6)), zorder=3)
    for i in x:
        F.marker(ax, i, fx[i], F.LOCAL_ALL, "o", filled=True, ms=3.6)
        F.marker(ax, i, rv[i], F.LOCAL_ALL, "o", filled=False, ms=3.6)
    dci, rci = fxd["diff_ci"], rsd["diff_ci"]
    ax.annotate(f"Fixed arm\n(risk set by design):\nhigh {fx[2]:.0f}% vs low {fx[0]:.0f}%\n"
                f"(difference {ci_words(pp(dci[0]), pp(dci[1])).replace(' CI ', ' CI' + chr(10))} points)",
                xy=(2, fx[2]), xytext=(2.12, fx[2] + 4), fontsize=5.2, va="center", color=F.INK)
    ax.annotate(f"Responsive arm\n(risk follows the model):\nhigh {rv[2]:.0f}% vs low {rv[0]:.0f}%\n"
                f"(difference {ci_words(pp(rci[0]), pp(rci[1])).replace(' CI ', ' CI' + chr(10))} points)",
                xy=(2, rv[2]), xytext=(2.12, rv[2] - 4), fontsize=5.2, va="center", color=F.INK)
    ax.set_xticks(x)
    ax.set_xticklabels([BAND_N[b] for b in bands])
    ax.set_xlim(-0.3, 3.3)
    ax.set_ylim(15, 60)
    F.grid(ax, "y")
    ax.set_xlabel("Expressed-risk band")
    ax.set_ylabel("Judge-flagged failure (%)")
    nres = [rsd["n_turns"][b] for b in bands]
    note(ax, 0.04, 0.03, "responsive-arm turns: " + " / ".join(f"{n:,}" for n in nres), fs=5.0, color=F.MUTED,
         transform=ax.transAxes, va="bottom")


def _fig6b(ax):
    T = F.J("reports/judge_dose_turn/judge_dose_turn.json")
    ct = T["crosstab"]
    for b in ("低", "中", "高"):
        pts = sorted((int(k.split("|")[1]), v) for k, v in ct.items() if k.split("|")[0] == b)
        xs = [t for t, _ in pts]
        ys = [pp(v["any"]) for _, v in pts]
        ax.plot(xs, ys, color=BAND_C[b], lw=1.0, zorder=3)
        for t, v in pts:
            ms = 1.6 + 2.4 * math.sqrt(v["n"] / 288)
            ax.plot([t], [pp(v["any"])], marker="o", ms=ms, mfc=BAND_C[b], mec="white", mew=0.4, ls="none", zorder=4)
        ax.annotate(BAND_N[b], xy=(xs[-1], ys[-1]), xytext=(4, 0), textcoords="offset points", fontsize=5.4,
                    color=F.INK, va="center")
    P = T["point"]["any"]["pairs"]
    cis = T["ci"]

    def bracket(x0, x1, ytop, text):
        ax.plot([x0, x0, x1, x1], [ytop - 1.2, ytop, ytop, ytop - 1.2], color=F.INK2, lw=0.5)
        ax.text((x0 + x1) / 2, ytop + 0.8, text, fontsize=5.0, ha="center", va="bottom", color=F.INK)

    lm = P["低|中"]["obs"]
    lmci = cis["any|('低', '中')|obs"]
    mh = P["中|高"]["obs"]
    mhci = cis["any|('中', '高')|obs"]
    bracket(3, 4, 49, f"mid vs low, same turns:\n{signed(pp(lm), 0)} points\n({ci_words(pp(lmci[0]), pp(lmci[1]))})")
    bracket(7, 12, 56, f"high vs mid, same turns:\n{signed(pp(mh), 0)} points ({ci_words(pp(mhci[0]), pp(mhci[1]))})")
    ch = T["point"]["any"]["chain"][0]
    chci = cis["any|chain|obs"]
    un = T["point"]["any"]["unstrat"][0]
    sl = T["slopes"]["中|any"][1]
    slci = cis["slope|中|any|within"]
    ax.text(0.02, 0.03, f"High vs low risk: {signed(pp(un), 0)} points overall,\n{signed(pp(ch), 0)} points "
            f"({ci_words(pp(chci[0]), pp(chci[1]))}) after adjusting for turn", transform=ax.transAxes, fontsize=5.2,
            color=F.INK, va="bottom", linespacing=1.25)
    ax.set_xlim(0.5, 13.2)
    ax.set_xticks(range(1, 13))
    ax.set_ylim(8, 62)
    F.grid(ax, "y")
    ax.set_xlabel("Turn")
    ax.set_ylabel("Judge-flagged failure (%)")


def parse_branch_pooled():
    """intervention_v2/diagnostics.md §3 表格的「抛弃率」行（跨三个被测按人设配对合并）。"""
    txt = (REPO / "reports/intervention_v2/diagnostics.md").read_text(encoding="utf-8")
    line = next(ln for ln in txt.splitlines() if ln.startswith("| 抛弃率 |"))
    num = r"([−\-+]?\d+\.\d+)"
    m = re.findall(num + r"\s*\[" + num + r",\s*" + num + r"\]", line)
    f = lambda s: float(s.replace("−", "-"))  # noqa: E731
    (a, al, ah), (b, bl, bh) = [(f(x), f(y), f(z)) for x, y, z in m[:2]]
    return {"regen_minus_orig": (a, al, ah), "intv_minus_regen": (b, bl, bh)}


def _fig6c(ax):
    B = F.J("reports/branch_v2/report.json")
    models = B["models"]
    conds = (("orig", "Original reply", F.ORIG), ("regen", "Regenerated only", F.RESAMPLE),
             ("intv", "Regenerated + prompt", F.ACCENT))
    rows = models + ["pooled"]
    ypos = {m: len(rows) - 1 - i for i, m in enumerate(rows)}
    ypos["pooled"] -= 0.35
    pooled = {}
    for c, _, _ in conds:
        n = sum(B["itt"][m][c]["abandon"]["n"] for m in models)
        pooled[c] = sum(B["itt"][m][c]["abandon"]["mean"] * B["itt"][m][c]["abandon"]["n"] for m in models) / n
    for m in rows:
        y = ypos[m]
        vals = [pp(pooled[c]) if m == "pooled" else pp(B["itt"][m][c]["abandon"]["mean"]) for c, _, _ in conds]
        for k, (c, _, col) in enumerate(conds):
            dy = (0.2, 0.0, -0.2)[k]
            if m != "pooled":
                lo, hi = B["itt"][m][c]["abandon"]["ci"]
                ax.plot([pp(lo), pp(hi)], [y + dy] * 2, color=col, lw=0.7, zorder=2, solid_capstyle="butt")
            ax.plot([vals[k]], [y + dy], marker="D", ms=3.4, mfc=col, mec="white", mew=0.5, ls="none", zorder=4)
    ax.set_yticks([ypos[m] for m in rows])
    ax.set_yticklabels([mname(m) if m != "pooled" else "Pooled" for m in rows])
    for t, m in zip(ax.get_yticklabels(), rows):
        if m != "pooled":
            t.set_color(F.GROUP_C[F.GROUP_OF[m]])
    F.ylabels_clean(ax)
    F.grid(ax, "x")
    ax.set_xlim(0, 100)
    ax.set_ylim(ypos["pooled"] - 2.55, len(rows) - 0.45)
    ax.set_xlabel("Later abandonment (%)")
    d = parse_branch_pooled()
    a, al, ah = d["regen_minus_orig"]
    b, bl, bh = d["intv_minus_regen"]
    o, r, i = (pp(pooled[c]) for c in ("orig", "regen", "intv"))
    ax.text(0.02, 0.02, f"Pooled: original {o:.0f}%\n→ regenerated only {r:.0f}%\n  (drop: {ci_words(100 * al, 100 * ah)} points)\n"
            f"→ with prompt {i:.0f}%\n  (further drop: {ci_words(100 * bl, 100 * bh)} points)",
            transform=ax.transAxes, fontsize=5.1, color=F.INK, va="bottom", linespacing=1.2)
    return conds


def fig6(out):
    fig = plt.figure(figsize=(171 * MM, 80 * MM))          # tight 后约 178 mm，不超过双栏 180 mm
    gs = fig.add_gridspec(1, 3, width_ratios=[0.95, 1.35, 1.05], wspace=0.55, left=0.065, right=0.985, top=0.74,
                          bottom=0.15)
    F.fig_title(fig, "Three evaluation-design choices that change the conclusion")
    ax = fig.add_subplot(gs[0])
    _fig6a(ax)
    F.label(ax, "a", dx=-28, dy=34)
    F.headline(ax, "Grouping by a risk level that\nfollows the model reverses the result", dy=17)
    F.caption(ax, "Local models, Qwen judge", dy=4)
    ax = fig.add_subplot(gs[1])
    _fig6b(ax)
    F.label(ax, "b", dx=-28, dy=34)
    F.headline(ax, "Risk level is tied to turn number; the dose–\nresponse holds after adjusting for turn", dy=17)
    F.caption(ax, "Fixed arm, Qwen judge; dot size = turns", dy=4)
    ax = fig.add_subplot(gs[2])
    conds = _fig6c(ax)
    F.label(ax, "c", dx=-68, dy=34)
    F.headline(ax, "Regenerating alone explains most\nof an apparent fix", dy=17)
    F.caption(ax, "113 branch points", dy=4)
    ax.legend(handles=[F.handle(col, "D", True, lab, 3.4) for _, lab, col in conds], loc="upper center",
              bbox_to_anchor=(0.4, -0.2), ncol=3, fontsize=5.0, columnspacing=0.7, handletextpad=0.2)
    return save300(fig, out, "Fig6_design_pitfalls")


# ================================================================ ED Fig. 1 模拟求助者效度
def edfig1(out):
    fig = plt.figure(figsize=(DOUBLE, 128 * MM))
    F.fig_title(fig, "Simulated help-seekers looked realistic to a psychiatrist, and the assigned risk level showed up "
                     "in what they said")
    gs = fig.add_gridspec(2, 3, width_ratios=[0.8, 1, 1.35], height_ratios=[1, 1], wspace=0.6, hspace=0.95 if F.SHOW_TITLE else 0.6,
                          left=0.07, right=0.98, top=0.86, bottom=0.1)
    if SIM_XLSX.exists():
        import openpyxl
        wb = openpyxl.load_workbook(SIM_XLSX, read_only=True, data_only=True)
        rows = [r for r in list(wb["待审(填)"].iter_rows(values_only=True))[1:] if r[4] not in (None, "")]
        scores = [int(r[4]) for r in rows]
        by = collections.defaultdict(list)
        for r in rows:
            by[r[0]].append(int(r[4]))
        ax = fig.add_subplot(gs[0, 0])
        cnt = collections.Counter(scores)
        for s in range(1, 6):
            ax.bar(s, 100 * cnt[s] / len(scores), 0.62, color=F.ACCENT, lw=0)
            ax.text(s, 100 * cnt[s] / len(scores) + 1.5, str(cnt[s]), ha="center", fontsize=5.4, color=F.INK2)
        ax.set_xticks(range(1, 6))
        ax.set_xlim(0.4, 5.6)
        ax.set_ylim(0, 92)
        F.grid(ax, "y")
        ax.set_xlabel("Realism rating (1–5)")
        ax.set_ylabel("Help-seeker turns (%)")
        ax.text(0.03, 0.97, f"mean {statistics.mean(scores):.2f}\n≥ 4: {100 * (cnt[4] + cnt[5]) / len(scores):.1f}%\n"
                f"n = {len(scores)} turns", transform=ax.transAxes, fontsize=5.6, va="top")
        F.caption(ax, "Psychiatrist rating of help-seeker turns")
        F.headline(ax, "Rated realistic: nearly nine in\nten turns scored 4–5", dy=14)
        F.label(ax, "a", dx=-26, dy=28)
        ax = fig.add_subplot(gs[0, 1])
        units = sorted(by)
        for i, u in enumerate(units):
            y = len(units) - 1 - i
            m = statistics.mean(by[u])
            ax.plot([1, m], [y, y], color=F.GRID, lw=1.0)
            F.marker(ax, m, y, F.ACCENT if m >= 4 else F.FAIL, ms=3.4)
        ax.set_yticks(range(len(units)))
        ax.set_yticklabels([f"p{int(u[1:]):02d}" for u in units][::-1], fontsize=5.2)
        F.ylabels_clean(ax)
        F.grid(ax, "x")
        ax.set_xlim(1, 5.2)
        ax.set_ylim(-0.6, len(units) - 0.4)
        ax.set_xlabel("Mean realism rating")
        F.caption(ax, "By persona (red: mean < 4)")
        F.headline(ax, f"{sum(statistics.mean(v) < 4 for v in by.values())} of {len(by)} personas "
                       "averaged below 4", dy=14)
        F.label(ax, "b", dx=-30, dy=28)
    else:
        for k, letter in ((0, "a"), (1, "b")):
            ax = fig.add_subplot(gs[0, k])
            ax.axis("off")
            ax.text(0.5, 0.5, "rating file not available", ha="center", fontsize=6, color=F.MUTED, transform=ax.transAxes)
            F.label(ax, letter, dx=-26)

    ax = fig.add_subplot(gs[0, 2])
    ps = F.J("reports/sim_vs_real/results.json")["markers_per_1000_chars"]
    th = F.J("reports/sim_vs_real_treehole/results.json")["markers_per_1000_chars"]
    keys = [("第一人称「我」", "first person 'I'"), ("否定（不/没/别）", "negation"), ("死亡/自杀词", "death / suicide words"),
            ("睡眠/身体", "sleep / body"), ("家人", "family"), ("语气词", "modal particles"), ("省略号", "ellipsis"),
            ("问号", "question mark")]
    srcs = [("Simulated help-seekers", ps["SIM"], F.ACCENT, "o", True), ("SOS-1K tree-hole posts", th["REAL-post"], F.NEUTRAL, "s", False),
            ("PsyQA help-seeking posts", ps["REAL-post"], F.MUTED, "^", False)]
    for i, (k, lab) in enumerate(keys):
        y = len(keys) - 1 - i
        vals = [s[1][k] for s in srcs]
        ax.plot([min(vals), max(vals)], [y, y], color=F.GRID, lw=1.6, zorder=1)
        for (_, d, col, mk, f), dy in zip(srcs, (0.0, 0.2, -0.2)):
            F.marker(ax, d[k], y + dy, col, mk, filled=f, ms=3.4)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([lab for _, lab in keys][::-1])
    F.ylabels_clean(ax)
    F.grid(ax, "x")
    ax.set_xscale("log")
    ax.set_xlim(0.8, 90)
    ax.set_xticks([1, 3, 10, 30])
    ax.set_xticklabels(["1", "3", "10", "30"])
    ax.minorticks_off()
    ax.set_xlabel("Occurrences per 1,000 characters")
    ax.legend(handles=[F.handle(c, m, f, l, 3.4) for l, _, c, m, f in srcs], loc="lower right", fontsize=5.0,
              borderaxespad=0.2)
    F.caption(ax, "Lexical markers vs two real Chinese corpora")
    F.headline(ax, "Wording resembles real posts, but death is\nmentioned less often (by design)", dy=14)
    F.label(ax, "c", dx=-70, dy=28)

    # d 操纵检验
    ax = fig.add_subplot(gs[1, :])
    _p = ax.get_position()
    ax.set_position([_p.x0, _p.y0, _p.width * 0.66, _p.height])
    T = F.J("reports/judge_dose_turn/judge_dose_turn.json")["manipulation"]["rates"]
    other = tint(F.MUTED, 0.35)
    series = (("P1", "Has stated a wish to die, a plan or means (cumulative)", F.ACCENT, "-", 1.3),
              ("P2", "", other, "-", 0.8), ("P3", "", other, "-", 0.8), ("P4", "", other, "-", 0.8))
    xs = list(range(len(F.STAGES)))
    for key, lab, col, ls, lw in series:
        ys = [pp(T[f"{s}|全部"][key]) for s in F.STAGES]
        ax.plot(xs, ys, color=col, ls=ls, lw=lw, zorder=3)
        for x, yv in zip(xs, ys):
            ax.plot([x], [yv], marker="o", ms=2.6, mfc=col, mec="white", mew=0.4, ls="none", zorder=4)
    ends = [pp(T[f"{F.STAGES[-1]}|全部"][k]) for k, *_ in series]
    ax.annotate(series[0][1], xy=(xs[-1], ends[0]), xytext=(xs[-1] + 0.12, ends[0]), textcoords="data",
                fontsize=5.2, va="center", color=F.INK, annotation_clip=False)
    glabs = ["States a new plan, time or means this turn", "Distorted appraisal this turn",
             "Wants to end the conversation"]
    for lab, yl, ye in zip(glabs, F.repel(ends[1:], 5.5), ends[1:]):
        ax.annotate(lab, xy=(xs[-1], ye), xytext=(xs[-1] + 0.12, yl), textcoords="data", fontsize=5.0, va="center",
                    color=F.INK2, annotation_clip=False)
    ns = [T[f"{s}|全部"]["n"] for s in F.STAGES]
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{n}\n({c:,})" for n, c in zip(F.STAGE_N, ns)], fontsize=5.2)
    ax.set_xlim(-0.3, len(xs) - 1 + 0.3)
    ax.set_ylim(0, 92)
    F.grid(ax, "y")
    ax.set_ylabel("Seeker turns with signal (%)")
    ax.set_xlabel("Assigned risk level, fixed arm (turns)")
    F.caption(ax, "Manipulation check: help-seeker signals extracted by the Qwen judge")
    F.headline(ax, "The higher the assigned risk, the more often the simulated help-seeker states a wish to die, "
                   "a plan or means", dy=14)
    F.label(ax, "d", dx=-26, dy=28)
    return save300(fig, out, "EDFig1_simulator_validity")


# ================================================================ ED Fig. 2 轮次分层的剂量反应
def edfig2(out):
    """ED Fig. 2：控制轮次后的剂量反应。a 上 = 档位 × 轮次示意（为什么要同轮次比较），a 下 = 直接比 vs 同轮次比；
    b = 同轮次估计按失败类型。两段分量、人设匹配与对话内斜率写在图注。"""
    TQ = F.J("reports/judge_dose_turn/judge_dose_turn.json")
    TD = F.J("reports/a4_deepseek/judge_dose_turn/judge_dose_turn.json")
    TT = (("Qwen", "o", TQ), ("DeepSeek", "s", TD))
    fig = plt.figure(figsize=(DOUBLE, 96 * MM))
    F.fig_title(fig, "The judge-observed dose–response, stratified by turn")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.85], height_ratios=[0.62, 1.0], wspace=0.75, hspace=1.35,
                          left=0.19, right=0.985, top=0.80, bottom=0.13)

    def est(T, outc, kind):
        P = T["point"][outc]
        if kind == "unstrat":
            return P["unstrat"][0], T["ci"][f"{outc}|unstrat|obs"]
        return P["chain"][0], T["ci"][f"{outc}|chain|obs"]

    def forest(ax, items, xlabel, values=False):
        n = len(items)
        for i, (lab, getter) in enumerate(items):
            y = n - 1 - i
            for k, (jn, mk, T) in enumerate(TT):
                e, ci = getter(T)
                yy = y + (0.17 if k == 0 else -0.17)
                ax.plot([pp(ci[0]), pp(ci[1])], [yy, yy], color=F.ACCENT, lw=0.8, solid_capstyle="butt", zorder=3)
                F.marker(ax, pp(e), yy, F.ACCENT, mk, filled=True, ms=3.6)
                if values:
                    ax.text(pp(ci[1]) + 1.2, yy, f"{signed(pp(e), 0)}", fontsize=5.4, va="center", color=F.INK)
        ax.set_yticks(range(n))
        ax.set_yticklabels([lab for lab, _ in items][::-1], linespacing=1.1)
        F.ylabels_clean(ax)
        F.refline(ax, x=0)
        F.grid(ax, "x")
        ax.set_ylim(-0.6, n - 0.4)
        ax.set_xlabel(xlabel)

    # a 上：哪一轮出现哪一档（fixed 臂，本地模型）
    ax = fig.add_subplot(gs[0, 0])
    ct = TQ["crosstab"]
    bands = ("低", "中", "高")
    for r, b in enumerate(bands):
        y = len(bands) - 1 - r
        for t in range(1, 13):
            v = ct.get(f"{b}|{t}")
            if v:
                ax.add_patch(Rectangle((t - 0.45, y - 0.36), 0.9, 0.72, fc=BAND_C[b], ec="none"))
    for (t0, t1, y0, y1) in ((3, 4, 1, 2), (7, 12, 0, 1)):
        ax.add_patch(Rectangle((t0 - 0.55, y0 - 0.48), t1 - t0 + 1.1, y1 - y0 + 0.96, fc="none", ec=F.INK,
                               lw=0.8, ls=(0, (2, 1.2)), zorder=4))
    ax.text(3.5, 2.62, "mid vs low", ha="center", va="bottom", fontsize=5.2, color=F.INK)
    ax.text(9.5, 1.62, "high vs mid", ha="center", va="bottom", fontsize=5.2, color=F.INK)
    ax.set_xlim(0.4, 12.6)
    ax.set_ylim(-0.6, 3.0)
    ax.set_xticks(range(1, 13))
    ax.set_yticks(range(3))
    ax.set_yticklabels([BAND_N[b] + " risk" for b in bands][::-1])
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Turn (fixed arm)")
    F.caption(ax, "Dashed boxes: turns where two levels meet", dy=4)
    F.headline(ax, "High risk comes late; mid risk overlaps both ends,\nso it bridges two same-turn comparisons", dy=14)
    F.label(ax, "a", dx=-80, dy=28)

    # a 下：直接比 vs 同轮次比
    ax = fig.add_subplot(gs[1, 0])
    forest(ax, [("High vs low,\ncompared directly", lambda T: est(T, "any", "unstrat")),
                ("High vs low, turn-adjusted\n(= mid vs low + high vs mid)", lambda T: est(T, "any", "chain"))],
           "Difference in failure rate (%)", values=True)
    ax.set_xticks([-10, 0, 10, 20, 30])
    ax.set_xlim(-12, 38)
    F.caption(ax, "Below minimum standard", dy=4)
    F.headline(ax, "After adjusting for turn, the Qwen judge keeps the rise;\nthe DeepSeek judge's estimate shrinks and crosses 0",
               dy=14)
    F.label(ax, "b", dx=-80, dy=28)

    # b：同轮次估计按失败类型
    dims = ["any", "d1", "d2", "d3", "d4", "d5"]
    ax = fig.add_subplot(gs[:, 1])
    forest(ax, [(F.DIM_N[d], (lambda dd: lambda T: est(T, dd, "chain"))(d)) for d in dims],
           "High vs low, turn-adjusted (%)")
    ax.set_xticks([-10, 0, 10, 20, 30])
    F.caption(ax, "By failure type", dy=4)
    F.headline(ax, "After adjusting for turn, the rise both judges\nagree on is in unrecognized risk", dy=14)
    F.label(ax, "c", dx=-100, dy=28)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=F.ACCENT, lw=0.8, marker="o", ms=3.4, mfc=F.ACCENT, mec="white", mew=0.5,
                 label="Qwen judge"),
          Line2D([], [], color=F.ACCENT, lw=0.8, marker="s", ms=3.1, mfc=F.ACCENT, mec="white", mew=0.5,
                 label="DeepSeek judge (lines, 95% CI)")]
    fig.legend(handles=hs, loc="lower left", bbox_to_anchor=(0.18, -0.005), ncol=2, fontsize=5.3, frameon=False,
               handlelength=2.0, columnspacing=2.0)
    fig.text(0.985, 0.012, "Local models (fine-tuned + control), fixed arm", fontsize=5.3, color=F.INK2, ha="right")
    return save300(fig, out, "EDFig2_dose_by_turn")


# ================================================================ ED Fig. 3 金标权重
def edfig3(out):
    R = F.J("reports/gold_prevalence_raked/prevalence_raked.json")
    S = R["schemes"]
    fig = plt.figure(figsize=(DOUBLE, 92 * MM))
    F.fig_title(fig, "Weighting the 180 gold turns back to the corpus")
    gs = fig.add_gridspec(1, 3, width_ratios=[0.9, 0.8, 0.85], wspace=1.05, left=0.125, right=0.875, top=0.75, bottom=0.17)
    # a 边际还原：三组之间空半行、左侧组标签；原权重 ●，raking ▽（全部落在 1 上）
    ax = fig.add_subplot(gs[0])
    rec, rk = S["orig"]["recon"], S["raked"]["recon"]
    arm_n = {"fixed": "Fixed", "responsive": "Responsive"}
    groups = [("Models", [(mname(m), rec["model"][m], rk["model"][m], F.GROUP_C[F.GROUP_OF[m]]) for m in MODELS]),
              ("Model groups", [(F.GROUP_N[g], rec["dgroup"][ZHG[g]], rk["dgroup"][ZHG[g]], F.GROUP_C[g])
                                for g in F.GROUPS]),
              ("Arm × risk band", [(f"{arm_n[k.split('|')[0]]}, {BAND_N[k.split('|')[1]].lower()} risk",
                                    rec["arm_band"][k], rk["arm_band"][k], F.ACCENT)
                                   for k in ("fixed|低", "fixed|中", "fixed|高", "responsive|低", "responsive|中",
                                             "responsive|高")])]
    ys, labs, y = [], [], 0.0
    ytr = ax.get_yaxis_transform()
    for gname, rows in groups:
        ax.text(-0.02, y + 0.15, gname, transform=ytr, ha="right", va="center", fontsize=5.4, fontweight="bold",
                color=F.MUTED)
        y -= 1.0
        for lab, v, vr, col in rows:
            ax.plot([1, v], [y, y], color=F.GRID, lw=1.0, zorder=1)
            F.marker(ax, v, y, col, "o", ms=3.3)
            F.raked_pt(ax, vr, y, F.INK2, ms=3.0)
            ys.append(y)
            labs.append(lab)
            y -= 1.0
        y -= 0.5
    F.refline(ax, x=1)
    ax.set_yticks(ys)
    ax.set_yticklabels(labs, fontsize=5.2)
    F.ylabels_clean(ax)
    F.grid(ax, "x")
    ax.set_ylim(y + 1.0, 0.8)
    ax.set_xlim(0.4, 2.25)
    ax.set_xticks([0.5, 1.0, 1.5, 2.0])
    ax.set_xlabel("Weighted total / corpus total")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], ls="none", marker="o", ms=3.3, mfc=F.INK2, mec="white", mew=0.5,
                              label="original weights"),
                       F.raked_handle(F.INK2, "raking weights", ms=3.0)],
              loc="upper right", fontsize=5.0, borderaxespad=0.1, handletextpad=0.4)
    F.caption(ax, "Margins reconstructed")
    F.headline(ax, "Original weights over- or under-represent\nsome models and risk bands; raking does not", dy=14)
    F.label(ax, "a", dx=-62, dy=28)

    schemes = (("orig", "Original", 2.3, 1.0), ("raked", "Raking", 1.3, 0.75), ("trim", "Trimmed raking", 0.8, 0.5))
    RAKE_C, TRIM_C = darker(F.ACCENT, 0.45), tint(F.MUTED, 0.3)

    def draw(ax, items, xlabel, ref=None):
        n = len(items)
        for i, (lab, get) in enumerate(items):
            y = n - 1 - i
            for k, (s, _, lw, al) in enumerate(schemes):
                e, ci = get(s)
                yy = y + (0.24, 0.0, -0.24)[k]
                if s == "orig":
                    phys(ax, yy, pp(e), pp(ci[0]), pp(ci[1]), F.ACCENT, scheme="orig")
                elif s == "raked":
                    ax.plot([pp(ci[0]), pp(ci[1])], [yy, yy], color=RAKE_C, lw=0.9, solid_capstyle="butt", zorder=3)
                    F.raked_pt(ax, pp(e), yy, RAKE_C, ms=3.0)
                else:
                    ax.plot([pp(ci[0]), pp(ci[1])], [yy, yy], color=TRIM_C, lw=0.8, ls=(0, (2.5, 1.2)), zorder=3)
                    ax.plot([pp(e)], [yy], marker="|", ms=3.6, mew=1.0, color=TRIM_C, ls="none", zorder=4)
        ax.set_yticks(range(n))
        ax.set_yticklabels([lab for lab, _ in items][::-1])
        F.ylabels_clean(ax)
        F.grid(ax, "x")
        if ref is not None:
            F.refline(ax, x=ref)
        ax.set_ylim(-0.6, n - 0.4)
        ax.set_xlabel(xlabel)

    ax = fig.add_subplot(gs[1])
    pv = [("any", "Below minimum standard"), ("any134", "Harm, risk or sycophancy"), ("d2", "No actionable safety step"),
          ("d3", "Risk not recognized"), ("d5", "Perfunctory reply")]
    draw(ax, [(lab, (lambda o: lambda s: (S[s]["point"]["全部"][o]["p"], S[s]["ci"]["全部"][o]))(o)) for o, lab in pv],
         "Failure rate, all turns (%)")
    F.caption(ax, "Failure rate, all turns")
    F.headline(ax, "Failure rates barely change\nwith the weighting", dy=14)
    F.label(ax, "b", dx=-80, dy=28)
    ax = fig.add_subplot(gs[2])
    dk = [("('设计组', '旗舰')|('组', '本地')|any", "Flagship vs local"),
          ("('设计组', '心理微调')|('设计组', '对照基座')|any", "Fine-tuned vs control"),
          ("('设计组', '陪伴')|('设计组', '旗舰')|any", "Companion vs flagship"),
          ("('风险段', '高')|('风险段', '低')|any", "High vs low risk, both arms"),
          ("('臂×风险段', 'fixed', '高')|('臂×风险段', 'fixed', '低')|any", "High vs low risk, fixed arm")]
    draw(ax, [(lab, (lambda kk: lambda s: (S[s]["diffs"][kk][0], S[s]["diffs"][kk][1]))(k)) for k, lab in dk],
         "Difference in failure rate (%)", ref=0)
    F.caption(ax, "Key contrasts (below minimum standard)")
    F.headline(ax, "Only flagship vs local holds\nunder all three weightings", dy=14)
    F.label(ax, "c", dx=-80, dy=28)
    ytr = ax.get_yaxis_transform()
    ax.text(1.04, len(dk) - 0.55, "Significant\nunder", transform=ytr, fontsize=5.0, color=F.MUTED, va="center",
            linespacing=1.1)
    for i, (k, _) in enumerate(dk):
        sig_ = [F.sig(S[s_]["diffs"][k][1]) for s_, *_ in schemes]
        v = ("all three" if all(sig_) else "none" if not any(sig_) else
             "original only" if sig_ == [True, False, False] else "all but raking" if sig_ == [True, False, True]
             else "/".join(n for (s_, n, *_), g in zip(schemes, sig_) if g))
        ax.text(1.04, len(dk) - 1 - i, v, transform=ytr, fontsize=5.4, va="center",
                color=F.INK if all(sig_) else F.INK2, fontweight="bold" if all(sig_) else "normal")
    # 三种权重：底部一行图例（不再直接标注在首行）
    from matplotlib.lines import Line2D
    neff = {s_: f"{S[s_]['weights']['kish_n_eff']:.0f}" for s_, *_ in schemes}
    hs = [Line2D([], [], color=tint(F.ACCENT, 0.72), lw=3.0, marker="|", ms=4.5, mew=1.2, mec=F.ACCENT,
                 label=f"Original weights (effective n {neff['orig']})"),
          Line2D([], [], color=RAKE_C, lw=0.9, marker="v", ms=3.0, mfc="white", mec=RAKE_C, mew=0.8,
                 label=f"Raking weights (effective n {neff['raked']})"),
          Line2D([], [], color=TRIM_C, lw=0.8, ls=(0, (2.5, 1.2)), marker="|", ms=3.6, mew=1.0, mec=TRIM_C,
                 label=f"Trimmed raking weights (effective n {neff['trim']})")]
    fig.legend(handles=hs, loc="lower center", bbox_to_anchor=(0.55, -0.01), ncol=3, fontsize=5.2, frameon=False,
               handlelength=2.2, handletextpad=0.7, columnspacing=2.0)
    return save300(fig, out, "EDFig3_gold_weights")


# ================================================================ ED Fig. 4 d5 封顶规则
def edfig4(out):
    D = F.J("reports/d5_uncapped/d5_uncapped.json")["judges"]
    fig = plt.figure(figsize=(DOUBLE, 92 * MM))
    F.fig_title(fig, "The two caps on 'perfunctory reply': whom they affect and whether they match psychiatrists")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.9], wspace=0.75, left=0.12, right=0.9, top=0.75, bottom=0.2)
    SHADE = (0.0, 0.3, 0.52, 0.72)                  # 两个都用 / 只去复读 / 只去 <20 字 / 都去：同色由深到浅（与 b 同序）
    DY = (0.33, 0.11, -0.11, -0.33)
    # a 各模型三种口径（Qwen）
    ax = fig.add_subplot(gs[0])
    ch = D["Qwen"]["changes"]
    n = len(MODELS)
    for i, m in enumerate(MODELS):
        y = n - 1 - i
        c = ch[m]
        col = F.GROUP_C[F.GROUP_OF[m]]
        for k, key in enumerate(("fail_capped", "fail_noloop", "fail_noshort", "fail_raw")):
            ax.plot([0, pp(c[key])], [y + DY[k]] * 2, color=tint(col, 0.55 + 0.3 * SHADE[k]), lw=0.7, zorder=2)
            F.marker(ax, pp(c[key]), y + DY[k], tint(col, SHADE[k]), "o", filled=True, ms=2.7)
    ax.set_yticks(range(n))
    ax.set_yticklabels([mname(m) for m in MODELS][::-1])
    F.ylabels_clean(ax)
    F.grid(ax, "x")
    ax.set_xlim(0, 62)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel("Perfunctory-reply failure, Qwen judge (%)")
    F.caption(ax, "Perfunctory-reply failure rate by model", dy=4)
    F.headline(ax, "The caps matter most for the companion model\n(short replies) and CPsyCounX (repetition)", dy=14)
    F.label(ax, "a", dx=-60, dy=28)
    # b 与医师一致性
    ax = fig.add_subplot(gs[1])
    rules = (("capped", "Both caps"), ("noloop", "No repetition cap"), ("noshort", "No <20-char cap"), ("raw", "No caps"))
    for k, (jn, mk, _) in enumerate(JUDGES):
        v = D[jn]["gold"]["variants"]
        for i, (rk, _) in enumerate(rules):
            y = len(rules) - 1 - i + (0.15 if k == 0 else -0.15)
            w = v[rk]["all"]["wkappa"]
            lo, hi = v[rk]["ci"]["wkappa"]
            ax.plot([lo, hi], [y, y], color=F.ACCENT, lw=0.8, solid_capstyle="butt", zorder=3)
            F.marker(ax, w, y, F.ACCENT, mk, filled=True, ms=3.4)
    ax.set_yticks(range(len(rules)))
    ax.set_yticklabels([lab for _, lab in rules][::-1])
    F.ylabels_clean(ax)
    F.refline(ax, x=0)
    F.grid(ax, "x")
    ax.set_xlim(-0.1, 1.0)
    ax.set_ylim(-0.6, len(rules) - 0.4)
    ax.set_xlabel("Weighted κ with psychiatrists (n = 180)")
    F.caption(ax, "Agreement with psychiatrists (● Qwen, ■ DeepSeek)", dy=4)
    F.headline(ax, "Each cap brings the judge\ncloser to psychiatrists", dy=14)
    F.label(ax, "b", dx=-70, dy=28)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], ls="none", marker="o", ms=3.4, mfc=tint(F.INK2, sh), mec="white", mew=0.5, label=lab)
          for sh, lab in zip(SHADE, ("both caps (as scored)", "without the repetition cap",
                                     "without the <20-character cap", "without either cap"))]
    fig.text(0.12, 0.035, "a: four rows per model, top to bottom (dark to light), in the same order as b",
             fontsize=5.0, color=F.INK2)
    fig.legend(handles=hs, loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=4, fontsize=5.2, frameon=False,
               handletextpad=0.5, columnspacing=2.4)
    return save300(fig, out, "EDFig4_d5_cap_rules")


JOBS = {
    "F4": ("figures/main", fig4), "F5": ("figures/main", fig5), "F6": ("figures/main", fig6),
    "E1": ("figures/extended_data", edfig1), "E2": ("figures/extended_data", edfig2),
    "E3": ("figures/extended_data", edfig3), "E4": ("figures/extended_data", edfig4),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="论文终稿图 B 组（Fig. 4–6 + ED Fig. 1–4）")
    ap.add_argument("--out", default=str(REPO / "paper_final"))
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--titles", action="store_true", help="画整张图顶部的一句话标题（审阅用；投稿稿不画）")
    a = ap.parse_args(argv)
    F.SHOW_TITLE = a.titles
    F.style()
    root = Path(a.out)
    for k, (sub, f) in JOBS.items():
        if not a.only or k in a.only:
            for p in f(root / sub):
                print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
