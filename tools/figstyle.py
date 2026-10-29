"""论文图的视觉系统（全文统一，改色只改这里）。被 tools/nmh_figures.py 使用。

  * 颜色只编码「模型组」：API / 旗舰 蓝、本地 / 对照基座 朱红、心理微调 绿、陪伴 靛紫
    （四色经 dataviz 校验脚本 all-pairs 色盲检验通过）；不属于任何模型组的数据（全部回复、测量工具本身、
    干预条件、显著性）一律用金色 ACCENT；数据标记不用黑色，黑色只给文字。
  * 图例只画图中真实出现的标记；形状 / 实心含义靠直接标注或面板说明，不画图中不存在的灰色样例。
  * 形状编码测量工具：Qwen 判官 ●、DeepSeek 判官 ■、C 评估器 ◆；实心 = CI 不含 0（OR 图为不含 1）。
  * 干预图「突出一个、其余灰」；风险档位用一条单色红梯度。
  * 文字一律墨色，颜色只在标记上；参考线、网格是淡实线；实心标记带白描边。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

MM = 1 / 25.4
DOUBLE = 180 * MM
SINGLE = 88 * MM
MINUS = "−"

INK = "#262626"          # 只用于文字
INK2 = "#595959"
MUTED = "#808080"
NEUTRAL = "#8c8c8c"      # 中性数据（医师间一致性刻度、稳定轨迹）
STEP = "#5f6b7a"
RULE = "#c4c4c4"
GRID = "#ebebeb"
ZEBRA = "#f6f6f5"
# 四个组色 + 金色强调色经 dataviz 校验（四组色 all-pairs 全部通过；金色与朱红同图时靠直接标注区分）
API = "#4A86CF"
LOCAL = "#E4572E"
LOCAL_ALL = "#7A6E66"   # 「本地开源模型」整组（心理微调 + 对照合并）：中性暖灰，避免被读成只有对照
PSYCH = "#1B9E77"
COMPANION = "#4E3D9E"
ACCENT = "#C49A00"       # 非模型组的数据：全部回复、测量工具、干预条件、显著性
FAIL = "#B2182B"
GROUP_C = {"A_flagship": API, "companion": COMPANION, "B2_psych": PSYCH, "C_base": LOCAL}
GROUP_N = {"A_flagship": "Flagship", "companion": "Companion", "B2_psych": "Fine-tuned", "C_base": "Same-base control"}
GROUPS = ("A_flagship", "companion", "B2_psych", "C_base")
JUDGE_M = {"qwen": "o", "deepseek": "s", "ceval": "D"}
PROMPT = ACCENT
ORIG = "#c8c8c8"
RESAMPLE = "#8c8c8c"
RISK = ["#fbe4dc", "#f5bfab", "#ec937a", "#db644f", "#bb3b31", "#88201c"]

STAGES = ["passive_ideation", "active_ideation", "method_inquiry", "intent", "plan", "means_timing"]
STAGE_N = ["Passive\nideation", "Active\nideation", "Method\ninquiry", "Intent", "Plan", "Means &\ntiming"]
DIM_N = {"d1": "Harmful content", "d2": "No actionable safety step", "d3": "Risk not recognized", "d4": "Sycophancy",
         "d5": "Perfunctory reply", "any": "Below minimum standard", "any134": "Harm, risk or sycophancy"}
PAIRS = (("cpsycoun", "internlm2-base", "CPsyCounX vs InternLM2"), ("mindchat", "qwen15-4b-base", "MindChat vs Qwen1.5"),
         ("soulchat", "qwen3-8b-base", "SoulChat vs Qwen3-8B"))
MODEL_N = {"openai/gpt-5.6-sol": "GPT-5.6 Sol", "anthropic/claude-opus-5": "Claude Opus 5",
           "deepseek/deepseek-v4-pro": "DeepSeek-V4-Pro", "minimax/minimax-m2-her": "MiniMax-M2-her",
           "cpsycoun": "CPsyCounX", "internlm2-base": "InternLM2", "mindchat": "MindChat",
           "qwen15-4b-base": "Qwen1.5-4B", "soulchat": "SoulChat", "qwen3-8b-base": "Qwen3-8B"}
GROUP_OF = {"openai/gpt-5.6-sol": "A_flagship", "anthropic/claude-opus-5": "A_flagship", "deepseek/deepseek-v4-pro": "A_flagship",
            "minimax/minimax-m2-her": "companion", "cpsycoun": "B2_psych", "mindchat": "B2_psych", "soulchat": "B2_psych",
            "internlm2-base": "C_base", "qwen15-4b-base": "C_base", "qwen3-8b-base": "C_base"}


def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 6.5, "axes.titlesize": 6.5, "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6,
        "legend.fontsize": 6, "axes.linewidth": 0.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelcolor": INK, "ytick.labelcolor": INK, "text.color": INK,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
        "xtick.major.pad": 1.8, "ytick.major.pad": 1.8, "axes.labelpad": 2.5,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "legend.handletextpad": 0.3, "legend.borderaxespad": 0.2, "legend.labelspacing": 0.35,
        "lines.solid_capstyle": "round", "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 400})


def J(path):
    return json.loads((REPO / path).read_text(encoding="utf-8"))


HEADLINE_PT = 8.0          # 一行面板结论句约占的高度（pt）


LABEL_GAP = 4.0            # 面板字母底边高出面板小标题顶边的距离（pt）


def _shift_label(ax):
    """投稿版：面板字母一律放在面板小标题（caption）之上 LABEL_GAP；没有小标题的面板放在轴顶之上 12 pt。
    同一行的面板小标题都在 dy = 4，字母因此在同一高度（label / caption 调用顺序不限）。"""
    art = getattr(ax, "_cl_label", None)
    if art is None:
        return
    top = getattr(ax, "_cl_captop", None)
    x, _ = art.xyann
    art.xyann = (x, (top + LABEL_GAP) if top is not None else 12.0)


def label(ax, letter, dx=-22, dy=4):
    art = ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(dx, dy), textcoords="offset points",
                      fontsize=8, fontweight="bold", ha="left", va="bottom")
    ax._cl_label = art
    if not SHOW_TITLE:
        _shift_label(ax)


def align_labels(fig, tol=0.06):
    """保存前统一摆放面板字母：每个字母放在其面板整块区域（含刻度标签、轴名、小标题）的左上角；
    同一行（轴顶相近）的字母取同一高度，同一列（绘图区左边相近）的字母取最左的位置。"""
    items = [(ax, ax._cl_label) for ax in fig.axes if getattr(ax, "_cl_label", None) is not None]
    if not items:
        return
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    pt = fig.dpi / 72
    rows = []
    for ax, art in items:
        art.set_visible(False)
        tb = ax.get_tightbbox(r)
        left = ax.patch.get_window_extent(r).x0
        try:
            yb = ax.yaxis.get_tightbbox(r)
            if yb is not None and yb.width > 0:
                left = min(left, yb.x0)
        except Exception:
            pass
        pe = ax.patch.get_window_extent(r)
        rows.append({"art": art, "x": left - 2 * pt, "y": tb.y1 + 2 * pt, "top": pe.y1, "px0": pe.x0})
    # 行：轴顶相近 → 取最高的 y；列：左边界相近 → 取最左的 x
    for grp_key, coord, span, agg in (("top", "y", H, max), ("px0", "x", W, min)):
        srt = sorted(rows, key=lambda d: d[grp_key])
        cluster = []
        for d in srt + [None]:
            if d is not None and (not cluster or abs(d[grp_key] - cluster[0][grp_key]) < tol * span):
                cluster.append(d)
                continue
            v = agg(c[coord] for c in cluster)
            for c in cluster:
                c[coord] = v
            cluster = [d] if d is not None else []
    for d in rows:
        art = d["art"]
        fig.text(d["x"] / W, d["y"] / H, art.get_text(), fontsize=8, fontweight="bold", ha="left", va="bottom")
    title = getattr(fig, "_cl_title", None)            # 审阅版整图标题：放到最高的面板字母之上
    if title is not None:
        top = max(d["y"] for d in rows) + 12 * pt
        x0, y0 = title.get_position()
        title.set_position((x0, max(y0, top / H)))


def caption(ax, text, dy=4):
    ax.annotate(text, xy=(0, 1), xycoords="axes fraction", xytext=(0, dy), textcoords="offset points",
                fontsize=6, color=MUTED, ha="left", va="bottom")
    ax._cl_captop = dy + 7.5 * (text.count("\n") + 1)
    if not SHOW_TITLE:
        _shift_label(ax)


def grid(ax, axis="x"):
    ax.grid(axis=axis, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def refline(ax, x=None, y=None):
    if x is not None:
        ax.axvline(x, color=RULE, lw=0.6, zorder=1)
    if y is not None:
        ax.axhline(y, color=RULE, lw=0.6, zorder=1)


def ylabels_clean(ax):
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)


def sig(ci, null=0.0):
    return bool(ci) and (ci[0] > null or ci[1] < null)


def marker(ax, x, y, color, m="o", filled=True, ms=3.8, z=4):
    if filled:
        ax.plot([x], [y], m, ms=ms, mfc=color, mec="white", mew=0.5, zorder=z, ls="none")
    else:
        ax.plot([x], [y], m, ms=ms * 0.92, mfc="white", mec=color, mew=0.9, zorder=z, ls="none")


def point_h(ax, y, est, lo, hi, color, m="o", filled=True, ms=3.8, lw=0.8, z=3):
    ax.plot([lo, hi], [y, y], color=color, lw=lw, zorder=z, solid_capstyle="butt")
    marker(ax, est, y, color, m, filled, ms, z + 1)


def dlabel(ax, x, y, text, color, dx=4, fs=5.8, bold=False):
    """直接标注：彩色圆点 + 墨色文字。"""
    ax.annotate("●", xy=(x, y), xytext=(dx, 0), textcoords="offset points", color=color, fontsize=5,
                va="center", ha="left", annotation_clip=False)
    ax.annotate(text, xy=(x, y), xytext=(dx + 6, 0), textcoords="offset points", color=INK, fontsize=fs,
                va="center", ha="left", fontweight="bold" if bold else "normal", annotation_clip=False)


#: 整张图顶部的一句话标题（学习 / 审阅版画，投稿版不画 —— Nature 系把图标题写在图注里）。
#: final_figs_a / final_figs_b 默认 False（paper_final/figures 是投稿稿），--titles 打开；final_figs_zh 设 True。
SHOW_TITLE = False


def fig_title(fig, text, y=1.01, x=0.01, fs=7.4):
    if SHOW_TITLE:
        fig._cl_title = fig.text(x, y, text, fontsize=fs, fontweight="bold", color=INK, va="bottom", ha="left")


def headline(ax, text, dy=14, fs=6.3, x=0.0, ha="left"):
    """面板级一句话结论：只画在审阅版 / 中文版（SHOW_TITLE）。投稿版按 Nature 惯例不画（2026-10-24 用户定，方案 A），
    结论写在图注；面板上只留灰色的描述性小标题（caption）。"""
    if not SHOW_TITLE:
        return
    ax.annotate(text, xy=(x, 1), xycoords="axes fraction", xytext=(0, dy), textcoords="offset points",
                fontsize=fs, fontweight="bold", color=INK, ha=ha, va="bottom", annotation_clip=False)


def raked_pt(ax, x, y, color, ms=3.0, dy=0.0, z=6):
    """raking 权重的医师金标：主图只画点估计 —— 空心倒三角，压在原权重 CI 带上方（raking 的 CI 在 ED Fig. 3 / ST3）。"""
    ax.plot([x], [y + dy], marker="v", ms=ms, mfc="white", mec=color, mew=0.8, ls="none", zorder=z)


def raked_handle(color, label_="", ms=3.0):
    return Line2D([], [], marker="v", ms=ms, mfc="white", mec=color, mew=0.8, ls="none", label=label_)


def handle(color, m="o", filled=True, label_="", ms=3.8):
    return Line2D([], [], color=color, marker=m, ms=ms, ls="none", mfc=color if filled else "white",
                  mec="white" if filled else color, mew=0.5 if filled else 0.9, label=label_)


def repel(ys, gap):
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for k in range(1, len(order)):
        a, b = order[k - 1], order[k]
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return out


def save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = out / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight", pad_inches=0.03)
        paths.append(p)
    plt.close(fig)
    return paths
