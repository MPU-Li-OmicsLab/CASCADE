"""按 d 维分数的富集抽样（6 维判官金标用）。纯逻辑、零 IO、零模型调用。

**为什么必须富集**：全语料 4796 轮里 `d1=1` 只有 **8 条**（0.17%）、`d1=0` 130 条、
`d6=0` 184 条。随机抽 240 轮，那几格会**整格为空** —— 零方差的格子上 κ 恒为 0，
无论判官多准。C 评估器那轮踩过同一个坑（baseline/high_risk 两层里人和机器判「抛弃」
都只有 0–4%，抛弃维结构上不可测，见 `enrich.py` 的长注释）。

**难点与解法**：一条轮次**同时**有 6 个维度的分，没法为每一维独立分层。
做法 = **按稀有度指派**：先把 (维, 档) 全部格子按语料里的条数升序排，每条轮次只归入
它所属的**最稀有那一格**，不重复计数。然后按稀有度轮转取样 —— 稀有格自然拿到地板，
常见格填满剩下的。

⚠️ 这样抽出来的样本**不是**总体的无偏样本（稀有格被过采样）。每条都带 `stratum`，
需要总体估计时按各层在语料中的占比重加权即可 —— 别直接把样本里的一致率当总体一致率。
"""
from __future__ import annotations
import random
from collections import Counter, defaultdict

from cascade.goldeval.enrich import scatter_order

DIMS: tuple[str, ...] = ("d1", "d2", "d3", "d4", "d5", "d6")
LEVELS: tuple[int, ...] = (0, 1, 2)


def cell_counts(turns, dims: tuple[str, ...] = DIMS) -> Counter:
    """每个 (维, 档) 格子在语料里有多少条。"""
    c: Counter = Counter()
    for t in turns:
        for d in dims:
            v = t["score"].get(d)
            if v is not None:
                c[(d, v)] += 1
    return c


def rarity_order(turns, dims: tuple[str, ...] = DIMS) -> list[tuple[str, int]]:
    """(维, 档) 按语料条数升序。并列时按 (维, 档) 定序 —— 保证可复现。"""
    c = cell_counts(turns, dims)
    return sorted(c, key=lambda k: (c[k], k[0], k[1]))


def assign_stratum(turn, order: list[tuple[str, int]]) -> tuple[str, int] | None:
    """这条轮次所属的**最稀有**格子。一条只进一个层，否则稀有层的配额会被稀释。"""
    for dim, level in order:
        if turn["score"].get(dim) == level:
            return (dim, level)
    return None


def enrich_by_dimension(turns, *, n: int, seed: int = 0, test_frac: float = 0.33,
                        dims: tuple[str, ...] = DIMS,
                        max_per_dialogue: int | None = None,
                        max_per_target: int | None = None,
                        must_include: dict | None = None,
                        exclude: set | None = None) -> list[dict]:
    """按稀有度轮转取最多 `n` 条。返回的每条 = 原 turn 的浅拷贝 + `stratum` + `split`。

    `max_per_dialogue`：同一段对话内的轮次共享人设/被测/上下文，强相关；不设帽的话
    n=240 可能只来自几十段，有效样本量远低于 240、κ 的精度被高估。**帽是硬的** ——
    顶到了宁可少给，让调用方去加宽语料，而不是悄悄超额。
    `must_include`：`{(维, 档): 最少条数}`，先于轮转满足。⚠️ **这个参数不是可选的
    锦上添花** —— 「指派到最稀有格」有个非显然的后果：**稀有度排最后的那一格永远为空**
    （任何轮次都会先被更稀有的格子吸走，而最后那格没有更后面的格子可退）。实测按
    d2/d3 富集 12 段时 `d2=0` 一条都没抽到，偏偏那是专家分歧最集中的格子。
    配额按**实际分数**计，不按指派到的层 —— 后者恒为 0，当配额用毫无意义。
    `exclude`：已经拿去讨论过的题不能再用来验证规则 —— 讨论过的题一致率必然上去。
    """
    exclude = exclude or set()
    pool = [t for t in turns if t["id"] not in exclude]
    order = rarity_order(pool, dims)
    buckets: dict = defaultdict(list)
    for t in pool:
        s = assign_stratum(t, order)
        if s is not None:
            buckets[s].append(t)

    rng = random.Random(seed)
    for s in buckets:
        buckets[s] = sorted(buckets[s], key=lambda t: t["id"])
        rng.shuffle(buckets[s])

    picked, per_dlg, per_tgt = [], Counter(), Counter()
    taken: set = set()

    def _take(t, stratum) -> bool:
        if t["id"] in taken:
            return False
        if max_per_dialogue is not None and per_dlg[t["dlg"]] >= max_per_dialogue:
            return False
        if max_per_target is not None and per_tgt[t["target"]] >= max_per_target:
            return False
        per_dlg[t["dlg"]] += 1
        per_tgt[t["target"]] += 1
        taken.add(t["id"])
        picked.append({**t, "stratum": stratum})
        return True

    # ① 先满足显式配额（按实际分数找人，不看指派到的层）
    for (dim, level), need in sorted((must_include or {}).items()):
        cands = sorted((t for t in pool if t["score"].get(dim) == level),
                       key=lambda t: t["id"])
        random.Random(f"{seed}-must-{dim}-{level}").shuffle(cands)
        got = 0
        for t in cands:
            if got >= need or len(picked) >= n:
                break
            if _take(t, assign_stratum(t, order)):
                got += 1

    # ② 再按稀有度轮转：稀有格先拿到地板，常见格填剩下的
    ranked = [s for s in order if buckets.get(s)]
    while len(picked) < n:
        took = False
        for s in ranked:
            if len(picked) >= n:
                break
            while buckets[s]:
                if _take(buckets[s].pop(), s):
                    took = True
                    break
        if not took:                      # 所有层都取空或被帽挡住 —— 给不满就是给不满
            break

    # dev/test 分层切分：每一层各自按 test_frac 切，避免某一层整层落在一边
    by_stratum: dict = defaultdict(list)
    for c in picked:
        by_stratum[c["stratum"]].append(c)
    for s, items in by_stratum.items():
        r = random.Random(f"{seed}-split-{s}")
        idx = list(range(len(items)))
        r.shuffle(idx)
        n_test = max(1, round(len(items) * test_frac)) if len(items) >= 2 else 0
        for rank, i in enumerate(idx):
            items[i]["split"] = "test" if rank < n_test else "dev"

    # 打散：相邻两条不来自同一段对话，否则升级轨迹会给判定打底
    return scatter_order(picked, seed=seed, key="dlg")
