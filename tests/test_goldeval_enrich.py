"""Enriched C-evaluator gold sampling (Plan 4.4): over-sample the scarce, informative
strata (evaluator disagreement, abandon candidates), quota high-risk, fill baseline,
split dev/test. Pure logic — caller precomputes each turn's stratum flags."""
from cascade.goldeval.enrich import enrich_sample, scatter_order


def _t(i, dis=False, ab=False, hr=False, lang="zh", dlg=None):
    return {"id": f"u{i}", "lang": lang, "disagree": dis, "abandon": ab, "high_risk": hr,
            "dlg": dlg if dlg is not None else f"d{i}"}


def test_takes_all_scarce_and_fills_baseline():
    turns = [_t(0, dis=True), _t(1, dis=True), _t(2, ab=True)] + [_t(i) for i in range(3, 40)]
    sel = enrich_sample(turns, n=10, seed=0)
    strata = {s["id"]: s["stratum"] for s in sel}
    assert {"u0", "u1", "u2"} <= set(strata)                 # all scarce kept
    assert strata["u0"] == "disagree" and strata["u2"] == "abandon"
    assert len(sel) == 10                                     # filled to n from baseline
    assert sum(1 for s in sel if s["stratum"] == "baseline") == 7


def test_priority_dedup_disagree_over_high_risk():
    turns = [_t(0, dis=True, hr=True)] + [_t(i) for i in range(1, 20)]
    sel = enrich_sample(turns, n=5, seed=0)
    s0 = next(s for s in sel if s["id"] == "u0")
    assert s0["stratum"] == "disagree"                       # highest priority, not double-counted


def test_disagree_oversampled_but_not_to_the_exclusion_of_baseline():
    # 契约变更（2026-08-02）：原先是「全取 disagree」，现在受 dis_frac 软配额约束。
    # 理由见 enrich_sample docstring：全取会让金标 100% 是分歧轮，估不出绝对一致率、
    # 也装不下「三家一致但都错」。这里断言的是「超配但不独占」。
    turns = [_t(i, dis=True) for i in range(8)] + [_t(i) for i in range(8, 30)]
    sel = enrich_sample(turns, n=10, seed=0, dis_frac=0.5)
    n_dis = sum(1 for s in sel if s["stratum"] == "disagree")
    assert n_dis == 5                                   # 0.5 × n
    assert n_dis / len(sel) > 8 / 30                    # 仍显著高于其在池中的占比 = 超配
    assert len(sel) == 10 and any(s["stratum"] == "baseline" for s in sel)


def test_high_risk_soft_cap_when_baseline_available():
    turns = [_t(i, hr=True) for i in range(30)] + [_t(i) for i in range(30, 60)]
    sel = enrich_sample(turns, n=20, hr_frac=0.2, seed=0)
    assert sum(1 for s in sel if s["stratum"] == "high_risk") == 4     # soft-capped at 0.2*n
    assert len(sel) == 20                                              # baseline fills the rest


def test_fills_n_when_baseline_exhausted_by_relaxing_quota():
    turns = [_t(i, hr=True) for i in range(30)]              # only high-risk, no baseline
    sel = enrich_sample(turns, n=20, hr_frac=0.2, seed=0)
    assert len(sel) == 20                                     # quota relaxed to still reach n


def test_dev_test_split_deterministic_and_stratified():
    turns = [_t(i, dis=True) for i in range(6)] + [_t(i) for i in range(6, 30)]
    sel = enrich_sample(turns, n=12, test_frac=0.5, seed=0)
    assert {s["split"] for s in sel} == {"dev", "test"}
    assert sum(1 for s in sel if s["split"] == "test") == 6          # ~half test per stratum
    sel2 = enrich_sample(turns, n=12, test_frac=0.5, seed=0)
    assert [s["id"] for s in sel] == [s["id"] for s in sel2]         # deterministic under seed


# —— 每段对弈配额（聚类控制）——————————————————————————————
# 同一段对弈内的交换共享人设/被测/上下文，高度相关；不设上限时 300 条可能只来自几十段，
# 有效样本量远小于 300，κ 的精度会被高估。上限是硬的：宁可取不满 n，也不偷偷突破。

def test_max_per_dialogue_caps_each_dialogue():
    turns = [_t(i, dlg="dA") for i in range(20)] + [_t(i + 100, dlg="dB") for i in range(20)]
    sel = enrich_sample(turns, n=10, seed=0, max_per_dialogue=2)
    from collections import Counter
    per = Counter(s["dlg"] for s in sel)
    assert max(per.values()) <= 2
    assert len(sel) == 4                                    # 2 段 × 2 = 硬上限，取不满 n


def test_max_per_dialogue_applies_across_strata():
    # 同一段里既有 disagree 又有 baseline：优先级仍生效，但合计不超过 cap
    turns = ([_t(0, dis=True, dlg="dA"), _t(1, dis=True, dlg="dA"), _t(2, dlg="dA")]
             + [_t(i, dlg=f"d{i}") for i in range(3, 30)])
    sel = enrich_sample(turns, n=10, seed=0, max_per_dialogue=2)
    got = {s["id"] for s in sel if s["dlg"] == "dA"}
    assert got == {"u0", "u1"}                              # 两条 disagree 优先，baseline u2 被挤掉
    assert len(sel) == 10


def test_no_cap_by_default_preserves_old_behaviour():
    turns = [_t(i, dlg="dA") for i in range(30)]
    assert len(enrich_sample(turns, n=10, seed=0)) == 10     # 不传 cap = 旧行为


# —— 打散呈现顺序（标注者内的携带效应）——————————————————————
def test_scatter_order_breaks_same_dialogue_adjacency():
    items = [_t(i, dlg=f"d{i % 20}") for i in range(60)]     # 20 段 × 3 条
    out = scatter_order(items, seed=0)
    assert len(out) == 60 and {o["id"] for o in out} == {i["id"] for i in items}
    assert all(a["dlg"] != b["dlg"] for a, b in zip(out, out[1:]))


def test_scatter_order_deterministic_under_seed():
    items = [_t(i, dlg=f"d{i % 7}") for i in range(28)]
    assert [o["id"] for o in scatter_order(items, seed=1)] == \
           [o["id"] for o in scatter_order(items, seed=1)]
    assert [o["id"] for o in scatter_order(items, seed=1)] != \
           [o["id"] for o in scatter_order(items, seed=2)]


def test_scatter_order_survives_one_dominant_dialogue():
    # 一段占了绝大多数 → 无法完全避免相邻，但不得丢件或重复
    items = [_t(i, dlg="dA") for i in range(10)] + [_t(100, dlg="dB")]
    out = scatter_order(items, seed=0)
    assert sorted(o["id"] for o in out) == sorted(i["id"] for i in items)


# —— 每被测配额（代表性控制）——————————————————————————————
# 分歧率与被测强相关（实测陪伴类 65% vs claude 14%），"优先全取 disagree" 会让
# 金标被一个被测淹没 → 选出的评估器只是"最擅长判那一家的"。

def test_max_per_model_caps_each_model():
    turns = ([_t(i, dis=True, dlg=f"a{i}") for i in range(20)]          # 全来自被测 A
             + [_t(i + 100, dlg=f"b{i}") for i in range(20)])
    for t in turns[:20]:
        t["model"] = "A"
    for t in turns[20:]:
        t["model"] = "B"
    sel = enrich_sample(turns, n=20, seed=0, max_per_model=6)
    from collections import Counter
    per = Counter(s["model"] for s in sel)
    assert max(per.values()) <= 6
    assert len(sel) == 12                                   # 2 家 × 6 = 硬上限


def test_max_per_model_composes_with_max_per_dialogue():
    turns = []
    for m in ("A", "B"):
        for i in range(10):
            for j in range(5):                              # 每家 10 段 × 每段 5 轮
                turns.append(dict(_t(f"{m}{i}_{j}", dlg=f"{m}{i}"), model=m))
    sel = enrich_sample(turns, n=24, seed=0, max_per_dialogue=2, max_per_model=12)
    from collections import Counter
    assert max(Counter(s["dlg"] for s in sel).values()) <= 2
    assert max(Counter(s["model"] for s in sel).values()) <= 12
    assert len(sel) == 24


# —— disagree 软配额（可加权还原 + 能看见"共识但错"）——————————————
# 全取 disagree 时金标 100% 落在分歧轮：只能比较评估器优劣，估不出绝对一致率，
# 且"三家一致但都错"这类系统性偏松永远进不了金标。

def test_disagree_soft_capped_leaving_room_for_other_strata():
    turns = ([_t(i, dis=True) for i in range(200)]
             + [_t(i + 500, hr=True) for i in range(100)]
             + [_t(i + 900) for i in range(200)])
    sel = enrich_sample(turns, n=100, seed=0, dis_frac=0.5, hr_frac=0.2)
    from collections import Counter
    per = Counter(s["stratum"] for s in sel)
    assert per["disagree"] == 50                     # 0.5 × n，不再吃满
    assert per["high_risk"] == 20                    # 0.2 × n
    assert per["baseline"] == 30                     # 其余回填
    assert len(sel) == 100


def test_disagree_quota_relaxed_when_nothing_else_available():
    turns = [_t(i, dis=True) for i in range(200)]     # 只有分歧轮
    sel = enrich_sample(turns, n=100, seed=0, dis_frac=0.5)
    assert len(sel) == 100                           # 放宽以达到 n（有总比缺好）


# —— connect_hard 层（2026-08 加）：abandon 维在 baseline/high_risk 层零方差，κ 恒为 0，
#    那部分标注对这一维没有信息；本层按【文本形态】挑，且不含任何评估器自身的判定。
def _tc(i, **kw):
    t = _t(i, **{k: v for k, v in kw.items() if k in ("dis", "ab", "hr", "lang", "dlg")})
    t["connect_hard"] = kw.get("ch", False)
    return t


def test_connect_hard_is_its_own_stratum_between_abandon_and_high_risk():
    turns = [_tc(0, ch=True)] + [_tc(i) for i in range(1, 20)]
    sel = enrich_sample(turns, n=5, seed=0)
    assert next(s for s in sel if s["id"] == "u0")["stratum"] == "connect_hard"


def test_disagree_and_abandon_still_outrank_connect_hard():
    """两层用评估器输出定义、且更稀缺，不能被 connect_hard 抢走。"""
    turns = [_tc(0, dis=True, ch=True), _tc(1, ab=True, ch=True)] + [_tc(i) for i in range(2, 20)]
    strata = {s["id"]: s["stratum"] for s in enrich_sample(turns, n=8, seed=0)}
    assert strata["u0"] == "disagree" and strata["u1"] == "abandon"


def test_connect_hard_outranks_high_risk():
    """高危层实测 abandon 维零方差（人 2%/机 0%），换成 connect_hard 才有信息。"""
    turns = [_tc(0, hr=True, ch=True)] + [_tc(i) for i in range(1, 20)]
    strata = {s["id"]: s["stratum"] for s in enrich_sample(turns, n=5, seed=0)}
    assert strata["u0"] == "connect_hard"


def test_ch_frac_quotas_connect_hard():
    turns = [_tc(i, ch=True) for i in range(50)] + [_tc(i) for i in range(50, 100)]
    sel = enrich_sample(turns, n=20, ch_frac=0.25, dis_frac=0.5, seed=0)
    n_ch = sum(1 for s in sel if s["stratum"] == "connect_hard")
    assert n_ch == 5                                          # round(0.25*20)
    assert sum(1 for s in sel if s["stratum"] == "baseline") == 15


def test_ch_quota_relaxes_when_other_pools_run_dry():
    turns = [_tc(i, ch=True) for i in range(30)]
    sel = enrich_sample(turns, n=20, ch_frac=0.2, seed=0)
    assert len(sel) == 20                                     # 放宽后仍取满 n
