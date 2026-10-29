"""终稿图 B 组冒烟测试：Fig. 4–6 与 ED Fig. 1–4 能从 reports/ 生成（ED Fig. 1 依赖未入库的 xlsx，缺失时只画词汇与操纵检验面板）。"""
import pytest

pytest.importorskip("matplotlib")

from tools import final_figs_b  # noqa: E402


def test_all_figures_render(tmp_path):
    final_figs_b.main(["--out", str(tmp_path)])
    main = tmp_path / "figures/main"
    ed = tmp_path / "figures/extended_data"
    for name in ("Fig4_why_judges_err", "Fig5_psych_finetuning", "Fig6_design_pitfalls"):
        for ext in ("pdf", "png"):
            assert (main / f"{name}.{ext}").stat().st_size > 10_000
    for name in ("EDFig1_simulator_validity", "EDFig2_dose_by_turn", "EDFig3_gold_weights", "EDFig4_d5_cap_rules"):
        for ext in ("pdf", "png"):
            assert (ed / f"{name}.{ext}").stat().st_size > 10_000


def test_derived_numbers_match_reports():
    # 两臂按臂直接计算（tools/judge_dose_by_arm）；responsive 臂须与「两臂合并 − fixed」回推一致
    fx, rs = final_figs_b.arm_rates()
    D = final_figs_b.F.J("reports/judge_dose_v7/judge_dose.json")
    for band in ("低", "中", "高"):
        nb, nf = D["n_turns"][f"本地|both|{band}"], D["n_turns"][f"本地|fixed|{band}"]
        rb, rf = D["point"]["本地|both|any"][band][0], D["point"]["本地|fixed|any"][band][0]
        assert abs((rb * nb - rf * nf) / (nb - nf) - rs["point"][band]) < 1e-3
        # 轮数口径：judge_dose 计入未打分复读轮，「任一」结局的分母不计入，差几轮
        assert abs(fx["point"][band] - rf) < 1e-3 and 0 <= (nb - nf) - rs["n_turns"][band] <= 10
    assert rs["diff_ci"][0] < rs["diff"] < rs["diff_ci"][1]
    b = final_figs_b.parse_branch_pooled()
    assert b["regen_minus_orig"][1] < b["regen_minus_orig"][0] < b["regen_minus_orig"][2]
    assert b["intv_minus_regen"][1] < b["intv_minus_regen"][0] < b["intv_minus_regen"][2]


def test_ci_words():
    assert final_figs_b.ci_words(12.7, 20.5) == "95% CI 13–20"
    assert final_figs_b.ci_words(-20.0, -9.0) == "95% CI 9–20"
    assert final_figs_b.ci_words(-2.7, 23.2).startswith("95% CI −3 to +23")
