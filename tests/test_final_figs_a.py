"""tools/final_figs_a.py：出 Fig. 1–3 到临时目录并检查文件存在（读真实报告 JSON，无网络、无模型）。"""
from __future__ import annotations

import pytest

pytest.importorskip("matplotlib")

from tools import final_figs_a  # noqa: E402

NAMES = ("Fig1_study_design", "Fig2_psychiatrist_failures", "Fig3_judge_vs_psychiatrist")


def test_all_figures_written(tmp_path):
    assert final_figs_a.main(["--out", str(tmp_path)]) == 0
    main = tmp_path / "figures" / "main"
    for n in NAMES:
        for ext in ("pdf", "png"):
            p = main / f"{n}.{ext}"
            assert p.exists() and p.stat().st_size > 1000, p


def test_only_subset(tmp_path):
    assert final_figs_a.main(["--out", str(tmp_path), "--only", "F2"]) == 0
    main = tmp_path / "figures" / "main"
    assert (main / "Fig2_psychiatrist_failures.png").exists()
    assert not (main / "Fig1_study_design.png").exists()


def test_mindchat_note_reads_holm():
    """d 面板注释：说人话（多重比较校正），并给出两套权重的 Holm 校正 p。"""
    note = final_figs_a.mindchat_note()
    assert note is None or ("multiple tests" in note and note.count("p = ") == 2 and " vs " in note)


def test_ed_fig5_written(tmp_path):
    assert final_figs_a.main(["--out", str(tmp_path), "--only", "E5"]) == 0
    p = tmp_path / "figures" / "extended_data" / "EDFig5_flip_table_full.png"
    assert p.exists() and p.stat().st_size > 1000


def test_diff_words():
    s = final_figs_a.diff_words(14.7, 41.1, (-43.0, -11.4))
    assert s.startswith("Flagship 15% vs local 41%") and "11–43 points" in s
    assert "no clear difference" in final_figs_a.diff_words(14.3, 7.3, (-7.7, 22.2))
