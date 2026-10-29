"""tools/final_tables.py 冒烟测试：输出到临时目录，检查投稿表格的文件与行数。只读 reports/，零模型调用。"""
import csv

import pytest

from tools import final_tables

NEEDED = [
    "judge_vs_gold/judge_vs_gold.json",
    "primary_tests/primary_tests.json",
    "gold_prevalence/prevalence.json",
    "gold_prevalence_raked/prevalence_raked.json",
    "judge_error_by_model/judge_error_by_model.json",
]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    if not all((final_tables.REPO / "reports" / p).exists() for p in NEEDED):
        pytest.skip("reports/ 下缺少所需 JSON")
    out = tmp_path_factory.mktemp("paper_final")
    counts = final_tables.main(["--out", str(out)])
    return out, counts


def _csv_rows(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.reader(f))[1:]


def test_files_exist(built):
    out, _ = built
    for stem in ("EDTable1_target_models", "STable1_flip_table", "STable2_primary_tests", "STable3_gold_prevalence",
                 "STable4_judge_error_by_model"):
        assert (out / "tables" / f"{stem}.csv").exists(), stem
        assert (out / "tables" / f"{stem}.md").exists(), stem


def test_row_counts(built):
    out, counts = built
    t = out / "tables"
    assert len(_csv_rows(t / "EDTable1_target_models.csv")) == 10
    flip = final_tables.json.loads((final_tables.REPO / "reports/judge_vs_gold/judge_vs_gold.json").read_text(encoding="utf-8"))
    assert len(_csv_rows(t / "STable1_flip_table.csv")) == len(flip["Qwen"]["rows"]) + len(flip["DeepSeek"]["rows"])
    assert len(_csv_rows(t / "STable2_primary_tests.csv")) == 21          # 7 个检验 × 原 / raking / 截断
    assert len(_csv_rows(t / "STable3_gold_prevalence.csv")) > 200
    assert len(_csv_rows(t / "STable4_judge_error_by_model.csv")) == 120  # 2 判官 × 10 模型 × 6 结局
    assert counts["EDTable1_target_models"] == 10


def test_ed_table_groups(built):
    out, _ = built
    rows = _csv_rows(out / "tables" / "EDTable1_target_models.csv")
    groups = {r[2] for r in rows}
    assert groups == {"旗舰", "陪伴", "心理微调", "对照基座"}
