"""tools/repetition_contrast：按人设配对的复读率差。"""
from tools import repetition_contrast as RC

LOOP = "我在这里陪着你，一直都在。" * 3
OK = "我听到了，你愿意多说一点吗？"


def _data():
    return {("cpsycoun", "fixed"): {"p1": [{"loop": 1, "truncated": 0}, {"loop": 0, "truncated": 0}],
                                    "p2": [{"loop": 1, "truncated": 1}]},
            ("internlm2-base", "fixed"): {"p1": [{"loop": 0, "truncated": 0}], "p2": [{"loop": 0, "truncated": 0}]}}


def test_metrics_detect_loop():
    assert RC.METRICS["loop"](LOOP) and not RC.METRICS["loop"](OK)


def test_analyse_paired_difference():
    r = RC.analyse(_data(), n_boot=200)
    row = next(x for x in r["rows"] if x["a"] == "cpsycoun" and x["metric"] == "loop")
    assert row["rate_a"] == 2 / 3 and row["rate_c"] == 0 and row["diff"] == 2 / 3
    assert row["ci"][0] <= row["diff"] <= row["ci"][1]
    assert row["personas_with_a"] == 2 and row["n_personas"] == 2
