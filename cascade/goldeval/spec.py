"""Field declarations that parameterize the whole gold-eval harness over the three
model judges (Plan 4.4 §3.2). Sampling, blinding, and agreement all iterate JUDGES —
adding a judge or field is a one-place edit here. empathy_aux is deliberately absent
(P4.3 'never aggregated'; not human-annotated this round)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Literal
from cascade.schema import TurnRecord, CSSRSStage

RISK_LEVELS: tuple[str, ...] = ("none",) + tuple(s.value for s in CSSRSStage)  # 7 档：none 底档 + 6 C-SSRS（none=index 0）


@dataclass(frozen=True)
class FieldSpec:
    name: str                                          # 内部字段名（机器侧语义）
    kind: Literal["binary", "ordinal"]
    levels: tuple                                       # binary=(0,1)；risk=RISK_LEVELS；d*=(0,1,2)
    extract: Callable[[TurnRecord], int | str | None]   # 机器判取值；None = 机器未判
    judge: str
    # —— 人标层：全项目统一「1 = 好 / 0 = 差」——
    # 机器侧 abandon 是【构念存在】标志（1 = 求助者感到被抛弃 = 差），与另两维反向。
    # 三位医生连填 150 行时「1=好」是极强的心理默认，翻号后的数据看起来完全正常、
    # 事后查不出来。所以人标列一律正向命名（abandon → connect），读入时取反一次，
    # 机器侧语义与既有数据（runs/*.jsonl 已序列化 abandonment_felt）保持不动。
    h_name: str = ""                                    # 人标列名去 h_ 前缀；空 = 同 name
    h_invert: bool = False                              # True: 表里 1=好，读入时取反成机器语义

    @property
    def h_col(self) -> str:
        return f"h_{self.h_name or self.name}"


def _c(name: str, attr: str, *, h_name: str = "", h_invert: bool = False) -> FieldSpec:
    return FieldSpec(name, "binary", (0, 1),
                     lambda t, a=attr: int(getattr(t.appraisal, a)), "c_evaluator",
                     h_name=h_name, h_invert=h_invert)


def _d(i: int) -> FieldSpec:
    return FieldSpec(f"d{i}", "ordinal", (0, 1, 2),
                     lambda t, i=i: getattr(t.score, f"d{i}") if t.score is not None else None,
                     "rubric_judge")


JUDGES: dict[str, list[FieldSpec]] = {
    "c_evaluator": [_c("empathy", "empathy_detected"),
                    _c("deesc", "deescalation_attempt"),
                    _c("abandon", "abandonment_felt", h_name="connect", h_invert=True)],
    "risk_labeler": [FieldSpec("expressed_risk", "ordinal", RISK_LEVELS,
                               lambda t: (t.expressed_risk.value if isinstance(t.expressed_risk, CSSRSStage)
                                          else t.expressed_risk),   # CSSRSStage -> value; "none"/None pass through
                               "risk_labeler",
                               # 人标列叫 h_risk：h_expressed_risk 会把机器字段名印进表头，
                               # 撞上 sample.py 的盲化断言（隐藏列名不得出现在给标注者的表里）。
                               h_name="risk")],
    "rubric_judge": [_d(i) for i in range(1, 7)],
}

ALL_FIELDS: list[FieldSpec] = [f for fields in JUDGES.values() for f in fields]
