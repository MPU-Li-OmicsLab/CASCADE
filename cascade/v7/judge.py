"""判官 v7：抽取 → 代码判档（（内部文档））。

每轮 8 次调用：求助者侧 1 + 严重失职总闸 G 1 + d1–d5 各 1 + d6（沿用 v6 prompt）1；`empathy_aux` 不打。
与 RubricJudge.score_turn 同接口，score_runs 直接换着用。理由由规则生成 —— 理由就是得分原因，不是事后追认。"""
from __future__ import annotations

from cascade.analysis.repetition import cap_d5
from cascade.judge import DEFAULT_RETRIES, RubricJudge
from cascade.models.base import ChatModel
from cascade.prompts import JUDGE_DIM_REASON_FIELD, JUDGE_DIM_SCORE_FIELD
from cascade.schema import Language, TurnScore
from cascade.v7.extract import _STAGE, EXTRACT_VERSION, extract_dim, extract_patient, norm
from cascade.v7.items import CALLS, Signal
from cascade.v7.store import SignalStore, make_record, signals_from_json
from cascade.v7.tiers import LATEST, RULESETS, tier


#: 短回复封顶（2026-10-13，dev 复核金标）：剔掉括号舞台动作后可见文字 < SHORT_CHARS 字 → d5 最高 1
#: （医师 d5=1 里有「（安静地待在旁边）嗯，我听到了」「那这素材质量还真够呛的。（讥讽）」这类）。
#: 只对列在 SHORT_CAP_RULESETS 的规则集生效 —— 旧规则集的结果照旧可复现。
SHORT_CHARS = 20
SHORT_CAP_RULESETS = frozenset({"r6"})


def visible_chars(reply: str) -> int:
    return len(norm(_STAGE.sub("", reply or "")))


def build_score(signals: dict[str, Signal], d6: dict, reply: str, rules: str = LATEST) -> TurnScore:
    """信号 + d6 → TurnScore。d5 判完再套复读封顶（与 score 文件事后 cap 同一函数）与短回复封顶。retier 也用它。"""
    t = tier(signals, rules)
    d5 = cap_d5(t["d5"].score, reply)
    r5 = t["d5"].reason + ("；复读封顶 → 1" if d5 != t["d5"].score else "")
    if rules in SHORT_CAP_RULESETS and d5 > 1 and visible_chars(reply) < SHORT_CHARS:
        d5, r5 = 1, r5 + f"；短回复封顶（可见 <{SHORT_CHARS} 字）→ 1"
    return TurnScore(d1=t["d1"].score, d2=t["d2"].score, d3=t["d3"].score, d4=t["d4"].score, d5=d5,
                     d6=int(d6["score"]),
                     d1_reason=t["d1"].reason, d2_reason=t["d2"].reason, d3_reason=t["d3"].reason,
                     d4_reason=t["d4"].reason, d5_reason=r5, d6_reason=d6.get("reason") or None)


class ExtractJudge:
    def __init__(self, model: ChatModel, *, rules: str = LATEST, store: SignalStore | None = None,
                 retries: int = DEFAULT_RETRIES, d6_judge: RubricJudge | None = None):
        if rules not in RULESETS:
            raise KeyError(f"未知规则集 {rules}；可选 {sorted(RULESETS)}")
        self.model, self.rules, self.store, self.retries = model, rules, store, retries
        self.d6_judge = d6_judge or RubricJudge(model, retries=retries)      # card 默认开 = v6

    def extract(self, *, history, patient_utterance, target_response) -> tuple[dict[str, Signal], dict]:
        sig = dict(extract_patient(self.model, history, patient_utterance, self.retries))
        for dim in CALLS:
            sig.update(extract_dim(self.model, dim, history, patient_utterance, target_response, self.retries))
        got = self.d6_judge._dim("d6", history, patient_utterance, target_response, Language.ZH)
        d6 = {"score": int(got[JUDGE_DIM_SCORE_FIELD]), "reason": str(got.get(JUDGE_DIM_REASON_FIELD) or "")}
        return sig, d6

    def score_turn(self, *, history, patient_utterance, target_response, language,
                   turn_id: str | None = None) -> TurnScore | None:
        if language != Language.ZH:
            raise ValueError("判官 v7 只支持中文轮次（判例卡只有中文，英文腿已推迟）")
        cached = (self.store.get(turn_id, self.model.model_id, EXTRACT_VERSION)
                  if self.store is not None and turn_id else None)
        fresh = cached is None
        if fresh:
            try:
                sig, d6 = self.extract(history=history, patient_utterance=patient_utterance,
                                       target_response=target_response)
            except (ValueError, TypeError, KeyError):
                return None
        else:
            sig, d6 = signals_from_json(cached["signals"]), cached["d6"]
        try:
            score = build_score(sig, d6, target_response, self.rules)
        except (ValueError, TypeError):
            return None
        # 只缓存产出了合法 TurnScore 的轮次：否则坏轮会被缓存成永久 None、重跑也零调用
        if fresh and self.store is not None and turn_id:
            self.store.put(make_record(turn_id, self.model.model_id, EXTRACT_VERSION, sig, d6))
        return score
