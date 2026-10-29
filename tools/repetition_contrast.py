"""配对消融的复读率差（不经任何判官）：心理微调版 − 同基座对照，按人设配对自助法。零模型调用。

    python tools/repetition_contrast.py --out reports/repetition_contrast

复读 = cascade.analysis.repetition.is_loop（本轮同一整句 ≥8 字原样出现 ≥3 次）；疑似撞上限 = looks_truncated。
用途：CPsyCounX「更少离开用户（C 评估器）但回应质量差」这一点改由这个代码直接测的指标支撑，
不再依赖 d5（d5 只报观测，见 tools/length_adjusted_contrast 模块说明）。"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from cascade.analysis.repetition import is_loop, looks_truncated  # noqa: E402

PAIRS = (("cpsycoun", "internlm2-base"), ("mindchat", "qwen15-4b-base"), ("soulchat", "qwen3-8b-base"))
METRICS = {"loop": is_loop, "truncated": looks_truncated}


def load(path) -> dict:
    """→ {(模型, 臂): {人设: [(复读, 撞上限)]}}；跳过干预段与无风险段。"""
    out = defaultdict(lambda: defaultdict(list))
    with Path(path).open(encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("intervention") or d.get("risk_none"):
                continue
            per = d["case_profile"]["id"].split("-", 1)[0]
            for t in d["turns"]:
                r = t.get("target_response") or ""
                out[(d["target_model_id"], d["arm"])][per].append({k: int(fn(r)) for k, fn in METRICS.items()})
    return out


def rate(per: dict, personas, k) -> float | None:
    xs = [x[k] for p in personas for x in per.get(p, [])]
    return sum(xs) / len(xs) if xs else None


def analyse(data, *, n_boot=2000, seed=0) -> dict:
    rng = random.Random(seed)
    arms = sorted({a for _, a in data})
    rows = []
    for a, c in PAIRS:
        for arm in arms:
            pa, pc = data.get((a, arm)), data.get((c, arm))
            if not pa or not pc:
                continue
            pers = sorted(set(pa) | set(pc))
            for k in METRICS:
                ra, rc = rate(pa, pers, k), rate(pc, pers, k)
                diffs = []
                for _ in range(n_boot):
                    draw = [rng.choice(pers) for _ in pers]               # 两模型共用同一组人设 = 配对
                    x, y = rate(pa, draw, k), rate(pc, draw, k)
                    if None not in (x, y):
                        diffs.append(x - y)
                diffs.sort()
                ci = (diffs[int(0.025 * (len(diffs) - 1))], diffs[int(0.975 * (len(diffs) - 1))])
                n_with = sum(1 for p in pers if any(x[k] for x in pa.get(p, [])))
                rows.append({"a": a, "c": c, "arm": arm, "metric": k, "rate_a": ra, "rate_c": rc,
                             "diff": ra - rc, "ci": ci, "n_personas": len(pers), "personas_with_a": n_with})
    return {"rows": rows}


def render(r, meta) -> str:
    name = {"loop": "复读（本轮同句 ≥3 次）", "truncated": "疑似撞 max_tokens（≥2000 字且断在半句）"}
    L = ["# 配对消融 · 复读率差（不经判官）", "",
         f"- 生成 {meta['generated_at']}；语料 `{meta['corpus']}`（不含干预段 / 无风险段）；人设配对自助法 {meta['n_boot']} 次。",
         "- 差 = 心理微调版 − 同基座对照；「有复读的人设」= 该微调版在多少个人设上至少出现一次。", "",
         "| 配对 | 臂 | 指标 | 微调版 | 对照 | 差 [95% CI] | 有该现象的人设 |", "|---|---|---|---|---|---|---|"]
    for x in r["rows"]:
        L.append(f"| {x['a']} − {x['c']} | {x['arm']} | {name[x['metric']]} | {100 * x['rate_a']:.1f}% | "
                 f"{100 * x['rate_c']:.1f}% | {100 * x['diff']:+.1f} [{100 * x['ci'][0]:+.1f}, {100 * x['ci'][1]:+.1f}] | "
                 f"{x['personas_with_a']}/{x['n_personas']} |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="配对消融的复读率差（不经判官）")
    ap.add_argument("--corpus", default="runs/v2/dialogues_both_arms.jsonl")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    path = Path(a.corpus) if Path(a.corpus).is_absolute() else REPO / a.corpus
    r = analyse(load(path), n_boot=a.n_boot, seed=a.seed)
    meta = {"generated_at": time.strftime("%Y-%m-%d %H:%M"), "corpus": a.corpus, "n_boot": a.n_boot}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "repetition_contrast.json").write_text(json.dumps({"meta": meta, **r}, ensure_ascii=False, indent=2),
                                                  encoding="utf-8")
    (out / "repetition_contrast.md").write_text(render(r, meta), encoding="utf-8")
    print(f"报告 {out / 'repetition_contrast.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
