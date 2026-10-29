"""判官 v7：用信号缓存按指定规则集重新判档，零模型调用（spec §2、§6 第 3 步）。

    python tools/retier.py --in runs/v2/v7_dev.jsonl --signals runs/v2/v7_dev.jsonl.signals.jsonl \
        --rules r1 --out runs/v2/v7_dev_r1.jsonl

缓存里混着多个 模型 / 问题版本、或缺规则要用的信号 → 直接报错退出，不静默套用。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from cascade.v7.extract import EXTRACT_VERSION                        # noqa: E402
from cascade.v7.judge import build_score                              # noqa: E402
from cascade.v7.store import SignalStore, signals_from_json           # noqa: E402
from cascade.v7.tiers import RULESETS, required_signals               # noqa: E402


class RetierError(ValueError):
    pass


def validate_cache(latest: dict, rules: str) -> tuple[str, str]:
    """校验信号缓存可用于规则集 rules，返回唯一的 (model, items_version)；否则抛 RetierError。"""
    if rules not in RULESETS:
        raise RetierError(f"未知规则集 {rules}；可选 {sorted(RULESETS)}")
    if not latest:
        raise RetierError("信号缓存为空（或没有该模型的记录）")
    combos = {(r["model"], r["items_version"]) for r in latest.values()}
    if len(combos) > 1:
        raise RetierError(f"缓存里混着多个 模型 / 问题版本：{sorted(combos)}；用 --model 指定或分开文件")
    need = required_signals(rules)
    for rec in latest.values():
        missing = need - set(rec["signals"])
        if missing:
            raise RetierError(f"{rec['id']} 缺信号 {sorted(missing)}：规则 {rules} 要用，请用新问题版本在 dev 重抽")
    (combo,) = combos
    return combo


def retier(src, signals_path, rules: str, out, model: str | None = None) -> dict:
    if rules not in RULESETS:
        raise RetierError(f"未知规则集 {rules}；可选 {sorted(RULESETS)}")
    latest = SignalStore(signals_path).latest_by_id(model)
    m, v = validate_cache(latest, rules)
    n = stale = 0
    if v != EXTRACT_VERSION:
        print(f"retier 警告：缓存的 items_version={v} ≠ 当前 EXTRACT_VERSION={EXTRACT_VERSION}；"
              "用旧问题版本的信号套新规则，确认是有意为之", file=sys.stderr)
    with Path(src).open(encoding="utf-8-sig") as f, Path(out).open("w", encoding="utf-8") as g:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            base = f'{d["target_model_id"]}|{d["arm"]}|{d["case_profile"]["id"]}'
            for t in d.get("turns", []):
                rec = latest.get(f'{base}#t{t["turn"]}')
                if rec is None:
                    if t.get("score") is not None:       # 有分却无缓存 = 旧分残留（旧规则 / 别的模型）：清掉，免得被当 rN 读
                        t["score"] = None
                        stale += 1
                    continue                              # 本来就没分的轮：原样，不计
                t["score"] = build_score(signals_from_json(rec["signals"]), rec["d6"],
                                         t.get("target_response") or "", rules).model_dump()
                n += 1
            g.write(json.dumps(d, ensure_ascii=False) + "\n")
    if stale:
        print(f"retier 警告：{stale} 轮在 --in 里有分但信号缓存里没有记录（已把它们的分清空）；"
              "先回第 1 步补抽", file=sys.stderr)
    info = {"rescored": n, "not_cached": stale, "rules": rules, "model": m, "items_version": v}
    Path(f"{out}.meta.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="判官 v7：用信号缓存按规则集重判档（零调用）")
    ap.add_argument("--in", dest="src", required=True, help="v7 打分文件或原始对话 jsonl")
    ap.add_argument("--signals", required=True, help="<out>.signals.jsonl")
    ap.add_argument("--rules", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=None, help="缓存里有多个模型时指定一个")
    a = ap.parse_args(argv)
    if Path(a.src).resolve() == Path(a.out).resolve():
        ap.error("--out 不能覆盖 --in")
    try:
        print(json.dumps(retier(a.src, a.signals, a.rules, a.out, a.model), ensure_ascii=False))
    except RetierError as e:
        print(f"retier 失败：{e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
