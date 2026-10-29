"""Stage-2 batch scorer (Plan 4.3 §6): read stage-1 dialogues.jsonl, run the
6-dim RubricJudge over every turn, write an enriched scored.jsonl. Stage-1 input
is left immutable, with per-turn idempotency (already-scored turns are skipped, so
a re-run fills only the gaps). Zero real backend in tests via injected judge.

Judge calls are independent — a turn's history is the prior (patient,target)
utterances, NOT scores — so `concurrency > 1` fans them out over a thread pool and a
local vLLM server batches the concurrent requests (continuous batching), turning a
long serial run into a parallel one. Both paths build the same (dialogue, turn) job
list and reassemble by index, not completion order, so serial and concurrent produce
identical output and isolate failures at the same granularity (one bad turn -> None,
the batch continues).

⚠️ 结果**不是逐位可复现**的：vLLM 连续批处理下同一输入在不同批次组成里可能吐出不同
token（temp=0 也一样）。实测同一文件同一命令两次跑出 4730 vs 4753 条有分。所以榜单、
图和判官验证必须全部引用**同一次**跑出来的 scored.jsonl —— 别把两次的结果混着用。

`checkpoint_every` 每 N 轮把当前结果原子落盘（先写 .tmp 再 os.replace），4-8 小时的
批中断后可以拿检查点当 `--in` 续跑，已打分的轮会被跳过。续跑不会拆散分数与理由：
每一轮的两者始终来自同一次生成。"""
from __future__ import annotations
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from cascade.io import read_jsonl, write_jsonl
from cascade.judge import RubricJudge, score_dialogue      # noqa: F401 (score_dialogue re-exported)


def turn_id(rec, turn) -> str:
    """model|arm|profile#tK —— 与 180 段 key / 金标 / repetition 同一种 id。
    分支打包（tools/branch_score_pack，intervention = "branch:<spec>:k<k>"）的三枝共用同一段对话 id，
    分叉后的回复各不相同 → 末尾加 "@<intervention>" 区分；否则 v7 信号缓存（按 id 存）会把一枝的抽取结果
    套到另一枝上（2026-10-15 实际发生过）。其余记录 id 不变。"""
    arm = getattr(rec.arm, "value", rec.arm)
    base = f"{rec.target_model_id}|{arm}|{rec.case_profile.id}#t{turn.turn}"
    tag = getattr(rec, "intervention", None) or ""
    return f"{base}@{tag}" if tag.startswith("branch:") else base


def duplicate_turn_ids(records) -> list[str]:
    """同一输入里出现不止一次的轮次 id（v7 信号缓存按 id 存，重复 id 会串用抽取结果）。"""
    from collections import Counter
    c = Counter(turn_id(r, t) for r in records for t in r.turns)
    return sorted(k for k, v in c.items() if v > 1)


def load_ids(path) -> set[str]:
    """一行一个 id；或 jsonl（每行取 "id" 字段，如 gold_judge180/gold_dev.jsonl）。"""
    import json
    out = set()
    for line in open(path, encoding="utf-8-sig"):
        line = line.strip()
        if line:
            out.add(json.loads(line)["id"] if line.startswith("{") else line)
    return out


def load_id_splits(path) -> set[str]:
    """--ids 文件里出现过的 split 值（jsonl 行里的 "split" 字段；纯 id 行没有）。用来拦 test 段。"""
    import json
    out = set()
    for line in open(path, encoding="utf-8-sig"):
        line = line.strip()
        if line.startswith("{"):
            sp = json.loads(line).get("split")
            if sp:
                out.add(sp)
    return out


def _jobs(records, force, only=None):
    """One independent job per to-score turn. History is the prior (patient,target)
    pairs (NOT scores), so every turn's judge calls can run in parallel.
    `only`（id 集合）：只打这些轮 —— 在金标 dev 段上反复试判官版本时用，别的轮原样不打。"""
    out = []
    for ri, rec in enumerate(records):
        history: list = []
        lang = rec.case_profile.language
        for ti, turn in enumerate(rec.turns):
            if (turn.score is None or force) and (only is None or turn_id(rec, turn) in only):
                out.append((ri, ti, list(history), turn, lang))
            history = history + [(turn.patient_utterance, turn.target_response)]
    return out


def _assemble(records, results):
    out = []
    for ri, rec in enumerate(records):
        turns = [t.model_copy(update={"score": results[(ri, ti)]})
                 if results.get((ri, ti)) is not None else t
                 for ti, t in enumerate(rec.turns)]
        out.append(rec.model_copy(update={"turns": turns}))
    return out


def _replace(tmp, out_path, tries: int = 8) -> None:
    """os.replace，Windows 上目标被别的进程短暂占着（Resilio 同步正在读）会报 PermissionError
    （2026-10-15 A4 在本机跑到 50/180 时撞上）→ 退避重试；Linux 上第一次就成功。"""
    import time
    for k in range(tries):
        try:
            os.replace(tmp, out_path)
            return
        except PermissionError:
            if k == tries - 1:
                raise
            time.sleep(0.5 * (k + 1))


def _checkpoint(records, results, out_path) -> None:
    """原子写：先落 .tmp 再 replace。直接写 out_path 的话，写到一半崩掉会把
    检查点本身毁掉 —— 那正是它要防的事。检查点替换失败只告警、不中断整批：
    已打的轮在内存里，结束时还会再写；v7 的抽取结果另有信号缓存。"""
    tmp = f"{out_path}.tmp"
    write_jsonl(_assemble(records, results), tmp)
    try:
        _replace(tmp, out_path)
    except PermissionError as e:
        print(f"⚠️ 检查点没写成（{e}），继续跑，结束时再写")


def _run_jobs(judge, records, jobs, out_path, *, concurrency, progress, checkpoint_every):
    results: dict = {}
    total, done = len(jobs), 0

    def _one(job):
        ri, _ti, hist, turn, lang = job
        return judge.score_turn(history=hist, patient_utterance=turn.patient_utterance,
                                target_response=turn.target_response, language=lang,
                                turn_id=turn_id(records[ri], turn))

    def _record(key, value):
        nonlocal done
        results[key] = value
        done += 1
        if progress:
            progress(done, total)
        if checkpoint_every and done % checkpoint_every == 0 and done < total:
            _checkpoint(records, results, out_path)

    def _label(ri, ti):
        return f"{records[ri].case_profile.id}#t{ti + 1}"

    if concurrency <= 1:
        for job in jobs:
            ri, ti = job[0], job[1]
            try:
                value = _one(job)
            except Exception as e:                       # isolate a single bad turn
                print(f"failed to score {_label(ri, ti)}: {e}")
                value = None
            _record((ri, ti), value)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = {ex.submit(_one, job): (job[0], job[1]) for job in jobs}
            for fut in as_completed(futs):
                ri, ti = futs[fut]
                try:
                    value = fut.result()
                except Exception as e:                   # isolate a single bad turn
                    print(f"failed to score {_label(ri, ti)}: {e}")
                    value = None
                _record((ri, ti), value)
    return _assemble(records, results)


def _finish(out, out_path) -> tuple[int, int]:
    tmp = f"{out_path}.tmp"
    write_jsonl(out, tmp)
    _replace(tmp, out_path)
    total = sum(len(d.turns) for d in out)
    scored = sum(1 for d in out for t in d.turns if t.score is not None)
    return scored, total


def score_run(judge: RubricJudge, in_path, out_path, *, force: bool = False,
              concurrency: int = 1, progress=None, checkpoint_every: int = 0,
              only: set[str] | None = None) -> tuple[int, int]:
    records = read_jsonl(in_path)
    out = _run_jobs(judge, records, _jobs(records, force, only), out_path,
                    concurrency=concurrency, progress=progress,
                    checkpoint_every=checkpoint_every)
    return _finish(out, out_path)


def _console_progress(total_hint: int):
    """进度 + ETA 打到 stderr。4-8 小时的批看不见进度就分不清「在跑」和「卡死」。"""
    import sys
    import time
    start = time.time()

    def report(done, total):
        if done % 25 and done != total:
            return
        el = time.time() - start
        rate = done / el if el else 0.0
        eta = (total - done) / rate if rate else 0.0
        print(f"\r  {done}/{total} ({done / total:.1%})  {rate * 60:.0f} 轮/分  "
              f"已用 {el / 60:.0f} 分  预计还需 {eta / 60:.0f} 分   ",
              end="\n" if done == total else "", file=sys.stderr, flush=True)
    return report


def _write_v7_meta(out, judge) -> None:
    """<out>.meta.json：规则集 / 抽取版本 / 模型 —— 打分文件本身只在理由里写 d1·R3，看不出这些（终审 M1）。"""
    import json
    from cascade.v7.extract import EXTRACT_VERSION
    meta = {"judge": "v7", "rules": judge.rules, "extract_version": EXTRACT_VERSION, "model": judge.model.model_id}
    with open(f"{out}.meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def main(argv: list[str] | None = None) -> None:
    import argparse
    from cascade.config import build_model
    p = argparse.ArgumentParser(prog="cascade.score_runs",
                                description="Stage-2 offline 6-dim scoring of stage-1 JSONL.")
    p.add_argument("--in", dest="inp", required=True, help="stage-1 dialogues.jsonl")
    p.add_argument("--out", required=True, help="scored.jsonl (enriched copy)")
    p.add_argument("--model", required=True, help="judge model_id in data/models.yaml")
    p.add_argument("--force", action="store_true", help="re-score turns already scored")
    p.add_argument("--concurrency", type=int, default=1,
                   help="parallel judge requests; a local vLLM server batches them (try 16-32)")
    p.add_argument("--checkpoint-every", type=int, default=200,
                   help="每 N 轮原子落盘一次；中断后拿输出文件当 --in 续跑。0 = 关闭")
    p.add_argument("--quiet", action="store_true", help="不打进度")
    p.add_argument("--no-card", action="store_true",
                   help="不附判例卡 = v5 行为（只用于给旧版补分做对照，正式打分别用）")
    p.add_argument("--ids", default=None,
                   help="只打这些轮：一行一个 model|arm|profile#tK，或 jsonl 取 id 字段（如 gold_judge180/gold_dev.jsonl）")
    p.add_argument("--final", action="store_true",
                   help="⛔ --ids 含 test 段（gold_test.jsonl）时必须显式给；test 只许在冻结之后跑一次")
    p.add_argument("--judge", choices=("v6", "v7"), default="v6",
                   help="v6 = RubricJudge（默认，现有命令不变）；v7 = 抽取 → 代码判档，信号缓存写 <out>.signals.jsonl")
    p.add_argument("--rules", default=None, help="v7 规则集版本（默认 cascade.v7.tiers.LATEST）")
    args = p.parse_args(argv)
    if args.ids and "test" in load_id_splits(args.ids) and not args.final:
        p.error("--ids 含 test 段：test 只许冻结之后跑一次，要跑必须显式 --final")
    if args.judge == "v6" and args.rules is not None:
        p.error("--rules 只对 --judge v7 有意义")
    if args.judge == "v7":
        from cascade.v7.tiers import RULESETS
        if args.rules is not None and args.rules not in RULESETS:
            p.error(f"未知规则集 {args.rules}；可选 {sorted(RULESETS)}")
        if args.no_card:
            p.error("--no-card 只对 v6 有意义")
        dup = duplicate_turn_ids(read_jsonl(args.inp))
        if dup:
            p.error(f"输入里有 {len(dup)} 个重复的轮次 id（如 {dup[:2]}）：v7 信号缓存按 id 存，会串用抽取结果。"
                    "分支数据请用 tools/branch_score_pack.py pack 打包")
        from cascade.v7.judge import ExtractJudge
        from cascade.v7.store import SignalStore
        from cascade.v7.tiers import LATEST
        judge_model = build_model(args.model)
        judge = ExtractJudge(judge_model, rules=args.rules or LATEST,
                             store=SignalStore(f"{args.out}.signals.jsonl"))
    else:
        judge = RubricJudge(build_model(args.model), card=not args.no_card)
    scored, total = score_run(judge, args.inp, args.out, force=args.force,
                              concurrency=args.concurrency,
                              progress=None if args.quiet else _console_progress(0),
                              checkpoint_every=args.checkpoint_every,
                              only=load_ids(args.ids) if args.ids else None)
    if args.judge == "v7":
        _write_v7_meta(args.out, judge)
    print(f"scored {scored}/{total} turns -> {args.out} "
          f"({total - scored} unscored: judge returned no parseable score)")


if __name__ == "__main__":
    main()
