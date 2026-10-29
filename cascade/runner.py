"""Batch stage-1 runner: profile × model × arm -> stage-1 JSONL.

Manifest-idempotent (resumable), streaming-append, single-dialogue failure
isolation (spec 3B §6). `concurrency>1` fans independent dialogues (each internally
serial) over a thread pool with locked append/manifest writes; a local vLLM server
batches the concurrent simulator/evaluator requests. Tests inject a fake
`model_factory` so the suite touches no real backend; production uses
cascade.config.build_model.
"""
from __future__ import annotations
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from pydantic import BaseModel, Field
from cascade.actor import PatientSimulator
from cascade.config import build_model
from cascade.content import load_taxonomies
from cascade.escalation import EscalationRates
from cascade.evaluator import CEvaluator
from cascade.io import append_jsonl
from cascade.orchestrator import run_dialogue
from cascade.session import turn_budget
from cascade.schema import Arm, CaseProfile


class BatchManifest(BaseModel):
    entries: dict[str, str] = Field(default_factory=dict)   # key -> "done" | "failed: <msg>"


def _key(profile_id: str, model_id: str, arm: Arm) -> str:
    return f"{profile_id}|{model_id}|{arm.value}"


def _seed(key: str) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)   # stable across processes


def _load_manifest(path: Path) -> BatchManifest:
    if path.exists():
        return BatchManifest.model_validate_json(path.read_text(encoding="utf-8"))
    return BatchManifest()


def run_batch(profiles: list[CaseProfile], model_ids: list[str], arm: Arm,
              max_turns: int, out_dir: str | Path, *,
              escalation: EscalationRates | None = None,
              simulator_model_id: str = "qwen-sim",
              evaluator_model_id: str = "qwen-eval",
              concurrency: int = 1, progress: bool = False,
              model_factory=None, taxonomies=None,
              rewrite_opening: bool = False,
              vary_length: bool = True,
              target_system_prompt: str | None = None,
              intervention_name: str | None = None,
              risk_none: bool = False) -> BatchManifest:
    if arm is Arm.FIXED and escalation is None and not risk_none:
        raise ValueError("fixed arm requires escalation rates")
    factory = model_factory or build_model
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dialogues_path = out_dir / "dialogues.jsonl"
    manifest_path = out_dir / "manifest.json"
    manifest = _load_manifest(manifest_path)
    if taxonomies is None:
        taxonomies = load_taxonomies()

    sim_model = factory(simulator_model_id)
    evaluator = CEvaluator(model=factory(evaluator_model_id))

    def _run_one(target, profile, key):     # one dialogue; run_dialogue is serial internally
        fixed_rate = (escalation.per_language[profile.language.value].rate
                      if (arm is Arm.FIXED and escalation is not None) else None)
        actor = PatientSimulator(model=sim_model, profile=profile, taxonomies=taxonomies)
        # 会话成形只给 responsive（生态效度）臂：fixed 臂是「无视 C 强制升到目标档」的
        # 人工臂，专供判官验证抽高危样本，缩短会让高危档覆盖崩掉（实测 37.5% → 0）。
        shape = vary_length and arm is Arm.RESPONSIVE
        rec = run_dialogue(profile=profile, target=target, actor=actor,
                           evaluator=evaluator, max_turns=max_turns, arm=arm,
                           seed=_seed(key), fixed_rate=fixed_rate,
                           rewrite_opening=rewrite_opening,
                           turn_budget=turn_budget(profile.id, max_turns) if shape else None,
                           allow_disengage=shape,
                           target_system_prompt=target_system_prompt,
                           risk_none=risk_none)
        if intervention_name:
            rec = rec.model_copy(update={"intervention": intervention_name})
        return rec

    jobs = []                               # (target, profile, key), skipping already-done
    for model_id in model_ids:
        target = factory(model_id)
        for profile in profiles:
            key = _key(profile.id, model_id, arm)
            if manifest.entries.get(key) != "done":
                jobs.append((target, profile, key))

    lock = threading.Lock()
    total = len(jobs)
    done_n = [0]

    def _record(key, rec, err):             # commit one result under a lock (thread-safe)
        with lock:
            if err is None:
                append_jsonl(rec, dialogues_path)
                manifest.entries[key] = "done"
            else:                           # isolate single-dialogue failure
                manifest.entries[key] = f"failed: {err}"
            manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
            if progress:
                done_n[0] += 1
                mark = "OK  " if err is None else "FAIL"
                line = f"[{done_n[0]:>4}/{total}] {mark} {key}"
                print(line if err is None else f"{line} :: {err}", flush=True)

    if concurrency <= 1:                    # serial path (behavior unchanged)
        for target, profile, key in jobs:
            try:
                _record(key, _run_one(target, profile, key), None)
            except Exception as e:
                _record(key, None, str(e))
    else:                                   # fan independent dialogues over a thread pool
        def _job(target, profile, key):
            try:
                return key, _run_one(target, profile, key), None
            except Exception as e:
                return key, None, str(e)
        ex = ThreadPoolExecutor(max_workers=concurrency)
        futs = [ex.submit(_job, t, p, k) for (t, p, k) in jobs]
        try:
            for fut in as_completed(futs):
                _record(*fut.result())
        except KeyboardInterrupt:           # Ctrl-C: drop queued work instead of waiting for all
            ex.shutdown(wait=False, cancel_futures=True)
            raise
        ex.shutdown(wait=True)
    return manifest


def main(argv: list[str] | None = None) -> None:
    import argparse
    from cascade.assemble import read_profiles_jsonl
    from cascade.content import DATA_DIR
    p = argparse.ArgumentParser(prog="cascade.runner",
                                description="Batch stage-1 runner (1.A responsive / 1.B fixed).")
    p.add_argument("--arm", required=True, choices=["responsive", "fixed"])
    p.add_argument("--models", required=True, help="comma-separated target model_ids")
    p.add_argument("--profiles", default=str(DATA_DIR / "profiles_pilot.jsonl"))
    p.add_argument("--max-turns", type=int, default=12)
    p.add_argument("--out", required=True, help="run dir (dialogues.jsonl + manifest.json)")
    p.add_argument("--escalation", default=None, help="escalation.json (required for fixed arm)")
    p.add_argument("--simulator-model", default="qwen-sim")
    p.add_argument("--evaluator-model", default="qwen-eval")
    p.add_argument("--concurrency", type=int, default=1,
                   help="parallel dialogues (independent per profile×model; local vLLM batches them; try 8-16)")
    p.add_argument("--no-vary-length", dest="vary_length", action="store_false",
                   help="关闭会话成形（responsive 臂改回全批定长 max_turns、不提前终止）。"
                        "默认开启：按档案抽 4..max_turns 的轮数预算 + 连续告辞即终止，"
                        "并记录 end_reason/turn_budget（disengaged = 右删失，可做 TTD 生存分析）")
    p.add_argument("--opening", choices=["seed", "chat"], default="seed",
                   help="第1条患者消息：seed=语料原文照搬（默认）；chat=改写成聊天体。"
                        "⚠️ chat 已被 simdata5 实测证伪：意念率 29%%→58%%（凭空加重 16/48）、"
                        "人设 few-shot 逐字复现 5/48、收信人错位 2/48。仅作消融分支保留，勿用于生产批")
    p.add_argument("--target-system-prompt-file", default=None,
                   help="A6 干预臂：把该 UTF-8 文件内容作为 system 消息喂被测（只喂被测；"
                        "DialogueRecord.intervention 记文件名去后缀）。对照臂不传")
    p.add_argument("--risk-none", action="store_true",
                   help="A7 无风险对照：模拟器全程无自杀意念（配 data/profiles_none_zh.jsonl，--arm fixed，无需 --escalation）")
    args = p.parse_args(argv)

    profiles = read_profiles_jsonl(args.profiles)
    escalation = None
    if args.escalation:
        escalation = EscalationRates.model_validate_json(
            Path(args.escalation).read_text(encoding="utf-8"))
    prompt_text, intervention_name = None, None
    if args.target_system_prompt_file:
        pf = Path(args.target_system_prompt_file)
        prompt_text = pf.read_text(encoding="utf-8").strip()
        intervention_name = pf.name.rsplit(".", 1)[0]     # present_v1.zh.txt -> present_v1.zh
    manifest = run_batch(profiles, args.models.split(","), Arm(args.arm), args.max_turns,
                         args.out, escalation=escalation,
                         simulator_model_id=args.simulator_model,
                         evaluator_model_id=args.evaluator_model,
                         concurrency=args.concurrency, progress=True,
                         rewrite_opening=(args.opening == "chat"),
                         vary_length=args.vary_length,
                         target_system_prompt=prompt_text, intervention_name=intervention_name,
                         risk_none=args.risk_none)
    done = sum(1 for v in manifest.entries.values() if v == "done")
    print(f"{done}/{len(manifest.entries)} dialogues done -> {args.out}")


if __name__ == "__main__":
    main()
