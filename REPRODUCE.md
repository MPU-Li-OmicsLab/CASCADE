# Reproduction commands

Run everything from the repository root with the gated dataset downloaded (see `DATA_ACCESS.md`). Every tool's docstring repeats its own usage line; the commands below are the ones used for the paper.

## 1. Psychiatrist estimates and primary tests

```bash
python tools/gold_prevalence.py --gold gold_judge180/gold_dev.jsonl --gold gold_judge180/gold_test.jsonl --corpus runs/v2/v7_all_qwen38_fill.jsonl --out reports/gold_prevalence
python tools/gold_prevalence_raked.py --gold gold_judge180/gold_dev.jsonl --gold gold_judge180/gold_test.jsonl --out reports/gold_prevalence_raked
python tools/primary_tests.py --gold gold_judge180/gold_dev.jsonl --gold gold_judge180/gold_test.jsonl --out reports/primary_tests
python tools/multiplicity.py --out reports/multiplicity
```

## 2. Judge conclusions against psychiatrists (Fig. 3, Extended Data Fig. 2, Supplementary Table 1)

```bash
python tools/judge_vs_gold_joint.py --out reports/judge_vs_gold_joint
python tools/judge_vs_gold.py --out reports/judge_vs_gold
```

## 3. Judge errors by model, length and retest (Fig. 4, Supplementary Table 4)

```bash
python tools/judge_error_by_model.py --out reports/judge_error_by_model
python tools/judge_sensitivity_groups.py --out reports/judge_sensitivity_groups
python tools/correction_sensitivity.py --out reports/correction_sensitivity
python tools/length_bias_v7.py --gold gold_judge180/gold_dev.jsonl --gold gold_judge180/gold_test.jsonl --signals runs/v2/v7_dev_qwen38_q4.jsonl.signals.jsonl --signals runs/v2/v7_test_qwen38.jsonl.signals.jsonl --out reports/length_bias_v7
python tools/judge_retest_v7.py --first runs/v2/v7_dev_qwen38_q4.jsonl.signals.jsonl --first runs/v2/v7_test_qwen38.jsonl.signals.jsonl --second runs/v2/a5_retest_gold.jsonl.signals.jsonl --out reports/judge_retest_v7
python tools/judge_spearman.py --out reports/judge_spearman
python tools/judge_spearman.py --gold gold_judge180/gold_test.jsonl --qwen-signals runs/v2/v7_test_qwen38.jsonl.signals.jsonl --out reports/judge_spearman_test
```

Judge version selection (`reports/judge_select_*`) uses `tools/judge_select.py`; see its docstring for the development-set and locked test-set invocations.

## 4. Fine-tuning, help-seeker-perspective rater and repetition (Fig. 5)

```bash
python tools/repetition_contrast.py --out reports/repetition_contrast
python tools/misclass_correction.py --n-boot 1000 --out reports/misclass_correction
python tools/misclass_correction_clustered.py --n-boot 1000 --out reports/misclass_correction_clustered
```

## 5. Evaluation design (Fig. 6, Extended Data Fig. 3)

```bash
python tools/judge_dose_by_arm.py --out reports/judge_dose_by_arm
python tools/judge_dose_turn.py --out reports/judge_dose_turn
python -m cascade.analysis.intervention --control runs/v2/dialogues_both_arms.jsonl --intv runs/v2_intv/dialogues.jsonl runs/v2_intv_api/dialogues.jsonl --none runs/v2_none/dialogues.jsonl runs/v2_none_api/dialogues.jsonl --out reports/intervention_v2
python tools/branch_intervention.py report --orig runs/v2_branch/orig.jsonl --regen runs/v2_branch/regen_all.jsonl --intv runs/v2_branch/intv_all.jsonl --out reports/branch_v2/report.md
```

## 6. Other robustness reports

```bash
python tools/d5_uncapped.py --out reports/d5_uncapped
python tools/six_dim_questions.py --out reports/six_dim_questions
python tools/sim_vs_real_text.py --out reports/sim_vs_real          # needs PsyQA (not redistributed)
```

## 7. Figures and tables

```bash
python tools/final_figs_a.py --out paper_final
python tools/final_figs_b.py --out paper_final
python tools/final_tables.py
python tools/export_submission_tables.py
```

## 8. Generating a new corpus

Serve the simulator and judge (Qwen3.6-27B and Qwen3.8-27B in the paper) and local target models with vLLM, then edit `data/models.yaml`. Fixed arm requires the responsive arm's escalation calibration first.

```bash
python -m cascade.runner --arm responsive --models "<model-id>" --profiles data/profiles_v2_zh.jsonl --max-turns 12 --evaluator-model qwen-eval-38 --concurrency 6 --out runs/<exp>
python -m cascade.escalation --in runs/<exp>/dialogues.jsonl --out runs/<exp>/escalation.json
python -m cascade.runner --arm fixed --escalation runs/<exp>/escalation.json --models "<model-id>" --profiles data/profiles_v2_zh.jsonl --max-turns 12 --evaluator-model qwen-eval-38 --concurrency 6 --out runs/<exp>
python -m cascade.score_runs --judge v7 --rules r6 --in runs/<exp>/dialogues.jsonl --model qwen-eval-38 --concurrency 64 --checkpoint-every 400 --out runs/<exp>/v7_scored.jsonl
```

Add `--target-system-prompt-file data/interventions/present_v1.zh.txt` to the runner for the "stay present" system-prompt condition, and `--risk-none --profiles data/profiles_none_zh.jsonl` for the no-risk control. Counterfactual branches: `tools/branch_intervention.py select | run | orig | report` (see its docstring).
