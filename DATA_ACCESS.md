# Access to the conversation dataset

The full texts of the 960 simulated conversations, the judge outputs that quote model replies, the follow-up experiments (system-prompt intervention, counterfactual branches, no-risk controls) and the anonymized annotation database are distributed as a **gated dataset** on Hugging Face:

**https://huggingface.co/datasets/<ORG>/CASCADE-conversations** *(link to be added)*

Access requires agreeing to research-use terms: the data may be used only for research on the safety evaluation of conversational AI, may not be used to train models to produce harmful content, and may not be redistributed outside these terms. The dataset is licensed under CC BY-NC 4.0.

After approval, download it into the root of this repository:

```bash
huggingface-cli download <ORG>/CASCADE-conversations --repo-type dataset --local-dir .
```

This places the files at the paths the analysis scripts expect:

| Path | Contents |
|---|---|
| `runs/v2/dialogues_both_arms.jsonl` | Main corpus: 960 conversations, 9,493 replies (both arms) |
| `runs/v2/dialogues_gold180.jsonl` | The 169 conversations containing the 180 psychiatrist-rated replies |
| `runs/v2/v7_all_qwen38*.jsonl` | Qwen judge on the full corpus (scores, extracted signals) |
| `runs/v2/a4_dsnt_*.jsonl` | DeepSeek judge |
| `runs/v2/v7_dev_*`, `v7_test_*`, `a5_retest_gold*` | Judge on the development / test gold sets and its retest |
| `runs/v2/scored_v5*.jsonl`, `scored_v6*.jsonl` | Earlier judge versions (used for gold-standard sampling strata and version selection) |
| `runs/v2/diag_dev_*.jsonl` | Frontier models as judges on the development set |
| `runs/v2_intv*`, `runs/v2_ctrl_api`, `runs/v2_branch*`, `runs/v2_none*` | System-prompt intervention, counterfactual branches, no-risk controls |
| `runs/valbatch_v2/`, `cmp_eval_v2/` | Data for the help-seeker-perspective rater's misclassification correction |
| `webapp/data/annotations.sqlite3` | Anonymized annotation database (items shown to psychiatrists and their ratings; credentials, logs and response times removed) |
| `docs/collaboration/sim_realism_*.xlsx` | Psychiatrist realism ratings of simulated help-seeker messages |
