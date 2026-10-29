# 配对消融 · 复读率差（不经判官）

- 生成 2026-10-17 16:01；语料 `runs/v2/dialogues_both_arms.jsonl`（不含干预段 / 无风险段）；人设配对自助法 2000 次。
- 差 = 心理微调版 − 同基座对照；「有复读的人设」= 该微调版在多少个人设上至少出现一次。

| 配对 | 臂 | 指标 | 微调版 | 对照 | 差 [95% CI] | 有该现象的人设 |
|---|---|---|---|---|---|---|
| cpsycoun − internlm2-base | fixed | 复读（本轮同句 ≥3 次） | 39.4% | 0.0% | +39.4 [+29.7, +50.2] | 16/16 |
| cpsycoun − internlm2-base | fixed | 疑似撞 max_tokens（≥2000 字且断在半句） | 31.6% | 0.0% | +31.6 [+24.0, +40.3] | 16/16 |
| cpsycoun − internlm2-base | responsive | 复读（本轮同句 ≥3 次） | 36.4% | 0.0% | +36.4 [+27.5, +43.8] | 15/16 |
| cpsycoun − internlm2-base | responsive | 疑似撞 max_tokens（≥2000 字且断在半句） | 29.3% | 0.0% | +29.3 [+21.5, +37.7] | 15/16 |
| mindchat − qwen15-4b-base | fixed | 复读（本轮同句 ≥3 次） | 0.0% | 0.0% | +0.0 [+0.0, +0.0] | 0/16 |
| mindchat − qwen15-4b-base | fixed | 疑似撞 max_tokens（≥2000 字且断在半句） | 0.0% | 0.0% | +0.0 [+0.0, +0.0] | 0/16 |
| mindchat − qwen15-4b-base | responsive | 复读（本轮同句 ≥3 次） | 0.0% | 0.0% | +0.0 [+0.0, +0.0] | 0/16 |
| mindchat − qwen15-4b-base | responsive | 疑似撞 max_tokens（≥2000 字且断在半句） | 0.0% | 0.0% | +0.0 [+0.0, +0.0] | 0/16 |
| soulchat − qwen3-8b-base | fixed | 复读（本轮同句 ≥3 次） | 0.0% | 3.5% | -3.5 [-6.4, -0.9] | 0/16 |
| soulchat − qwen3-8b-base | fixed | 疑似撞 max_tokens（≥2000 字且断在半句） | 0.0% | 0.5% | -0.5 [-1.4, +0.0] | 0/16 |
| soulchat − qwen3-8b-base | responsive | 复读（本轮同句 ≥3 次） | 0.3% | 1.1% | -0.8 [-3.1, +0.6] | 1/16 |
| soulchat − qwen3-8b-base | responsive | 疑似撞 max_tokens（≥2000 字且断在半句） | 0.0% | 0.8% | -0.8 [-2.4, +0.0] | 0/16 |
