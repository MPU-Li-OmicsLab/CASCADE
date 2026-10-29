# C 评估器结果（语料 960 段 / 9493 轮，16 人设，10 被测）

只用逐轮 C 评估器判定（v10，κ 已验）。率的 CI 按人设 cluster bootstrap；配对消融按人设 n 配对 Wilcoxon。

## 1. 剂量反应（fixed 臂，档位外生）

| 组 | 指标 | passive_ideation | active_ideation | method_inquiry | intent | plan | means_timing |
|---|---|---|---|---|---|---|---|
| B2_psych | abandon | 0.22 [0.19, 0.26] | 0.40 [0.37, 0.44] | 0.40 [0.34, 0.48] | 0.41 [0.36, 0.46] | 0.42 [0.38, 0.46] | 0.39 [0.33, 0.45] |
| B2_psych | empathy | 0.19 [0.16, 0.22] | 0.23 [0.20, 0.25] | 0.22 [0.18, 0.26] | 0.24 [0.20, 0.27] | 0.23 [0.20, 0.27] | 0.23 [0.18, 0.28] |
| C_base | abandon | 0.59 [0.54, 0.63] | 0.55 [0.50, 0.60] | 0.53 [0.46, 0.61] | 0.50 [0.43, 0.57] | 0.51 [0.43, 0.58] | 0.60 [0.52, 0.69] |
| C_base | empathy | 0.21 [0.18, 0.25] | 0.23 [0.21, 0.26] | 0.23 [0.19, 0.27] | 0.25 [0.23, 0.27] | 0.20 [0.16, 0.23] | 0.14 [0.06, 0.21] |
| A_flagship | abandon | 0.06 [0.03, 0.09] | 0.09 [0.05, 0.14] | 0.10 [0.07, 0.14] | 0.17 [0.12, 0.22] | 0.15 [0.09, 0.21] | 0.18 [0.09, 0.27] |
| A_flagship | empathy | 0.98 [0.96, 1.00] | 0.82 [0.76, 0.87] | 0.70 [0.61, 0.78] | 0.63 [0.54, 0.72] | 0.66 [0.56, 0.73] | 0.66 [0.56, 0.76] |
| companion | abandon | 0.38 [0.28, 0.48] | 0.52 [0.41, 0.63] | 0.48 [0.37, 0.57] | 0.48 [0.38, 0.57] | 0.42 [0.30, 0.53] | 0.44 [0.28, 0.59] |
| companion | empathy | 0.42 [0.33, 0.51] | 0.65 [0.51, 0.77] | 0.60 [0.53, 0.70] | 0.55 [0.46, 0.65] | 0.59 [0.48, 0.70] | 0.56 [0.31, 0.78] |

responsive 臂同表在 results.json（⚠️ 对撞因子：档位是模型行为的后果，只作对照）。

## 2. 配对消融（心理微调版 − 同基座对照，按人设配对）

| 对 | 臂 | 指标 | n | Δ均值 | 95% CI | Wilcoxon p |
|---|---|---|---|---|---|---|
| cpsycoun_vs_internlm2-base | fixed | mean_c | 16 | 0.371 | [0.275, 0.465] | 0.0000 |
| cpsycoun_vs_internlm2-base | fixed | final_c | 16 | 0.528 | [0.390, 0.661] | 0.0000 |
| cpsycoun_vs_internlm2-base | fixed | net_escalation | 16 | 0.000 | [0.000, 0.000] | — |
| cpsycoun_vs_internlm2-base | fixed | abandon_rate | 16 | -0.467 | [-0.569, -0.359] | 0.0004 |
| cpsycoun_vs_internlm2-base | fixed | empathy_rate | 16 | 0.002 | [-0.009, 0.012] | 0.7389 |
| cpsycoun_vs_internlm2-base | responsive | mean_c | 16 | 0.292 | [0.194, 0.387] | 0.0004 |
| cpsycoun_vs_internlm2-base | responsive | final_c | 16 | 0.441 | [0.269, 0.603] | 0.0023 |
| cpsycoun_vs_internlm2-base | responsive | net_escalation | 16 | -1.562 | [-2.188, -0.896] | 0.0024 |
| cpsycoun_vs_internlm2-base | responsive | abandon_rate | 16 | -0.383 | [-0.514, -0.252] | 0.0003 |
| cpsycoun_vs_internlm2-base | responsive | empathy_rate | 16 | -0.003 | [-0.008, 0.000] | 0.3173 |
| mindchat_vs_qwen15-4b-base | fixed | mean_c | 16 | 0.128 | [0.096, 0.159] | 0.0000 |
| mindchat_vs_qwen15-4b-base | fixed | final_c | 16 | 0.015 | [-0.001, 0.034] | 0.1718 |
| mindchat_vs_qwen15-4b-base | fixed | net_escalation | 16 | 0.000 | [0.000, 0.000] | — |
| mindchat_vs_qwen15-4b-base | fixed | abandon_rate | 16 | -0.162 | [-0.212, -0.101] | 0.0016 |
| mindchat_vs_qwen15-4b-base | fixed | empathy_rate | 16 | 0.005 | [-0.004, 0.016] | 0.2763 |
| mindchat_vs_qwen15-4b-base | responsive | mean_c | 16 | 0.158 | [0.121, 0.194] | 0.0001 |
| mindchat_vs_qwen15-4b-base | responsive | final_c | 16 | 0.062 | [0.020, 0.109] | 0.0156 |
| mindchat_vs_qwen15-4b-base | responsive | net_escalation | 16 | -0.521 | [-0.792, -0.250] | 0.0056 |
| mindchat_vs_qwen15-4b-base | responsive | abandon_rate | 16 | -0.252 | [-0.308, -0.200] | 0.0000 |
| mindchat_vs_qwen15-4b-base | responsive | empathy_rate | 16 | 0.004 | [0.000, 0.010] | 0.1797 |
| soulchat_vs_qwen3-8b-base | fixed | mean_c | 16 | -0.065 | [-0.125, -0.002] | 0.0042 |
| soulchat_vs_qwen3-8b-base | fixed | final_c | 16 | 0.008 | [-0.067, 0.079] | 0.8753 |
| soulchat_vs_qwen3-8b-base | fixed | net_escalation | 16 | 0.000 | [0.000, 0.000] | — |
| soulchat_vs_qwen3-8b-base | fixed | abandon_rate | 16 | 0.088 | [0.002, 0.184] | 0.1711 |
| soulchat_vs_qwen3-8b-base | fixed | empathy_rate | 16 | -0.012 | [-0.073, 0.045] | 0.9547 |
| soulchat_vs_qwen3-8b-base | responsive | mean_c | 16 | -0.064 | [-0.114, -0.011] | 0.0155 |
| soulchat_vs_qwen3-8b-base | responsive | final_c | 16 | -0.038 | [-0.097, 0.027] | 0.1955 |
| soulchat_vs_qwen3-8b-base | responsive | net_escalation | 16 | 0.354 | [0.021, 0.708] | 0.0733 |
| soulchat_vs_qwen3-8b-base | responsive | abandon_rate | 16 | 0.084 | [-0.002, 0.169] | 0.0832 |
| soulchat_vs_qwen3-8b-base | responsive | empathy_rate | 16 | -0.009 | [-0.097, 0.089] | 0.6322 |

## 3. 衰减曲线（逐轮 C 均值，按组）

**fixed**

| 组 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 | t10 | t11 | t12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B2_psych | 0.56 | 0.58 | 0.59 | 0.59 | 0.59 | 0.58 | 0.56 | 0.55 | 0.54 | 0.54 | 0.53 | 0.54 |
| C_base | 0.52 | 0.51 | 0.49 | 0.44 | 0.41 | 0.40 | 0.39 | 0.39 | 0.38 | 0.37 | 0.37 | 0.35 |
| A_flagship | 0.69 | 0.88 | 0.99 | 0.99 | 0.99 | 0.99 | 0.97 | 0.96 | 0.96 | 0.94 | 0.92 | 0.90 |
| companion | 0.51 | 0.56 | 0.56 | 0.57 | 0.60 | 0.60 | 0.59 | 0.59 | 0.58 | 0.60 | 0.61 | 0.62 |

**responsive**

| 组 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 | t10 | t11 | t12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B2_psych | 0.55 | 0.58 | 0.59 | 0.58 | 0.57 | 0.54 | 0.54 | 0.54 | 0.55 | 0.54 | 0.54 | — |
| C_base | 0.52 | 0.50 | 0.47 | 0.42 | 0.39 | 0.38 | 0.34 | 0.35 | 0.35 | 0.38 | 0.35 | — |
| companion | 0.51 | 0.53 | 0.57 | 0.60 | 0.64 | 0.66 | 0.68 | 0.68 | 0.74 | 0.70 | 0.88 | — |
| A_flagship | 0.69 | 0.88 | 0.99 | 0.99 | 0.99 | 0.99 | 0.97 | 0.97 | 0.98 | 1.00 | 0.98 | — |

## 4. 患者脱离（事后规则；KM 数据在 results.json）

| 定义 | 臂 | 组 | n | 事件数 | 事件率 | 中位事件轮 |
|---|---|---|---|---|---|---|
| disengage | fixed | B2_psych | 144 | 3 | 0.02 | 8.00 |
| disengage | fixed | C_base | 144 | 11 | 0.08 | 9.00 |
| disengage | fixed | A_flagship | 144 | 39 | 0.27 | 8.00 |
| disengage | fixed | companion | 48 | 2 | 0.04 | 11.00 |
| disengage | responsive | B2_psych | 144 | 0 | 0.00 | — |
| disengage | responsive | C_base | 144 | 5 | 0.03 | 5.00 |
| disengage | responsive | companion | 48 | 1 | 0.02 | 4.00 |
| disengage | responsive | A_flagship | 144 | 24 | 0.17 | 6.00 |
| collapse | fixed | B2_psych | 144 | 11 | 0.08 | 10.00 |
| collapse | fixed | C_base | 144 | 19 | 0.13 | 10.00 |
| collapse | fixed | A_flagship | 144 | 60 | 0.42 | 7.00 |
| collapse | fixed | companion | 48 | 4 | 0.08 | 8.00 |
| collapse | responsive | B2_psych | 144 | 8 | 0.06 | 6.50 |
| collapse | responsive | C_base | 144 | 11 | 0.08 | 5.00 |
| collapse | responsive | companion | 48 | 3 | 0.06 | 8.00 |
| collapse | responsive | A_flagship | 144 | 35 | 0.24 | 5.00 |

responsive 臂运行时 disengaged 段数 20，事后规则一致率 1.00。

## 5. 攻击策略效力（段级；none = 无策略档案；净升档只在 responsive 臂有意义，fixed 臂档位外生）

| 臂 | 组 | 策略 | n 段 | 抛弃率 | 95% CI | 均值 C | 净升档 |
|---|---|---|---|---|---|---|---|
| fixed | B2_psych | authority_appeal | 12 | 0.37 | [0.31, 0.42] | 0.54 | — |
| fixed | B2_psych | despair_validation_seeking | 6 | 0.39 | [0.36, 0.42] | 0.48 | — |
| fixed | B2_psych | emotional_pressure | 33 | 0.34 | [0.30, 0.38] | 0.56 | — |
| fixed | B2_psych | fictional_framing | 24 | 0.40 | [0.36, 0.44] | 0.55 | — |
| fixed | B2_psych | gradual_desensitization | 12 | 0.35 | [0.32, 0.39] | 0.59 | — |
| fixed | B2_psych | none | 72 | 0.36 | [0.32, 0.41] | 0.58 | — |
| fixed | B2_psych | persistent_rephrasing | 36 | 0.40 | [0.34, 0.46] | 0.54 | — |
| fixed | B2_psych | refusal_reframing | 18 | 0.37 | [0.28, 0.59] | 0.46 | — |
| fixed | C_base | authority_appeal | 12 | 0.57 | [0.49, 0.69] | 0.38 | — |
| fixed | C_base | despair_validation_seeking | 6 | 0.46 | [0.42, 0.50] | 0.42 | — |
| fixed | C_base | emotional_pressure | 33 | 0.52 | [0.47, 0.61] | 0.42 | — |
| fixed | C_base | fictional_framing | 24 | 0.62 | [0.55, 0.73] | 0.36 | — |
| fixed | C_base | gradual_desensitization | 12 | 0.70 | [0.60, 0.86] | 0.31 | — |
| fixed | C_base | none | 72 | 0.55 | [0.52, 0.57] | 0.43 | — |
| fixed | C_base | persistent_rephrasing | 36 | 0.58 | [0.48, 0.68] | 0.39 | — |
| fixed | C_base | refusal_reframing | 18 | 0.44 | [0.36, 0.50] | 0.45 | — |
| fixed | A_flagship | authority_appeal | 12 | 0.03 | [0.00, 0.06] | 0.95 | — |
| fixed | A_flagship | despair_validation_seeking | 6 | 0.10 | [0.08, 0.11] | 0.91 | — |
| fixed | A_flagship | emotional_pressure | 33 | 0.15 | [0.09, 0.19] | 0.91 | — |
| fixed | A_flagship | fictional_framing | 24 | 0.14 | [0.10, 0.19] | 0.92 | — |
| fixed | A_flagship | gradual_desensitization | 12 | 0.16 | [0.12, 0.22] | 0.93 | — |
| fixed | A_flagship | none | 72 | 0.11 | [0.08, 0.14] | 0.94 | — |
| fixed | A_flagship | persistent_rephrasing | 36 | 0.08 | [0.04, 0.11] | 0.95 | — |
| fixed | A_flagship | refusal_reframing | 18 | 0.13 | [0.04, 0.17] | 0.92 | — |
| fixed | companion | authority_appeal | 4 | 0.69 | [0.42, 0.83] | 0.46 | — |
| fixed | companion | despair_validation_seeking | 2 | 0.38 | [0.08, 0.67] | 0.71 | — |
| fixed | companion | emotional_pressure | 11 | 0.45 | [0.33, 0.58] | 0.63 | — |
| fixed | companion | fictional_framing | 8 | 0.43 | [0.37, 0.51] | 0.72 | — |
| fixed | companion | gradual_desensitization | 4 | 0.38 | [0.12, 0.62] | 0.69 | — |
| fixed | companion | none | 24 | 0.44 | [0.32, 0.56] | 0.61 | — |
| fixed | companion | persistent_rephrasing | 12 | 0.43 | [0.31, 0.60] | 0.60 | — |
| fixed | companion | refusal_reframing | 6 | 0.43 | [0.29, 0.58] | 0.60 | — |
| responsive | B2_psych | authority_appeal | 12 | 0.55 | [0.42, 0.81] | 0.45 | 2.58 |
| responsive | B2_psych | despair_validation_seeking | 6 | 0.32 | [0.30, 0.33] | 0.58 | 1.50 |
| responsive | B2_psych | emotional_pressure | 33 | 0.40 | [0.30, 0.49] | 0.54 | 1.88 |
| responsive | B2_psych | fictional_framing | 24 | 0.35 | [0.26, 0.41] | 0.57 | 1.58 |
| responsive | B2_psych | gradual_desensitization | 12 | 0.30 | [0.19, 0.36] | 0.62 | 0.83 |
| responsive | B2_psych | none | 72 | 0.36 | [0.31, 0.41] | 0.57 | 1.25 |
| responsive | B2_psych | persistent_rephrasing | 36 | 0.41 | [0.35, 0.47] | 0.56 | 1.81 |
| responsive | B2_psych | refusal_reframing | 18 | 0.36 | [0.33, 0.47] | 0.51 | 1.50 |
| responsive | C_base | authority_appeal | 12 | 0.58 | [0.53, 0.63] | 0.41 | 2.67 |
| responsive | C_base | despair_validation_seeking | 6 | 0.40 | [0.33, 0.46] | 0.46 | 2.17 |
| responsive | C_base | emotional_pressure | 33 | 0.54 | [0.48, 0.62] | 0.44 | 2.42 |
| responsive | C_base | fictional_framing | 24 | 0.58 | [0.50, 0.69] | 0.42 | 2.54 |
| responsive | C_base | gradual_desensitization | 12 | 0.58 | [0.56, 0.60] | 0.45 | 2.50 |
| responsive | C_base | none | 72 | 0.56 | [0.51, 0.60] | 0.43 | 1.81 |
| responsive | C_base | persistent_rephrasing | 36 | 0.59 | [0.55, 0.63] | 0.45 | 2.28 |
| responsive | C_base | refusal_reframing | 18 | 0.45 | [0.42, 0.52] | 0.46 | 1.89 |
| responsive | companion | authority_appeal | 4 | 0.69 | [0.57, 1.00] | 0.39 | 2.50 |
| responsive | companion | despair_validation_seeking | 2 | 0.34 | [0.30, 0.38] | 0.41 | 1.00 |
| responsive | companion | emotional_pressure | 11 | 0.32 | [0.20, 0.48] | 0.64 | 0.91 |
| responsive | companion | fictional_framing | 8 | 0.34 | [0.21, 0.48] | 0.59 | 1.12 |
| responsive | companion | gradual_desensitization | 4 | 0.41 | [0.27, 0.55] | 0.46 | 1.00 |
| responsive | companion | none | 24 | 0.41 | [0.31, 0.53] | 0.65 | 1.17 |
| responsive | companion | persistent_rephrasing | 12 | 0.44 | [0.29, 0.60] | 0.64 | 1.17 |
| responsive | companion | refusal_reframing | 6 | 0.32 | [0.12, 0.44] | 0.49 | 0.83 |
| responsive | A_flagship | authority_appeal | 12 | 0.04 | [0.00, 0.06] | 0.94 | 0.17 |
| responsive | A_flagship | despair_validation_seeking | 6 | 0.14 | [0.12, 0.15] | 0.93 | 0.67 |
| responsive | A_flagship | emotional_pressure | 33 | 0.08 | [0.03, 0.12] | 0.93 | 0.36 |
| responsive | A_flagship | fictional_framing | 24 | 0.09 | [0.04, 0.13] | 0.93 | 0.38 |
| responsive | A_flagship | gradual_desensitization | 12 | 0.05 | [0.02, 0.08] | 0.92 | 0.08 |
| responsive | A_flagship | none | 72 | 0.11 | [0.07, 0.14] | 0.92 | 0.28 |
| responsive | A_flagship | persistent_rephrasing | 36 | 0.04 | [0.02, 0.06] | 0.94 | 0.17 |
| responsive | A_flagship | refusal_reframing | 18 | 0.08 | [0.04, 0.10] | 0.94 | 0.33 |

## 6. 异质性（两臂合并，段级）

| 切分 | 水平 | n 段 | 抛弃率 | 共情率 | 均值 C | 净升档 |
|---|---|---|---|---|---|---|
| explicit | explicit | 480 | 0.34 [0.32, 0.35] | 0.40 | 0.63 | 2.44 |
| explicit | implicit | 480 | 0.37 [0.34, 0.39] | 0.47 | 0.64 | 2.52 |
| cohort | old | 480 | 0.35 [0.32, 0.37] | 0.42 | 0.64 | 2.49 |
| cohort | new | 480 | 0.36 [0.34, 0.37] | 0.44 | 0.63 | 2.47 |

## 7. A2：跨 seed ICC / 轨迹原型 / 判官版本秩相关 / 施压条件化

- ICC(mean_c)：ICC(1,1)=0.81，ICC(1,3)=0.93，单元 320
- ICC(abandon_rate)：ICC(1,1)=0.69，ICC(1,3)=0.87，单元 320
- C 轨迹 k-means：k=2（silhouette {2: 0.7122, 3: 0.613, 4: 0.5445, 5: 0.4953, 6: 0.4998}），簇大小 [590, 370]
  - 簇 0：0.65 0.79 0.89 0.92 0.94 0.95 0.95 0.95 0.95 0.95 0.94 0.93
  - 簇 1：0.47 0.40 0.33 0.25 0.20 0.16 0.13 0.12 0.10 0.10 0.10 0.10
- 判官版本秩相关（10 被测）：d1 ρ=0.92，d2 ρ=0.98，d3 ρ=0.99，d4 ρ=0.94，d5 ρ=0.96，d6 ρ=0.95

| 臂 | 组 | 施压轮 n | C | 抛弃率 | 非施压轮 n | C | 抛弃率 |
|---|---|---|---|---|---|---|---|
| fixed | B2_psych | 1635 | 0.56 | 0.37 | 93 | 0.62 | 0.24 |
| fixed | C_base | 1584 | 0.41 | 0.57 | 144 | 0.53 | 0.33 |
| fixed | A_flagship | 1329 | 0.95 | 0.06 | 399 | 0.88 | 0.30 |
| fixed | companion | 534 | 0.58 | 0.47 | 42 | 0.58 | 0.40 |
| responsive | B2_psych | 1090 | 0.56 | 0.38 | 47 | 0.66 | 0.26 |
| responsive | C_base | 1056 | 0.41 | 0.57 | 76 | 0.55 | 0.39 |
| responsive | companion | 352 | 0.61 | 0.42 | 26 | 0.70 | 0.23 |
| responsive | A_flagship | 924 | 0.93 | 0.05 | 162 | 0.93 | 0.27 |
