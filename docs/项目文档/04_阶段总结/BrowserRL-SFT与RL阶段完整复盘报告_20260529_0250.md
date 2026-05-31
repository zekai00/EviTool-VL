# BrowserRL SFT 与 RL 阶段完整复盘报告

- 生成时间：2026-05-29 02:50 CST
- 项目目录：`/root/Workspace/VLM/EviTool-VL`
- 当前主线：真正 on-policy VLM agentic RL，即让本地视觉语言模型看浏览器截图、输出动作、由环境执行并用 verifier reward 更新策略。

## 0. 口径说明

本报告只统计当前 BrowserRL 主线。早期 `GUI candidate` 候选框选择实验属于前置探索，目标是“从候选框里选正确框”，不再作为当前 agentic RL 的主线指标。

术语：

- `SFT`：监督微调，用专家轨迹教模型在每一步输出正确动作。
- `RL`：强化学习，用模型自己在环境中试出来的动作和 reward 训练模型。
- `VLM`：视觉语言模型，这里主要是 Qwen2.5-VL-3B-Instruct。
- `LoRA adapter`：低秩增量权重。底模不全量更新，只训练并保存 adapter。
- `on-policy`：在线策略采样，训练数据来自当前模型自己 rollout，而不是离线专家数据。
- `GRPO`：Group Relative Policy Optimization，组内相对策略优化。同一状态采多个动作，用 reward 相对高低更新策略。
- `verifier`：自动判分器，读取浏览器状态判断任务是否成功。
- `success_rate`：可执行任务完成率，是本报告最主要分数。
- `val`：验证集，用于调参和模型选择。
- `test`：测试集，只用于阶段性最终验收，不应反复调参。

当前最终模型组合：

```text
base model:
  /root/models/Qwen2.5-VL-3B-Instruct

current adapter:
  outputs/onpolicy_browser_rl_grpo_table_advanced_repair_289tg_safe_20260529_0101/adapter
```

这个最终 adapter 是完整 LoRA adapter。推理/评测时加载 base model + 这个 adapter 即可，不需要再叠前面的 SFT adapter 或 RL adapter。

## 1. 当前任务与评测集

正式 BrowserRL 任务套件：

```text
/root/datasets/browser_rl/task_suites/browser_rl_task_suite_2000_20260528_1344
```

规模：

| split | 数量 |
|---|---:|
| train | 1600 |
| val | 200 |
| test | 200 |
| total | 2000 |

任务大类：

| family | total | train | val | test |
|---|---:|---:|---:|---:|
| form | 450 | 360 | 45 | 45 |
| search | 400 | 320 | 40 | 40 |
| choice | 350 | 281 | 36 | 33 |
| menu | 250 | 200 | 25 | 25 |
| table | 250 | 200 | 25 | 25 |
| todo | 200 | 160 | 20 | 20 |
| advanced | 100 | 79 | 9 | 12 |

所有 2000 个任务已经用 scripted oracle 跑通，oracle center success_rate=1.0。这说明任务本身、reset/step/verifier 链路是闭环可解的。

## 2. 原始 Qwen2.5-VL 底模分数

原始底模指不加载任何 LoRA adapter 的：

```text
/root/models/Qwen2.5-VL-3B-Instruct
```

本报告补测了 `val_balanced_70`。注意：`val_balanced_70` 实际为 69 条，因为 val split 里 advanced 只有 9 条。这个集合覆盖全部 family，适合快速判断原始底模是否已经会本项目动作协议。

结果：

| 模型 | eval set | rollouts | success_rate | avg_steps | valid_json_rate | valid_action_rate | policy_error_rate |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-VL-3B-Instruct 原始底模 | val_balanced_70 | 69 | 0.0000 | 4.6812 | 0.6749 | 0.5728 | 0.4272 |

结论：原始 instruct 底模不是不会“看图”，而是没有学会本项目要求的可执行 GUI action JSON 协议，也没有学会多步状态闭环。它在同一 prompt 下 success_rate=0，因此必须先做 SFT warmup。

## 3. SFT 阶段

按 BrowserRL 主线计，SFT 一共有 6 个阶段。其中第 0/1 阶段主要是训练链路和协议调试；真正成为 RL 起点的是 v3 1000-task augmented SFT。

| 阶段 | 目的 | 训练集 | 规模 | 输出模型 | 主要分数 | 是否采用 |
|---|---|---|---|---|---|---|
| 0 Action JSON smoke | 验证训练链路 | `/root/datasets/browser_rl/sft/local_qwen_action_sft_warmup_20260527_2224` | 90 rows | `checkpoints/qwen25vl_3b_local_action_sft_warmup_smoke` | 只做 1 step smoke | 否 |
| 1 Action full warmup/clean oracle | 学合法 JSON，排查 teacher 噪声 | warmup 90 rows；oracle 60 rows | 90/60 rows | `qwen25vl_3b_local_action_sft_warmup_lora`；`qwen25vl_3b_local_action_sft_oracle_smoke_lora` | smoke success_rate=0 | 否 |
| 2 History-aware v1 | 加入 step/history/progress | `/root/datasets/browser_rl/sft/local_qwen_history_aware_sft_v1_20260527_2344` | 60 rows | `checkpoints/qwen25vl_3b_history_aware_sft_v1_lora` | eval20=0.3000 | 否 |
| 3 History-aware v2 300 | 扩到 300 tasks，严格 split | `/root/datasets/browser_rl/sft/local_qwen_history_aware_sft_v2_300_20260528_0017` | 900 rows | `checkpoints/qwen25vl_3b_history_aware_sft_v2_300_lora` | test30=0.3333 | 否 |
| 4 History-aware v3 1000 augmented | 1000 tasks + 坐标扰动 + 失败恢复 | `/root/datasets/browser_rl/sft/local_qwen_history_aware_sft_v3_1000_aug_20260528_0140` | 7942 rows | `checkpoints/qwen25vl_3b_history_aware_sft_v3_1000_aug_lora` | test100=0.8200；val200=0.7450 | 是，RL 起点 |
| 5 Form/Choice repair SFT | 进入 RL 前尝试修 form/choice | `/root/datasets/browser_rl/sft/local_qwen_history_aware_sft_v3_1000_aug_form_choice_repair_v1_20260528_1104` | 9786 rows | `checkpoints/qwen25vl_3b_history_aware_sft_v3_form_choice_repair_v1_lora` | test100=0.7600 | 否，整体退化 |

关键判断：

- 原始底模 `val_balanced_70=0.0000`。
- 最终采用的 SFT 是 v3 1000 augmented：`test100=0.8200`，`val200=0.7450`。
- repair SFT 虽然提升 choice，但伤害 search/table，因此没有作为 RL 起点。

## 4. RL 阶段

| RL 阶段 | 起点 adapter | 训练集 | 训练规模 | 输出 adapter | 分数 | 是否进入主线 |
|---|---|---|---:|---|---|---|
| 0 smoke aggressive | SFT v3 | `outputs/onpolicy_browser_rl_grpo_smoke_v1_20260528_1305/groups.jsonl` | 5 trainable groups | `outputs/onpolicy_browser_rl_grpo_smoke_v1_20260528_1305/adapter` | test30=0.5667 | 否 |
| 1 smoke safe | SFT v3 | 同上 | 5 trainable groups | `outputs/onpolicy_browser_rl_grpo_smoke_v1_safe_20260528_1318/adapter` | test30=0.7000 | 否，只证明安全更新 |
| 2 100tg safe | SFT v3 | `/root/datasets/browser_rl/onpolicy/onpolicy_browser_rl_grpo_100tg_merged_20260528_1645/groups.jsonl` | 100 trainable groups | `outputs/onpolicy_browser_rl_grpo_100tg_safe_20260528_1646/adapter` | val200=0.7900 | 是 |
| 3 213tg replay | 100tg adapter | `/root/datasets/browser_rl/onpolicy/onpolicy_browser_rl_grpo_replay_213tg_merged_20260528_2040/groups.jsonl` | 213 trainable groups + SFT replay | `outputs/onpolicy_browser_rl_grpo_replay_213tg_safe_20260528_2041/adapter` | val200=0.8500 | 是 |
| 4 289tg table/advanced repair | 213tg adapter | `/root/datasets/browser_rl/onpolicy/onpolicy_browser_rl_table_advanced_replay_289tg_merged_20260529_0100/groups.jsonl` | 289 trainable groups + SFT replay | `outputs/onpolicy_browser_rl_grpo_table_advanced_repair_289tg_safe_20260529_0101/adapter` | val200=0.9100；test200=0.8900 | 是，当前最强 |

正式 val200 对比：

| 阶段 | success_rate | avg_steps |
|---|---:|---:|
| SFT v3 baseline | 0.7450 | 3.8100 |
| RL 100tg safe | 0.7900 | 3.6800 |
| RL 213tg replay | 0.8500 | 3.5600 |
| RL 289tg table/advanced repair | 0.9100 | 3.4450 |

同一 `val_balanced_70` 对比：

| 阶段 | success_rate |
|---|---:|
| 原始底模，无 adapter | 0.0000 |
| SFT v3 | 0.8406 |
| RL 100tg safe | 0.8261 |
| RL 213tg replay | 0.8406 |
| RL 289tg repair | 0.9420 |

阶段性最终 test200：

| 模型 | success_rate | avg_steps |
|---|---:|---:|
| RL 289tg table/advanced repair | 0.8900 | 3.4650 |

## 5. 当前最强模型能力与短板

当前最强模型：

```text
/root/models/Qwen2.5-VL-3B-Instruct
+ outputs/onpolicy_browser_rl_grpo_table_advanced_repair_289tg_safe_20260529_0101/adapter
```

test200 family：

| family | count | success_rate |
|---|---:|---:|
| advanced | 12 | 0.6667 |
| choice | 33 | 0.8788 |
| form | 45 | 0.9111 |
| menu | 25 | 0.9200 |
| search | 40 | 0.9000 |
| table | 25 | 0.8400 |
| todo | 20 | 1.0000 |

test200 template：

| template | count | success_rate |
|---|---:|---:|
| advanced_dialog | 4 | 1.0000 |
| advanced_scroll | 4 | 0.0000 |
| advanced_tab | 4 | 1.0000 |
| choice_checkbox | 11 | 1.0000 |
| choice_radio | 11 | 0.8182 |
| choice_select | 11 | 0.8182 |
| form_fill | 45 | 0.9111 |
| menu_select | 25 | 0.9200 |
| search_select | 40 | 0.9000 |
| table_action | 25 | 0.8400 |
| todo_add | 20 | 1.0000 |

主要短板：

- `advanced_scroll` 在 test200 为 0。
- 新采集中 `advanced_scroll` 没有采到 trainable group。
- 后续应先改 scroll reward shaping 和显式 scroll candidate，再继续 RL。

