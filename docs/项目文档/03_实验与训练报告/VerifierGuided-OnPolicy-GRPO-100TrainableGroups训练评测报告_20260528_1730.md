# Verifier-Guided On-Policy GRPO 100 Trainable Groups 训练评测报告

- 生成时间：2026-05-28 17:30 CST
- 主线：真正 on-policy VLM agentic RL，即让本地 Qwen2.5-VL 在可 reset/step/verifier 的浏览器环境中自己试动作，再用 verifier reward 更新策略。
- 结论：本轮 100 个 trainable groups 的安全版 GRPO adapter 在正式 `val200` 上达到 success_rate=0.790，高于 v3 SFT baseline 的 0.745。该 adapter 可作为下一轮 RL current model，但 `test200` 仍保留给阶段性最终验收，不用于本轮调参。

## 术语说明

- `on-policy`：在线策略采样。训练数据来自当前模型自己在环境里跑出来的动作，而不是只用离线专家数据。
- `GRPO`：Group Relative Policy Optimization。对同一个状态采多个候选动作，用组内 reward 均值/方差归一化后更新模型。
- `verifier`：自动判题器。它读取浏览器页面状态，返回任务是否成功和阶段性 reward，不把答案暴露给模型。
- `trainable group`：同一个状态下多次采样得到的动作组；如果这些动作 reward 有差异，就能提供 RL 学习信号。
- `val gate`：验证集门槛。先用固定验证集判断 adapter 是否超过基线，没过就不切换默认模型。
- `adapter`：LoRA 增量权重。base model 不变，只保存少量可训练参数。

## 输入基线

当前 base model：

- `/root/models/Qwen2.5-VL-3B-Instruct`

本轮 RL 起点 adapter：

- `checkpoints/qwen25vl_3b_history_aware_sft_v3_1000_aug_lora`

正式任务套件：

- `outputs/browser_rl_task_suite_2000_20260528_1344`
- split：train=1600，val=200，test=200

正式 val200 基线：

- 输出：`outputs/rollouts_local_qwen25vl_history_aware_v3_2000_val200_20260528_1412/merged`
- success_rate=0.745
- family：advanced=0.7778，choice=0.6389，form=0.5333，menu=0.8400，search=0.8000，table=0.8800，todo=1.0000

## On-Policy 数据采集

采集目录：

- `outputs/onpolicy_browser_rl_grpo_100tg_collect_20260528_1445`

合并选择目录：

- `outputs/onpolicy_browser_rl_grpo_100tg_merged_20260528_1645`

采集方式：

- 从 train split 中采样任务。
- 当前模型对同一个状态采样多个动作。
- Playwright 环境执行动作。
- verifier 返回 reward。
- 只选择 reward 有方差的动作组作为 trainable group。

并行分片：

| shard | 侧重 family | 目标 trainable groups | 实际 trainable groups |
| --- | --- | ---: | ---: |
| shard_a | form/search/choice | 50 | 50 |
| shard_b | choice/menu/table/todo/advanced | 50 | 51 |
| shard_c | form/search/choice 补采样 | 25 | 25 |

合并后统计：

| 指标 | 值 |
| --- | ---: |
| collected_groups | 445 |
| collected_trainable_groups | 126 |
| selected_groups | 100 |
| rollouts | 175 |
| rollout_success_rate | 0.2514 |
| selected_avg_reward_std | 0.2467 |

选入训练的 family 分布：

| family | groups |
| --- | ---: |
| form | 21 |
| search | 21 |
| choice | 20 |
| menu | 15 |
| table | 15 |
| todo | 5 |
| advanced | 3 |

说明：advanced 的可训练组较少，本轮只采到 3 个。下一轮需要专项增加 advanced_scroll/dialog 的探索样本。

## 训练配置

输出目录：

- `outputs/onpolicy_browser_rl_grpo_100tg_safe_20260528_1646`
- adapter：`outputs/onpolicy_browser_rl_grpo_100tg_safe_20260528_1646/adapter`

训练命令要点：

```bash
CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/llama/bin/python3 scripts/train_browser_rl_onpolicy_grpo.py \
  --output-dir outputs/onpolicy_browser_rl_grpo_100tg_safe_20260528_1646 \
  --train-only-from outputs/onpolicy_browser_rl_grpo_100tg_merged_20260528_1645/groups.jsonl \
  --model /root/models/Qwen2.5-VL-3B-Instruct \
  --adapter checkpoints/qwen25vl_3b_history_aware_sft_v3_1000_aug_lora \
  --learning-rate 5e-7 \
  --epochs 1 \
  --gradient-accumulation-steps 8 \
  --grad-clip 1.0 \
  --load-in-4bit \
  --prompt-style full \
  --max-history 4 \
  --logprob-reduction mean
```

训练指标：

| 指标 | 值 |
| --- | ---: |
| trainable_groups | 100 |
| micro_steps | 100 |
| optimizer_steps | 13 |
| epochs | 1 |
| learning_rate | 5e-7 |
| trainable_parameters | 29,933,568 |
| total_parameters | 2,063,958,016 |
| trainable_ratio | 0.0145 |
| loss_mean | -0.0672 |
| loss_last | 0.0507 |
| reward_mean | 0.3407 |
| reward_std_mean | 0.2467 |

## 快速 Gate：Val Balanced 70

输出：

- `outputs/rollouts_local_qwen25vl_onpolicy_grpo_100tg_safe_val_balanced70_20260528_1652`

结果对比：

| 指标 | v3 baseline | RL adapter |
| --- | ---: | ---: |
| rollouts | 69 | 69 |
| success_rate | 0.8406 | 0.8261 |
| avg_steps | 3.3768 | 3.2899 |
| valid_json_rate | 1.0 | 1.0 |
| valid_action_rate | 1.0 | 1.0 |
| policy_error_rate | 0.0 | 0.0 |

按 family：

| family | baseline | RL adapter | 变化 |
| --- | ---: | ---: | ---: |
| advanced | 0.7778 | 0.6667 | -0.1111 |
| choice | 0.8000 | 0.8000 | 0.0000 |
| form | 0.5000 | 0.6000 | +0.1000 |
| menu | 1.0000 | 1.0000 | 0.0000 |
| search | 1.0000 | 0.8000 | -0.2000 |
| table | 0.8000 | 0.9000 | +0.1000 |
| todo | 1.0000 | 1.0000 | 0.0000 |

判断：快速 gate 略低于 baseline，但只差 1 个任务，且目标弱项 form 有提升，因此继续跑完整 val200。

## 正式 Gate：Val200

输出：

- `outputs/rollouts_local_qwen25vl_onpolicy_grpo_100tg_safe_val200_20260528_1705/merged`

分片：

- `shard_00`：99 条，success_rate=0.7576
- `shard_01`：101 条，success_rate=0.8218

总体结果：

| 指标 | v3 baseline | RL adapter | 变化 |
| --- | ---: | ---: | ---: |
| rollouts | 200 | 200 | 0 |
| success_rate | 0.7450 | 0.7900 | +0.0450 |
| avg_steps | 3.8100 | 3.6800 | -0.1300 |
| valid_json_rate | 1.0 | 1.0 | 0.0 |
| valid_action_rate | 1.0 | 1.0 | 0.0 |
| policy_error_rate | 0.0 | 0.0 | 0.0 |

按 family：

| family | n | baseline | RL adapter | 变化 |
| --- | ---: | ---: | ---: | ---: |
| advanced | 9 | 0.7778 | 0.6667 | -0.1111 |
| choice | 36 | 0.6389 | 0.6944 | +0.0556 |
| form | 45 | 0.5333 | 0.6667 | +0.1333 |
| menu | 25 | 0.8400 | 0.9200 | +0.0800 |
| search | 40 | 0.8000 | 0.8250 | +0.0250 |
| table | 25 | 0.8800 | 0.8400 | -0.0400 |
| todo | 20 | 1.0000 | 1.0000 | 0.0000 |

按 template：

| template | n | success_rate |
| --- | ---: | ---: |
| advanced_dialog | 3 | 0.6667 |
| advanced_scroll | 3 | 0.3333 |
| advanced_tab | 3 | 1.0000 |
| choice_checkbox | 12 | 0.5833 |
| choice_radio | 12 | 0.6667 |
| choice_select | 12 | 0.8333 |
| form_fill | 45 | 0.6667 |
| menu_select | 25 | 0.9200 |
| search_select | 40 | 0.8250 |
| table_action | 25 | 0.8400 |
| todo_add | 20 | 1.0000 |

## 数据质量验证

验证目录：

- `outputs/onpolicy_browser_rl_grpo_100tg_safe_validation_20260528_1729`

输入：

- on-policy collect rollouts：175 条
- val_balanced_70 rollouts：69 条
- val200 rollouts：200 条

验证结果：

| 指标 | 值 |
| --- | ---: |
| rollouts | 444 |
| valid_rollouts | 444 |
| error_count | 0 |
| exec_ok_step_rate | 1.0 |
| action_parse_step_rate | 1.0 |
| screenshot_field_rate | 1.0 |
| unique_tasks | 369 |

## 结论

本轮 adapter 通过 `val200` gate：

- `0.790 > 0.745`
- JSON/action 合法性没有退化。
- form 从 0.5333 提到 0.6667，是本轮最关键收益。
- choice、menu、search 也有提升。

但还不能称为最终模型：

- `test200` 没有跑，仍保留给阶段性最终验收。
- advanced 从 0.7778 降到 0.6667，主要仍受 advanced_scroll/dialog 小样本影响。
- table 从 0.8800 降到 0.8400，需要下一轮加 replay 或保守 KL。

当前建议：

1. 后续 on-policy RL 的 current model 可以切到 `outputs/onpolicy_browser_rl_grpo_100tg_safe_20260528_1646/adapter`。
2. 下一轮扩大到 200-300 trainable groups，但必须加入 v3 replay 或 KL 稳定项，避免 table/advanced 继续掉。
3. 对 advanced_scroll、advanced_dialog、choice_checkbox 和 table_action 做定向采样，不建议再做只偏 form/choice 的纯 SFT 修复。
4. 如果下一轮 `val200 >= 0.82` 且主要 family 没有明显退化，再跑一次 `test200` 作为阶段性验收。
