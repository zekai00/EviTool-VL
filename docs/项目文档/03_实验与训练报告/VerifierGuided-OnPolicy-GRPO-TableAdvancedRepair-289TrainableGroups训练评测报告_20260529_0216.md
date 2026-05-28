# Verifier-Guided On-Policy GRPO Table/Advanced Repair 289 Trainable Groups 训练评测报告

- 生成时间：2026-05-29 02:16 CST
- 主线：真正的 VLM agentic RL。VLM 是视觉语言模型；agentic RL 是让模型在可执行环境中看截图、输出动作、获得 reward 并更新策略。
- 当前结论：本轮 table/advanced 修复版 adapter 在 `val200` 上达到 `0.9100`，在首次打开的 `test200` 上达到 `0.8900`。整体已经超过 `0.75` 目标，table 在 test 上达到 `0.8400`；但 `advanced_scroll` 在 test 上仍为 `0.0000`，不能说 advanced scroll 已解决。

## 当前模型

```text
base model:
  /root/models/Qwen2.5-VL-3B-Instruct

adapter:
  outputs/onpolicy_browser_rl_grpo_table_advanced_repair_289tg_safe_20260529_0101/adapter
```

这个 adapter 是从上一轮 213tg replay adapter 继续训练得到的完整 LoRA adapter。LoRA adapter 是低秩增量权重；推理时只需要加载 base model 和这个 adapter，不需要叠两层 adapter。

## 本轮为什么做

上一轮 213tg replay adapter 的 `val200=0.8500`，但存在两个明显短板：

- table 从 v3 SFT 的 `0.8800` 降到 `0.7600`。
- advanced 只有 `0.6667`，低于 v3 SFT 的 `0.7778`。

因此本轮只做 table/advanced repair。Repair 是定向修复，目标是在不明显伤害整体能力的前提下，把弱项 family 拉回来。

## 新采集数据

`on-policy`：在线策略采样，指当前模型自己在环境中 rollout 产生训练数据。
`trainable group`：同一状态下采多个候选动作，如果 reward 有差异，就能用于 GRPO 学习。

本轮从当前 213tg adapter 出发，专项采集：

- `table_action`
- `advanced_dialog`
- `advanced_scroll`

采集目录：

```text
/root/datasets/browser_rl/onpolicy/onpolicy_browser_rl_table_advanced_collect_20260529_0040
/root/datasets/browser_rl/onpolicy/onpolicy_browser_rl_advanced_collect_20260529_0050
```

采集统计：

| collect | groups | trainable groups | rollouts | rollout success_rate |
| --- | ---: | ---: | ---: | ---: |
| table_advanced | 42 | 27 | 28 | 0.7143 |
| advanced_only | 22 | 11 | 16 | 0.4375 |
| total | 64 | 38 | 44 | 0.6136 |

新增 trainable template：

| template | trainable groups |
| --- | ---: |
| table_action | 23 |
| advanced_dialog | 15 |
| advanced_scroll | 0 |

结论：table 和 dialog 能采到有 reward 差异的训练信号；advanced_scroll 仍没有采到 trainable group，说明 scroll 不是简单多采样就能解决，后续需要改 action sampling 或加入显式 scroll candidate。

## 合并训练数据

合并目录：

```text
/root/datasets/browser_rl/onpolicy/onpolicy_browser_rl_table_advanced_replay_289tg_merged_20260529_0100
```

合并策略：

- 保留上一轮 213 个 trainable groups，作为 broad replay，防止遗忘。
- 新增 38 个 table/advanced trainable groups 复制一份，相当于轻微加权。
- 最终训练集：`213 + 38 * 2 = 289` trainable groups。

最终 selected template：

| template | groups |
| --- | ---: |
| table_action | 93 |
| advanced_dialog | 50 |
| advanced_scroll | 6 |
| choice_checkbox | 42 |
| form_fill | 30 |
| search_select | 28 |
| menu_select | 16 |
| choice_radio | 10 |
| choice_select | 6 |
| todo_add | 5 |
| advanced_tab | 3 |

## 训练配置

```bash
CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/llama/bin/python3 scripts/train_browser_rl_onpolicy_grpo.py \
  --output-dir outputs/onpolicy_browser_rl_grpo_table_advanced_repair_289tg_safe_20260529_0101 \
  --train-only-from /root/datasets/browser_rl/onpolicy/onpolicy_browser_rl_table_advanced_replay_289tg_merged_20260529_0100/groups.jsonl \
  --model /root/models/Qwen2.5-VL-3B-Instruct \
  --adapter outputs/onpolicy_browser_rl_grpo_replay_213tg_safe_20260528_2041/adapter \
  --learning-rate 2e-7 \
  --epochs 1 \
  --gradient-accumulation-steps 8 \
  --grad-clip 1.0 \
  --load-in-4bit \
  --prompt-style full \
  --max-history 4 \
  --logprob-reduction mean \
  --replay-sft-jsonl /root/datasets/browser_rl/sft/local_qwen_history_aware_sft_v3_1000_aug_20260528_0140/local_qwen_history_aware_sft_train.jsonl \
  --replay-loss-weight 0.06 \
  --replay-ratio 0.30 \
  --replay-max-rows 2048 \
  --seed 62
```

训练结果：

| 指标 | 值 |
| --- | ---: |
| trainable_groups | 289 |
| micro_steps | 289 |
| optimizer_steps | 37 |
| learning_rate | 2e-7 |
| replay_loss_weight | 0.06 |
| replay_ratio | 0.30 |
| trainable_parameters | 29,933,568 |
| loss_mean | -0.0413 |
| policy_loss_mean | -0.0472 |
| replay_loss_mean | 0.2871 |
| reward_mean | 0.3838 |
| reward_std_mean | 0.2921 |

## 快速 Gate：Val Balanced 70

输出：

```text
outputs/rollouts_local_qwen25vl_onpolicy_table_advanced_repair_289tg_val_balanced70_20260529_0119
```

结果：

| 指标 | 213tg replay | table/advanced repair |
| --- | ---: | ---: |
| success_rate | 0.8406 | 0.9420 |
| avg_steps | 3.1884 | 3.0290 |
| valid_json_rate | 1.0 | 1.0 |
| valid_action_rate | 1.0 | 1.0 |
| policy_error_rate | 0.0 | 0.0 |

结论：快速 gate 明显通过，继续跑正式 `val200`。

## 正式 Gate：Val200

输出：

```text
outputs/rollouts_local_qwen25vl_onpolicy_table_advanced_repair_289tg_val200_20260529_0130/merged
```

分片：

| shard | rollouts | success_rate |
| --- | ---: | ---: |
| shard_00 | 99 | 0.9091 |
| shard_01 | 101 | 0.9109 |
| merged | 200 | 0.9100 |

总体：

| 指标 | v3 SFT | 100tg RL | 213tg replay | 289tg repair |
| --- | ---: | ---: | ---: | ---: |
| success_rate | 0.7450 | 0.7900 | 0.8500 | 0.9100 |
| avg_steps | 3.8100 | 3.6800 | 3.5600 | 3.4450 |
| valid_json_rate | 1.0 | 1.0 | 1.0 | 1.0 |
| valid_action_rate | 1.0 | 1.0 | 1.0 | 1.0 |
| policy_error_rate | 0.0 | 0.0 | 0.0 | 0.0 |

Val200 family：

| family | 213tg replay | 289tg repair |
| --- | ---: | ---: |
| advanced | 0.6667 | 0.8889 |
| choice | 0.8611 | 0.8611 |
| form | 0.7778 | 0.8889 |
| menu | 0.8800 | 0.9200 |
| search | 0.9250 | 0.9500 |
| table | 0.7600 | 0.8800 |
| todo | 1.0000 | 1.0000 |

Val200 template：

| template | success_rate |
| --- | ---: |
| advanced_dialog | 1.0000 |
| advanced_scroll | 0.6667 |
| advanced_tab | 1.0000 |
| choice_checkbox | 0.9167 |
| choice_radio | 0.7500 |
| choice_select | 0.9167 |
| form_fill | 0.8889 |
| menu_select | 0.9200 |
| search_select | 0.9500 |
| table_action | 0.8800 |
| todo_add | 1.0000 |

结论：val gate 通过，且 table/advanced 均达到进入 test200 的条件。

## 最终验收：Test200

这是本阶段首次使用 `test200`。Test set 是测试集，只用于阶段性最终验收，不应频繁用于调参。

输出：

```text
outputs/rollouts_local_qwen25vl_onpolicy_table_advanced_repair_289tg_test200_20260529_0154/merged
```

分片：

| shard | rollouts | success_rate |
| --- | ---: | ---: |
| shard_00 | 100 | 0.9000 |
| shard_01 | 100 | 0.8800 |
| merged | 200 | 0.8900 |

总体：

| 指标 | 值 |
| --- | ---: |
| success_rate | 0.8900 |
| avg_steps | 3.4650 |
| valid_json_rate | 1.0 |
| valid_action_rate | 1.0 |
| policy_error_rate | 0.0 |

Test200 family：

| family | count | success_rate |
| --- | ---: | ---: |
| advanced | 12 | 0.6667 |
| choice | 33 | 0.8788 |
| form | 45 | 0.9111 |
| menu | 25 | 0.9200 |
| search | 40 | 0.9000 |
| table | 25 | 0.8400 |
| todo | 20 | 1.0000 |

Test200 template：

| template | count | success_rate |
| --- | ---: | ---: |
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

## 质量校验

新采集 rollout 校验：

```text
outputs/onpolicy_table_advanced_collect_validation_20260529_0100
```

- rollouts：44
- valid_rollouts：44
- error_count：0
- exec_ok_step_rate：1.0
- action_parse_step_rate：1.0

Val200 校验：

```text
outputs/onpolicy_table_advanced_repair_289tg_val200_validation_20260529_0153
```

- rollouts：200
- valid_rollouts：200
- error_count：0
- exec_ok_step_rate：1.0
- action_parse_step_rate：1.0

Test200 校验：

```text
outputs/onpolicy_table_advanced_repair_289tg_test200_validation_20260529_0216
```

- rollouts：200
- valid_rollouts：200
- error_count：0
- exec_ok_step_rate：1.0
- action_parse_step_rate：1.0

## 结论

1. 本轮 adapter 可以作为当前阶段最强模型：`val200=0.9100`，`test200=0.8900`。
2. table 回修成功：`val table=0.8800`，`test table=0.8400`。
3. advanced 在 val 上明显恢复，但 test advanced 仍只有 `0.6667`，主要因为 `advanced_scroll=0.0000`。
4. 后续不要继续盲采 scroll；应改 `advanced_scroll` 的 action sampling，例如显式加入 `scroll` 候选、滚动幅度扰动、scroll 后状态分支比较，或者构建少量 scroll recovery SFT/replay 数据。
5. `test200` 已经用于本阶段最终验收，下一轮调参仍应回到 train/val，不应继续用 test200 做选择。
