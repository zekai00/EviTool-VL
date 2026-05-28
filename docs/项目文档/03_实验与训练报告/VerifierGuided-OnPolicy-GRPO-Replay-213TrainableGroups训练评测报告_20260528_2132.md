# Verifier-Guided On-Policy GRPO Replay 213 Trainable Groups 训练评测报告

- 生成时间：2026-05-28 21:32 CST
- 主线：继续推进真正 on-policy VLM agentic RL，让本地 Qwen2.5-VL 在 BrowserRL 环境中自己采样动作，并用 verifier reward 更新 LoRA adapter。
- 结论：本轮 replay 稳定版 adapter 在正式 `val200` 上达到 success_rate=0.850，高于 v3 SFT baseline 0.745 和上一轮 100tg RL adapter 0.790；但 table 从 0.88 降到 0.76，advanced 仍为 0.6667，因此本轮不跑 `test200`，继续保留 test set 给更稳定版本做最终验收。

## 术语说明

- `on-policy`：在线策略采样。训练数据来自当前模型自己在环境里跑出来的动作。
- `GRPO`：Group Relative Policy Optimization。对同一状态采多个动作，用组内 reward 相对高低更新模型。
- `replay`：重放旧的高质量数据或旧策略采集数据，防止新训练把已有能力遗忘。
- `SFT replay loss`：把旧 SFT 样本的下一步 action 当作监督目标，训练时加一个小权重的负 logprob loss。
- `template quota`：按任务子类型设置采样配额，例如 `advanced_scroll`、`choice_checkbox`、`table_action`。
- `val gate`：验证集门槛。通过 `val_balanced_70` 和 `val200` 判断 adapter 是否值得进入下一阶段。

## 代码变更

修改脚本：

- `scripts/train_browser_rl_onpolicy_grpo.py`

新增能力：

- `--template-quotas-json`：按 task template 控制可训练组配额。
- `--include-families` / `--include-templates`：采样前按 family/template 过滤任务。
- `--replay-sft-jsonl`、`--replay-loss-weight`、`--replay-ratio`、`--replay-max-rows`：训练时加入 history-aware SFT replay loss。

修复问题：

- 原采集逻辑在 committed action 成功时先 `break`，导致 `target_trainable_groups` / `max_groups` 停止条件没有立刻触发。
- 已调整为先检查停止条件，再根据 task success 退出当前任务。

仍需改进：

- collect-only 当前仍是结束后统一写 `groups.jsonl`，长采集时无法实时看到 trainable group 数量。
- 下一轮应实现流式落盘和周期性 summary，避免盲等和中断后无 partial groups 的问题。

## 当前模型组合

本轮训练起点：

```text
base model:
  /root/models/Qwen2.5-VL-3B-Instruct

adapter:
  outputs/onpolicy_browser_rl_grpo_100tg_safe_20260528_1646/adapter
```

本轮输出 adapter：

```text
outputs/onpolicy_browser_rl_grpo_replay_213tg_safe_20260528_2041/adapter
```

## 采集与合并

本轮先跑弱项 template 分片：

```text
outputs/onpolicy_browser_rl_grpo_replay_250tg_collect_20260528_1756/weak_templates
```

弱项分片结果：

| 指标 | 值 |
| --- | ---: |
| groups | 159 |
| trainable_groups | 87 |
| zero_std_groups | 72 |
| samples | 623 |
| rollouts | 81 |
| rollout_success_rate | 0.6173 |

弱项 template trainable groups：

| template | trainable groups |
| --- | ---: |
| table_action | 31 |
| choice_checkbox | 30 |
| advanced_dialog | 20 |
| advanced_scroll | 6 |

说明：advanced_scroll 仍然难采到有 reward 差异的动作组，是后续重点瓶颈。

general 分片情况：

- 原 `general` 分片因采集脚本停止条件问题被中断，不采用。
- 修复后的 `general_fixed` 分片仍因 collect-only 不流式落盘而耗时过长，被中断，不采用。
- 后续必须先做流式落盘再继续扩大采集规模。

最终训练集合并目录：

```text
outputs/onpolicy_browser_rl_grpo_replay_213tg_merged_20260528_2040
```

合并来源：

| source | trainable groups |
| --- | ---: |
| current_weak | 87 |
| prev_all | 126 |
| total | 213 |

其中：

- `current_weak` 来自当前 100tg RL adapter 的弱项 template on-policy 采样。
- `prev_all` 来自上一轮已验证的 on-policy collected groups，用作 RL replay/stability 数据。

合并后 family 分布：

| family | groups |
| --- | ---: |
| choice | 58 |
| table | 47 |
| form | 30 |
| advanced | 29 |
| search | 28 |
| menu | 16 |
| todo | 5 |

## 训练配置

训练命令要点：

```bash
CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/llama/bin/python3 scripts/train_browser_rl_onpolicy_grpo.py \
  --output-dir outputs/onpolicy_browser_rl_grpo_replay_213tg_safe_20260528_2041 \
  --train-only-from outputs/onpolicy_browser_rl_grpo_replay_213tg_merged_20260528_2040/groups.jsonl \
  --model /root/models/Qwen2.5-VL-3B-Instruct \
  --adapter outputs/onpolicy_browser_rl_grpo_100tg_safe_20260528_1646/adapter \
  --learning-rate 3e-7 \
  --epochs 1 \
  --gradient-accumulation-steps 8 \
  --grad-clip 1.0 \
  --load-in-4bit \
  --prompt-style full \
  --max-history 4 \
  --logprob-reduction mean \
  --replay-sft-jsonl /root/models/datasets/local_qwen_history_aware_sft_v3_1000_aug_20260528_0140/local_qwen_history_aware_sft_train.jsonl \
  --replay-loss-weight 0.05 \
  --replay-ratio 0.25 \
  --replay-max-rows 2048
```

训练结果：

| 指标 | 值 |
| --- | ---: |
| trainable_groups | 213 |
| micro_steps | 213 |
| optimizer_steps | 27 |
| epochs | 1 |
| learning_rate | 3e-7 |
| trainable_parameters | 29,933,568 |
| trainable_ratio | 0.0145 |
| loss_mean | -0.0502 |
| policy_loss_mean | -0.0536 |
| replay_loss_mean | 0.2687 |
| replay_rows | 2048 |
| replay_ratio | 0.25 |
| replay_loss_weight | 0.05 |
| reward_mean | 0.3390 |
| reward_std_mean | 0.2531 |

## 快速 Gate：Val Balanced 70

输出：

```text
outputs/rollouts_local_qwen25vl_onpolicy_grpo_replay_213tg_safe_val_balanced70_20260528_2056
```

对比：

| 指标 | v3 SFT | 100tg RL | 213tg replay RL |
| --- | ---: | ---: | ---: |
| success_rate | 0.8406 | 0.8261 | 0.8406 |
| avg_steps | 3.3768 | 3.2899 | 3.1884 |
| valid_json_rate | 1.0 | 1.0 | 1.0 |
| valid_action_rate | 1.0 | 1.0 | 1.0 |
| policy_error_rate | 0.0 | 0.0 | 0.0 |

按 family：

| family | success_rate |
| --- | ---: |
| advanced | 0.6667 |
| choice | 0.8000 |
| form | 0.6000 |
| menu | 1.0000 |
| search | 1.0000 |
| table | 0.8000 |
| todo | 1.0000 |

结论：快速 gate 通过，继续跑正式 `val200`。

## 正式 Gate：Val200

输出：

```text
outputs/rollouts_local_qwen25vl_onpolicy_grpo_replay_213tg_safe_val200_20260528_2108/merged
```

分片：

| shard | rollouts | success_rate |
| --- | ---: | ---: |
| shard_00 | 99 | 0.8485 |
| shard_01 | 101 | 0.8515 |

总体指标：

| 指标 | v3 SFT | 100tg RL | 213tg replay RL |
| --- | ---: | ---: | ---: |
| success_rate | 0.7450 | 0.7900 | 0.8500 |
| avg_steps | 3.8100 | 3.6800 | 3.5600 |
| valid_json_rate | 1.0 | 1.0 | 1.0 |
| valid_action_rate | 1.0 | 1.0 | 1.0 |
| policy_error_rate | 0.0 | 0.0 | 0.0 |

按 family：

| family | v3 SFT | 100tg RL | 213tg replay RL |
| --- | ---: | ---: | ---: |
| advanced | 0.7778 | 0.6667 | 0.6667 |
| choice | 0.6389 | 0.6944 | 0.8611 |
| form | 0.5333 | 0.6667 | 0.7778 |
| menu | 0.8400 | 0.9200 | 0.8800 |
| search | 0.8000 | 0.8250 | 0.9250 |
| table | 0.8800 | 0.8400 | 0.7600 |
| todo | 1.0000 | 1.0000 | 1.0000 |

按 template：

| template | success_rate |
| --- | ---: |
| advanced_dialog | 0.6667 |
| advanced_scroll | 0.3333 |
| advanced_tab | 1.0000 |
| choice_checkbox | 0.9167 |
| choice_radio | 0.7500 |
| choice_select | 0.9167 |
| form_fill | 0.7778 |
| menu_select | 0.8800 |
| search_select | 0.9250 |
| table_action | 0.7600 |
| todo_add | 1.0000 |

## 数据质量验证

验证目录：

```text
outputs/onpolicy_browser_rl_grpo_replay_213tg_safe_validation_20260528_2132
```

输入：

- weak template collect rollouts：81 条
- merged train provenance rollouts：256 条
- val_balanced_70 rollouts：69 条
- val200 rollouts：200 条

验证结果：

| 指标 | 值 |
| --- | ---: |
| rollouts | 606 |
| valid_rollouts | 606 |
| error_count | 0 |
| exec_ok_step_rate | 1.0 |
| action_parse_step_rate | 1.0 |
| screenshot_field_rate | 1.0 |
| unique_tasks | 444 |

## 是否跑 Test200

本轮不跑 `test200`。

原因：

1. `val200=0.85` 已达到进入下一阶段的总体门槛。
2. 但 table 从 v3 SFT 的 0.88 降到 0.76，退化明显。
3. advanced 仍为 0.6667，低于 v3 SFT 的 0.7778。
4. `test200` 应保留给“总体高且主要 family 不明显退化”的版本。

## 结论

本轮证明 replay 版 GRPO 可以显著提高总体 `val200`：

- v3 SFT：0.745
- 100tg RL：0.790
- 213tg replay RL：0.850

关键收益：

- form：0.5333 -> 0.7778
- choice：0.6389 -> 0.8611
- search：0.8000 -> 0.9250

关键问题：

- table：0.8800 -> 0.7600
- advanced：0.7778 -> 0.6667
- collect-only 需要流式落盘，否则长采集中断会损失 partial groups。

当前建议：

1. `outputs/onpolicy_browser_rl_grpo_replay_213tg_safe_20260528_2041/adapter` 可作为下一轮 RL current model，但不能称为最终模型。
2. 下一轮先修采集器：流式写 `groups.jsonl.tmp`、周期性写 `live_summary.json`、支持安全 resume。
3. 然后专门采 table_action 与 advanced_scroll/dialog，训练时增加 table/advanced replay 或更高权重。
4. 目标是保持 `val200 >= 0.85`，同时 table 回到至少 0.84，advanced 回到至少 0.7778。
5. 达到后再跑 `test200` 做阶段性验收。
