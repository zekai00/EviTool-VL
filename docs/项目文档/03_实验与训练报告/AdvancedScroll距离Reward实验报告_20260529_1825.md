# AdvancedScroll 距离 Reward 实验报告

生成时间：2026-05-29 18:25 CST

## 结论

本次实验按“不要注入候选动作，让 VLM 自己采样动作”的原则修改了 on-policy GRPO 采集与训练 reward。新增目标可见后的点击距离奖励和重复点击惩罚后，采集阶段确实产生了更细的组内 reward 差异，但 20 个 trainable groups 的小步训练没有提升 `advanced_scroll` 专项验证集。

因此本次 adapter 不采用。当前主线 current model 仍保持：

`outputs/onpolicy_browser_rl_grpo_table_advanced_repair_289tg_safe_20260529_0101/adapter`

## 本次修改

修改文件：

- `scripts/train_browser_rl_onpolicy_grpo.py`

新增参数：

- `--target-distance-reward-weight`：目标可见后，点击越接近 `#target` 中心，额外 reward 越高。
- `--target-distance-reward-sigma`：距离 reward 的高斯宽度，单位是 0-1000 归一化截图坐标。
- `--target-distance-reward-templates`：允许使用距离 reward 的任务模板，默认 `advanced_scroll`。
- `--repeat-click-penalty`：如果当前点击重复此前相近点击，扣分。
- `--repeat-click-radius`：判断重复点击的半径。

同时确认：

- `--inject-scroll-candidates` 默认关闭。
- `--inject-target-click-candidates` 默认关闭。
- 单一唯一动作状态可以继续 rollout，但不会形成 GRPO group。

## Reward 形式

基础 reward 来自环境 verifier：

```text
r_env = verifier_reward
```

新增目标距离奖励：

```text
d = || click_xy - target_center_xy ||_2
r_dist = w * exp(-d^2 / (2 * sigma^2))
```

本次参数：

```text
w = 0.35
sigma = 90
```

重复点击惩罚：

```text
r_repeat = -0.15, if current click is within radius 25 of a previous click
r_repeat = 0, otherwise
```

总 reward：

```text
r = r_env + r_dist + r_repeat - step_cost + success_bonus - invalid_action_penalty
```

本次没有把目标中心写进 prompt，也没有把目标中心变成候选动作。目标中心只用于训练时计算 reward，模型推理时仍然只能看截图、任务目标和历史。

## 采集数据

采集目录：

`/root/datasets/browser_rl/onpolicy/onpolicy_browser_rl_advanced_scroll_distance_reward_collect_20260529_1756`

采集命令核心参数：

```bash
CUDA_VISIBLE_DEVICES=1 /opt/conda/envs/llama/bin/python3 scripts/train_browser_rl_onpolicy_grpo.py \
  --tasks /root/datasets/browser_rl/task_suites/browser_rl_advanced_scroll_progress_900_20260529_0300/train_advanced_scroll_tasks.jsonl \
  --output-dir /root/datasets/browser_rl/onpolicy/onpolicy_browser_rl_advanced_scroll_distance_reward_collect_20260529_1756 \
  --model /root/models/Qwen2.5-VL-3B-Instruct \
  --adapter outputs/onpolicy_browser_rl_grpo_table_advanced_repair_289tg_safe_20260529_0101/adapter \
  --collect-only \
  --limit 80 \
  --max-groups 80 \
  --target-trainable-groups 30 \
  --max-steps 4 \
  --num-generations 8 \
  --max-sample-attempts 24 \
  --sampling-temperature 0.9 \
  --prompt-style full \
  --max-history 4 \
  --load-in-4bit \
  --headless \
  --commit-strategy best \
  --no-inject-scroll-candidates \
  --no-inject-target-click-candidates \
  --continue-on-single-sample \
  --target-distance-reward-weight 0.35 \
  --target-distance-reward-sigma 90 \
  --repeat-click-penalty 0.15 \
  --repeat-click-radius 25 \
  --stream-collect \
  --stream-flush-every 5 \
  --seed 101
```

为了快速验证 reward 是否有效，在达到 20 个 trainable groups 后停止采集，并固化临时文件。

采集统计：

| 指标 | 数值 |
|---|---:|
| groups | 23 |
| trainable_groups | 20 |
| zero_std_groups | 3 |
| samples | 142 |
| rollout rows | 12 |
| rollout success_rate | 0.8333 |
| validation error_count | 0 |
| exec_ok_step_rate | 1.0 |
| reward_std_mean | 0.1921 |
| reward_min / reward_max | 0.0 / 1.3500 |
| target_distance_bonus samples | 128 |
| target_distance_bonus mean | 0.2412 |
| target_distance mean | 74.84 |
| target_distance min / max | 1.41 / 202.01 |
| repeat_click_penalty count | 3 |

质量检查目录：

`outputs/advanced_scroll_distance_reward_collect_validation_20260529_1815`

## 训练

训练输出：

`outputs/onpolicy_browser_rl_grpo_advanced_scroll_distance_reward_20tg_safe_20260529_1816/adapter`

训练命令核心参数：

```bash
CUDA_VISIBLE_DEVICES=1 /opt/conda/envs/llama/bin/python3 scripts/train_browser_rl_onpolicy_grpo.py \
  --output-dir outputs/onpolicy_browser_rl_grpo_advanced_scroll_distance_reward_20tg_safe_20260529_1816 \
  --train-only-from /root/datasets/browser_rl/onpolicy/onpolicy_browser_rl_advanced_scroll_distance_reward_collect_20260529_1756/groups.jsonl \
  --model /root/models/Qwen2.5-VL-3B-Instruct \
  --adapter outputs/onpolicy_browser_rl_grpo_table_advanced_repair_289tg_safe_20260529_0101/adapter \
  --learning-rate 1e-7 \
  --epochs 1 \
  --gradient-accumulation-steps 4 \
  --grad-clip 1.0 \
  --load-in-4bit \
  --prompt-style full \
  --max-history 4 \
  --logprob-reduction mean \
  --replay-sft-jsonl /root/datasets/browser_rl/sft/local_qwen_history_aware_sft_v3_1000_aug_20260528_0140/local_qwen_history_aware_sft_train.jsonl \
  --replay-loss-weight 0.08 \
  --replay-ratio 0.50 \
  --replay-max-rows 2048 \
  --seed 103
```

训练统计：

| 指标 | 数值 |
|---|---:|
| trainable_groups | 20 |
| micro_steps | 20 |
| optimizer_steps | 5 |
| learning_rate | 1e-7 |
| trainable_parameters | 29,933,568 |
| trainable_ratio | 0.0145 |
| loss_mean | -0.0213 |
| policy_loss_mean | -0.0370 |
| replay_loss_mean | 0.3265 |
| reward_mean | 0.4777 |
| reward_std_mean | 0.2210 |

## 评测

评测目录：

`outputs/rollouts_local_qwen25vl_distance_reward_20tg_advanced_scroll_progress_val30_20260529_1819`

评测集：

`/root/datasets/browser_rl/task_suites/browser_rl_advanced_scroll_progress_900_20260529_0300/val_advanced_scroll_tasks.jsonl`

评测命令核心参数：

```bash
CUDA_VISIBLE_DEVICES=1 /opt/conda/envs/llama/bin/python3 scripts/run_browser_rl_rollouts.py \
  --env-source local_smoke \
  --policy local_qwen \
  --tasks /root/datasets/browser_rl/task_suites/browser_rl_advanced_scroll_progress_900_20260529_0300/val_advanced_scroll_tasks.jsonl \
  --output-dir outputs/rollouts_local_qwen25vl_distance_reward_20tg_advanced_scroll_progress_val30_20260529_1819 \
  --max-steps 4 \
  --headless \
  --local-model /root/models/Qwen2.5-VL-3B-Instruct \
  --local-adapter outputs/onpolicy_browser_rl_grpo_advanced_scroll_distance_reward_20tg_safe_20260529_1816/adapter \
  --local-load-in-4bit \
  --local-max-new-tokens 96 \
  --local-temperature 0 \
  --local-max-history 4 \
  --local-prompt-style full
```

评测结果：

| 模型 | advanced_scroll progress val30 |
|---|---:|
| 当前主线 289tg adapter baseline | 0.1667 |
| pure VLM 20tg adapter | 0.1667 |
| 本次 distance reward 20tg adapter | 0.1667 |

## 失败分析

本次 adapter 的 val30 行为：

| 指标 | 数值 |
|---|---:|
| rollouts | 30 |
| success | 5 |
| success_rate | 0.1667 |
| first action | 30/30 为 `scroll` |
| action 总数 | `scroll=30`, `click=80` |
| 主失败序列 | `scroll > click > click > click`，25/30 |
| 成功序列 | `scroll > click`，5/30 |
| 最终 verifier progress | 29/30 为 `scrolled_down=true,target_visible=true` |

说明：

1. 模型已经稳定学会第一步滚动。
2. 失败主要不是没有看到目标，而是目标可见后点击坐标仍偏移。
3. 小规模 distance reward 让采样 group 的训练信号更细，但 20 个 trainable groups、5 个 optimizer steps 不足以稳定改变推理行为。
4. 本次实验没有恢复候选注入，因此失败不来自“规则候选泄漏”，而是当前 VLM 对 after-scroll 目标定位仍弱。

## 是否采用

不采用。

原因：

- `advanced_scroll progress val30` 没有超过 0.1667 baseline。
- 没有必要继续跑 val_balanced_70 或 val200。
- 当前全局主线仍应保留 289tg adapter。

## 下一步建议

1. 保留本次 reward 代码，但不要把本次 20tg adapter 设为 current model。
2. 若继续走纯 VLM on-policy 路线，应扩大 distance reward 采集到至少 80-150 trainable groups，并提高任务覆盖，而不是只用 20tg 判断最终效果。
3. 可尝试更强的 after-scroll click 定向 SFT replay：只用模型自己可见目标后的截图和成功点击动作，不注入候选到 RL 采样。
4. 可增加“重复点击无进展”的 history penalty，让模型在连续错误点击后倾向重新观察、调整点击位置或结束失败，而不是在同一坐标附近反复点击。
5. 在达到 `advanced_scroll val30 >= 0.40` 前，不建议把相关 adapter 送入 val200/test200 作为主线候选。
