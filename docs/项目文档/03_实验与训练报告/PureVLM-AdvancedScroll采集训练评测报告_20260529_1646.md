# Pure VLM AdvancedScroll 采集训练评测报告

- 生成时间：2026-05-29 16:46 CST
- 仓库：`/root/Workspace/VLM/EviTool-VL`
- GPU：`CUDA_VISIBLE_DEVICES=1`
- 目标：验证关闭 injected candidates 后，当前本地 Qwen2.5-VL 是否能通过纯 VLM 自采样继续做 advanced_scroll RL。

## 术语说明

- `Pure VLM`：纯视觉语言模型自采样。这里指候选动作都来自本地 Qwen2.5-VL 自己生成，不注入规则 scroll 或 DOM 中心 click 候选。
- `Injected candidates`：注入候选。训练脚本额外塞入规则动作，例如固定 `scroll(dy=900)` 或通过 DOM 查询目标中心点后塞入 click。此类动作已从默认采集关闭。
- `GRPO group`：同一个状态下多个候选动作及其 reward 的集合。只有一个唯一动作时不能形成 group。

## 本轮代码修正

### 1. 关闭 injected candidates 默认开关

文件：

```text
scripts/train_browser_rl_onpolicy_grpo.py
```

变更：

```text
--inject-scroll-candidates 默认 False
--inject-target-click-candidates 默认 False
```

后续默认采集只用 VLM 自己生成动作；如果要做探索增强，必须显式打开。

### 2. 允许单一唯一动作继续 rollout

发现问题：

关闭 injected candidates 后，advanced_scroll 的 step0 经常 24 次采样都只输出同一个动作：

```json
{"action":"scroll","dy":900.0}
```

旧逻辑会因为“只有 1 个唯一动作”直接终止任务，导致后续 step1 点击阶段无法被采集。

修正后：

- 如果一个状态只有 1 个唯一动作：继续执行该动作，推进 rollout。
- 但这个状态不写 GRPO group，因为没有候选间比较。
- 如果后续状态采出多个不同动作，再形成 group。

这符合 pure VLM on-policy 采集：模型自己决定动作，训练只利用有比较信号的状态。

## 纯 VLM 采集

命令核心参数：

```bash
CUDA_VISIBLE_DEVICES=1 python3 scripts/train_browser_rl_onpolicy_grpo.py \
  --tasks /root/datasets/browser_rl/task_suites/browser_rl_advanced_scroll_progress_900_20260529_0300/train_advanced_scroll_tasks.jsonl \
  --output-dir /root/datasets/browser_rl/onpolicy/onpolicy_browser_rl_advanced_scroll_pure_vlm_collect_20260529_1611 \
  --model /root/models/Qwen2.5-VL-3B-Instruct \
  --adapter outputs/onpolicy_browser_rl_grpo_table_advanced_repair_289tg_safe_20260529_0101/adapter \
  --collect-only \
  --num-generations 8 \
  --max-sample-attempts 24 \
  --sampling-temperature 0.9 \
  --no-inject-scroll-candidates \
  --no-inject-target-click-candidates \
  --continue-on-single-sample
```

采集被人工截断在 20 个 trainable groups，用于快速判断方向。

路径：

```text
/root/datasets/browser_rl/onpolicy/onpolicy_browser_rl_advanced_scroll_pure_vlm_collect_20260529_1611
```

统计：

| 指标 | 数值 |
| --- | ---: |
| rollouts | 15 |
| rollout_success_rate | 0.9333 |
| groups | 27 |
| trainable_groups | 20 |
| zero_std_groups | 7 |
| samples | 184 |
| avg_reward_std | 0.1832 |

动作来源：

```text
local_qwen: 184
verifier_guided_injected_candidate: 0
```

动作分布：

| action | 数量 |
| --- | ---: |
| click | 176 |
| scroll | 6 |
| wait | 1 |
| type | 1 |

质量校验：

```text
outputs/advanced_scroll_pure_vlm_collect_validation_20260529_1636
```

结果：

- valid_rollouts：15/15
- error_count：0
- success_rate：0.9333
- exec_ok_step_rate：1.0
- action_parse_step_rate：1.0

## 训练

训练输出：

```text
outputs/onpolicy_browser_rl_grpo_advanced_scroll_pure_vlm_20tg_safe_20260529_1637/adapter
```

训练参数：

```text
base model: /root/models/Qwen2.5-VL-3B-Instruct
initial adapter: outputs/onpolicy_browser_rl_grpo_table_advanced_repair_289tg_safe_20260529_0101/adapter
trainable_groups: 20
learning_rate: 1e-7
epochs: 1
gradient_accumulation_steps: 4
replay_sft_jsonl: /root/datasets/browser_rl/sft/local_qwen_history_aware_sft_v3_1000_aug_20260528_0140/local_qwen_history_aware_sft_train.jsonl
replay_loss_weight: 0.08
replay_ratio: 0.50
```

训练统计：

| 指标 | 数值 |
| --- | ---: |
| micro_steps | 20 |
| optimizer_steps | 5 |
| policy_loss_mean | -0.1154 |
| replay_loss_mean | 0.2838 |
| reward_mean | 0.2825 |
| reward_std_mean | 0.2473 |

## 评测

专项验证集：

```text
/root/datasets/browser_rl/task_suites/browser_rl_advanced_scroll_progress_900_20260529_0300/val_advanced_scroll_tasks.jsonl
```

输出：

```text
outputs/rollouts_local_qwen25vl_pure_vlm_20tg_advanced_scroll_progress_val30_20260529_1640
```

结果：

| 模型 | advanced_scroll progress val30 |
| --- | ---: |
| 289tg baseline | 0.1667 |
| pure VLM 20tg adapter | 0.1667 |

本轮 pure VLM 20tg 没有带来专项提升。

## 失败分析

采集集：

```text
15 rollouts, 14 success, success_rate=0.9333
first action: scroll 15/15
主要成功序列: scroll > click
```

val30：

```text
30 rollouts, 5 success, success_rate=0.1667
first action: scroll 30/30
失败主序列: scroll > click > click > click
```

失败样本的 verifier progress 大多是：

```json
{"scrolled_down": true, "target_visible": true}
```

结论：

1. 模型已经会先 scroll。
2. 失败集中在目标已经可见后，点击坐标仍然偏。
3. 当前 reward 对“目标可见但点偏”的动作大多给同样的 0.2，无法区分“差一点点”和“偏很多”。
4. 20 个 pure VLM groups 太少，且采集集和 val30 的按钮位置分布仍有差异，导致没有泛化提升。

## 决策

不采用：

```text
outputs/onpolicy_browser_rl_grpo_advanced_scroll_pure_vlm_20tg_safe_20260529_1637/adapter
```

当前主线仍保留：

```text
outputs/onpolicy_browser_rl_grpo_table_advanced_repair_289tg_safe_20260529_0101/adapter
```

## 下一步

不要恢复 injected candidates 作为默认方案。更合理的 pure VLM 下一步是改 reward：

1. 保持候选动作全部由 VLM 自己采样。
2. 在 reward 中加入点击点到目标中心的距离奖励。
3. 对重复点击同一个失败坐标加惩罚。
4. 重新采集 pure VLM groups，再训练。

建议 reward：

```text
r = r_success
  + 0.2 * progress_ratio
  + beta * 1[target_visible] * exp(-d(click, target_center)^2 / (2 * sigma^2))
  - gamma * 1[repeat_same_failed_click]
```

这里的 `target_center` 只用于 verifier/reward，不作为候选动作暴露给模型。这样仍然是 VLM 自己选择动作，但 reward 更细，可以学习“点击更接近目标中心”。

