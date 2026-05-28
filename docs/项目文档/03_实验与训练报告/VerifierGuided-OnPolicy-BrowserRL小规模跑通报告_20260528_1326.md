# Verifier-Guided On-Policy Browser RL 小规模跑通报告

- 生成时间：2026-05-28 13:26 CST
- 主线：真正 on-policy VLM agentic RL
- 结论：链路已跑通；激进更新会退化；安全更新没有退化但也没有带来 test30 增益。当前生产起点仍建议使用 `checkpoints/qwen25vl_3b_history_aware_sft_v3_1000_aug_lora`。

## 术语说明

- `verifier-guided`：用自动 verifier 判分器给环境执行结果打 reward，而不是靠人工标注每一步。
- `on-policy`：用当前模型自己采样出来的动作和轨迹训练当前模型。
- `GRPO`：Group Relative Policy Optimization，同一个状态下采样一组动作，用组内 reward 的相对高低构造 advantage。
- `advantage`：优势值，表示某个动作比同组平均动作好多少；本实验用 `(r_i - mean(r)) / std(r)`。
- `rollout`：模型在环境里完成一次任务尝试的完整轨迹。

## 新增实现

新增脚本：

- `scripts/train_browser_rl_onpolicy_grpo.py`

脚本做三件事：

1. 从当前本地 Qwen2.5-VL LoRA 采样 GUI action。
2. 对同一截图状态采样多个 action，用 Playwright reset/replay 回到同一状态后逐个执行，由 verifier 给 reward。
3. 对 reward 有差异的组做 group-relative policy-gradient 更新，并保存新 LoRA adapter。

训练目标：

```text
A_i = (r_i - mean(r)) / (std(r) + eps)
L = - mean_i A_i * log pi_theta(a_i | s)
```

其中 `s` 是当前截图状态，`a_i` 是模型采样出的第 `i` 个 action，`r_i` 是 verifier reward。第一版使用 completion 总 logprob，后续改成 token mean logprob 来降低 JSON 长度带来的梯度放大。

## 采集与训练

基础 current model：

- base：`/root/models/Qwen2.5-VL-3B-Instruct`
- adapter：`checkpoints/qwen25vl_3b_history_aware_sft_v3_1000_aug_lora`

on-policy 采集：

- 输出：`outputs/onpolicy_browser_rl_grpo_smoke_v1_20260528_1305`
- tasks：从 train split 打乱后取 8 条
- max_groups：16
- num_generations：3
- max_steps：4
- commit_strategy：first，即用当前模型采样的第一个动作推进真实 on-policy 轨迹

采集结果：

| 指标 | 值 |
| --- | ---: |
| groups | 16 |
| trainable_groups | 5 |
| zero_std_groups | 11 |
| samples | 48 |
| rollouts | 8 |
| rollout_success_rate | 0.75 |
| groups_by_family | choice 4, menu 4, table 4, todo 4 |
| trainable_groups_by_family | choice 2, menu 1, table 2 |

## 两版训练对照

### 激进版

- 输出 adapter：`outputs/onpolicy_browser_rl_grpo_smoke_v1_20260528_1305/adapter`
- learning_rate：`5e-6`
- logprob_reduction：旧版 sum
- trainable_groups：5
- optimizer_steps：3

test30：

| 模型 | success_rate | form | search |
| --- | ---: | ---: | ---: |
| v3 baseline 前 30 条 | 0.7000 | 0.6364 | 0.8750 |
| on-policy aggressive | 0.5667 | 0.5000 | 0.7500 |

结论：激进版退化，不采用。

### 安全版

- 输出 adapter：`outputs/onpolicy_browser_rl_grpo_smoke_v1_safe_20260528_1318/adapter`
- 数据：复用同一批 on-policy groups
- learning_rate：`5e-7`
- logprob_reduction：mean
- trainable_groups：5
- optimizer_steps：3

test30：

| 模型 | success_rate | form | search |
| --- | ---: | ---: | ---: |
| v3 baseline 前 30 条 | 0.7000 | 0.6364 | 0.8750 |
| on-policy safe | 0.7000 | 0.6364 | 0.8750 |

结论：安全版没有破坏前 30 条 test 表现，但也没有产生增益。它可作为 RL 链路 smoke checkpoint，不应直接替换 v3 baseline。

## 质量校验

校验输出：

- `outputs/onpolicy_browser_rl_grpo_smoke_v1_validation_20260528_1326`

结果：

- rollouts：68
- valid_rollouts：68
- error_count：0
- exec_ok_step_rate：1.0
- action_parse_step_rate：1.0
- screenshot_field_rate：1.0

说明环境执行、动作解析、截图记录都是正常的。

## 当前判断

本轮真正完成的是“可执行 on-policy RL 闭环”：

```text
current model -> sample actions -> browser step -> verifier reward -> group-relative update -> new LoRA -> held-out rollout eval
```

但本轮还不是有效提升模型性能的 RL 版本，主要原因：

1. trainable_groups 只有 5 个，更新信号太少。
2. 采集 family 是 choice/menu/table/todo，而快速 test30 主要是 form/search，分布不一致。
3. 没有 KL anchor 或 family early stop，激进版容易破坏旧能力。
4. 当前只是一步动作的组内 reward，还没有把整条 trajectory return 充分利用起来。

## 下一步建议

1. 保留 v3 baseline 作为 current model，不把本轮 adapter 升级为默认模型。
2. 下一轮采集做 family-balanced，至少包含 form/search/choice/menu/table。
3. 把 trainable_groups 扩到 50-100，再每轮只做 5-10 个 optimizer step。
4. 加入 evaluation gate：每轮后跑 test30，如果低于 baseline 就丢弃 adapter。
5. 再加入 reference KL 或 SFT replay，避免 search/table/form 被 RL 更新带崩。

## 主要命令

采集并训练激进版：

```bash
CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/llama/bin/python3 scripts/train_browser_rl_onpolicy_grpo.py \
  --tasks outputs/browser_rl_task_suite_1000_20260528_0058/train_tasks.jsonl \
  --output-dir outputs/onpolicy_browser_rl_grpo_smoke_v1_20260528_1305 \
  --model /root/models/Qwen2.5-VL-3B-Instruct \
  --adapter checkpoints/qwen25vl_3b_history_aware_sft_v3_1000_aug_lora \
  --limit 8 \
  --max-groups 16 \
  --max-steps 4 \
  --num-generations 3 \
  --sampling-temperature 0.8 \
  --learning-rate 5e-6 \
  --epochs 1 \
  --gradient-accumulation-steps 2 \
  --headless \
  --load-in-4bit \
  --prompt-style full \
  --max-history 4
```

训练安全版：

```bash
CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/llama/bin/python3 scripts/train_browser_rl_onpolicy_grpo.py \
  --output-dir outputs/onpolicy_browser_rl_grpo_smoke_v1_safe_20260528_1318 \
  --train-only-from outputs/onpolicy_browser_rl_grpo_smoke_v1_20260528_1305/groups.jsonl \
  --model /root/models/Qwen2.5-VL-3B-Instruct \
  --adapter checkpoints/qwen25vl_3b_history_aware_sft_v3_1000_aug_lora \
  --learning-rate 5e-7 \
  --epochs 1 \
  --gradient-accumulation-steps 2 \
  --load-in-4bit \
  --prompt-style full \
  --max-history 4 \
  --logprob-reduction mean
```

快速评测：

```bash
CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/llama/bin/python3 scripts/run_browser_rl_rollouts.py \
  --env-source local_smoke \
  --policy local_qwen \
  --tasks outputs/browser_rl_task_suite_1000_20260528_0058/test_tasks.jsonl \
  --output-dir outputs/rollouts_local_qwen25vl_onpolicy_grpo_smoke_v1_safe_test30_20260528_1319 \
  --limit 30 \
  --max-steps 6 \
  --headless \
  --local-model /root/models/Qwen2.5-VL-3B-Instruct \
  --local-adapter outputs/onpolicy_browser_rl_grpo_smoke_v1_safe_20260528_1318/adapter \
  --local-load-in-4bit \
  --local-max-new-tokens 96 \
  --local-temperature 0 \
  --local-max-history 4 \
  --local-prompt-style full
```
