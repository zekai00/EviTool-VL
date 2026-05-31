# Step-wise GRPO Ratio Clip / Reference KL 小实验报告

时间：2026-05-31 01:04 CST

## 目的

> 重要更正：本报告是“289tg adapter 上继续训练”的代码路径 smoke，不是公平的 vanilla vs clipped GRPO 算法对比。公平对比已补做，见 `StepWiseGRPO-SFTv3同起点VanillaVsClippedKL对比报告_20260531_0246.md`。

此前 `EviTool-VL` 的 step-wise GRPO 训练使用的是简化损失：

```text
L_RL = - mean_i(A_i * log pi_theta(a_i | s))
```

没有 old-policy ratio clip，也没有 reference KL。本次实验目标是验证：在当前 289tg BrowserRL adapter 上，补入 old-policy ratio clip 和 reference KL 后，是否能稳定或提升 `val_balanced_70`。

## 实现

修改文件：

- `scripts/train_browser_rl_onpolicy_grpo.py`

新增参数：

- `--grpo-loss-type {vanilla,clipped}`
- `--clip-epsilon`
- `--kl-beta`
- `--old-logprob-key`
- `--reference-logprob-key`
- `--skip-logprob-cache`
- `--train-max-groups`

本次实现采用“缓存式 reference/old logprob”，不是常驻第二个 3B VLM。训练开始前，用初始 289tg adapter 对每个采样动作计算：

```text
log pi_old(a_i | s)
log pi_ref(a_i | s)
```

本次 smoke 实验里 `pi_old = pi_ref = 初始 289tg adapter`。这样能实现 PPO/GRPO 风格的 ratio clip 与 KL 约束，同时避免显存里同时放两份 Qwen2.5-VL-3B。

对每个 group，reward 标准化为：

```text
A_i = (r_i - mean(r)) / std(r)
```

clipped GRPO 目标：

```text
rho_i = exp(log pi_theta(a_i | s) - log pi_old(a_i | s))

L_policy = - mean_i min(
  rho_i * A_i,
  clip(rho_i, 1 - epsilon, 1 + epsilon) * A_i
)

L_KL = 0.5 * mean_i (log pi_theta(a_i | s) - log pi_ref(a_i | s))^2

L_RL = L_policy + beta * L_KL
```

仍保留 SFT replay：

```text
L_total = L_RL + lambda * L_SFT
```

## 训练命令

```bash
CUDA_VISIBLE_DEVICES=1 timeout 7200s /opt/conda/envs/llama/bin/python3 -u scripts/train_browser_rl_onpolicy_grpo.py \
  --output-dir outputs/onpolicy_browser_rl_grpo_289tg_clipkl_smoke20_20260531_0017 \
  --train-only-from /root/datasets/browser_rl/onpolicy/onpolicy_browser_rl_table_advanced_replay_289tg_merged_20260529_0100/groups.jsonl \
  --tasks /root/datasets/browser_rl/task_suites/browser_rl_task_suite_2000_20260528_1344/train_tasks.jsonl \
  --model /root/models/Qwen2.5-VL-3B-Instruct \
  --adapter /root/models/adapters/browser_rl/qwen25vl_3b_browserrl_289tg_table_advanced_repair_adapter_backup_20260530_1313 \
  --train-max-groups 20 --epochs 1 --learning-rate 1e-7 \
  --gradient-accumulation-steps 8 --logprob-reduction mean \
  --grpo-loss-type clipped --clip-epsilon 0.2 --kl-beta 0.01 \
  --replay-sft-jsonl /root/datasets/browser_rl/sft/local_qwen_history_aware_sft_v3_1000_aug_20260528_0140/local_qwen_history_aware_sft_train.jsonl \
  --replay-loss-weight 0.06 --replay-ratio 0.3 --replay-max-rows 2048 \
  --sampling-temperature 0.8 --max-new-tokens 96 --max-history 4 \
  --prompt-style full --image-max-pixels 262144 --load-in-4bit --torch-dtype bf16
```

输出 adapter：

```text
outputs/onpolicy_browser_rl_grpo_289tg_clipkl_smoke20_20260531_0017/adapter
```

## 训练结果

训练 summary：

- trainable_groups：20
- micro_steps：20
- optimizer_steps：3
- learning_rate：1e-7
- clip_epsilon：0.2
- kl_beta：0.01
- replay_ratio：0.30
- replay_loss_weight：0.06
- loss_mean：0.0046808
- policy_loss_mean：0.0004808
- replay_loss_mean：0.2800
- reward_mean：0.3613
- reward_std_mean：0.3105
- approx_kl_mean：4.36e-6
- clip_frac_mean：0.0
- ratio_mean：1.0006

解释：

- `approx_kl_mean` 极小，说明新 adapter 没有明显偏离 289tg 初始策略。
- `clip_frac_mean=0`，说明本次 20-group smoke 规模下 ratio clip 几乎没有实际触发。
- 从训练统计看，ratio clip / KL 更像稳定器，还没有带来可见优化信号。

## 评测

必须使用与 baseline 一致的 `--local-prompt-style full`。一次错误评测使用了默认 `sft_minimal`，模型退回 Qwen-VL 原生 GUI 输出格式，例如 `left_click + coordinate`，导致 30 条 0 分；该结果不计入正式对比。

正式评测：

```bash
CUDA_VISIBLE_DEVICES=1 /opt/conda/envs/llama/bin/python3 -u scripts/run_browser_rl_rollouts.py \
  --env-source local_smoke --policy local_qwen \
  --tasks /root/datasets/browser_rl/task_suites/browser_rl_task_suite_2000_20260528_1344/val_balanced_70_tasks.jsonl \
  --limit 30 --max-steps 6 --seed 42 --headless \
  --local-model /root/models/Qwen2.5-VL-3B-Instruct \
  --local-adapter outputs/onpolicy_browser_rl_grpo_289tg_clipkl_smoke20_20260531_0017/adapter \
  --local-load-in-4bit --local-torch-dtype auto --local-max-new-tokens 128 \
  --local-temperature 0.0 --local-max-history 4 --local-prompt-style full --local-image-max-pixels 262144
```

随后补跑剩余 39 条并合并：

```text
outputs/rollouts_local_qwen25vl_289tg_clipkl_smoke20_valbalanced70_fullprompt_20260531_0100/merged
```

## 对比结果

| 模型 | val_balanced_70 | success | avg_steps |
|---|---:|---:|---:|
| 289tg baseline | 0.9420 | 65/69 | 3.029 |
| 289tg + clipped/KL smoke20 | 0.8986 | 62/69 | 3.072 |

分类型对比：

| 类型 | baseline | clipped/KL smoke20 |
|---|---:|---:|
| advanced | 9/9 | 8/9 |
| choice | 8/10 | 8/10 |
| form | 9/10 | 8/10 |
| menu | 10/10 | 10/10 |
| search | 10/10 | 10/10 |
| table | 9/10 | 8/10 |
| todo | 10/10 | 10/10 |

新增退化任务：

- `suite_advanced_scroll_085`
- `suite_form_367`
- `suite_table_223`

## 结论

本次小实验不升级 current model。

原因：

- 20 个 group 的 clipped/KL smoke 训练没有带来收益，完整 `val_balanced_70` 从 0.9420 下降到 0.8986。
- `clip_frac_mean=0`，说明 ratio clip 在当前学习率和训练规模下没有真正发挥作用。
- `approx_kl_mean` 极小，reference KL 也没有提供有效改进，只是提供了安全约束。

当前 BrowserRL 主线模型继续保持：

```text
/root/models/adapters/browser_rl/qwen25vl_3b_browserrl_289tg_table_advanced_repair_adapter_backup_20260530_1313
```

## 下一步建议

1. 保留 ratio clip / reference KL 代码路径，但默认仍使用 `vanilla`，不要把本次 smoke adapter 作为 current model。
2. 如果继续验证 clipped GRPO，应使用更大规模、类型均衡的 100-300 trainable groups，而不是只取前 20 个 group。
3. 训练前需要保证 eval 命令固定 `--local-prompt-style full`，否则会把 action schema 问题误判为模型能力问题。
4. 如果要让 clip 真正生效，可以提高训练步数或略增学习率，但必须先用 `val30 -> val70 -> val200` gate，防止 form/table/advanced_scroll 退化。
