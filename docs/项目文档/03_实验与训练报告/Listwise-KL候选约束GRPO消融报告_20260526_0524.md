# Listwise + KL Candidate-Constrained GRPO 实验报告

日期：2026-05-26

## 1. 本轮目标

本轮不是单纯继续跑 GRPO，而是在算法层面做一个更适合 GUI candidate selection 的版本：

1. 先做 candidate-level listwise warmup，让模型学习“整组候选的相对分布”，不只学习单个正例。
2. 再做 Candidate-Constrained GRPO，但在 loss 里加入对 listwise reference policy 的 KL 约束。
3. 加入 rank-balanced sampler，避免短训练中过度集中在 oracle rank 早期或晚期样本。
4. 用双卡 3090 通过 PyTorch DDP 跑训练。

核心判断问题：这些改动是否只是在 `cc_action_ids` 子候选集合内有效，还是能泛化到 full candidates。

## 2. 代码改动

本轮新增/修改如下：

- `rl/candidate_constrained.py`
  - 新增 `rank_balanced_indices()`。
  - 作用：按 oracle rank bucket（early/middle/late/unknown）打散并 round-robin 交错样本，降低短 run 中的 rank 分布偏置。

- `scripts/train_gui_candidate_listwise.py`
  - 新增 candidate-level listwise warmup 训练脚本。
  - 目标分布：`softmax(candidate_reward / target_temperature)`。
  - 损失：`cross_entropy(target_distribution, policy_distribution)`，附带可选 entropy 项。

- `scripts/cache_gui_candidate_reference_scores.py`
  - 新增 reference logprobs 缓存脚本。
  - 作用：提前用 listwise adapter 给候选 action 计算 logprob，写入 `reference_logprobs`，后续 GRPO 不需要再加载第二个 frozen reference VLM。

- `scripts/train_gui_candidate_cc_grpo.py`
  - 新增 `--kl-coef`、`--reference-logprobs-key`、`--rank-balanced-sampler`。
  - GRPO loss 改为：
    - `policy_loss + kl_coef * KL(pi_current || pi_reference) - entropy_coef * entropy`
  - KL 是 candidate action 分布上的 forward KL，不是 token-level KL。

这些不是直接套现成 GRPO，而是围绕你的 GUI candidate selection 任务做的 task-specific GRPO 改造。

## 3. 双卡训练方式

使用方式：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 ...
```

实际分布式方法：

- PyTorch `DistributedDataParallel`
- backend：`NCCL`
- `world_size=2`
- 每张 3090 上各放一份完整 Qwen2.5-VL-3B + LoRA adapter replica
- 每个进程处理不同数据 shard
- backward 后通过 NCCL all-reduce 同步 LoRA 梯度

这属于数据并行，不是模型并行。因此双 3090 提升吞吐，但单卡仍需要能放下完整模型副本；它不会把 24GB + 24GB 自动合成一个 48GB 显存池。

## 4. 数据与训练设置

训练数据：

- 基础训练集：`outputs/gui_candidate_rl_os_atlas_linux_2k_c60_cc_pref/train.jsonl`
- train rows：906
- val rows：266
- 本轮快速验证时：
  - listwise warmup 使用前 256 train rows 跑 100 step
  - reference cache 使用 256 rows
  - KL + CC-GRPO 使用带 `reference_logprobs` 的 256 rows 跑 100 step

关键参数：

- listwise：
  - `max_steps=100`
  - `max_actions=8`
  - `max_hard_negatives=6`
  - `target_temperature=0.2`
  - `learning_rate=2e-6`
  - `rank_balanced_sampler=True`

- KL + CC-GRPO：
  - `max_steps=100`
  - `num_generations=4`
  - `max_actions=8`
  - `max_hard_negatives=6`
  - `policy_temperature=1.2`
  - `entropy_coef=0.01`
  - `kl_coef=0.02`
  - `rank_balanced_sampler=True`

## 5. 训练日志摘要

### 5.1 Listwise warmup 100 step

输出 adapter：

`outputs/gui_candidate_rl_os_atlas_linux_2k_c60_cc_pref/listwise_a8_100step`

训练均值：

| Metric | Value |
|---|---:|
| ce_loss | 2.020957 |
| expected_reward | 0.164191 |
| oracle_prob | 0.136038 |
| top1_prob | 0.142625 |
| policy_entropy | 2.062558 |
| target_entropy | 0.550113 |
| grad_norm | 2.244811 |

解释：

- `oracle_prob` 仍低，说明 100 step listwise 还没有把 oracle 候选强推到最高概率。
- `top1_prob` 也不高，说明它没有明显回到 top1/c00 shortcut。
- 这一步更像分布校准，不适合作为最终策略。

### 5.2 Listwise + KL + CC-GRPO 100 step

输出 adapter：

`outputs/gui_candidate_rl_os_atlas_linux_2k_c60_cc_pref/cc_grpo_listwise_kl_g4_a8_100step`

训练均值：

| Metric | Value |
|---|---:|
| reward_mean | 0.188660 |
| reward_std | 0.218042 |
| frac_reward_zero_std | 0.060000 |
| best_action_reward | 0.902132 |
| oracle_sample_rate | 0.161250 |
| top1_bias_rate | 0.115000 |
| entropy | 2.060333 |
| kl_ref | 0.001904 |
| grad_norm | 1.469971 |

解释：

- `frac_reward_zero_std=0.06`，说明绝大多数 group 有 reward 差异，GRPO 有有效训练信号。
- `top1_bias_rate=0.115`，比旧 c00/top1 shortcut 明显健康。
- `kl_ref=0.001904` 很小，说明 KL 没有压死训练，只是轻约束。
- `best_action_reward=0.902` 很高，但 `reward_mean=0.189` 不高，说明候选池里经常有好动作，但采样策略并不总能抽到。这是后续算法改进重点。

## 6. 验证结果

### 6.1 `cc_action_ids val100`

这个评估只在 RL 构建的 action 子候选集合内选。它回答的问题是：在候选子集已经比较合理时，policy 能不能把好候选排上来。

| Model / Adapter | Avg reward | Pointing | IoU@0.5 | Avg IoU | Avg selected rank | c00 rate |
|---|---:|---:|---:|---:|---:|---:|
| SFT warmup | 0.2552 | 11.00% | 9.00% | 0.0807 | 2.26 | 70.00% |
| Old CC-GRPO g4/a8/100 | 0.3457 | 20.00% | 17.00% | 0.1379 | 13.66 | 16.00% |
| Listwise warmup | 0.3367 | 20.00% | 16.00% | 0.1345 | 16.40 | 21.00% |
| Listwise + KL + CC-GRPO | **0.3656** | **24.00%** | **18.00%** | **0.1572** | 23.25 | **0.00%** |

结论：

- 新算法在 `cc_action_ids val100` 上超过旧 CC-GRPO。
- c00 rate 从 warmup 的 70% 降到 0%，说明 c00 shortcut 被彻底打破。
- 但 avg selected rank 到 23.25，表示模型更愿意选高编号/后排候选。这在子候选集合内有收益，但在 full candidates 中可能变成过度探索。

### 6.2 `full candidates val30`

这个评估不使用 `cc_action_ids`，而是在完整候选集合里选。它更接近真实部署。

| Model / Adapter | Avg reward | Pointing | IoU@0.5 | Avg IoU | Avg selected rank | c00 rate |
|---|---:|---:|---:|---:|---:|---:|
| SFT warmup | 0.2529 | 10.00% | 6.67% | 0.0689 | 2.50 | 80.00% |
| Old CC-GRPO g4/a8/100 | **0.2958** | **13.33%** | **6.67%** | **0.0747** | 16.10 | 3.33% |
| Listwise warmup | 0.2614 | 10.00% | 3.33% | 0.0485 | 19.20 | 10.00% |
| Listwise + KL + CC-GRPO | 0.2471 | 6.67% | 3.33% | 0.0394 | 23.00 | 0.00% |

结论：

- 新算法没有提升 full candidates，反而低于旧 CC-GRPO。
- 它确实消除了 c00 shortcut，但代价是更容易选择后排高编号候选。
- 这说明当前瓶颈不是“GRPO 还不够多跑”，而是 candidate set / action sampling / hard negative 构建方式没有和 full candidate 场景对齐。

## 7. 总结结论

本轮算法改进有效，但只在 constrained candidate subset 上有效：

- 成功点：
  - `cc_action_ids val100` 从旧 CC-GRPO 的 20% pointing 提升到 24%。
  - IoU@0.5 从 17% 提升到 18%。
  - Avg reward 从 0.3457 提升到 0.3656。
  - c00 shortcut 从 16% 降到 0%。

- 失败点：
  - `full candidates val30` 从旧 CC-GRPO 的 13.33% pointing 降到 6.67%。
  - Avg reward 从 0.2958 降到 0.2471。
  - 模型更倾向高 rank 候选，说明探索方向在 full set 上过强。

因此，不建议现在直接跑 300 step。更合理的下一步是先改 candidate RL 数据构建和采样算法，否则继续训练会放大“后排候选偏置”。

## 8. 下一步算法改进计划

### 8.1 把 action set 从 `cc_action_ids` 改成 full-aware constrained set

当前 `cc_action_ids` 对训练是友好的，但和 full candidates eval 有分布差异。下一版应构建：

- always include oracle/high-IoU candidate
- include rank1/top visual candidate
- include same-region hard negatives
- include distant false positives
- include model-current top predicted candidates

这样 RL group 内既有正确答案，也有模型真正会误选的 full-set hard negatives。

### 8.2 做 adaptive hard-negative mining

流程：

1. 用当前 adapter 在 train subset 的 full candidates 上打分。
2. 找出模型高分但 reward 低的候选。
3. 把这些候选写回下一轮 `cc_action_ids`。
4. 再跑 Candidate-Constrained GRPO。

这会把 RL 从“固定候选子集优化”变成“针对当前模型错误动态构建候选子集”。

### 8.3 改 GRPO group sampler

当前采样是从 policy 分布里抽 `num_generations=4`，可能抽不到高 reward 动作。下一版可以做任务定制 group：

- 1 个 oracle/high reward action
- 1 个 rank1/top visual action
- 1-2 个 current-policy high-score false positive
- 1-2 个 random negative

这样每个 group 保证有 reward contrast，减少 `reward_mean` 低但 `best_action_reward` 高的问题。

### 8.4 加 candidate-rank regularization，但不能回到 c00 shortcut

可以加入轻量 rank penalty：

`reward_final = reward_iou + reward_pointing - alpha * log(1 + rank)`

但 alpha 必须很小，并且只用于 full-aware action set。否则会把模型重新拉回 rank1/c00 shortcut。

### 8.5 数据规模

目前 906 train rows 仍偏小，而且本轮 KL 训练只用 256 rows 快速验证。正式 RL 至少需要：

- 2k+ GUI candidate RL rows
- 每条 row 有 full candidates
- 每条 row 有 oracle/high-IoU 标注
- 每条 row 有 current-model hard negatives
- 验证集保持 fixed，不参与 hard-negative mining

建议下一阶段先构建 2k full-aware candidate RL 数据，再跑：

1. listwise 300 step
2. hard-negative mining
3. KL + CC-GRPO 300 step
4. full candidates val100

## 9. 本轮最终判断

Candidate-Constrained GRPO 的方向是对的，但下一步不能只加训练步数。需要把算法改进推进到“candidate construction + group sampling”层面。

本轮最重要的结论是：

`Listwise + KL + rank-balanced CC-GRPO` 能提升候选子集排序，但 full candidates 失败说明候选池构建才是当前主瓶颈。

