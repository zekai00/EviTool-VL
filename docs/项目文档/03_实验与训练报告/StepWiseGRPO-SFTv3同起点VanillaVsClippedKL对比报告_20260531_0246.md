# Step-wise GRPO：SFT v3 同起点 Vanilla vs Clipped+KL 对比报告

时间：2026-05-31 02:46 CST

## 先纠正一个实验定义问题

上一份 `StepWiseGRPO-RatioClip-ReferenceKL小实验报告_20260531_0104.md` 跑的是：

```text
289tg adapter -> 继续训练 20 groups clipped/KL -> 与 289tg adapter 对比
```

它只能证明代码路径能跑通，以及“在 289tg 上继续小步训练不一定有收益”。它不是公平的算法对比。

更合理的对比应该是：

```text
同一个 SFT v3 起点
同一批 SFT v3 on-policy groups
同样训练超参
只改变 GRPO loss：
  A. vanilla step-wise GRPO
  B. old-policy ratio clipped GRPO + reference KL
```

本报告补做的就是这个版本。

## 实验设计

底模：

```text
/root/models/Qwen2.5-VL-3B-Instruct
```

共同起点 adapter：

```text
checkpoints/qwen25vl_3b_history_aware_sft_v3_1000_aug_lora
```

reference model：

```text
SFT v3 adapter 本身
```

old policy：

```text
SFT v3 adapter 本身
```

原因：本次 groups 是用 SFT v3 作为当前策略采样出来的，所以 old policy 应该是 SFT v3。这样才符合 clipped policy optimization 的基本设定。

## 采集数据

输出：

```text
/root/datasets/browser_rl/onpolicy/onpolicy_browser_rl_sftv3_probe_60tg_20260531_0213
```

采集过程：

- 原计划每类尽量 10 个 trainable groups。
- 由于 SFT v3 在部分任务上采样动作重复较多，继续追 todo/search quota 会浪费时间，因此在已有 60 个 trainable groups 时停止，作为 smoke 对比数据。

采集统计：

| 指标 | 数值 |
|---|---:|
| groups | 173 |
| trainable_groups | 60 |
| zero_std_groups | 113 |
| samples | 501 |
| rollouts | 64 |
| rollout_success_rate | 0.5625 |

trainable group 分布：

| family | trainable_groups |
|---|---:|
| form | 11 |
| choice | 11 |
| menu | 10 |
| table | 10 |
| search | 8 |
| advanced | 7 |
| todo | 3 |

这个分布说明：从 SFT v3 出发时，todo/search/advanced 的有效 group 不如 form/choice/menu/table 容易得到；弱策略下 on-policy exploration 本身就是瓶颈。

## Loss 定义

### Vanilla Step-wise GRPO

对同一状态下的多个候选动作，用 verifier reward 做组内标准化：

```text
A_i = (r_i - mean(r)) / std(r)
```

训练目标：

```text
L_vanilla = - mean_i A_i * log pi_theta(a_i | s)
```

### Clipped+KL GRPO

先用 SFT v3 给每个采样动作缓存：

```text
log pi_old(a_i | s)
log pi_ref(a_i | s)
```

本实验中：

```text
pi_old = pi_ref = SFT v3 adapter
```

训练时：

```text
rho_i = exp(log pi_theta(a_i | s) - log pi_old(a_i | s))

L_policy = - mean_i min(
  rho_i * A_i,
  clip(rho_i, 1 - epsilon, 1 + epsilon) * A_i
)

L_KL = 0.5 * mean_i (log pi_theta(a_i | s) - log pi_ref(a_i | s))^2

L_clipped = L_policy + beta * L_KL
```

本次参数：

```text
epsilon = 0.2
beta = 0.01
```

两组都保留 SFT replay：

```text
L_total = L_GRPO + 0.06 * L_SFT
replay_ratio = 0.30
```

## 训练结果

### Vanilla

输出：

```text
outputs/onpolicy_browser_rl_sftv3_probe60_vanilla_20260531_0214/adapter
```

关键指标：

| 指标 | 数值 |
|---|---:|
| trainable_groups | 60 |
| optimizer_steps | 8 |
| learning_rate | 2e-7 |
| policy_loss_mean | -0.06839 |
| replay_loss_mean | 0.27965 |
| reward_mean | 0.35972 |
| reward_std_mean | 0.26382 |

### Clipped+KL

输出：

```text
outputs/onpolicy_browser_rl_sftv3_probe60_clippedkl_20260531_0217/adapter
```

关键指标：

| 指标 | 数值 |
|---|---:|
| trainable_groups | 60 |
| optimizer_steps | 8 |
| learning_rate | 2e-7 |
| clip_epsilon | 0.2 |
| kl_beta | 0.01 |
| policy_loss_mean | -0.00040 |
| replay_loss_mean | 0.27948 |
| approx_kl_mean | 3.84e-6 |
| clip_frac_mean | 0.0 |
| ratio_mean | 1.00005 |

解释：

- `clip_frac_mean=0.0`，说明这次小步训练里 ratio 没有越过 clip 边界。
- `approx_kl_mean` 很小，说明策略没有明显偏离 SFT v3 reference。
- 因此，这次 clipped+KL 的收益不是“clip 边界强烈生效”的结果，而更像是 ratio-form objective + reference 约束带来的更保守更新。

## Val Balanced 70 对比

统一评测集：

```text
/root/datasets/browser_rl/task_suites/browser_rl_task_suite_2000_20260528_1344/val_balanced_70_tasks.jsonl
```

注意：实际 rollouts 数为 69。

| 模型 | 起点 | 训练数据 | val_balanced_70 | success | avg_steps |
|---|---|---|---:|---:|---:|
| SFT v3 baseline | SFT v3 | 无 RL | 0.8261 | 57/69 | 3.391 |
| Vanilla GRPO 60tg | SFT v3 | SFT v3 on-policy 60tg | 0.8406 | 58/69 | 3.377 |
| Clipped+KL GRPO 60tg | SFT v3 | 同一批 60tg | 0.8841 | 61/69 | 3.319 |
| 历史 289tg staged adapter | SFT v3 -> 100tg -> 213tg -> 289tg | 分阶段累计 groups | 0.9420 | 65/69 | 3.029 |

按类型：

| family | SFT v3 | Vanilla 60tg | Clipped+KL 60tg | 289tg staged |
|---|---:|---:|---:|---:|
| advanced | 6/9 | 6/9 | 7/9 | 9/9 |
| choice | 8/10 | 7/10 | 7/10 | 8/10 |
| form | 5/10 | 7/10 | 8/10 | 9/10 |
| menu | 10/10 | 10/10 | 10/10 | 10/10 |
| search | 10/10 | 9/10 | 10/10 | 10/10 |
| table | 8/10 | 9/10 | 9/10 | 9/10 |
| todo | 10/10 | 10/10 | 10/10 | 10/10 |

Clipped+KL 相比 vanilla 多成功 3 条：

- `suite_advanced_scroll_088`
- `suite_form_363`
- `suite_search_343`

Clipped+KL 仍落后 289tg 的主要失败：

- `suite_advanced_scroll_082`
- `suite_advanced_scroll_085`
- `suite_choice_radio_301`
- `suite_form_367`
- `suite_form_405`
- `suite_table_223`

## Val200 补跑对比

统一评测集：

```text
/root/datasets/browser_rl/task_suites/browser_rl_task_suite_2000_20260528_1344/val_tasks.jsonl
```

补跑时间：2026-05-31 03:00-03:36 CST。

| 模型 | 起点 | 训练数据 | val200 success | avg_steps | 输出 |
|---|---|---|---:|---:|---|
| SFT v3 baseline | SFT v3 | 无 RL | 0.745 | 3.810 | `outputs/rollouts_local_qwen25vl_history_aware_v3_2000_val200_20260528_1412/merged` |
| Vanilla GRPO 60tg | SFT v3 | SFT v3 on-policy 60tg | 0.750 | 3.805 | `outputs/rollouts_local_qwen25vl_sftv3_probe60_vanilla_val200_20260531_0300` |
| Clipped+KL GRPO 60tg | SFT v3 | 同一批 60tg | 0.770 | 3.765 | `outputs/rollouts_local_qwen25vl_sftv3_probe60_clippedkl_val200_20260531_0300` |
| 历史 289tg staged adapter | SFT v3 -> 100tg -> 213tg -> 289tg | 分阶段累计 groups | 0.910 | 3.445 | `outputs/rollouts_local_qwen25vl_onpolicy_table_advanced_repair_289tg_val200_20260529_0130/merged` |

结论：

- val200 上 clipped+KL 仍小幅优于 vanilla：0.770 vs 0.750。
- 60tg 小实验只证明 clipped+KL 方向更稳，不能替代 289tg staged adapter。
- 与 289tg 的 0.910 相比，主要差距更可能来自训练数据规模和 targeted repair 覆盖，而不是单纯 loss 形式。

## 结论

1. 用户指出的问题是对的：上一轮 289tg 上继续训练不是公平算法对比。
2. 在正确的 SFT v3 同起点小实验里，Clipped+KL GRPO 明显优于 vanilla GRPO：
   - vanilla：58/69
   - clipped+KL：61/69
3. 但 clipped+KL 60tg 仍低于历史 289tg staged adapter：
   - clipped+KL 60tg：61/69
   - 289tg staged：65/69
4. 当前还不能把 clipped+KL 60tg 升级为主线 current model。
5. 下一步值得做的是把 clipped+KL 按同样 staged 路线扩到 200-300 trainable groups，并跑 val200/test200。

## 下一步

建议按这个顺序继续：

1. 用 SFT v3 重新采集 200-300 trainable groups，增加 advanced_scroll、form、choice_radio、table 的覆盖。
2. 跑三组对比：
   - vanilla GRPO，beta=0
   - clipped GRPO，beta=0
   - clipped GRPO，beta=0.01
3. 先用 `val_balanced_70` gate，目标不低于 0.94。
4. 达标后跑 val200/test200；只有超过当前 289tg staged adapter，才替换 current model。
