# AdvancedScroll 滚动点击专项修复实验报告

- 生成时间：2026-05-29 05:40 CST
- 仓库：`/root/Workspace/VLM/EviTool-VL`
- 数据根目录：`/root/datasets/browser_rl`
- 训练与评测 GPU：`CUDA_VISIBLE_DEVICES=1`，避开 GPU0 上已有模型

## 术语说明

- `advanced_scroll`：BrowserRL 任务模板之一，任务要求模型先向下滚动页面，再点击滚动后出现的目标按钮。
- `LoRA adapter`：低秩适配器，只训练一小部分增量参数，不改动 Qwen2.5-VL 底模本体。
- `on-policy rollout`：在线策略轨迹，当前模型自己看截图、输出动作、环境执行动作、验证器返回奖励。
- `verifier-guided candidate`：验证器辅助候选动作。它不是最终推理时使用的外部工具，而是在采集阶段额外注入可执行候选动作，让同一个状态下有好坏样本可比较。
- `val_balanced_70`：快速验证集，按任务类型均衡抽样，用于快速判断 adapter 是否明显退化；它不是最终 test。
- `test200`：阶段性最终验收集。本轮没有继续使用 test200 调参。

## 背景

上一阶段已通过验收的 289 trainable groups adapter：

```text
outputs/onpolicy_browser_rl_grpo_table_advanced_repair_289tg_safe_20260529_0101/adapter
```

已有指标：

- `val_balanced_70`：0.9420
- `val200`：0.9100
- `test200`：0.8900
- `test200 advanced_scroll`：0.0000

这说明总体 BrowserRL 能力已经较强，但 `advanced_scroll` 仍是明显短板。进一步分析发现，问题不是单纯不会输出合法 JSON，而是滚动与滚动后点击两个子能力没有稳定结合。

## 本轮代码修改

### 1. 为 advanced_scroll 增加过程奖励

修改文件：

```text
scripts/build_browser_rl_task_suite.py
```

新增 verifier progress：

- `scrolled_down`：页面滚动距离是否达到阈值。
- `target_visible`：目标按钮是否已经出现在 viewport 内。

目的：让 reward 不只在最终点击成功时才给 1 分，而是在“已经滚动”“目标已可见”两个中间状态提供可学习信号。

### 2. 为 on-policy 采集加入滚动与目标点击候选动作

修改文件：

```text
scripts/train_browser_rl_onpolicy_grpo.py
```

新增参数：

```text
--inject-scroll-candidates / --no-inject-scroll-candidates
--inject-target-click-candidates / --no-inject-target-click-candidates
--scroll-candidate-dys
```

行为：

- 如果目标还不可见，额外注入多个 `scroll(dy)` 候选。
- 如果目标已经可见，使用 Playwright 查询 `#target` 的真实中心点，并在中心点附近注入若干 `click(x,y)` 候选。

注意：这些候选只用于采集和训练阶段，最终评测仍是本地 Qwen2.5-VL adapter 自己根据截图输出 direct GUI action。

## 新构建数据

### advanced_scroll progress 任务套件

路径：

```text
/root/datasets/browser_rl/task_suites/browser_rl_advanced_scroll_progress_900_20260529_0300
```

构成：

| 模板 | 数量 |
| --- | ---: |
| `advanced_dialog` | 300 |
| `advanced_scroll` | 300 |
| `advanced_tab` | 300 |
| 总计 | 900 |

切分：

| split | 数量 |
| --- | ---: |
| train | 720 |
| val | 90 |
| test | 90 |

专项过滤文件：

| 文件 | 数量 |
| --- | ---: |
| `train_advanced_scroll_tasks.jsonl` | 240 |
| `val_advanced_scroll_tasks.jsonl` | 30 |
| `test_advanced_scroll_tasks.jsonl` | 30 |

本轮只使用 train 和 val。`test_advanced_scroll_tasks.jsonl` 没有用于调参。

## 质量验证

### Oracle 验证

Oracle 指脚本化正确策略，用来验证任务本身是否可 reset、可执行、可被 verifier 判定成功。

```text
outputs/rollouts_oracle_advanced_scroll_progress_val30_20260529_0300
```

结果：

- rollouts：30
- success_rate：1.0000
- avg_steps：2.0

结论：任务本身没有问题，脚本正确动作可以 100% 完成。

### 289tg baseline 在新 scroll val30 上的表现

```text
outputs/rollouts_local_qwen25vl_289tg_advanced_scroll_progress_val30_20260529_0301
```

结果：

- success_rate：0.1667
- avg_steps：3.6667
- valid_json/action：1.0

失败分析：

- 第一动作为 `scroll` 的比例：30/30
- 主要失败序列：`scroll > click > click > click`

这说明 289tg adapter 在 progress reward 版本上已经会滚动，但滚动后点击位置不稳定。

## 第一轮修复：409tg advanced_scroll progress repair

### 采集

路径：

```text
/root/datasets/browser_rl/onpolicy/onpolicy_browser_rl_advanced_scroll_progress_collect_20260529_0308
```

关键参数：

```bash
CUDA_VISIBLE_DEVICES=1 python3 scripts/train_browser_rl_onpolicy_grpo.py \
  --collect-only \
  --adapter outputs/onpolicy_browser_rl_grpo_table_advanced_repair_289tg_safe_20260529_0101/adapter \
  --tasks /root/datasets/browser_rl/task_suites/browser_rl_advanced_scroll_progress_900_20260529_0300/train_advanced_scroll_tasks.jsonl \
  --target-trainable-groups 40 \
  --num-generations 4 \
  --scroll-candidate-dys 600,900,1200,1500
```

采集结果：

- groups：57
- trainable_groups：40
- zero_std_groups：17
- samples：273
- rollouts：23
- rollout_success_rate：0.8696
- action_distribution：scroll=138，click=135

### 训练

合并训练集：

```text
/root/datasets/browser_rl/onpolicy/onpolicy_browser_rl_advanced_scroll_replay_409tg_merged_20260529_0357
```

训练输出：

```text
outputs/onpolicy_browser_rl_grpo_advanced_scroll_repair_409tg_safe_20260529_0358/adapter
```

训练统计：

- trainable_groups：409
- optimizer_steps：52
- learning_rate：1e-7
- replay_loss_weight：0.08
- replay_ratio：0.35

评测：

| 验证集 | success_rate | 说明 |
| --- | ---: | --- |
| advanced_scroll progress val30 | 0.3000 | 从 289tg 的 0.1667 上升 |
| val_balanced_70 | 0.9710 | 高于 289tg 的 0.9420 |

结论：第一轮修复有效，且快速均衡验证集没有退化。但 advanced_scroll 仍远低于 0.75 目标。

## 第二轮修复：220tg scroll-click target injection repair

### 采集

路径：

```text
/root/datasets/browser_rl/onpolicy/onpolicy_browser_rl_advanced_scroll_click_progress_collect_20260529_0444
```

本轮新增目标点击候选：目标可见后，根据 `#target` 的真实中心点注入中心点和附近偏移点击。

采集结果：

- groups：47
- trainable_groups：40
- zero_std_groups：7
- samples：328
- rollouts：23
- rollout_success_rate：0.9565
- action_distribution：scroll=136，click=191，wait=1

质量校验：

```text
outputs/advanced_scroll_click_collect_validation_20260529_0540
```

- valid_rollouts：23/23
- error_count：0
- exec_ok_step_rate：1.0
- action_parse_step_rate：1.0

### 训练

合并训练集：

```text
/root/datasets/browser_rl/onpolicy/onpolicy_browser_rl_advanced_scroll_click_replay_220tg_merged_20260529_0500
```

训练输出：

```text
outputs/onpolicy_browser_rl_grpo_advanced_scroll_click_repair_220tg_safe_20260529_0501/adapter
```

训练统计：

- trainable_groups：220
- optimizer_steps：28
- learning_rate：1e-7
- replay_loss_weight：0.08
- replay_ratio：0.40
- reward_mean：0.3989
- reward_std_mean：0.2599

评测：

| 验证集 | success_rate | avg_steps | valid_action_rate |
| --- | ---: | ---: | ---: |
| advanced_scroll progress val30 | 0.4000 | 3.20 | 1.0 |
| val_balanced_70 | 0.9420 | 3.01 | 1.0 |

质量校验：

| 输出目录 | valid_rollouts | error_count | exec_ok_step_rate | action_parse_step_rate |
| --- | ---: | ---: | ---: | ---: |
| `outputs/advanced_scroll_click_eval_val30_validation_20260529_0540` | 30/30 | 0 | 1.0 | 1.0 |
| `outputs/advanced_scroll_click_eval_val70_validation_20260529_0540` | 69/69 | 0 | 1.0 | 1.0 |

## 对比结论

| 版本 | adapter | advanced_scroll progress val30 | val_balanced_70 | 是否作为主线 |
| --- | --- | ---: | ---: | --- |
| 289tg 已验收版 | `outputs/onpolicy_browser_rl_grpo_table_advanced_repair_289tg_safe_20260529_0101/adapter` | 0.1667 | 0.9420 | 已通过 val200/test200 阶段验收 |
| 409tg 第一轮修复 | `outputs/onpolicy_browser_rl_grpo_advanced_scroll_repair_409tg_safe_20260529_0358/adapter` | 0.3000 | 0.9710 | 可作为下一轮候选基座，但还未跑 val200 |
| 220tg 第二轮修复 | `outputs/onpolicy_browser_rl_grpo_advanced_scroll_click_repair_220tg_safe_20260529_0501/adapter` | 0.4000 | 0.9420 | scroll 专项更好，但不建议替换主线 |

本轮最重要的发现：

1. 模型已经稳定学会第一步滚动：advanced_scroll val30 中 30/30 第一动作都是 `scroll`。
2. 主要失败已转移到滚动后的目标点击：失败样本多为 `scroll > click > click > click`，目标已经可见，但点击坐标偏高或偏离按钮。
3. 第二轮 target click 注入让专项成功率从 0.30 到 0.40，但均衡 val70 从 0.971 回落到 0.942。
4. 220tg adapter 不能作为新的全局主线；它是有效的专项实验，但还没有达到 0.75 目标，也没有通过 val200。

## 典型失败例子

样本：

```text
suite_advanced_scroll_20722
```

模型动作：

```text
scroll(dy=900) -> click(x=253,y=719) -> click(x=253,y=719) -> click(x=253,y=719)
```

截图中按钮已经位于页面下方可见区域，但模型点击高度偏上，没有点中按钮中心。归一化坐标下，目标更接近 `y≈790-820`，模型经常输出 `y≈719`。

这说明下一步不应继续只增加“是否滚动”的奖励，而要加强“目标可见后的精确点击坐标”学习。

## 当前推荐状态

- 已通过阶段验收的主线 adapter 仍是：

```text
outputs/onpolicy_browser_rl_grpo_table_advanced_repair_289tg_safe_20260529_0101/adapter
```

- 409tg adapter 可以作为下一轮 scroll 修复的训练起点，因为它在快速均衡验证集上最好：

```text
outputs/onpolicy_browser_rl_grpo_advanced_scroll_repair_409tg_safe_20260529_0358/adapter
```

- 220tg adapter 只作为 scroll-click 专项实验版本保留：

```text
outputs/onpolicy_browser_rl_grpo_advanced_scroll_click_repair_220tg_safe_20260529_0501/adapter
```

## 下一步计划

1. 构建“滚动后点击”专项 SFT/replay 小数据：只取目标已经可见的 step1 状态，assistant completion 使用 verifier 验证过的目标中心点击。
2. 在 GRPO reward 中加入目标可见后的坐标距离奖励，例如：

```text
r = r_success + alpha * 1[target_visible] + beta * exp(-d(click,target)^2 / (2*sigma^2)) - gamma * 1[repeat_same_failed_click]
```

其中 `d(click,target)` 是点击点到目标中心的距离。这样错误点击也能按距离给连续信号，而不是只有 0 或 1。

3. 加入重复无效点击惩罚：如果同一状态下连续点击近似同一个错误坐标，降低 reward，避免 `click > click > click` 死循环。
4. 先在 advanced_scroll val30 上达到 0.60-0.70，再跑 val_balanced_70；只有两者同时不退化，才跑 val200。
5. 继续保留 test200，不用于日常调参。

