# History-Aware Trajectory SFT 实现与本地 Qwen 评测报告

- 时间：2026-05-27 23:56 CST
- 仓库：`/root/Workspace/VLM/EviTool-VL`
- 主线：真正 on-policy VLM agentic RL

## 术语说明

- `History-aware`：带历史感知，指训练输入里包含当前 step、历史动作、上一轮 verifier progress、动作空间和坐标规则。
- `Trajectory SFT`：轨迹监督微调，让模型按专家轨迹逐步模仿下一步动作，而不是只看单张截图预测动作。
- `Local Qwen`：本地 Qwen2.5-VL/Qwen3-VL current model，参数可通过 LoRA/SFT/RL 更新。
- `Rollout`：一次完整任务尝试，模型反复看截图、输出动作、环境执行、返回 reward，直到成功或超步数。
- `Verifier`：验证器，用程序检查任务是否完成，并返回 reward/progress。

## 本次完成

1. 增强 full prompt。
   - 修改：`envs/browser_rl/qwen_policy.py`
   - 新增字段：`最大 step`、`当前 verifier progress`
   - 将 `viewport` 和 `action_space` 改成标准 JSON 字符串。
   - 增加提示：根据截图和历史判断下一步，不要重复无进展动作。

2. 新增 history-aware SFT 构建器。
   - 新增：`scripts/build_local_qwen_history_aware_sft.py`
   - 输入：`rollouts.jsonl` 和可选 `tasks.jsonl`
   - 输出：LlamaFactory sharegpt multimodal 格式
   - 每条样本包含：
     - screenshot
     - task goal
     - current step
     - max steps
     - viewport
     - action space
     - verifier progress
     - recent history
     - target next action JSON

3. 构建 v1 数据集。
   - 数据集：`/root/models/datasets/local_qwen_history_aware_sft_v1_20260527_2344`
   - rows: 60
   - train_rows: 55
   - val_rows: 5
   - unique_tasks: 20
   - train_tasks: 18
   - val_tasks: 2
   - task_overlap: 0
   - action_distribution: click 40, type 16, press 4
   - history_len_distribution: 0:20, 1:16, 2:12, 3:8, 4:4

4. 训练本地 Qwen2.5-VL history-aware LoRA。
   - 配置：`configs/sft_local_qwen_history_aware_qwen25vl_3b_lora.yaml`
   - 底模：`/root/models/Qwen2.5-VL-3B-Instruct`
   - checkpoint：`checkpoints/qwen25vl_3b_history_aware_sft_v1_lora`
   - train_rows: 55
   - global_step: 35
   - epoch: 5.0
   - train_loss: 0.4766

5. 完成本地 current model rollout 评测。
   - eval5 输出：`outputs/rollouts_local_qwen25vl_history_aware_v1_eval5_20260527_2349`
   - eval20 输出：`outputs/rollouts_local_qwen25vl_history_aware_v1_eval20_20260527_2352`

## 运行命令

构建 history-aware SFT 数据：

```bash
python3 scripts/build_local_qwen_history_aware_sft.py \
  --input outputs/browser_rl_smoke_rollouts_20260527_1w/rollouts.jsonl \
  --tasks outputs/browser_rl_smoke_tasks_20260527_1w/all_tasks.jsonl \
  --output-dir /root/models/datasets/local_qwen_history_aware_sft_v1_20260527_2344 \
  --include-policies scripted_oracle \
  --val-ratio 0.1 \
  --seed 42 \
  --max-history 4 \
  --dataset-name local_qwen_history_aware_sft_train \
  --val-dataset-name local_qwen_history_aware_sft_val
```

训练 history-aware SFT：

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/root/Workspace/VLM/EviTool-VL/third_party/LlamaFactory/src:${PYTHONPATH:-} \
python3 -m llamafactory.cli train \
  configs/sft_local_qwen_history_aware_qwen25vl_3b_lora.yaml
```

评测本地 Qwen current model：

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_browser_rl_rollouts.py \
  --env-source local_smoke \
  --policy local_qwen \
  --tasks outputs/browser_rl_smoke_tasks_20260527_1w/all_tasks.jsonl \
  --output-dir outputs/rollouts_local_qwen25vl_history_aware_v1_eval20_20260527_2352 \
  --limit 20 \
  --max-steps 6 \
  --headless \
  --local-model /root/models/Qwen2.5-VL-3B-Instruct \
  --local-adapter checkpoints/qwen25vl_3b_history_aware_sft_v1_lora \
  --local-load-in-4bit \
  --local-max-new-tokens 96 \
  --local-temperature 0 \
  --local-prompt-style full
```

## 评测结果

### 与前一阶段对比

| 阶段 | rollouts | success_rate | valid_json_rate | valid_action_rate | 说明 |
|---|---:|---:|---:|---:|---|
| action JSON warmup | 1 | 0.0 | 1.0 | 1.0 | 能输出合法动作，但不会完成任务 |
| clean-oracle short prompt | 1 | 0.0 | 1.0 | 1.0 | 仍不会稳定 click/type 顺序 |
| history-aware v1 eval5 | 5 | 0.6 | 1.0 | 1.0 | 首次出现稳定成功轨迹 |
| history-aware v1 eval20 | 20 | 0.3 | 1.0 | 1.0 | 全量 smoke 成功率仍偏低 |

### eval20 分类结果

| 任务类型 | 成功数 | success_rate | avg_reward |
|---|---:|---:|---:|
| smoke_form | 2/4 | 0.50 | 0.85 |
| smoke_todo | 4/4 | 1.00 | 1.20 |
| smoke_search | 0/4 | 0.00 | 0.00 |
| smoke_menu | 0/4 | 0.00 | 0.00 |
| smoke_table | 0/4 | 0.00 | 0.00 |

## 关键观察

1. history-aware SFT 是有效的。
   - 之前本地 Qwen 基本只能输出合法 JSON，success_rate 为 0。
   - 现在 eval20 success_rate 达到 0.3，说明模型开始学会多步操作。

2. 当前主要失败不是 JSON 格式问题。
   - eval20 valid_json_rate = 1.0
   - eval20 valid_action_rate = 1.0
   - 失败主要来自坐标定位和任务类型覆盖不足。

3. form 任务失败集中在第一步 click 坐标偏高。
   - smoke_form_02 / smoke_form_04 中 Name 输入框没有聚焦，后续 type 没有写入 name。
   - code_value 变成 true，但 name_value 仍 false。

4. search/menu/table 完全失败，说明 v1 数据覆盖太少。
   - 每类只有 4 个任务，table 每个任务只有 1 条 step-level 样本。
   - 现在还不能直接认为模型有泛化 GUI agent 能力。

## 结论

`history-aware trajectory SFT` 已经把本地 Qwen 从“只能输出合法 action JSON”推进到“能完成部分可 verifier 浏览器任务”。这一步是进入 on-policy RL 前必要且有效的 warmup。

但目前数据量只有 60 条 step-level 样本，成功率只有 0.3，仍不适合直接大规模 RL。下一步应该先扩任务和轨迹数据，让 current model 有更稳定的非零 reward，再进入 Verifier-Guided GRPO。

## 下一步计划

1. 扩充 local browser tasks 到 100-300 个。
   - 优先扩 form/search/menu/table，因为这些是当前瓶颈。
   - 每类至少 20-50 个任务。

2. 重建 history-aware SFT v2。
   - 目标至少 500-1500 条 step-level 样本。
   - 保持 task-level train/val/test 切分。

3. 加入坐标扰动增强。
   - 对 oracle click 坐标做小范围安全扰动，提升模型对输入框/button 区域的鲁棒性。

4. 评测 local current model baseline。
   - 目标：eval20 success_rate ≥ 0.6，valid_action_rate ≥ 0.95。
   - 达到后再接小规模 on-policy RL。

5. 开始 easy curriculum on-policy RL。
   - 先从 form/todo/search easy tasks 开始。
   - 避免在 0 reward 任务上直接做大规模 GRPO。
