# HistoryAwareSFTv3-FormChoice 定向修复实验报告

- 生成时间：2026-05-28 12:19 CST
- 目标：在进入 on-policy RL 前，先补少量 form/choice 定向示范，观察是否能降低后续 RL 初始失败率。
- 结论：本次修复版不建议作为 RL 起点；当前最稳的 current model 仍是 `checkpoints/qwen25vl_3b_history_aware_sft_v3_1000_aug_lora`。

## 术语说明

- `SFT`：监督微调，用 oracle 或人工示范轨迹教模型在每一步输出正确 action JSON。
- `LoRA`：低秩适配器，只训练少量增量参数，不全量更新 Qwen2.5-VL-3B。
- `rollout`：一次任务执行轨迹，包含截图、模型动作、环境执行结果、reward 和 verifier 状态。
- `verifier`：自动判分器，读取网页状态判断任务是否成功，不把答案泄露给模型。
- `family`：任务大类，例如 form、choice、search、table，用来观察不同任务类型是否互相干扰。

## 实验设计

本次不是正式 on-policy RL，而是一次小规模定向 SFT 修复实验。做法是从已有 v3 1000-task history-aware SFT 模型继续训练，额外加入 form/choice 修复数据，同时保留原 v3 数据做 replay。

修复任务：

| 数据 | 数量 | train/val/test | 说明 |
| --- | ---: | --- | --- |
| form repair v1 | 120 | 96/12/12 | 表单填写任务 |
| choice repair v1 | 60 | 48/6/6 | select/checkbox/radio 各 20 |

oracle 轨迹：

| 数据 | rollouts | success_rate | trainable_steps | 变体 |
| --- | ---: | ---: | ---: | --- |
| form repair oracle | 312 | 1.0 | 1560 | center 120, jitter 96, recovery 96 |
| choice repair oracle | 108 | 1.0 | 284 | center 60, jitter 48 |

choice 没有加入 recovery，因为 checkbox/radio 的错误点击会改变页面状态，简单恢复示范容易引入状态歧义。

混合 SFT 数据：

- 路径：`/root/models/datasets/local_qwen_history_aware_sft_v3_1000_aug_form_choice_repair_v1_20260528_1104`
- rows：9786
- train/val/test rows：9010/388/388
- unique_tasks：1180
- 原 v3 replay rows：7942
- repair form rows：1560
- repair choice rows：284
- max_history：4
- task_overlap：[]

训练：

- base model：`/root/models/Qwen2.5-VL-3B-Instruct`
- 起点 adapter：`checkpoints/qwen25vl_3b_history_aware_sft_v3_1000_aug_lora`
- 输出 adapter：`checkpoints/qwen25vl_3b_history_aware_sft_v3_form_choice_repair_v1_lora`
- max_steps：240
- learning_rate：2.0e-5
- best checkpoint：`checkpoint-200`
- best eval_loss：0.152206
- train_loss：0.233011

## Test100 结果

三组都在同一批 `outputs/browser_rl_task_suite_1000_20260528_0058/test_tasks.jsonl` 的前 100 条上评测。

| 模型 | success_rate | avg_steps | 结论 |
| --- | ---: | ---: | --- |
| v3 baseline | 0.82 | 3.59 | 当前最稳 |
| repair checkpoint-100 | 0.66 | 3.82 | 明显退化 |
| repair best/root checkpoint-200 | 0.76 | 3.62 | choice 提升，但整体退化 |

按 family 拆分：

| family | n | v3 baseline | repair ckpt100 | repair ckpt200/root |
| --- | ---: | ---: | ---: | ---: |
| advanced | 5 | 0.8000 | 0.8000 | 0.8000 |
| choice | 9 | 0.6667 | 0.6667 | 0.8889 |
| form | 22 | 0.6364 | 0.4091 | 0.6364 |
| menu | 17 | 0.8235 | 0.6471 | 0.8235 |
| search | 22 | 0.8636 | 0.6364 | 0.5909 |
| table | 17 | 1.0000 | 0.8235 | 0.8824 |
| todo | 8 | 1.0000 | 1.0000 | 1.0000 |

关键变化：

- choice：`6/9 -> 8/9`，确实提升。
- form：`14/22 -> 14/22`，没有提升。
- search：`19/22 -> 13/22`，明显退化。
- table：`17/17 -> 15/17`，退化。
- valid_json_rate 和 valid_action_rate 均为 1.0，说明退化不是 JSON 格式问题，而是策略动作选择问题。

## 质量校验

校验路径：`outputs/browser_rl_repair_v1_validation_20260528_1220`

- 输入 rollouts：620
- valid_rollouts：620
- error_count：0
- exec_ok_step_rate：1.0
- action_parse_step_rate：1.0
- screenshot_field_rate：1.0

这说明数据结构、图片路径和动作字段没有坏；性能下降来自模型策略变化。

## 失败模式

repair checkpoint-200 的失败主要表现为：

- search：输入 query 后反复点击搜索按钮或错误结果区域，没有稳定点击目标结果。
- table：行/列坐标出现偏移，原来能点中的行按钮变成点击邻近行。
- form：部分样本输入框点击顺序或焦点不稳，form 总体没有改善。
- choice：select 类动作更稳定，这是本次唯一明确收益。

一个合理解释是：repair 数据提高了模型对 choice/select 的点击倾向，但 token-level SFT loss 没有直接约束 search/table 的旧能力保持。即使加入原 v3 replay，240 step、2e-5 learning rate 对当前 3B LoRA 仍然会造成行为漂移。

## 当前判断

本次小修复达到了“看看效果”的目的，但结果不是正向：

1. 不能用 `qwen25vl_3b_history_aware_sft_v3_form_choice_repair_v1_lora` 作为后续 RL current model。
2. 应继续使用 `qwen25vl_3b_history_aware_sft_v3_1000_aug_lora` 作为 current model。
3. form/choice 修复数据暂时只保留为分析材料，不直接混入下一轮主训练。

## 下一步建议

1. 先用 v3 baseline 做小规模 verifier-guided on-policy RL。
   - `on-policy RL` 指让当前本地模型自己 rollout，再用 verifier reward 更新同一个模型。
   - 这样能直接针对当前模型真实失败分布学习，而不是继续依赖静态 repair SFT。

2. RL 前采集当前模型失败轨迹。
   - 对 v3 baseline 分 family 采样失败 rollout，特别是 form/choice/search/table。
   - 从真实失败中生成 recovery 示范，而不是只从 synthetic repair 任务生成。

3. 如果继续做 SFT 修复，应改成更保守的 constrained repair。
   - repair loss 权重不超过 5%。
   - 每个 repair batch 强制配 search/table replay。
   - learning_rate 降到 `5e-6` 或更低。
   - 每 50 step 跑 family test，按 success_rate 而不是 eval_loss 选 checkpoint。

4. RL 奖励里对 form/choice 加进度 reward，但保留 search/table 成功率作为 early stop。
   - 如果 form/choice 上升但 search/table 掉，就停止更新或回滚。

## 主要命令

构建修复任务：

```bash
python3 scripts/build_browser_rl_task_suite.py \
  --output-dir outputs/browser_rl_repair_form_v1_20260528_1045 \
  --timestamp 20260528_1045 \
  --seed-offset 10000 \
  --counts-json '{"form":120,"search":0,"menu":0,"table":0,"todo":0,"choice":0,"advanced":0}'

python3 scripts/build_browser_rl_task_suite.py \
  --output-dir outputs/browser_rl_repair_choice_v1_20260528_1045 \
  --timestamp 20260528_1045 \
  --seed-offset 10000 \
  --counts-json '{"form":0,"search":0,"menu":0,"table":0,"todo":0,"choice":60,"advanced":0}'
```

构建混合 SFT 数据：

```bash
/opt/conda/envs/llama/bin/python3 scripts/build_local_qwen_history_aware_sft.py \
  --input outputs/browser_rl_task_suite_1000_augmented_oracle_20260528_0104/rollouts.jsonl \
  --input outputs/browser_rl_repair_form_aug_oracle_v1_20260528_1046/rollouts.jsonl \
  --input outputs/browser_rl_repair_choice_aug_oracle_v1_20260528_1046/rollouts.jsonl \
  --tasks outputs/browser_rl_task_suite_1000_20260528_0058/all_tasks.jsonl \
  --tasks outputs/browser_rl_repair_form_v1_20260528_1045/all_tasks.jsonl \
  --tasks outputs/browser_rl_repair_choice_v1_20260528_1045/all_tasks.jsonl \
  --output-dir /root/models/datasets/local_qwen_history_aware_sft_v3_1000_aug_form_choice_repair_v1_20260528_1104 \
  --respect-task-splits \
  --max-history 4
```

训练：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/root/Workspace/VLM/EviTool-VL/third_party/LlamaFactory/src:${PYTHONPATH:-} \
  /opt/conda/envs/llama/bin/python3 -m llamafactory.cli train \
  configs/sft_local_qwen_history_aware_v3_repair_form_choice_qwen25vl_3b_lora.yaml
```

评测：

```bash
CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/llama/bin/python3 scripts/run_browser_rl_rollouts.py \
  --env-source local_smoke \
  --policy local_qwen \
  --tasks outputs/browser_rl_task_suite_1000_20260528_0058/test_tasks.jsonl \
  --limit 100 \
  --max-steps 6 \
  --headless \
  --local-model /root/models/Qwen2.5-VL-3B-Instruct \
  --local-adapter checkpoints/qwen25vl_3b_history_aware_sft_v3_form_choice_repair_v1_lora \
  --local-load-in-4bit \
  --local-max-new-tokens 96 \
  --local-temperature 0 \
  --local-max-history 4 \
  --local-prompt-style full
```
