# 本地 Qwen 完整 SFT 与 Local Policy 接入评测报告

- 时间：2026-05-27 23:02 CST
- 仓库：`/root/Workspace/VLM/EviTool-VL`
- 主线：真正 on-policy VLM agentic RL

## 术语说明

- `SFT`：监督微调，先让模型模仿专家动作，学会输出合法的 GUI action JSON。
- `LoRA`：低秩适配训练，只训练少量 adapter 参数，不全量更新底模。
- `Local policy`：本地策略，指本地 Qwen2.5-VL/Qwen3-VL checkpoint 自己看截图并输出动作。
- `Current model`：当前模型，指后续真正要被 SFT/RL 更新参数的本地模型，不是 DashScope 云端 teacher。
- `Rollout`：一次完整尝试轨迹，从 reset 后看截图、输出动作、执行、拿 reward，直到成功或超步数。
- `Verifier`：验证器，用程序判断任务是否完成，并返回 reward。

## 本次完成

1. 完成 90 条 full warmup SFT。
   - 数据集：`/root/models/datasets/local_qwen_action_sft_warmup_20260527_2224`
   - rows: 90
   - train_rows: 81
   - val_rows: 9
   - 数据来源：scripted oracle 60 条、DashScope Qwen teacher 30 条
   - checkpoint：`checkpoints/qwen25vl_3b_local_action_sft_warmup_lora`
   - global_step: 11
   - epoch: 1.0
   - train_loss: 1.4807

2. 接入 `--policy local_qwen`。
   - 新增：`envs/browser_rl/local_qwen_policy.py`
   - 修改：`envs/browser_rl/__init__.py`
   - 修改：`scripts/run_browser_rl_rollouts.py`
   - 能加载本地 `/root/models/Qwen2.5-VL-3B-Instruct` + LoRA adapter。
   - 能把截图和目标转成 Qwen-VL 输入，生成动作 JSON，再交给 Playwright 环境执行。

3. 增加本地策略提示词模式。
   - `sft_minimal`：和 SFT 数据提示词一致，格式为 `<image>\n任务：...\n请输出下一步 GUI action JSON。`
   - `full`：包含动作空间、坐标规范、step、history 等 rollout 信息。

4. 额外构建 clean-oracle debug 数据并训练。
   - 目的：排除 DashScope teacher 数据噪声，检查本地 Qwen 是否能在干净 scripted oracle 数据上学到基本坐标/动作。
   - 数据集：`/root/models/datasets/local_qwen_action_sft_oracle_smoke_20260527_2300`
   - rows: 60
   - train_rows: 54
   - val_rows: 6
   - checkpoint：`checkpoints/qwen25vl_3b_local_action_sft_oracle_smoke_lora`
   - global_step: 35
   - epoch: 5.0
   - train_loss: 0.7441

## 运行命令

完整 full warmup SFT：

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/root/Workspace/VLM/EviTool-VL/third_party/LlamaFactory/src:${PYTHONPATH:-} \
python3 -m llamafactory.cli train \
  configs/sft_local_qwen_action_warmup_qwen25vl_3b_lora.yaml
```

本地 current model rollout：

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_browser_rl_rollouts.py \
  --env-source local_smoke \
  --policy local_qwen \
  --tasks outputs/browser_rl_smoke_tasks_20260527_1w/all_tasks.jsonl \
  --output-dir outputs/rollouts_local_qwen25vl_sft_system_20260527_2258 \
  --limit 1 \
  --max-steps 6 \
  --headless \
  --local-model /root/models/Qwen2.5-VL-3B-Instruct \
  --local-adapter checkpoints/qwen25vl_3b_local_action_sft_warmup_lora \
  --local-load-in-4bit \
  --local-prompt-style sft_minimal
```

clean-oracle debug SFT：

```bash
python3 scripts/build_local_qwen_action_sft_warmup.py \
  --input outputs/browser_rl_smoke_rollouts_20260527_1w/rollouts.jsonl \
  --output-dir /root/models/datasets/local_qwen_action_sft_oracle_smoke_20260527_2300 \
  --include-policies scripted_oracle \
  --val-ratio 0.1 \
  --seed 42 \
  --dataset-name local_qwen_action_sft_train \
  --val-dataset-name local_qwen_action_sft_val

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/root/Workspace/VLM/EviTool-VL/third_party/LlamaFactory/src:${PYTHONPATH:-} \
python3 -m llamafactory.cli train \
  configs/sft_local_qwen_action_warmup_qwen25vl_3b_lora_oracle_smoke.yaml
```

## 评测结果

| checkpoint | prompt | rollout | success_rate | valid_json_rate | valid_action_rate | error_rate | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| full warmup | full | 1 | 0.0 | 1.0000 | 1.0000 | 0.0000 | 接口通，坐标偏上 |
| full warmup | sft_minimal | 1 | 0.0 | 1.0000 | 1.0000 | 0.0000 | 反复点击错误位置 |
| full warmup + system | sft_minimal | 1 | 0.0 | 1.0000 | 1.0000 | 0.0000 | system 对结果无明显改善 |
| clean oracle | sft_minimal | 1 | 0.0 | 1.0000 | 1.0000 | 0.0000 | 反复输出 type，未先 click |
| clean oracle | full | 1 | 0.0 | 0.8333 | 0.8333 | 0.1667 | 有一步 JSON 语法错误 |

关键输出目录：

- `outputs/rollouts_local_qwen25vl_sft_warmup_20260527_2248`
- `outputs/rollouts_local_qwen25vl_sft_minimal_20260527_2254`
- `outputs/rollouts_local_qwen25vl_sft_system_20260527_2258`
- `outputs/rollouts_local_qwen25vl_oracle_smoke_20260527_2300`
- `outputs/rollouts_local_qwen25vl_oracle_smoke_fullprompt_20260527_2301`

## 失败样例

任务：`Fill Name with user_1, Code with pass-17, then submit.`

full warmup + `sft_minimal` 输出：

```json
{"action":"click","coordinate":[238,69]}
```

解析后动作合法，但坐标落在页面上方空白区域，后续输入没有进入文本框。

clean-oracle 输出：

```json
{"action":"type","text":"user_1"}
```

动作合法，但初始页面没有聚焦输入框，所以 type 不产生任务进展。

## 判断

1. `local_qwen` 接入已经完成。
   - 本地 Qwen2.5-VL checkpoint 能作为 current model 被 rollout 脚本调用。
   - 环境能执行本地模型输出动作，并记录 reward、screenshot、policy_info。

2. 当前模型还没有学会任务。
   - 1-step action JSON SFT 能提高合法 JSON 输出率。
   - 但模型还不能稳定判断“当前 step 应该 click 还是 type”，也不能稳定定位输入框坐标。

3. 现在不应直接进入大规模 RL。
   - 如果直接 on-policy RL，当前模型会产生大量 0 reward 轨迹，学习信号太稀疏。
   - 更合理的是先构建 history-aware trajectory SFT，再进入 curriculum on-policy rollout。

## 下一步计划

1. 重建 history-aware SFT 数据。
   - `history-aware` 指提示词里包含 step、历史动作、上一轮 verifier progress、动作空间和坐标规则。
   - 训练目标仍然是下一步 direct GUI action JSON。
   - 这样模型能学会“先 click 聚焦，再 type，再 submit”，不是只学单张截图到单个动作。

2. 扩大本地可 verifier 任务。
   - 从 20 个 smoke tasks 扩到 100-300 个 browser tasks。
   - 每个任务必须有 reset、step、screenshot、verifier、reward。

3. 加动作后处理和格式鲁棒解析。
   - 修复常见输出如 `"y=343.0"` 这种近似 JSON。
   - 对 `coordinate`、`point`、`bbox` 等模型常见字段做稳定归一化。

4. 做 local current model baseline。
   - 在 history-aware SFT 后，用本地 Qwen checkpoint 跑 20-50 个 local_smoke/browsergym tasks。
   - 指标至少包括 success_rate、valid_json_rate、valid_action_rate、avg_reward、avg_steps。

5. 再进入 on-policy RL。
   - 先用 easy curriculum，让模型能拿到非零 reward。
   - 再接 Verifier-Guided GRPO，而不是从当前 0 success 状态直接做大规模 RL。
