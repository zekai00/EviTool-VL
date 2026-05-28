# 本地 Qwen Action SFT Warmup 与 Smoke 训练报告

- 生成时间：2026-05-27 22:34 CST
- 工作区：`/root/Workspace/VLM`
- 仓库：`/root/Workspace/VLM/EviTool-VL`
- 主线目标：让本地 Qwen2.5-VL/Qwen3-VL 成为真正 current model，而不是继续依赖 DashScope teacher。

## 1. 本轮结论

本轮完成了三个关键步骤：

1. 新增 `rollout schema validator`（轨迹格式检查器），并验证当前 30 条 rollout 全部合法。
2. 从成功 rollout 中构建本地 Qwen action JSON SFT warmup 数据，共 90 条单步样本。
3. 使用本地 `Qwen2.5-VL-3B-Instruct` 跑通 1 step LoRA SFT smoke，证明本地模型训练链路可用。

这里的 `SFT` 指监督微调：让本地模型先模仿正确 action JSON。它不是最终 RL，但它是进入 on-policy RL 前的必要 warmup。

## 2. 新增文件

1. `scripts/validate_browser_rl_rollouts.py`
   - 输入：一个或多个 `rollouts.jsonl`。
   - 输出：
     - `summary.json`
     - `errors.jsonl`
     - `valid_rollouts.jsonl`
   - 检查内容：top-level 字段、trajectory、screenshot 文件、action parse、exec_status、reward、verifier、policy_info。

2. `scripts/build_local_qwen_action_sft_warmup.py`
   - 输入：一个或多个通过 verifier 的 rollout 文件。
   - 默认只收 `success=true` 的 rollout。
   - 输出 LLaMAFactory sharegpt 格式数据，包括 `dataset_info.json`。

3. `configs/sft_local_qwen_action_warmup_qwen25vl_3b_lora.yaml`
   - 正式 warmup 配置，目标模型：`/root/models/Qwen2.5-VL-3B-Instruct`。

4. `configs/sft_local_qwen_action_warmup_qwen3vl_4b_lora.yaml`
   - 正式 warmup 配置，目标模型：`/root/models/Qwen3-VL-4B-Instruct`。

5. `configs/sft_local_qwen_action_warmup_qwen25vl_3b_lora_smoke.yaml`
   - 1 step smoke 配置，只验证训练链路。

## 3. Rollout Schema Validation

执行命令：

```bash
python3 scripts/validate_browser_rl_rollouts.py \
  --input outputs/browser_rl_smoke_rollouts_20260527_1w/rollouts.jsonl \
  --input outputs/rollouts_local_qwen_dashscope_eval5_20260527_2206/rollouts.jsonl \
  --input outputs/rollouts_miniwob_qwen_dashscope_eval5_20260527_2206/rollouts.jsonl \
  --output-dir outputs/browser_rl_rollout_validation_20260527_2224
```

结果：

| 指标 | 数值 |
|---|---:|
| rollouts | 30 |
| valid_rollouts | 30 |
| error_count | 0 |
| success_rate | 0.9 |
| steps | 105 |
| exec_ok_step_rate | 1.0 |
| action_parse_step_rate | 1.0 |
| screenshot_field_rate | 1.0 |
| qwen_steps | 45 |
| qwen_valid_json_rate | 1.0 |
| qwen_valid_action_rate | 1.0 |
| unique_tasks | 25 |

输出目录：

```text
outputs/browser_rl_rollout_validation_20260527_2224
```

## 4. 本地 Qwen Action SFT Warmup 数据

执行命令：

```bash
python3 scripts/build_local_qwen_action_sft_warmup.py \
  --input outputs/browser_rl_smoke_rollouts_20260527_1w/rollouts.jsonl \
  --input outputs/rollouts_local_qwen_dashscope_eval5_20260527_2206/rollouts.jsonl \
  --input outputs/rollouts_miniwob_qwen_dashscope_eval5_20260527_2206/rollouts.jsonl \
  --output-dir /root/models/datasets/local_qwen_action_sft_warmup_20260527_2224 \
  --require-success \
  --val-ratio 0.1 \
  --seed 42 \
  --absolute-image-paths \
  --dataset-name local_qwen_action_sft_train \
  --val-dataset-name local_qwen_action_sft_val
```

结果：

| 指标 | 数值 |
|---|---:|
| rows | 90 |
| train_rows | 81 |
| val_rows | 9 |
| unique_tasks | 22 |
| scripted_oracle rows | 60 |
| qwen_dashscope teacher rows | 30 |
| skipped rollout_not_success | 3 |
| missing images | 0 |

输出目录：

```text
/root/models/datasets/local_qwen_action_sft_warmup_20260527_2224
```

主要文件：

```text
dataset_info.json
local_qwen_action_sft_train.json
local_qwen_action_sft_val.json
messages.json
summary.json
```

## 5. 本地模型检查

已发现本地模型：

```text
/root/models/Qwen2.5-VL-3B-Instruct
/root/models/Qwen3-VL-4B-Instruct
/root/models/Qwen3-VL-8B-Instruct
```

本轮优先用 `Qwen2.5-VL-3B-Instruct` 做 smoke，因为它最轻，适合先验证训练链路。

## 6. Qwen2.5-VL-3B LoRA SFT Smoke

执行命令：

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/root/Workspace/VLM/EviTool-VL/third_party/LlamaFactory/src:${PYTHONPATH:-} \
python3 -m llamafactory.cli train \
  configs/sft_local_qwen_action_warmup_qwen25vl_3b_lora_smoke.yaml
```

结果：

| 指标 | 数值 |
|---|---:|
| max_steps | 1 |
| train samples | 4 |
| trainable params | 14,966,784 |
| all params | 3,769,589,760 |
| trainable% | 0.3970 |
| train_loss | 2.6845 |
| train_runtime | 3.04s |
| checkpoint saved | true |

输出 checkpoint：

```text
checkpoints/qwen25vl_3b_local_action_sft_warmup_smoke
```

这个 smoke 证明：

1. 数据格式能被 LLaMAFactory 读取。
2. 图片路径能加载。
3. Qwen2.5-VL tokenizer/template 能处理 action JSON 数据。
4. 4-bit LoRA 初始化成功。
5. 单步反传和 checkpoint 保存成功。

## 7. 当前还没有完成什么

1. 还没有跑完整 SFT warmup。
   - 当前只跑了 1 step smoke。
   - 下一步可以跑完整 90 条小数据 warmup。

2. 还没有 `local_qwen` rollout wrapper。
   - 本地 checkpoint 还没有接回 `run_browser_rl_rollouts.py`。
   - 这一步完成后，本地模型才能成为真正 current model。

3. 还没有 GRPO/RL。
   - 需要先用本地 SFT checkpoint 做 rollout baseline。

## 8. 下一步建议

1. 跑完整 Qwen2.5-VL-3B action JSON SFT warmup。
2. 实现 `--policy local_qwen`，加载本地 base + LoRA checkpoint 进行 rollout。
3. 用本地 current model 跑 20 个 Playwright smoke tasks 和 5-20 个 MiniWoB++ tasks。
4. 根据本地 rollout 的 valid JSON/action rate 和 success rate，再决定是否进入 Verifier-Guided GRPO。
