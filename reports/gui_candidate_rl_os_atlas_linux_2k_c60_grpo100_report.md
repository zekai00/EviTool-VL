# OS-Atlas Linux 2k GUI Candidate RL + GRPO 100-step Report

日期：2026-05-25

## 结论

本轮完成了 2k GUI candidate RL 数据构建、60-candidate 版本候选集生成、SFT-v2 基线评估、以及 100-step GRPO 训练。

最重要结论：**GRPO 管线已跑通，但这次 100 step 没有带来可测提升**。在同一 100 条验证样本上，GRPO 前后模型输出完全一致，prediction diff 为 `0/100`，指标也完全相同：

| Adapter | Avg reward | Valid | Pointing | IoU@0.5 | Avg IoU |
|---|---:|---:|---:|---:|---:|
| SFT-v2 LoRA | 0.3693 | 96.00% | 18.00% | 7.00% | 0.0908 |
| GRPO 100-step LoRA | 0.3693 | 96.00% | 18.00% | 7.00% | 0.0908 |

我不建议在**同一配置**下直接扩到 300 step。训练日志显示大量 step 的 `reward_std=0`、`grad_norm=0`，也就是同一 prompt 的多次采样得分相同，GRPO 没有优势信号可学。下一轮应该先改训练数据过滤、采样探索和 reward 设计，再扩步数。

## 数据

源数据使用 `OS-Copilot/OS-Atlas-data` 的 Linux desktop raw subset：

- JSON：`/root/models/datasets/os_atlas_linux_raw/desktop_domain/linux_splited.json`
- 图片 zip：Hugging Face 远程 zip 按需读取
- 输出：`/root/models/datasets/os_atlas_linux_2k/source_os_atlas_linux_2k.jsonl`
- 图片输出：`/root/models/datasets/os_atlas_linux_2k/images/os_atlas_linux`

抽样策略：

- `sampling_mode=image-random`
- 每张截图最多 `32` 个 target
- 生成 `2000` 行 GUI grounding source rows
- 覆盖 `100` 张唯一截图
- 过滤 tiny boxes：`21`
- 缺失图片成员：`3`

主要元素类型：

| Type | Count |
|---|---:|
| push-button | 744 |
| menu | 191 |
| label | 169 |
| link | 167 |
| tree-item | 125 |
| page-tab | 117 |
| table-cell | 105 |
| toggle-button | 91 |

说明：这是为了先完成 2k 行和 GRPO 试跑而构建的紧凑子集。它不是最终正式大规模 RL 数据集，因为唯一截图数只有 100。

## Candidate 构建

我先构建了 `max_candidates=30`，发现候选召回明显被截断：

| Candidate cap | Split | Avg candidates | Oracle hit | Oracle pointing | Oracle IoU@0.5 | Avg oracle IoU |
|---:|---|---:|---:|---:|---:|---:|
| 30 | val | 26.55 | 52.96% | 52.01% | 43.50% | 0.3715 |
| 60 | val | 39.40 | 62.88% | 61.70% | 52.25% | 0.4426 |

因此正式 GRPO 使用 `max_candidates=60`：

- Candidate output：`outputs/gui_candidate_rl_os_atlas_linux_2k_c60`
- Train / val：`1577 / 423`
- Group split key：`image`
- Unique candidate images：`100`
- Candidate cache hits：`1900`
- Overlay images：`1955`

`group_split_key=image` 的作用：同一截图可能有多个 instruction/target，按 image 分组切分可以避免同一截图同时出现在 train 和 val。

## 基线

60-candidate val 全量 `423` 条：

| Policy | Avg reward | Valid | Pointing | IoU@0.5 | Avg IoU | Avg selected rank |
|---|---:|---:|---:|---:|---:|---:|
| top1 | 0.2988 | 96.69% | 2.60% | 2.13% | 0.0201 | 1.00 |
| random | 0.2954 | 96.69% | 2.13% | 1.65% | 0.0152 | 20.14 |
| oracle | 0.6873 | 96.69% | 61.70% | 52.25% | 0.4426 | 13.92 |

SFT-v2 模型基线只跑了 val 前 `100` 条，作为 GRPO 前后对照：

- Base model：`/root/models/Qwen2.5-VL-3B-Instruct`
- Adapter：`checkpoints/qwen25vl_3b_evitool_sft_v2_tool_gui_lora`
- Avg reward：`0.3693`
- Valid：`96.00%`
- Pointing：`18.00%`
- IoU@0.5：`7.00%`
- Avg IoU：`0.0908`

## GRPO 设置

训练命令核心配置：

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/train_gui_candidate_grpo.py \
  --model /root/models/Qwen2.5-VL-3B-Instruct \
  --adapter checkpoints/qwen25vl_3b_evitool_sft_v2_tool_gui_lora \
  --train-data outputs/gui_candidate_rl_os_atlas_linux_2k_c60/train.jsonl \
  --eval-data outputs/gui_candidate_rl_os_atlas_linux_2k_c60/val.jsonl \
  --output-dir outputs/gui_candidate_rl_os_atlas_linux_2k_c60/grpo_3b_sftv2_g4_t12_100step \
  --limit 1577 \
  --eval-limit 64 \
  --num-generations 4 \
  --max-steps 100 \
  --learning-rate 1e-6 \
  --temperature 1.2 \
  --top-p 0.95
```

实际可训练样本：

- Train examples loaded：`1546`
- Eval examples loaded：`64`
- 少于 1577 的原因：无 candidates / 无 overlay 的行被跳过。

训练结果：

- Output adapter：`outputs/gui_candidate_rl_os_atlas_linux_2k_c60/grpo_3b_sftv2_g4_t12_100step`
- Runtime：`1871.38s`
- Train steps/s：`0.053`
- Train samples/s：`0.214`
- Train loss：`0.001185`
- GPU：RTX 3090，显存峰值接近 `23GB`

训练观察：

- 大量 step 出现 `reward_std=0` 和 `grad_norm=0`。
- 模型 completion 通常是短 JSON，格式问题较少。
- 有少量 step 出现有效优势信号，例如 `reward_std=0.35/0.40`、`pointing=0.25~0.75`。
- 但有效 step 占比不足，100 step 后没有改变 greedy inference 行为。

## GRPO 后评估

同一 val 前 `100` 条：

| Adapter | Avg reward | Valid | Pointing | IoU@0.5 | Avg IoU | Avg selected rank |
|---|---:|---:|---:|---:|---:|---:|
| SFT-v2 LoRA | 0.3693 | 96.00% | 18.00% | 7.00% | 0.0908 | 3.75 |
| GRPO 100-step LoRA | 0.3693 | 96.00% | 18.00% | 7.00% | 0.0908 | 3.75 |

逐条输出对比：

- `prediction_diff = 0/100`
- 也就是 100 条验证样本的模型输出完全一致。

## 指标含义

- `Policy`：评估策略。`top1` 永远选第一个候选，`random` 随机选，`oracle` 从候选里选与 GT 最匹配的候选，`model` 使用 VLM 生成 candidate id。
- `Adapter`：LoRA 权重路径。这里对比的是 SFT-v2 LoRA 与 GRPO 后 LoRA。
- `Valid`：输出的 candidate id 是否存在于候选列表里。
- `Pointing`：所选候选框中心点是否落入 GT bbox。
- `IoU@0.5`：所选候选框与 GT bbox 的 IoU 是否大于等于 0.5。
- `Avg IoU`：所选候选框与 GT bbox 的平均 IoU。
- `Avg reward`：当前 reward 函数总分，包含格式、候选合法性、pointing、IoU@0.5 和 IoU shaped reward。
- `Oracle`：在候选集合固定的情况下理论上可达到的上限，不代表模型能力。

## 主要问题

1. Candidate 召回仍然限制上限。

   60-candidate oracle 的 val IoU@0.5 只有 `52.25%`。OS-Atlas 含大量文字、menu、label、table-cell 等目标，而当前候选主要来自 OmniParser icon detector。即使模型选择完美，也无法超过候选召回上限。

2. GRPO 采样缺少有效方差。

   当前模型经常在同一 prompt 的 4 次采样中输出相同或同分 candidate，导致 `reward_std=0`，GRPO 没有优势信号。

3. 训练数据里包含不少低可学样本。

   一些样本的 oracle reward 也接近 `0.3`，只有“格式合法 + candidate 合法”分，空间定位奖励很弱。它们会稀释训练信号。

4. 100 张唯一截图不足以作为最终正式 RL 数据集。

   这批适合先跑通 pipeline 和定位算法问题，不适合作为最终结论。

## 下一步建议

不要直接同配置跑 300 step。建议先做以下改造：

1. 构建 filtered RL train set。

   只保留 `oracle_hit=True` 或 `oracle_iou50=True` 的训练样本，先让 candidate selector 学会在“答案存在于候选中”的场景做离散选择。val 保留全量和 filtered 两套。

2. 改 GRPO 采样策略。

   提高探索：`temperature=1.5~2.0`、更高 `top_p`、必要时增加 `num_generations`，或加入 candidate-id constrained decoding，避免模型只输出 `c00/c01`。

3. 做 pairwise/listwise reward 增强。

   对 oracle candidate 与 model candidate 的 rank 差、IoU 差做 shaped reward；同时对反复输出 top1 但不命中的行为加轻惩罚。

4. 增强候选生成。

   加 OCR/text boxes 或 OmniParser OCR/caption 版本，把 label/menu/table-cell 目标纳入候选，否则 OS-Atlas 的文字类任务上限会被候选召回卡死。

5. 再跑 300 step。

   在 filtered set + 更强探索 + 更高候选召回后，再跑 `300 step` 才有意义。当前配置已经证明 100 step 无可测提升。

## 产物路径

- Source JSONL：`/root/models/datasets/os_atlas_linux_2k/source_os_atlas_linux_2k.jsonl`
- Source summary：`/root/models/datasets/os_atlas_linux_2k/source_os_atlas_linux_2k.summary.json`
- Candidate RL 30：`outputs/gui_candidate_rl_os_atlas_linux_2k`
- Candidate RL 60：`outputs/gui_candidate_rl_os_atlas_linux_2k_c60`
- GRPO adapter：`outputs/gui_candidate_rl_os_atlas_linux_2k_c60/grpo_3b_sftv2_g4_t12_100step`
- Pre-GRPO model eval：`reports/gui_candidate_policy_os_atlas_linux_2k_c60_3b_sftv2_val100.md`
- Post-GRPO model eval：`reports/gui_candidate_policy_os_atlas_linux_2k_c60_3b_grpo100_val100.md`
