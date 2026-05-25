# GUI Candidate RL Five-Step Report

日期：2026-05-26

## 总结论

这轮五步工作已经完成：候选 RL 数据整理、reward 改造、SFT warmup、100-step GRPO 训练、以及 GRPO 后验证。

最重要结论：**GUI candidate RL 管线已经跑通，但这次 100 step GRPO 没有带来可测提升**。在同一 val 前 100 条上：

| Adapter | Avg reward | Valid | Pointing | IoU@0.5 | Avg IoU | Avg rank |
|---|---:|---:|---:|---:|---:|---:|
| SFT warmup LoRA | 0.2756 | 96.00% | 19.00% | 7.00% | 0.0934 | 4.9583 |
| SFT warmup + GRPO g2 100-step | 0.2731 | 96.00% | 18.00% | 7.00% | 0.0911 | 5.1354 |

因此我不建议直接在当前设置上继续堆 300 step。当前主要问题不是训练脚本没跑通，而是：

- 候选召回上限仍低，60-candidate val 的 oracle IoU@0.5 只有 `52.25%`。
- 24GB 单卡下 `num_generations=4` 容易 OOM，降到 `num_generations=2` 后 GRPO 的组内奖励方差偏弱。
- 高温探索会破坏 JSON/candidate-id 格式，低温探索又不够。
- 训练数据只有 `100` 张唯一截图，适合打通流程，不适合作为正式 RL 结论。

## Step 1：数据与候选集

源数据仍使用 OS-Atlas Linux 2k 子集：

- Source：`/root/models/datasets/os_atlas_linux_2k/source_os_atlas_linux_2k.jsonl`
- 样本数：`2000`
- 唯一截图数：`100`
- 原始 candidate RL：`outputs/gui_candidate_rl_os_atlas_linux_2k_c60`
- Train / val：`1577 / 423`
- Candidate cap：`60`

原始 60-candidate val 上限：

| Split | Oracle hit | Oracle pointing | Oracle IoU@0.5 | Avg oracle IoU |
|---|---:|---:|---:|---:|
| val 423 | 62.88% | 61.70% | 52.25% | 0.4426 |

我额外构建了 fused OCR + OmniParser ablation：

- Output：`outputs/gui_candidate_rl_os_atlas_linux_2k_fused_c80`
- Candidate cap：`80`
- OCR：EasyOCR
- Val oracle hit：`52.72%`
- Val oracle IoU@0.5：`31.44%`
- Val avg oracle IoU：`0.3217`

结论：fused c80 这版比原始 c60 更差，不能作为正式训练集，只保留为反例/ablation。

随后我从 c60 中过滤出高信号训练集：

- Output：`outputs/gui_candidate_rl_os_atlas_linux_2k_c60_filtered_hit`
- Train：`1577 -> 907`
- Val：保持全量 `423`
- 过滤条件：有 overlay、有 candidates、oracle hit、oracle IoU > 0.05

过滤后 train 上限：

| Split | Rows | Oracle hit | Oracle pointing | Oracle IoU@0.5 | Avg oracle IoU |
|---|---:|---:|---:|---:|---:|
| train filtered | 907 | 100.00% | 98.02% | 83.13% | 0.6961 |

这一步的目的不是美化 val，而是让训练先集中在“答案确实存在于候选集合里”的样本上，否则 RL reward 会被大量不可学样本稀释。

## Step 2：Reward 与数据工具

新增了 `reward_version=v2`，核心在 `rl/gui_candidate_env.py`：

- `format`：JSON 格式正确，`+0.05`
- `valid`：candidate id 存在，`+0.10`
- `pointing`：候选中心点落入 GT bbox，`+0.30`
- `IoU@0.5`：候选与 GT 的 IoU >= 0.5，`+0.25`
- `IoU shaped`：连续 IoU 奖励，最高 `+0.15`
- `center shaped`：中心距离越近奖励越高，最高 `+0.15`
- `first wrong penalty`：反复选 top1 但不命中时轻惩罚 `-0.05`

相关脚本改动：

- `scripts/build_gui_candidate_rl_data.py`：支持 `--generator fused-ui`、OCR/OmniParser 融合、query-aware cache。
- `scripts/filter_gui_candidate_rl_data.py`：新增，用于过滤高信号 RL 样本。
- `scripts/prepare_gui_candidate_sft_data.py`：新增，把 candidate RL 数据转成 ShareGPT 风格 SFT 数据。
- `scripts/train_gui_candidate_grpo.py`：支持 `--reward-version v1/v2`。
- `scripts/eval_gui_candidate_policy.py`：支持 `--reward-version v1/v2`。

代码里已经给 reward、candidate 融合、过滤条件和 SFT 目标构建加了注释，后续继续写算法代码时会保持这种注释粒度。

## Step 3：SFT Warmup

SFT 数据：

- Output：`/root/models/datasets/gui_candidate_sft_os_atlas_c60_filtered`
- Train items：`907`
- Val items：`266`
- Val items 少于 423 的原因：SFT 目标需要 oracle candidate，val 中无 oracle hit 的样本不能作为监督目标。

训练配置：

- Config：`configs/sft_gui_candidate_warmup_qwen25vl_3b_lora.yaml`
- Base：`/root/models/Qwen2.5-VL-3B-Instruct`
- Init adapter：`checkpoints/qwen25vl_3b_evitool_sft_v2_tool_gui_lora`
- Output adapter：`checkpoints/qwen25vl_3b_gui_candidate_c60_sft_warmup_lora`
- Epoch：`1`
- Batch：`1`
- Grad accumulation：`8`
- LoRA rank：`16`
- 4-bit quant：开启

训练结果：

- Runtime：`1776.04s`
- Train loss：`0.3789`
- Train samples/s：`0.511`
- Train steps/s：`0.064`

Warmup 后 val100：

| Metric | Value |
|---|---:|
| Avg reward | 0.2756 |
| Parseable | 100.00% |
| Valid candidate | 96.00% |
| Pointing | 19.00% |
| IoU@0.5 | 7.00% |
| Avg IoU | 0.0934 |
| Avg selected rank | 4.9583 |

结论：warmup 没有明显提升定位指标，但它让输出格式稳定，适合作为 GRPO 起点。

## Step 4：GRPO 训练

我尝试了三组 GRPO 配置：

| 配置 | 结果 | 结论 |
|---|---|---|
| g4, temperature 1.8, len 48 | 手动停止于约 14 step | 探索太强，JSON/candidate 格式被破坏，reward 变负 |
| g4, temperature 1.0, len 16 | step 5 OOM | 格式和 reward 信号较好，但 24GB 单卡承载不了 |
| g2, temperature 1.0, len 16 | 完成 100 step | 稳定可跑，但组内方差不足 |

最终完成的 GRPO：

- Output adapter：`outputs/gui_candidate_rl_os_atlas_linux_2k_c60/grpo_3b_candidate_c60_sft_rewardv2_g2_t10_len16_100step`
- Train data：`outputs/gui_candidate_rl_os_atlas_linux_2k_c60_filtered_hit/train.jsonl`
- Eval data：`outputs/gui_candidate_rl_os_atlas_linux_2k_c60/val.jsonl`
- Steps：`100`
- Num generations：`2`
- Temperature：`1.0`
- Max completion length：`16`
- Learning rate：`2e-6`
- Reward：`v2`

训练日志汇总：

| Metric | Value |
|---|---:|
| Runtime | 990.37s |
| Train loss | -0.0011 |
| Avg train reward | 0.3235 |
| Avg reward std | 0.1011 |
| Zero-std rows | 36 / 100 |
| Avg valid | 99.00% |
| Avg pointing | 19.50% |
| Avg IoU@0.5 | 16.00% |
| Clipped ratio | 0.00% |
| Avg completion length | 9.56 |

解释：`Zero-std rows = 36 / 100` 表示 36 个 step 的两个 sampled completions 得分完全相同。GRPO 依赖组内 reward 差异计算 advantage，这种情况下即使 loss 正常记录，也没有多少有效学习信号。

## Step 5：最终验证与算法判断

GRPO 后 val100：

- Report：`reports/gui_candidate_policy_os_atlas_linux_2k_c60_3b_candidate_sft_grpo_g2_rewardv2_val100.md`
- Eval output：`outputs/gui_candidate_rl_os_atlas_linux_2k_c60/eval_val_100_3b_candidate_c60_sft_grpo_g2_rewardv2_model.jsonl`

| Metric | Value |
|---|---:|
| Avg reward | 0.2731 |
| Parseable | 100.00% |
| Valid candidate | 96.00% |
| Pointing | 18.00% |
| IoU@0.5 | 7.00% |
| Avg IoU | 0.0911 |
| Avg selected rank | 5.1354 |

和 warmup 相比，GRPO 后略降，不构成有效提升。我的判断是：**当前任务不适合直接用普通 sequence-level GRPO 继续硬跑**。模型输出空间表面上是文本，实际动作空间是离散 candidate id；把它当普通文本生成来 RL，会同时遇到格式噪声、探索不足、显存压力和低 reward 方差。

## 下一步建议：改算法，而不是只加 step

我建议下一轮做一个任务定制版本，暂定叫 **Candidate-Constrained GRPO**：

1. 约束动作空间。

   训练时不再让模型自由生成任意文本，而是把动作限制为当前样本的候选 id，例如 `c00...c59`。这样可以去掉 JSON 格式噪声，并允许更高温度或更多 group samples。

2. 用候选级 reward，而不是纯文本 completion reward。

   每个 action 直接对应一个 candidate bbox，reward 可以稳定计算：valid、pointing、IoU、center distance、rank penalty。这样 reward 方差来自候选质量差异，而不是来自模型是否输出合法 JSON。

3. 构造 hard-negative group。

   每个训练样本保留 oracle candidate，并采样若干相似但错误的候选：中心接近但不命中、IoU 接近但不足 0.5、rank 很靠前但错误、OCR/text 相似但位置错误。这样模型学的是候选比较，而不是记住常见 id。

4. 做 rank-balanced sampling。

   现在模型经常偏向 `c00/c01`。训练 batch 应平衡 oracle rank，让 oracle 分布覆盖前、中、后位置，避免模型把候选 id 当位置先验。

5. 扩大正式数据。

   当前 `2k rows / 100 images` 只够 pipeline 验证。正式 RL 至少应该扩到更多唯一截图，优先目标是 `10k+ rows` 和 `1k+ unique images`，并保留两套验证集：全量 val 看真实能力，oracle-hit val 看选择器能力。

6. 先提升 candidate coverage。

   如果候选集合里没有正确框，RL 选得再好也没用。下一步应优先把 OS-Atlas 的 label/menu/table-cell/text 类目标召回提升到更高，再做长步数 RL。

## 当前产物

- Filtered RL train：`outputs/gui_candidate_rl_os_atlas_linux_2k_c60_filtered_hit/train.jsonl`
- Filtered/full val：`outputs/gui_candidate_rl_os_atlas_linux_2k_c60_filtered_hit/val.jsonl`
- SFT dataset：`/root/models/datasets/gui_candidate_sft_os_atlas_c60_filtered`
- SFT warmup adapter：`checkpoints/qwen25vl_3b_gui_candidate_c60_sft_warmup_lora`
- GRPO adapter：`outputs/gui_candidate_rl_os_atlas_linux_2k_c60/grpo_3b_candidate_c60_sft_rewardv2_g2_t10_len16_100step`
- Warmup eval report：`reports/gui_candidate_policy_os_atlas_linux_2k_c60_3b_candidate_sft_warmup_val100.md`
- GRPO eval report：`reports/gui_candidate_policy_os_atlas_linux_2k_c60_3b_candidate_sft_grpo_g2_rewardv2_val100.md`
