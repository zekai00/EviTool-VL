# Codex Worklog

更新日期：2026-05-27

## 2026-05-27 13:36 CST 更新：候选池增强 + full-aware 重建完成

本阶段按用户确认的顺序执行：先增强候选池（本地 + VLM），再重建更高召回 full-aware 数据，最后做质量评测；真实 current-model full-candidate score cache 留到下一阶段。

完成内容：

- 新增并使用 `scripts/build_gui_candidate_enhanced_data.py` 构建 local enhanced c100 候选池。
- 新增并使用 `scripts/augment_gui_candidates_dashscope.py` 调用 DashScope `gui-plus-2026-02-26` 给低 IoU 样本补候选框。
- 新增并使用 `scripts/diagnose_gui_candidate_misses.py` 输出 c60/local/DashScope 阶段 miss diagnosis。
- 用最终增强候选池重建 full-aware action 数据。
- 将 `.env` / `.env.*` 加入 `.gitignore`，避免 API key 被误纳入版本管理。
- 在 `/root/Workspace/VLM` 输出最终报告，后续已归档为：`项目文档/03_实验与训练报告/GUI候选池增强与Full-Aware数据集报告_20260527_1336.md`。

最终候选池：

- 路径：`outputs/gui_candidate_rl_os_atlas_linux_2k_enhanced_c100_base_dashscope_20260527_1210`
- `all`: 2000 rows, avg candidates 88.7125, oracle hit 83.15%, IoU@0.5 56.10%, avg IoU 0.5208
- `train`: 1577 rows, avg candidates 91.1249, oracle hit 82.44%, IoU@0.5 55.10%, avg IoU 0.5162
- `val`: 423 rows, avg candidates 79.7187, oracle hit 85.82%, IoU@0.5 59.81%, avg IoU 0.5379
- DashScope 合计调用 960 条，追加 1240 个框，错误数 0

最终 full-aware 数据：

- 路径：`outputs/gui_candidate_rl_os_atlas_linux_2k_enhanced_c100_base_dashscope_full_aware_20260527_1329`
- `all.jsonl`: 2000 rows
- `train.jsonl`: 1294 rows
- `val.jsonl`: 423 rows
- `val_oracle_hit.jsonl`: 362 rows
- train avg actions 15.9776, reward std 0.2185, zero std rate 0
- train/val image overlap 0
- `scripts/train_gui_candidate_cc_grpo.py --dry-run` 通过

主要评测结论：

- c60 baseline val oracle hit: 62.88%, IoU@0.5: 52.25%
- local + base_c60 val oracle hit: 78.72%, IoU@0.5: 54.85%
- local + base_c60 + DashScope val oracle hit: 85.82%, IoU@0.5: 59.81%
- 最大剩余瓶颈是 `table-cell`：final val 21 条，oracle hit 4.76%，IoU@0.5 0.00%
- `menu` / `menu-item` / `check-menu-item` 多数能点中，但 tight IoU 仍不足

当前实验命令：

```bash
python3 scripts/build_gui_candidate_full_aware_data.py \
  --input-dir outputs/gui_candidate_rl_os_atlas_linux_2k_enhanced_c100_base_dashscope_20260527_1210 \
  --output-dir outputs/gui_candidate_rl_os_atlas_linux_2k_enhanced_c100_base_dashscope_full_aware_20260527_1329 \
  --report reports/gui_candidate_full_aware_dataset_enhanced_c100_base_dashscope_20260527_1329.md \
  --max-actions 16 \
  --min-train-oracle-iou 0.05 \
  --require-train-oracle-hit \
  --seed 42
```

未解决问题：

- 当前 full-aware 数据仍没有真实 current-model full-candidate score cache，`model_score_coverage = 0.0`。
- `table-cell` 需要专门的 table/grid 候选生成或更强的结构化识别，否则仍会限制召回上限。
- menu 类样本需要 tight box 后处理，否则点击命中可以提升，但 IoU@0.5 指标仍受限。

下一步计划：

1. 基于最终增强 full-aware 数据跑 current-model full-candidate score cache。
2. 用 `--model-score-jsonl` 重建 full-aware 数据，把 proxy false positives 替换成真实 current-model false positives。
3. 跑 listwise warmup。
4. 缓存 reference logprobs。
5. 跑 KL + CC-GRPO。
6. 在 val100/val423 full candidates 上和旧 c60 SFT/GRPO 对比。

## 2026-05-27 14:11 CST 更新：Menu/Table targeted refine 完成

本阶段针对 `table-cell`、`menu`、`check-menu-item`、`menu-item` 做候选池后处理增强，并重建 full-aware 数据。

新增脚本：

- `scripts/refine_gui_candidate_menu_table.py`

新增报告：

- `/root/Workspace/VLM/项目文档/03_实验与训练报告/GUI候选池MenuTable专项增强报告_20260527_1411.md`

最终 refined 候选池：

- 路径：`outputs/gui_candidate_rl_os_atlas_linux_2k_enhanced_c100_base_dashscope_menu_table_refined_20260527_1407`
- `all`: 2000 rows, avg candidates 92.1885, oracle hit 87.60%, IoU@0.5 66.30%, avg IoU 0.5955
- `train`: 1577 rows, avg candidates 94.9486, oracle hit 87.00%, IoU@0.5 65.44%, avg IoU 0.5903
- `val`: 423 rows, avg candidates 81.8983, oracle hit 89.83%, IoU@0.5 69.50%, avg IoU 0.6151

四类 targeted val 指标：

- `table-cell`: oracle hit 71.43%, IoU@0.5 71.43%, avg IoU 0.7040
- `menu`: oracle hit 100.00%, IoU@0.5 90.00%, avg IoU 0.8966
- `check-menu-item`: oracle hit 100.00%, IoU@0.5 100.00%, avg IoU 0.7986
- `menu-item`: oracle hit 100.00%, IoU@0.5 80.00%, avg IoU 0.5991

最终 refined full-aware 数据：

- 路径：`outputs/gui_candidate_rl_os_atlas_linux_2k_enhanced_c100_base_dashscope_menu_table_refined_full_aware_20260527_1407`
- `train.jsonl`: 1367 rows
- `val.jsonl`: 423 rows
- `val_oracle_hit.jsonl`: 379 rows
- train/val image overlap 0
- reward zero-std rate 0
- CC-GRPO dry-run 通过

当前未解决问题：

- 真实 current-model full-candidate score cache 仍未生成，`model_score_coverage = 0.0`。
- refined 候选数增加到 all avg 92.19，后续 full-candidate scoring 成本会更高。
- 非 Calc 格式的 `table-cell` 还需要更通用的表格结构解析。

## 本轮完成内容

0. 更新 `AGENTS.md`。
   - 保留用户新增的“先检查 git status / 读 worklog / 阶段完成更新 worklog”方向。
   - 修正仓库根目录说明：工作区是 `/root/Workspace/VLM`，仓库是 `/root/Workspace/VLM/EviTool-VL`。
   - 补充当前 full-aware GUI candidate RL 主线、`rl/` 模块、full-aware 数据构建命令和质量检查要求。
   - 删除/替换过时的笼统项目结构描述。

1. 核查 `rl/gui_candidate_env.py` 冲突状态。
   - `git diff --name-only --diff-filter=U` 无输出。
   - `git ls-files -u` 无输出。
   - 未发现 `<<<<<<< / ======= / >>>>>>>` 冲突标记。
   - `python3 -m py_compile rl/gui_candidate_env.py` 通过。
   - 结论：该文件当前没有未解决的 git merge conflict，也没有语法错误。

2. 构建 OS-Atlas Linux 2k full-aware GUI candidate RL 数据。
   - 基础输入：`outputs/gui_candidate_rl_os_atlas_linux_2k_c60`
   - 输出目录：`outputs/gui_candidate_rl_os_atlas_linux_2k_c60_full_aware`
   - 构建报告：`reports/gui_candidate_full_aware_dataset_os_atlas_linux_2k_c60.md`
   - 新数据保留完整 `candidates`，并新增 full-aware action subset：
     - `candidate_rewards_v3`
     - `full_aware_action_ids`
     - `full_aware_action_roles`
     - `full_aware_action_reward_std`
     - `positive_ids`
     - `hard_negative_ids`
     - 兼容旧训练脚本的 `cc_action_ids` / `cc_hard_negative_ids` / `cc_reward_std`

3. 完成数据质量评测。
   - `all.jsonl`: 2000 rows
   - `train.jsonl`: 906 rows
   - `val.jsonl`: 423 rows
   - `val_oracle_hit.jsonl`: 266 rows
   - Train/val image overlap: 0
   - Train avg candidates: 47.1049
   - Train avg actions: 15.5607
   - Train oracle hit: 100.00%
   - Train oracle IoU@0.5: 83.11%
   - Train avg action reward std: 0.2242
   - Train zero-std rate: 0.00%

4. 做了兼容性检查。
   - `scripts/train_gui_candidate_cc_grpo.py` dry-run 能加载新 `train.jsonl`。
   - 所有 `full_aware_action_ids` 都存在于对应 row 的完整 `candidates` 中。
   - 所有 action 都有 `candidate_rewards_v3`。
   - `cc_action_ids` 与 `full_aware_action_ids` 保持一致。

## 修改或新增文件

代码：

- `AGENTS.md`
  - 更新 agent 工作约定、项目上下文、当前实验注意事项和常用检查命令。
- `scripts/build_gui_candidate_full_aware_data.py`
  - 新增 full-aware candidate RL 数据构建脚本。
  - 支持可选 `--model-score-jsonl`，后续可接入真实 current-model full-candidate score cache。
  - 当前没有 score cache 时，会用 top-ranked detector mistakes 作为 current-model false-positive proxy。

文档：

- `reports/gui_candidate_full_aware_dataset_os_atlas_linux_2k_c60.md`
  - 新增数据集构建与质量评测报告。
- `docs/codex-worklog.md`
  - 本工作日志。

生成数据：

- `outputs/gui_candidate_rl_os_atlas_linux_2k_c60_full_aware/all.jsonl`
- `outputs/gui_candidate_rl_os_atlas_linux_2k_c60_full_aware/train.jsonl`
- `outputs/gui_candidate_rl_os_atlas_linux_2k_c60_full_aware/val.jsonl`
- `outputs/gui_candidate_rl_os_atlas_linux_2k_c60_full_aware/val_oracle_hit.jsonl`
- `outputs/gui_candidate_rl_os_atlas_linux_2k_c60_full_aware/summary.json`

注意：`AGENTS.md` 已在本轮后续请求中更新，用于保持 agent 指南和当前 full-aware GUI candidate RL 阶段一致。

## 当前实验命令

核查 `gui_candidate_env.py`：

```bash
git diff --name-only --diff-filter=U
git ls-files -u
grep -nE '^(<<<<<<<|=======|>>>>>>>)' rl/gui_candidate_env.py || true
python3 -m py_compile rl/gui_candidate_env.py
```

构建 full-aware 数据：

```bash
python3 scripts/build_gui_candidate_full_aware_data.py \
  --input-dir outputs/gui_candidate_rl_os_atlas_linux_2k_c60 \
  --output-dir outputs/gui_candidate_rl_os_atlas_linux_2k_c60_full_aware \
  --report reports/gui_candidate_full_aware_dataset_os_atlas_linux_2k_c60.md \
  --max-actions 16 \
  --min-train-oracle-iou 0.05 \
  --require-train-oracle-hit \
  --seed 42
```

基础质量检查：

```bash
wc -l outputs/gui_candidate_rl_os_atlas_linux_2k_c60_full_aware/*.jsonl
python3 -m py_compile scripts/build_gui_candidate_full_aware_data.py \
  scripts/train_gui_candidate_cc_grpo.py \
  scripts/train_gui_candidate_listwise.py \
  scripts/eval_gui_candidate_scores.py
python3 scripts/train_gui_candidate_cc_grpo.py \
  --train-data outputs/gui_candidate_rl_os_atlas_linux_2k_c60_full_aware/train.jsonl \
  --output-dir /tmp/full_aware_cc_grpo_dryrun \
  --max-train-rows 8 \
  --dry-run
```

建议的下一轮 listwise warmup：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,1 \
torchrun --standalone --nproc_per_node=2 scripts/train_gui_candidate_listwise.py \
  --model /root/models/Qwen2.5-VL-3B-Instruct \
  --adapter checkpoints/qwen25vl_3b_gui_candidate_c60_sft_warmup_lora \
  --train-data outputs/gui_candidate_rl_os_atlas_linux_2k_c60_full_aware/train.jsonl \
  --output-dir outputs/gui_candidate_rl_os_atlas_linux_2k_c60_full_aware/listwise_a16_300step \
  --max-steps 300 \
  --per-device-batch-size 1 \
  --max-actions 16 \
  --max-hard-negatives 12 \
  --target-temperature 0.20 \
  --learning-rate 2e-6 \
  --rank-balanced-sampler \
  --gradient-checkpointing \
  --no-ddp-find-unused-parameters
```

建议的 full candidates 评估：

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/eval_gui_candidate_scores.py \
  --data outputs/gui_candidate_rl_os_atlas_linux_2k_c60_full_aware/val.jsonl \
  --output outputs/gui_candidate_rl_os_atlas_linux_2k_c60_full_aware/eval_val100_listwise_a16_full_candidates.jsonl \
  --report reports/gui_candidate_scores_full_aware_listwise_a16_val100.md \
  --model /root/models/Qwen2.5-VL-3B-Instruct \
  --adapter outputs/gui_candidate_rl_os_atlas_linux_2k_c60_full_aware/listwise_a16_300step \
  --limit 100 \
  --score-batch-size 12 \
  --use-overlay
```

建议的 KL + CC-GRPO：

```bash
python3 scripts/cache_gui_candidate_reference_scores.py \
  --input outputs/gui_candidate_rl_os_atlas_linux_2k_c60_full_aware/train.jsonl \
  --output outputs/gui_candidate_rl_os_atlas_linux_2k_c60_full_aware/train_listwise_ref_a16.jsonl \
  --model /root/models/Qwen2.5-VL-3B-Instruct \
  --adapter outputs/gui_candidate_rl_os_atlas_linux_2k_c60_full_aware/listwise_a16_300step \
  --max-actions 16 \
  --score-batch-size 8

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,1 \
torchrun --standalone --nproc_per_node=2 scripts/train_gui_candidate_cc_grpo.py \
  --model /root/models/Qwen2.5-VL-3B-Instruct \
  --adapter outputs/gui_candidate_rl_os_atlas_linux_2k_c60_full_aware/listwise_a16_300step \
  --train-data outputs/gui_candidate_rl_os_atlas_linux_2k_c60_full_aware/train_listwise_ref_a16.jsonl \
  --output-dir outputs/gui_candidate_rl_os_atlas_linux_2k_c60_full_aware/cc_grpo_listwise_kl_g4_a16_300step \
  --max-steps 300 \
  --per-device-batch-size 1 \
  --num-generations 4 \
  --max-actions 16 \
  --max-hard-negatives 12 \
  --learning-rate 2e-6 \
  --policy-temperature 1.2 \
  --entropy-coef 0.01 \
  --kl-coef 0.02 \
  --reference-logprobs-key reference_logprobs \
  --rank-balanced-sampler \
  --gradient-checkpointing \
  --no-ddp-find-unused-parameters
```

## 未解决问题

1. 当前 full-aware 数据没有真实 current-model full-candidate score cache。
   - `model_score_coverage = 0.0`
   - 这次用 `current_model_proxy_false_positive` 近似当前模型高分误选项。
   - 后续应先跑 full-candidate scoring cache，再用 `--model-score-jsonl` 重建数据。

2. 候选召回仍是上限瓶颈。
   - full val oracle hit: 62.88%
   - full val oracle IoU@0.5: 52.25%
   - 如果候选池没有正确框，RL selector 不可能选对。

3. full candidates 评估成本高。
   - 之前 val30 full-candidate scoring 平均延迟约 18s/row。
   - 后续需要 shortlist 或缓存视觉特征/候选 scores，否则 val423 全量评估很慢。

4. 工作区有未纳入本轮处理的 `AGENTS.md` 修改。
   - 需要用户确认是否保留、提交或恢复。

## 下一步计划

1. 先跑 full-aware listwise 300 step。
2. 在 `val.jsonl` 上跑 full candidates val100，确认是否比旧 CC-GRPO 改善。
3. 生成 current-model full-candidate score cache。
4. 用 `--model-score-jsonl` 重建 full-aware 数据，把 proxy false positives 替换成真实模型误选候选。
5. 缓存 listwise reference logprobs。
6. 跑 KL + CC-GRPO 300 step。
7. 对比四组结果：
   - SFT warmup
   - old CC-GRPO a8
   - full-aware listwise a16
   - full-aware listwise + KL + CC-GRPO a16

## 2026-05-27 19:40 CST - 1000 unique GUI tool trajectory v2 数据集

### 已完成

1. 从 OS-Atlas Linux raw 数据重新抽取 1000 条 source，每条对应 1 张 unique screenshot。
   - Source: `/root/models/datasets/os_atlas_linux_1k_unique_20260527_1824/source_os_atlas_linux_1k_unique.jsonl`
   - rows: 1000
   - unique images: 1000

2. 重建 base C60 候选池。
   - Output: `outputs/gui_candidate_rl_os_atlas_linux_1k_unique_c60_20260527_1853`
   - all oracle hit: 65.50%
   - all IoU@0.5: 53.20%
   - avg candidates: 38.84

3. 做 menu/table 本地候选增强，并作为本轮最终候选池。
   - Output: `outputs/gui_candidate_rl_os_atlas_linux_1k_unique_c60_menu_table_refined_20260527_1853`
   - all oracle hit: 69.70%
   - all IoU@0.5: 58.20%
   - avg candidates: 41.24
   - table-cell IoU@0.5: 5.26% -> 55.26%
   - menu IoU@0.5: 34.94% -> 72.29%

4. 构建 full-aware 数据。
   - Output: `outputs/gui_candidate_rl_os_atlas_linux_1k_unique_c60_menu_table_refined_full_aware_20260527_1853`
   - train: 547 条可训练样本
   - val: 200 条，其中 oracle hit 146 条
   - train/val image overlap: 0

5. 构建 v2 trajectory 严格版，作为推荐主数据。
   - RL: `outputs/gui_tool_trajectory_v2_os_atlas_linux_1k_unique_strict_20260527_1933`
   - SFT: `/root/models/datasets/gui_tool_trajectory_v2_os_atlas_linux_1k_unique_strict_20260527_1933`
   - train/val/test: 442 / 95 / 22
   - total: 559 trajectories / 559 unique images
   - final IoU@0.5: 100%
   - quality validation issues: 0

6. 构建 v2 trajectory 宽松辅助版。
   - RL: `outputs/gui_tool_trajectory_v2_os_atlas_linux_1k_unique_relaxed_20260527_1935`
   - SFT: `/root/models/datasets/gui_tool_trajectory_v2_os_atlas_linux_1k_unique_relaxed_20260527_1935`
   - train/val/test: 489 / 95 / 30
   - total: 614 trajectories / 614 unique images
   - final IoU@0.5: 91.04%

7. 已输出总报告并更新项目文档索引。
   - Report: `/root/Workspace/VLM/项目文档/02_指标与数据/GUI工具调用轨迹v2-1000Unique数据集构建总报告_20260527_1936.md`
   - Index: `/root/Workspace/VLM/项目文档/文档索引_20260527_1411.md`

### 修改文件

- `tools/ocr.py`
  - 增加 EasyOCR reader cache，避免每次 OCR 都重复初始化 reader。
- `scripts/augment_gui_candidates_dashscope.py`
  - 增加 `--workers` 并发参数，用于并行尝试 DashScope/VLM 补框。
- `/root/Workspace/VLM/项目文档/文档索引_20260527_1411.md`
  - 加入本轮 1000 unique 三份报告入口。

### 未解决问题

1. DashScope 补框本轮未成功进入最终候选池。
   - `gui-plus-2026-02-26` 返回免费额度耗尽。
   - `qwen3.7-max*` 当前调用格式返回 image content 兼容性错误。

2. enhanced visual/layout 候选池在 1000 unique 上仍偏慢。
   - 主要瓶颈是候选数膨胀后的 dedupe 和 OCR。
   - 本轮采用 base C60 + menu/table 本地增强作为可控版本。

3. 候选池召回仍是 trajectory 数据规模上限。
   - 1000 unique source 中，严格版最终留下 559 条高质量 trajectory。
   - 后续若要接近 1000 条严格 trajectory，需要继续提升候选召回，尤其是 menu-item、check-menu-item、细小控件和弱标注文本区域。

### 下一步计划

1. 用严格版 v2 先跑 trajectory SFT。
2. 用宽松辅助版做低权重混合或 curriculum 对照，不直接替代严格版。
3. 修正 DashScope multimodal 调用格式或换可用视觉模型，再尝试 VLM 补框。
4. 优化 enhanced candidate 的 dedupe/shortlist，使 OCR/layout 增强可以在 1000+ unique 上稳定运行。
5. SFT 后评估模型是否能稳定输出 `select_candidate` 工具调用轨迹，再决定是否进入 GRPO/RL 阶段。

## 2026-05-27 20:13 CST - 主线切换到真正 on-policy VLM agentic RL

### 已完成

1. 新增详细规划文档。
   - `/root/Workspace/VLM/项目文档/01_规划与路线/真正OnPolicy-VLM-Agentic-RL浏览器环境与算法规划_20260527_2008.md`

2. 更新 `AGENTS.md`。
   - 将过时的 full-aware candidate RL 主线改为 direct-action on-policy VLM agentic RL。
   - 明确主动作空间为 `click/type/press/scroll/drag/wait/finish` 等 direct GUI action。
   - 明确 candidate/bbox 数据只作为 SFT warmup、辅助 grounding loss、debug 和 ablation，不再作为默认 RL action space。
   - 明确下一阶段优先实现本地 Playwright GUI-RL Gym，再接 BrowserGym/MiniWoB++，不要直接从 OSWorld/真实桌面重环境开始。

3. 更新项目文档索引。
   - `/root/Workspace/VLM/项目文档/文档索引_20260527_1411.md`

### 当前结论

- 本地 browser Playwright Gym 指的是一个可 `reset/step/screenshot/verifier/reward` 的浏览器 GUI RL 环境。
- 现成数据可以用于 warmup 或接入外部可执行任务，但静态截图/轨迹数据不能替代 on-policy RL。
- 第一阶段应构建 100-200 个高可信 verifier 的本地任务，并混合 BrowserGym/MiniWoB++；不是继续扩候选框或补框。

### 下一步计划

1. 新增 `envs/browser_rl/` 最小环境接口。
2. 实现 Playwright `reset/step/screenshot/verifier/recorder`。
3. 做 20 个 smoke tasks 和 scripted oracle。
4. 把 strict v2 数据转换为 direct-click SFT warmup。
5. 实现 on-policy rollout recorder，再接 Verifier-Guided GRPO。
