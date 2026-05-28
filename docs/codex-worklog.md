# Codex Worklog

更新日期：2026-05-29

## 2026-05-29 00:33 CST 更新：整理提交前状态并迁移 BrowserRL 正式数据目录

- 报告：`/root/Workspace/VLM/项目文档/03_实验与训练报告/代码提交与BrowserRL数据目录迁移报告_20260529_0033.md`
- 仓库副本：`/root/Workspace/VLM/EviTool-VL/docs/项目文档/03_实验与训练报告/代码提交与BrowserRL数据目录迁移报告_20260529_0033.md`
- 正式数据根目录：`/root/datasets/browser_rl/`
- 已迁移：2000 task suite、v1/v2/v3 history-aware SFT 数据、action warmup 数据、100tg/213tg on-policy groups 和 weak template collect 数据。
- 代码更新：`scripts/train_browser_rl_onpolicy_grpo.py` 默认 `--tasks` 指向正式 2000 task suite；`configs/sft_local_qwen_*.yaml` 的 `dataset_dir` 指向 `/root/datasets/browser_rl/sft/...`。
- 验证：`py_compile` 通过；`/opt/conda/envs/llama/bin/python3 scripts/train_browser_rl_onpolicy_grpo.py --dry-run --limit 1 --max-groups 1 --headless` 通过，输出 `groups=1`、`trainable_groups=1`。
- 长期约定：`/root/datasets/browser_rl/` 保存正式可复用数据，repo 的 `outputs/` 只保存临时实验产物。

## 2026-05-28 23:42 CST 更新：VLM-Agentic-RL 阶段复盘与简历面试版同步到最新 RL 状态

- 详细版：`/root/Workspace/VLM/项目文档/04_阶段总结/VLM-Agentic-RL阶段复盘详细版_20260528_2342.md`
- 简历面试版：`/root/Workspace/VLM/项目文档/04_阶段总结/VLM-Agentic-RL简历面试版阶段总结_20260528_2342.md`
- 仓库副本：`/root/Workspace/VLM/EviTool-VL/docs/项目文档/04_阶段总结/`
- 内容更新到 213 trainable groups replay GRPO 阶段，明确 current model 是 `/root/models/Qwen2.5-VL-3B-Instruct` + `outputs/onpolicy_browser_rl_grpo_replay_213tg_safe_20260528_2041/adapter`。
- 关键状态：val200 success_rate=0.850，但 table=0.760、advanced=0.6667 仍是短板，因此 `test200` 仍保留给后续主要 family 不明显退化的版本。
- 补充：记录 on-policy 采集器流式落盘和 resume 能力，并明确后续可复用数据集统一放到 `/root/datasets/browser_rl/`，仓库 `outputs/` 只放临时实验产物。

## 2026-05-28 22:46 CST 更新：on-policy 采集器支持流式落盘和 resume

- 报告：`/root/Workspace/VLM/项目文档/03_实验与训练报告/OnPolicy采集器流式落盘与Resume实现报告_20260528_2246.md`
- 仓库副本：`/root/Workspace/VLM/EviTool-VL/docs/项目文档/03_实验与训练报告/OnPolicy采集器流式落盘与Resume实现报告_20260528_2246.md`
- 修改：`scripts/train_browser_rl_onpolicy_grpo.py`
- 新增参数：`--stream-collect`、`--no-stream-collect`、`--stream-flush-every`、`--resume-collect`。
- 新增实时文件：`groups.jsonl.tmp`、`rollouts.jsonl.tmp`、`live_summary.json`。
- 行为：每采到一个 group/rollout 立即追加写入临时 JSONL；周期性写 live summary；resume 时读取临时文件并跳过已覆盖 task；如果恢复后目标已满足，直接退出且不加载本地 Qwen。
- 验证：`outputs/onpolicy_collect_stream_smoke_20260528_check`，真实小采集 1 group/1 rollout；resume 后未重复采集，且不重新加载模型。

## 2026-05-28 21:32 CST 更新：213 trainable groups replay GRPO 完成，val200 达到 0.85

- 报告：`/root/Workspace/VLM/项目文档/03_实验与训练报告/VerifierGuided-OnPolicy-GRPO-Replay-213TrainableGroups训练评测报告_20260528_2132.md`
- 仓库副本：`/root/Workspace/VLM/EviTool-VL/docs/项目文档/03_实验与训练报告/VerifierGuided-OnPolicy-GRPO-Replay-213TrainableGroups训练评测报告_20260528_2132.md`
- 代码：`scripts/train_browser_rl_onpolicy_grpo.py` 增加 template 配额、family/template 过滤、SFT replay loss，并修复成功动作提前 break 导致停止条件不触发的问题。
- weak template collect：`outputs/onpolicy_browser_rl_grpo_replay_250tg_collect_20260528_1756/weak_templates`，87 trainable groups；table_action=31，choice_checkbox=30，advanced_dialog=20，advanced_scroll=6。
- merged train groups：`outputs/onpolicy_browser_rl_grpo_replay_213tg_merged_20260528_2040`，213 groups=current_weak 87 + previous collected trainable 126。
- 训练 adapter：`outputs/onpolicy_browser_rl_grpo_replay_213tg_safe_20260528_2041/adapter`，lr=3e-7，1 epoch，optimizer_steps=27，replay_loss_weight=0.05，replay_ratio=0.25。
- 快速 gate：`val_balanced_70` success_rate=0.8406，等于 v3 SFT baseline，高于 100tg RL 的 0.8261。
- 正式 gate：`val200` success_rate=0.850，高于 v3 SFT 0.745 和 100tg RL 0.790；valid_json_rate=1.0，valid_action_rate=1.0。
- val200 family：form=0.7778，choice=0.8611，search=0.925，menu=0.88，table=0.76，advanced=0.6667，todo=1.0。
- validation：`outputs/onpolicy_browser_rl_grpo_replay_213tg_safe_validation_20260528_2132`，606/606 valid，error_count=0。
- 决策：不跑 test200，因为 table 从 0.88 降到 0.76，advanced 仍低于 v3 baseline；test200 保留给主要 family 不明显退化的版本。

下一步：先修 collect-only 流式落盘和 live summary，然后专门采 table_action 与 advanced_scroll/dialog，目标保持 val200>=0.85，同时 table>=0.84、advanced>=0.7778。

## 2026-05-28 17:46 CST 更新：阶段复盘和简历面试版总结更新

- 新增详细版：`/root/Workspace/VLM/项目文档/04_阶段总结/VLM-Agentic-RL阶段复盘详细版_20260528_1746.md`
- 新增简历面试版：`/root/Workspace/VLM/项目文档/04_阶段总结/VLM-Agentic-RL简历面试版阶段总结_20260528_1746.md`
- 仓库副本：`/root/Workspace/VLM/EviTool-VL/docs/项目文档/04_阶段总结/`
- 内容更新到 100 trainable groups verifier-guided on-policy GRPO 阶段，明确 current model 是 `/root/models/Qwen2.5-VL-3B-Instruct` + `outputs/onpolicy_browser_rl_grpo_100tg_safe_20260528_1646/adapter`。
- 文档中补充了为什么推理时不需要叠两层 adapter、如何 headless 体验当前模型、当前不足和下一步计划。

## 2026-05-28 17:30 CST 更新：100 trainable groups on-policy GRPO 通过 val200 gate

- 报告：`/root/Workspace/VLM/项目文档/03_实验与训练报告/VerifierGuided-OnPolicy-GRPO-100TrainableGroups训练评测报告_20260528_1730.md`
- 仓库副本：`/root/Workspace/VLM/EviTool-VL/docs/项目文档/03_实验与训练报告/VerifierGuided-OnPolicy-GRPO-100TrainableGroups训练评测报告_20260528_1730.md`
- on-policy collect：`outputs/onpolicy_browser_rl_grpo_100tg_collect_20260528_1445`，合并选择目录 `outputs/onpolicy_browser_rl_grpo_100tg_merged_20260528_1645`。
- 采集统计：collected_groups=445，collected_trainable_groups=126，selected_groups=100，rollouts=175，rollout_success_rate=0.2514。
- 训练 adapter：`outputs/onpolicy_browser_rl_grpo_100tg_safe_20260528_1646/adapter`，1 epoch，lr=5e-7，optimizer_steps=13，trainable_parameters=29,933,568。
- 快速 gate：`val_balanced_70` success_rate=0.8261，略低于 v3 baseline 0.8406，但只差 1 个任务。
- 正式 gate：`val200` success_rate=0.790，高于 v3 baseline 0.745；valid_json_rate=1.0，valid_action_rate=1.0，policy_error_rate=0.0。
- val200 family：form 0.5333->0.6667，choice 0.6389->0.6944，menu 0.84->0.92，search 0.80->0.825，table 0.88->0.84，advanced 0.7778->0.6667，todo 1.0->1.0。
- validation：`outputs/onpolicy_browser_rl_grpo_100tg_safe_validation_20260528_1729`，444/444 valid，error_count=0。

结论：本轮 adapter 通过 val200 gate，可作为下一轮 RL current model；但 `test200` 仍保留作阶段性最终验收。下一步扩大到 200-300 trainable groups，并加入 v3 replay/KL 稳定项，重点补 advanced_scroll/dialog、choice_checkbox 和 table_action。

## 2026-05-28 12:19 CST 更新：form/choice 定向 SFT 修复实验完成但不采用

本轮按“先修复一点看看效果”的目标，构建了少量 form/choice 修复任务，混入 v3 1000-task history-aware SFT 数据后继续训练 Qwen2.5-VL-3B LoRA，并在同一批 test100 上评测。

新增/修改文件：

- `configs/sft_local_qwen_history_aware_v3_repair_form_choice_qwen25vl_3b_lora.yaml`
- `docs/项目文档/03_实验与训练报告/HistoryAwareSFTv3-FormChoice定向修复实验报告_20260528_1219.md`
- `docs/项目文档/文档索引_20260527_1411.md`

主要产物：

- form repair tasks：`outputs/browser_rl_repair_form_v1_20260528_1045`
- choice repair tasks：`outputs/browser_rl_repair_choice_v1_20260528_1045`
- repair oracle：`outputs/browser_rl_repair_form_aug_oracle_v1_20260528_1046`、`outputs/browser_rl_repair_choice_aug_oracle_v1_20260528_1046`
- mixed SFT dataset：`/root/models/datasets/local_qwen_history_aware_sft_v3_1000_aug_form_choice_repair_v1_20260528_1104`
- repair adapter：`checkpoints/qwen25vl_3b_history_aware_sft_v3_form_choice_repair_v1_lora`
- validation：`outputs/browser_rl_repair_v1_validation_20260528_1220`

关键指标：

- mixed SFT rows：9786，train/val/test rows=9010/388/388。
- 训练 best checkpoint：`checkpoint-200`，best eval_loss=0.152206。
- v3 baseline test100 success_rate=0.82。
- repair checkpoint-100 test100 success_rate=0.66。
- repair checkpoint-200/root test100 success_rate=0.76。
- checkpoint-200 family 指标：choice 0.6667 -> 0.8889，但 form 仍为 0.6364，search 0.8636 -> 0.5909，table 1.0 -> 0.8824。
- rollout validation：620/620 valid，error_count=0。

当前结论：

- 不采用 `qwen25vl_3b_history_aware_sft_v3_form_choice_repair_v1_lora` 作为 RL current model。
- 继续使用 `checkpoints/qwen25vl_3b_history_aware_sft_v3_1000_aug_lora` 作为当前最稳的 current model。
- 下一步应直接用 v3 baseline 做小规模 verifier-guided on-policy RL，并从当前模型真实失败 rollout 中构建恢复数据。

## 2026-05-28 13:26 CST 更新：Verifier-guided on-policy Browser RL 小规模跑通

本轮新增并跑通真正 direct-action browser on-policy RL 闭环：当前本地 Qwen2.5-VL LoRA 自己采样动作，Playwright 环境执行，verifier 给 reward，同一状态多动作组内归一化 advantage 后更新 LoRA。

新增/修改文件：

- `scripts/train_browser_rl_onpolicy_grpo.py`
- `docs/项目文档/03_实验与训练报告/VerifierGuided-OnPolicy-BrowserRL小规模跑通报告_20260528_1326.md`
- `docs/项目文档/文档索引_20260527_1411.md`
- `AGENTS.md`

主要产物：

- 激进版：`outputs/onpolicy_browser_rl_grpo_smoke_v1_20260528_1305`
- 安全版：`outputs/onpolicy_browser_rl_grpo_smoke_v1_safe_20260528_1318`
- 激进版 test30：`outputs/rollouts_local_qwen25vl_onpolicy_grpo_smoke_v1_test30_20260528_1309`
- 安全版 test30：`outputs/rollouts_local_qwen25vl_onpolicy_grpo_smoke_v1_safe_test30_20260528_1319`
- validation：`outputs/onpolicy_browser_rl_grpo_smoke_v1_validation_20260528_1326`

关键指标：

- on-policy collect：8 rollouts，16 groups，48 sampled actions，5 trainable groups，rollout_success_rate=0.75。
- 激进版：learning_rate=5e-6，completion logprob=sum，test30 success_rate=0.5667，低于 baseline first30=0.70，不采用。
- 安全版：learning_rate=5e-7，completion logprob=mean，test30 success_rate=0.70，等于 baseline first30=0.70。
- 校验：68/68 valid rollouts，error_count=0，exec_ok_step_rate=1.0，action_parse_step_rate=1.0。

当前结论：

- on-policy RL 工程闭环已经跑通。
- 本轮 adapter 还没有超过 v3 baseline；当前默认 current model 仍应是 `checkpoints/qwen25vl_3b_history_aware_sft_v3_1000_aug_lora`。
- 下一轮需要 family-balanced collect、50-100 个 trainable groups、KL/replay 稳定项和 test30 evaluation gate。

## 2026-05-28 14:38 CST 更新：2000 tasks 与 val200 baseline 建立完成

本轮构建了新的 BrowserRL 2000 task suite，修复了 task split 的 template 覆盖问题，并建立 v3 SFT baseline 的 val200 指标。

新增/修改文件：

- `scripts/build_browser_rl_task_suite.py`
  - split 从 family-level 改为 template-level，避免 choice/advanced 子模板只落在单一 split。
- `scripts/build_browser_rl_eval_subset.py`
  - 新增 family-balanced eval subset 构建脚本。
- `scripts/train_browser_rl_onpolicy_grpo.py`
  - 增加 `--family-quotas-json`、`--target-trainable-groups`、action 去重和补采样参数。
- `docs/项目文档/03_实验与训练报告/BrowserRL-2000任务套件与Val200基线评测报告_20260528_1438.md`

主要产物：

- 任务套件：`outputs/browser_rl_task_suite_2000_20260528_1344`
- oracle center：`outputs/browser_rl_task_suite_2000_oracle_center_20260528_1346/merged`
- val200 baseline：`outputs/rollouts_local_qwen25vl_history_aware_v3_2000_val200_20260528_1412/merged`
- validation：`outputs/browser_rl_task_suite_2000_validation_20260528_1437`

关键指标：

- tasks：train/val/test=1600/200/200。
- oracle center：2000/2000 success，avg_steps=3.1665。
- v3 baseline val200：success_rate=0.745，valid_json_rate=1.0，valid_action_rate=1.0。
- val200 family：form=0.5333，choice=0.6389，search=0.8，menu=0.84，table=0.88，todo=1.0，advanced=0.7778。
- val_balanced_70：69 条，success_rate=0.8406。
- validation：2200/2200 valid，error_count=0。

当前结论：

- 后续 RL 的正式 evaluation gate 使用 val200 baseline `0.745`。
- test200 暂时保留，不用于调参。
- 下一步先采 100 trainable groups，通过 val200 后再扩大到 200 trainable groups。

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

## 2026-05-27 20:51 CST - On-policy VLM agentic RL 第一周目标

### 已完成

1. 更新真正 on-policy VLM agentic RL 规划文档。
   - 第一周目标改为：BrowserGym/MiniWoB++、CUA-Gym、WebArena-Infinity、自建 Playwright smoke tasks 并行验证。
   - 更新路径：
     - `/root/Workspace/VLM/项目文档/01_规划与路线/真正OnPolicy-VLM-Agentic-RL浏览器环境与算法规划_20260527_2008.md`
     - `docs/项目文档/01_规划与路线/真正OnPolicy-VLM-Agentic-RL浏览器环境与算法规划_20260527_2008.md`

2. 新增 `envs/browser_rl/` 最小 browser GUI-RL 环境。
   - `actions.py`: direct GUI action parser，支持 `click/type/press/hotkey/scroll/drag/wait/finish`。
   - `task_spec.py`: browser task JSONL schema。
   - `verifier.py`: DOM/JS verifier。
   - `recorder.py`: rollout recorder。
   - `playwright_env.py`: Playwright `reset/step/screenshot/verifier` 环境。

3. 构建并执行 20 个本地 Playwright smoke tasks。
   - 构建输出：`outputs/browser_rl_smoke_tasks_20260527_1w`
   - rollout 输出：`outputs/browser_rl_smoke_rollouts_20260527_1w`
   - templates: form/todo/search/menu/table
   - rollouts: 20
   - scripted oracle success rate: 100%
   - generated SFT rows: 60

4. 验证 BrowserGym/MiniWoB++。
   - 安装 `browsergym-miniwob`、`gymnasium`、`playwright`。
   - 下载 MiniWoB++ HTML zip 到 `/root/models/datasets/miniwob-plusplus-main`。
   - 检查输出：`outputs/browsergym_miniwob_check_20260527_1w.json`
   - attempted/reset_ok/step_ok: 5/5/5

5. 验证 CUA-Gym。
   - 使用 `HF_ENDPOINT=https://hf-mirror.com` 成功加载 metadata。
   - 检查输出：`outputs/cua_gym_inspection_20260527_1w_mirror`
   - metadata rows: 7897
   - inspected bundles: 5
   - reward.py py_compile ok: 5/5

6. 实现 WebArena-Infinity 转换入口。
   - 脚本：`scripts/convert_webarena_infinity_trajectories.py`
   - 输出：`outputs/webarena_infinity_sft_sample_20260527_1w`
   - 本机 datasets streaming 访问 HF repo tree 反复超时，converted=0，已记录 blocker。

7. 输出第一周执行报告并更新文档索引。
   - `/root/Workspace/VLM/项目文档/03_实验与训练报告/OnPolicy-VLM-Agentic-RL第一周目标执行报告_20260527_2051.md`
   - `docs/项目文档/03_实验与训练报告/OnPolicy-VLM-Agentic-RL第一周目标执行报告_20260527_2051.md`

### 修改/新增文件

- `envs/__init__.py`
- `envs/browser_rl/__init__.py`
- `envs/browser_rl/actions.py`
- `envs/browser_rl/task_spec.py`
- `envs/browser_rl/verifier.py`
- `envs/browser_rl/recorder.py`
- `envs/browser_rl/playwright_env.py`
- `scripts/build_browser_rl_smoke_tasks.py`
- `scripts/run_browser_rl_smoke.py`
- `scripts/check_browsergym_miniwob.py`
- `scripts/inspect_cua_gym.py`
- `scripts/convert_webarena_infinity_trajectories.py`
- `requirements-browser-rl.txt`

### 验证

```bash
python3 -m py_compile \
  envs/browser_rl/actions.py \
  envs/browser_rl/task_spec.py \
  envs/browser_rl/verifier.py \
  envs/browser_rl/recorder.py \
  envs/browser_rl/playwright_env.py \
  scripts/build_browser_rl_smoke_tasks.py \
  scripts/run_browser_rl_smoke.py \
  scripts/check_browsergym_miniwob.py \
  scripts/inspect_cua_gym.py \
  scripts/convert_webarena_infinity_trajectories.py
```

结果：通过。

### 未解决问题

1. WebArena-Infinity 本机转换未完成。
   - 原因：`datasets` streaming 遍历 HF repo tree 超时。
   - 需要改用离线分片下载、镜像同步或手动预下载。

2. CUA-Gym 还没有动态执行 `reward.py`。
   - 当前只做 metadata、task bundle 解包和 `py_compile`。
   - 下一步必须在容器/VM 隔离环境中运行 setup/reward。

3. BrowserGym/MiniWoB++ 目前只验证 reset/step。
   - 还未把 observation/action 映射成统一 rollout schema。

### 下一步计划

1. 实现 CUA-Gym sandbox runner，优先筛选 `mock_web` 子集。
2. 实现 BrowserGym adapter，把 MiniWoB observation/action/reward 转成本项目 rollout schema。
3. 用 20 个 smoke tasks 的 60 条 SFT rows 跑 action JSON SFT smoke。
4. 实现 on-policy rollout collector，先支持 random/scripted policy，再接 Qwen2.5-VL。

## 2026-05-27 21:30 CST - 第二阶段启动：BrowserGym adapter 与统一 rollout collector

### 已完成

1. 更新 `AGENTS.md`。
   - 增加面向用户说明的术语解释要求：英文术语或缩写第一次出现必须先用中文解释。
   - 要求报告区分环境接入、脚本接入、模型训练和模型学会任务。

2. 新增 BrowserGym adapter。
   - 文件：`envs/browser_rl/browsergym_adapter.py`
   - `Adapter` 指适配器，把 BrowserGym/MiniWoB++ 的 observation/action/reward 转成本项目统一格式。
   - 支持 direct GUI action：`click/double_click/type/press/hotkey/scroll/drag/wait/finish`。

3. 新增统一 rollout collector。
   - 文件：`scripts/run_browser_rl_rollouts.py`
   - `Rollout` 指一次完整尝试轨迹：看截图、输出动作、环境执行、得到新截图和 reward，直到成功、失败或超时。
   - 支持 `local_smoke` 和 `browsergym_miniwob` 两类环境源。
   - 支持 `scripted_oracle/random/noop` 三种 policy。
   - 输出 `rollouts.jsonl`、`sft_messages.json` 和 `summary.json`。

4. 新增 CUA-Gym mock_web inspector。
   - 文件：`scripts/inspect_cua_gym_mock_web.py`
   - 从 `xlangai/CUA-Gym` 中筛选 mock_web 任务，抽样解包 task bundle，并检查 `reward.py` 是否能通过 Python 语法编译。

5. 新增手动任务执行脚本。
   - 文件：`scripts/play_browser_rl_task.py`
   - 可以用 headless 模式手动输入 action JSON，验证本地任务的 reset/step/screenshot/verifier/reward。

6. 输出第二阶段启动报告并更新项目文档索引。
   - `/root/Workspace/VLM/项目文档/03_实验与训练报告/OnPolicy-VLM-Agentic-RL第二阶段启动实现报告_20260527_2130.md`
   - `docs/项目文档/03_实验与训练报告/OnPolicy-VLM-Agentic-RL第二阶段启动实现报告_20260527_2130.md`

### 本轮验证

1. 本地 smoke tasks + scripted oracle。
   - 输出：`outputs/rollouts_local_oracle_20260527_2125`
   - rollouts: 5
   - success rate: 100%
   - avg steps: 4.6
   - valid action rate: 100%
   - SFT rows: 23

2. BrowserGym/MiniWoB++ + noop policy。
   - 输出：`outputs/rollouts_miniwob_noop_20260527_2125`
   - rollouts: 3
   - success rate: 0%，符合预期，因为 noop 只等待，不会完成任务。
   - valid action rate: 100%

3. BrowserGym/MiniWoB++ + random policy。
   - 输出：`outputs/rollouts_miniwob_random_20260527_2125`
   - rollouts: 2
   - success rate: 0%，符合预期，因为随机动作只验证接口可执行。
   - valid action rate: 100%

4. CUA-Gym mock_web inspection。
   - 输出：`outputs/cua_gym_mock_web_inspection_20260527_2125`
   - metadata rows: 7897
   - mock_web rows: 1068
   - inspected bundles: 10
   - reward.py py_compile ok: 10/10

5. Python 语法检查。
   - 覆盖 `envs/browser_rl/*.py` 和新增脚本。
   - 结果：通过。

### 当前限制

1. 还没有接 Qwen policy；当前只是脚本、随机和 noop 策略。
2. BrowserGym/MiniWoB++ 当前验证了 adapter 和 rollout schema，没有证明模型能完成 MiniWoB 任务。
3. CUA-Gym mock_web 还没有动态运行 reward，只完成 metadata、bundle 和 `reward.py` 静态检查。
4. WebArena-Infinity 仍受 Hugging Face repo tree 访问超时影响。

### 下一步计划

1. 实现 CUA-Gym mock_web sandbox runner，让 mock_web 任务真正支持 reset/step/reward。
2. 实现 Qwen policy wrapper，先接 DashScope Qwen 输出 action JSON，再接本地 Qwen2.5-VL/Qwen3-VL。
3. 增加 rollout schema validator，检查截图、动作、reward、终止状态、错误信息和 policy version。
4. 跑 action JSON SFT smoke，让模型先稳定输出合法 GUI action。
5. 在可执行任务上启动 Verifier-Guided GRPO。

## 2026-05-27 21:51 CST - CUA-Gym mock_web 弱隔离 sandbox 跑通

### 已完成

1. 新增 CUA-Gym mock_web 弱隔离 runner。
   - `Sandbox` 指隔离沙箱；本轮使用当前 Docker 容器内部弱隔离，不需要宿主机权限。
   - 文件：`envs/browser_rl/cua_gym_sandbox.py`
   - 能复制 task bundle、patch CUA-Gym URL placeholder、启动本地 state server、子进程执行 `initial_setup.py` 和 `reward.py`、解析 `REWARD: x`。

2. 新增命令行入口。
   - 文件：`scripts/run_cua_gym_mock_web_sandbox.py`
   - 默认读取 `outputs/cua_gym_mock_web_inspection_20260527_2125/mock_web_tasks.jsonl` 和已解包 bundles。

3. 更新依赖文件。
   - `requirements-browser-rl.txt` 增加 `requests`，用于 CUA-Gym setup/reward 脚本访问本地 state server。

4. 输出跑通报告并更新文档索引。
   - `/root/Workspace/VLM/项目文档/03_实验与训练报告/OnPolicy-VLM-Agentic-RL弱隔离CUAGymSandbox跑通报告_20260527_2151.md`
   - `docs/项目文档/03_实验与训练报告/OnPolicy-VLM-Agentic-RL弱隔离CUAGymSandbox跑通报告_20260527_2151.md`

### 验证

1. 语法检查通过。

```bash
python3 -m py_compile \
  envs/browser_rl/cua_gym_sandbox.py \
  envs/browser_rl/__init__.py \
  scripts/run_cua_gym_mock_web_sandbox.py
```

2. 2 个样本 smoke 通过。
   - 输出：`outputs/cua_gym_mock_web_sandbox_20260527_2145`
   - setup_ok: 2/2
   - reward_ok: 2/2

3. 10 个已解包样本通过。
   - 输出：`outputs/cua_gym_mock_web_sandbox_20260527_2151`
   - setup_ok: 10/10
   - reward_ok: 10/10
   - avg_initial_reward: 0.0，符合预期，因为还没有 agent 执行动作。

### 当前限制

1. 这还不是完整 CUA-Gym GUI 环境。
   - 当前 state server 只提供 `/post` 和 `/go` 状态 API，以及 debug HTML。
   - 还没有 CUA-Gym 原始 mock web 前端，因此不能直接让 VLM 在 CUA-Gym 上点击真实控件。

2. 还没有 Qwen policy。
   - 当前只验证 setup/reward 动态执行，不是模型 rollout。

3. 当前隔离是弱隔离。
   - 不传 API key，限制资源和超时，但仍在当前容器内运行。
   - 没有宿主机权限时这是可行的第一版；后续有权限再升级 sibling container/gVisor/VM。

### 下一步计划

1. 优先实现 Qwen policy wrapper，接到本地 Playwright smoke tasks 和 BrowserGym/MiniWoB++。
2. 并行查找或接入 CUA-Gym mock web 前端；如果短期拿不到前端，则做 state-level action bridge，把 CUA-Gym 作为 verifier/reward 任务池。
3. 增加 rollout schema validator，为后续 on-policy RL 训练做数据质量门禁。

## 2026-05-27 22:17 CST - Qwen policy wrapper 接入 on-policy rollout

### 已完成

1. 新增 Qwen policy wrapper。
   - `Qwen policy wrapper` 指 Qwen 策略封装器：把截图和任务目标发给 Qwen，再解析成 GUI action JSON。
   - 文件：`envs/browser_rl/qwen_policy.py`
   - 使用 DashScope OpenAI-compatible 接口。
   - 从 `.env` 读取 `DASHSCOPE_API_KEY`，不输出 key。
   - 支持模型 fallback：`qwen3.6-flash-2026-04-16,qwen3.7-max-2026-05-20,qwen3.7-max`。

2. 更新统一 rollout collector。
   - 文件：`scripts/run_browser_rl_rollouts.py`
   - 新增 `--policy qwen_dashscope`。
   - summary 增加 `qwen_valid_json_rate`、`qwen_valid_action_rate`、`qwen_error_rate` 和 `qwen_models`。

3. 改进动作解析与 prompt。
   - 兼容 `x:[90]` 这类模型输出。
   - 对小截图加入像素坐标到 0-1000 归一化坐标的换算提示，修正 MiniWoB++ 坐标误用问题。

4. 输出报告并更新文档索引。
   - `/root/Workspace/VLM/项目文档/03_实验与训练报告/OnPolicy-VLM-Agentic-RL-QwenPolicyWrapper跑通报告_20260527_2217.md`
   - `docs/项目文档/03_实验与训练报告/OnPolicy-VLM-Agentic-RL-QwenPolicyWrapper跑通报告_20260527_2217.md`

### 验证

1. 语法检查通过。

```bash
python3 -m py_compile \
  envs/browser_rl/qwen_policy.py \
  envs/browser_rl/__init__.py \
  scripts/run_browser_rl_rollouts.py \
  envs/browser_rl/cua_gym_sandbox.py \
  scripts/run_cua_gym_mock_web_sandbox.py
```

2. 本地 Playwright smoke tasks 小评测。
   - 输出：`outputs/rollouts_local_qwen_dashscope_eval5_20260527_2206`
   - rollouts: 5
   - success_rate: 1.0
   - avg_steps: 4.8
   - qwen_valid_json_rate: 1.0
   - qwen_valid_action_rate: 1.0
   - qwen_error_rate: 0.0

3. BrowserGym/MiniWoB++ 小评测。
   - 输出：`outputs/rollouts_miniwob_qwen_dashscope_eval5_20260527_2206`
   - rollouts: 5
   - success_rate: 0.4
   - avg_steps: 4.2
   - qwen_valid_json_rate: 1.0
   - qwen_valid_action_rate: 1.0
   - qwen_error_rate: 0.0

### 当前限制

1. MiniWoB++ 的 `click-checkboxes`、`enter-text`、`choose-list` 仍失败。
2. 远程 VLM 每步调用较慢，MiniWoB++ 5 任务评测耗时明显偏长。
3. 当前只是 policy rollout，还没有做 RL 参数更新。

### 下一步计划

1. 实现 rollout schema validator。
2. 针对 checkbox、text input、list/select 增强 prompt 和动作后处理。
3. 扩大到 20 个本地 smoke tasks 和 20 个 MiniWoB++ tasks，建立稳定 baseline。
4. 采集第一批 on-policy 成功/失败轨迹，准备接 Verifier-Guided GRPO。

## 2026-05-27 22:23 CST - 修正 teacher/current model 口径

### 已完成

1. 明确 DashScope Qwen 的角色。
   - `Teacher policy` 指教师策略：远程 `qwen3.6-flash-2026-04-16`、`qwen3.7-max` 等模型只用于环境验证、示范轨迹和 baseline。
   - 它们不是本地要训练的 current model。

2. 明确 current model 的定义。
   - `Current model` 指当前模型：真正会被 SFT/RL 更新参数的本地 Qwen2.5-VL/Qwen3-VL checkpoint。
   - 只有本地 current model 自己 rollout 后再被 verifier reward 更新，才是严格 on-policy RL。

3. 更新文件。
   - `AGENTS.md`
   - `envs/browser_rl/qwen_policy.py`
   - `scripts/run_browser_rl_rollouts.py`
   - `docs/项目文档/03_实验与训练报告/OnPolicy-VLM-Agentic-RL-QwenPolicyWrapper跑通报告_20260527_2217.md`
   - `/root/Workspace/VLM/项目文档/03_实验与训练报告/OnPolicy-VLM-Agentic-RL-QwenPolicyWrapper跑通报告_20260527_2217.md`

4. 新增下一步计划文档。
   - `/root/Workspace/VLM/项目文档/01_规划与路线/本地Qwen当前模型训练下一步计划_20260527_2223.md`
   - `docs/项目文档/01_规划与路线/本地Qwen当前模型训练下一步计划_20260527_2223.md`

### 当前建议

1. 先实现 `scripts/validate_browser_rl_rollouts.py`，检查 rollout schema。
2. 汇总高质量 action JSON SFT warmup 数据。
3. 检查 `/root/models` 中可用的本地 Qwen2.5-VL/Qwen3-VL base model。
4. 跑本地 Qwen LoRA SFT smoke。
5. 接 `--policy local_qwen`，让本地 checkpoint 成为真正 current model。

## 2026-05-27 22:34 CST - 本地 Qwen action SFT warmup 数据与 smoke 训练

### 已完成

1. 新增 rollout schema validator。
   - `Schema validator` 指轨迹格式检查器。
   - 文件：`scripts/validate_browser_rl_rollouts.py`
   - 验证输出：`outputs/browser_rl_rollout_validation_20260527_2224`
   - 30 条 rollout 全部有效，error_count=0。

2. 新增本地 Qwen action JSON SFT warmup 构建器。
   - `SFT` 指监督微调。
   - 文件：`scripts/build_local_qwen_action_sft_warmup.py`
   - 输出数据集：`/root/models/datasets/local_qwen_action_sft_warmup_20260527_2224`
   - rows: 90
   - train_rows: 81
   - val_rows: 9
   - unique_tasks: 22

3. 新增本地 Qwen SFT 配置。
   - `configs/sft_local_qwen_action_warmup_qwen25vl_3b_lora.yaml`
   - `configs/sft_local_qwen_action_warmup_qwen3vl_4b_lora.yaml`
   - `configs/sft_local_qwen_action_warmup_qwen25vl_3b_lora_smoke.yaml`

4. 本地模型检查完成。
   - `/root/models/Qwen2.5-VL-3B-Instruct`
   - `/root/models/Qwen3-VL-4B-Instruct`
   - `/root/models/Qwen3-VL-8B-Instruct`

5. 跑通 Qwen2.5-VL-3B LoRA SFT smoke。
   - 命令：`python3 -m llamafactory.cli train configs/sft_local_qwen_action_warmup_qwen25vl_3b_lora_smoke.yaml`
   - max_steps: 1
   - train samples: 4
   - train_loss: 2.6845
   - checkpoint: `checkpoints/qwen25vl_3b_local_action_sft_warmup_smoke`

6. 输出报告。
   - `/root/Workspace/VLM/项目文档/03_实验与训练报告/本地QwenActionSFTWarmup与Smoke训练报告_20260527_2234.md`
   - `docs/项目文档/03_实验与训练报告/本地QwenActionSFTWarmup与Smoke训练报告_20260527_2234.md`

### 当前限制

1. 当前只完成 1 step smoke，还没有完整 warmup 训练。
2. 本地 checkpoint 还没有通过 `--policy local_qwen` 接回 rollout collector。
3. 还没有开始 GRPO/RL。

### 下一步计划

1. 跑完整 Qwen2.5-VL-3B action JSON SFT warmup。
2. 实现 `--policy local_qwen`，加载本地 base + LoRA checkpoint 做 rollout。
3. 用本地 current model 跑本地 Playwright 和 BrowserGym/MiniWoB++ baseline。
4. 根据本地 rollout 成功率决定是否进入 Verifier-Guided GRPO。

## 2026-05-27 23:02 CST - 本地 Qwen 完整 SFT 与 local_qwen 接入评测

### 已完成

1. 跑完整 Qwen2.5-VL-3B action JSON SFT warmup。
   - `SFT` 指监督微调。
   - 数据集：`/root/models/datasets/local_qwen_action_sft_warmup_20260527_2224`
   - train_rows: 81
   - val_rows: 9
   - checkpoint: `checkpoints/qwen25vl_3b_local_action_sft_warmup_lora`
   - global_step: 11
   - epoch: 1.0
   - train_loss: 1.4807

2. 实现本地 Qwen policy。
   - `Policy` 指策略，也就是决定下一步 GUI 动作的组件。
   - `local_qwen` 指本地 Qwen2.5-VL/Qwen3-VL current model 策略，不调用 DashScope。
   - 新增：`envs/browser_rl/local_qwen_policy.py`
   - 修改：`envs/browser_rl/__init__.py`
   - 修改：`scripts/run_browser_rl_rollouts.py`
   - 新增 CLI：`--policy local_qwen`
   - 支持参数：`--local-model`、`--local-adapter`、`--local-load-in-4bit`、`--local-prompt-style`。

3. 增加本地策略提示词模式。
   - `sft_minimal`：和 warmup SFT 数据一致的短提示词。
   - `full`：包含动作空间、坐标规范、step 和 history 的完整 rollout 提示词。

4. 额外做 clean-oracle debug SFT。
   - `clean-oracle` 指只使用 scripted oracle 成功轨迹，不混入 DashScope teacher 轨迹。
   - 数据集：`/root/models/datasets/local_qwen_action_sft_oracle_smoke_20260527_2300`
   - rows: 60
   - train_rows: 54
   - val_rows: 6
   - checkpoint: `checkpoints/qwen25vl_3b_local_action_sft_oracle_smoke_lora`
   - global_step: 35
   - epoch: 5.0
   - train_loss: 0.7441

5. 完成 local current model 小评测。
   - full warmup + `sft_minimal`：success_rate=0.0，valid_action_rate=1.0。
   - clean-oracle + `sft_minimal`：success_rate=0.0，valid_action_rate=1.0。
   - clean-oracle + `full`：success_rate=0.0，valid_action_rate=0.8333。
   - 结论：接口接通，动作 JSON 基本合法，但当前模型还没有稳定学会多步状态判断和坐标定位。

6. 输出报告。
   - `/root/Workspace/VLM/项目文档/03_实验与训练报告/本地Qwen完整SFT与LocalPolicy接入评测报告_20260527_2302.md`
   - `docs/项目文档/03_实验与训练报告/本地Qwen完整SFT与LocalPolicy接入评测报告_20260527_2302.md`

### 运行命令

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/root/Workspace/VLM/EviTool-VL/third_party/LlamaFactory/src:${PYTHONPATH:-} \
python3 -m llamafactory.cli train \
  configs/sft_local_qwen_action_warmup_qwen25vl_3b_lora.yaml
```

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

### 修改文件

- `envs/browser_rl/local_qwen_policy.py`
- `envs/browser_rl/__init__.py`
- `scripts/run_browser_rl_rollouts.py`
- `configs/sft_local_qwen_action_warmup_qwen25vl_3b_lora_oracle_smoke.yaml`
- `/root/Workspace/VLM/项目文档/03_实验与训练报告/本地Qwen完整SFT与LocalPolicy接入评测报告_20260527_2302.md`
- `docs/项目文档/03_实验与训练报告/本地Qwen完整SFT与LocalPolicy接入评测报告_20260527_2302.md`
- `/root/Workspace/VLM/项目文档/文档索引_20260527_1411.md`
- `docs/项目文档/文档索引_20260527_1411.md`
- `docs/codex-worklog.md`

### 当前限制

1. `local_qwen` 已接入，但当前 Qwen2.5-VL LoRA 还不能完成 smoke_form_01。
2. 当前短 prompt SFT 缺少 step/history/verifier progress，模型不知道当前应该 click 还是 type。
3. 仅 60-90 条单步数据不足以支撑稳定多步 GUI agent 能力。
4. 现在直接做大规模 on-policy RL 会遇到 reward 极稀疏问题。

### 下一步计划

1. 重建 history-aware trajectory SFT 数据。
2. 扩大 local browser tasks 到 100-300 个可 reset、可 verifier 的任务。
3. 加强动作解析和后处理，修复常见近似 JSON。
4. 用 history-aware SFT 后的 local current model 跑 20-50 个任务 baseline。
5. baseline 有非零成功率后再接 Verifier-Guided GRPO。

## 2026-05-27 23:56 CST - History-aware trajectory SFT v1 实现、训练与评测

### 已完成

1. 增强 full prompt。
   - `History-aware` 指带历史感知。
   - 修改：`envs/browser_rl/qwen_policy.py`
   - 新增 `最大 step`、`当前 verifier progress`。
   - `viewport` 和 `action_space` 改为标准 JSON 字符串。
   - 增加“不要重复无进展动作”的提示。

2. 新增 history-aware SFT 数据构建脚本。
   - `Trajectory SFT` 指轨迹监督微调，让模型按专家轨迹逐步模仿下一步动作。
   - 文件：`scripts/build_local_qwen_history_aware_sft.py`
   - 输入：`rollouts.jsonl` 和可选 `tasks.jsonl`
   - 输出：LlamaFactory sharegpt multimodal 格式。

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

4. 训练本地 Qwen2.5-VL history-aware LoRA。
   - 配置：`configs/sft_local_qwen_history_aware_qwen25vl_3b_lora.yaml`
   - checkpoint: `checkpoints/qwen25vl_3b_history_aware_sft_v1_lora`
   - global_step: 35
   - epoch: 5.0
   - train_loss: 0.4766

5. 完成本地 current model 评测。
   - eval5：`outputs/rollouts_local_qwen25vl_history_aware_v1_eval5_20260527_2349`
   - eval5 success_rate: 0.6
   - eval20：`outputs/rollouts_local_qwen25vl_history_aware_v1_eval20_20260527_2352`
   - eval20 success_rate: 0.3
   - eval20 valid_json_rate: 1.0
   - eval20 valid_action_rate: 1.0
   - eval20 policy_error_rate: 0.0

6. 输出报告。
   - `/root/Workspace/VLM/项目文档/03_实验与训练报告/HistoryAwareTrajectorySFT实现与本地Qwen评测报告_20260527_2356.md`
   - `docs/项目文档/03_实验与训练报告/HistoryAwareTrajectorySFT实现与本地Qwen评测报告_20260527_2356.md`

### 运行命令

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

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/root/Workspace/VLM/EviTool-VL/third_party/LlamaFactory/src:${PYTHONPATH:-} \
python3 -m llamafactory.cli train \
  configs/sft_local_qwen_history_aware_qwen25vl_3b_lora.yaml
```

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

### 修改文件

- `envs/browser_rl/qwen_policy.py`
- `scripts/build_local_qwen_history_aware_sft.py`
- `configs/sft_local_qwen_history_aware_qwen25vl_3b_lora.yaml`
- `AGENTS.md`
- `/root/Workspace/VLM/项目文档/03_实验与训练报告/HistoryAwareTrajectorySFT实现与本地Qwen评测报告_20260527_2356.md`
- `docs/项目文档/03_实验与训练报告/HistoryAwareTrajectorySFT实现与本地Qwen评测报告_20260527_2356.md`
- `/root/Workspace/VLM/项目文档/文档索引_20260527_1411.md`
- `docs/项目文档/文档索引_20260527_1411.md`
- `docs/codex-worklog.md`

### 当前限制

1. eval20 success_rate=0.3，仍不适合直接大规模 RL。
2. todo 已经 4/4，form 2/4，但 search/menu/table 都是 0/4。
3. 主要瓶颈是任务覆盖和坐标定位，不是 JSON 格式。

### 下一步计划

1. 扩充 local browser tasks 到 100-300 个。
2. 针对 form/search/menu/table 重点补充 scripted oracle 轨迹。
3. 加安全坐标扰动增强，提升点击区域鲁棒性。
4. 重建 history-aware SFT v2，目标 500-1500 条 step-level 样本。
5. eval20 success_rate 到 0.6 左右后再接小规模 on-policy RL。

## 2026-05-28 00:38 CST - History-aware SFT v2 300 任务构建、训练与评测

### 完成内容

1. 新增 300 条本地 browser GUI-RL 任务。
   - 脚本：`scripts/build_browser_rl_task_suite.py`
   - 输出：`outputs/browser_rl_task_suite_300_20260528_0000`
   - split：train 240、val 30、test 30
   - family 配比：form 60、search 60、menu 50、table 50、todo 30、choice 30、advanced 20

2. 用 scripted oracle 跑通全部 300 个任务。
   - 输出：`outputs/browser_rl_task_suite_300_oracle_rollouts_20260528_0001`
   - success_rate: 1.0
   - avg_steps: 3.0
   - step-level SFT rows: 900

3. 重建 history-aware SFT v2 数据。
   - 数据目录：`/root/models/datasets/local_qwen_history_aware_sft_v2_300_20260528_0017`
   - train rows: 718
   - val rows: 91
   - test rows: 91
   - unique tasks: 300
   - 修复点：`scripts/build_local_qwen_history_aware_sft.py` 新增 `--respect-task-splits`，严格按任务文件内 split 切分，避免 test task 泄漏进 train。

4. 训练本地 current model。
   - current model：`/root/models/Qwen2.5-VL-3B-Instruct`
   - LoRA checkpoint：`checkpoints/qwen25vl_3b_history_aware_sft_v2_300_lora`
   - 配置：`configs/sft_local_qwen_history_aware_v2_300_qwen25vl_3b_lora.yaml`
   - 训练：1 epoch、90 steps、train_loss=0.3847、runtime=571.81s

5. 在独立 test 任务上评测 local_qwen。
   - 输出：`outputs/rollouts_local_qwen25vl_history_aware_v2_300_test30_20260528_0028`
   - rollouts: 30
   - success_rate: 0.3333
   - valid_json_rate: 1.0
   - valid_action_rate: 1.0
   - policy_error_rate: 0.0

6. 完成质量校验和报告。
   - 校验输出：`outputs/browser_rl_v2_300_validation_20260528_0036`
   - valid_rollouts: 330/330
   - error_count: 0
   - 报告：`/root/Workspace/VLM/项目文档/03_实验与训练报告/HistoryAwareSFTv2-300任务构建训练评测报告_20260528_0038.md`
   - 仓库副本：`docs/项目文档/03_实验与训练报告/HistoryAwareSFTv2-300任务构建训练评测报告_20260528_0038.md`

### 运行命令

```bash
python3 scripts/build_browser_rl_task_suite.py \
  --output-dir outputs/browser_rl_task_suite_300_20260528_0000 \
  --timestamp 20260528_0000
```

```bash
python3 scripts/run_browser_rl_smoke.py \
  --tasks outputs/browser_rl_task_suite_300_20260528_0000/all_tasks.jsonl \
  --output-dir outputs/browser_rl_task_suite_300_oracle_rollouts_20260528_0001 \
  --headless \
  --policy-version scripted_oracle_v2
```

```bash
python3 scripts/build_local_qwen_history_aware_sft.py \
  --input outputs/browser_rl_task_suite_300_oracle_rollouts_20260528_0001/rollouts.jsonl \
  --tasks outputs/browser_rl_task_suite_300_20260528_0000/all_tasks.jsonl \
  --output-dir /root/models/datasets/local_qwen_history_aware_sft_v2_300_20260528_0017 \
  --include-policies scripted_oracle_v2 \
  --respect-task-splits \
  --max-history 4 \
  --dataset-name local_qwen_history_aware_sft_train \
  --val-dataset-name local_qwen_history_aware_sft_val \
  --test-dataset-name local_qwen_history_aware_sft_test
```

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/root/Workspace/VLM/EviTool-VL/third_party/LlamaFactory/src:${PYTHONPATH:-} \
python3 -m llamafactory.cli train \
  configs/sft_local_qwen_history_aware_v2_300_qwen25vl_3b_lora.yaml
```

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_browser_rl_rollouts.py \
  --env-source local_smoke \
  --policy local_qwen \
  --tasks outputs/browser_rl_task_suite_300_20260528_0000/test_tasks.jsonl \
  --output-dir outputs/rollouts_local_qwen25vl_history_aware_v2_300_test30_20260528_0028 \
  --limit 30 \
  --max-steps 6 \
  --headless \
  --local-model /root/models/Qwen2.5-VL-3B-Instruct \
  --local-adapter checkpoints/qwen25vl_3b_history_aware_sft_v2_300_lora \
  --local-load-in-4bit \
  --local-max-new-tokens 96 \
  --local-temperature 0 \
  --local-max-history 4 \
  --local-prompt-style full
```

### 修改文件

- `scripts/build_browser_rl_task_suite.py`
- `scripts/build_local_qwen_history_aware_sft.py`
- `configs/sft_local_qwen_history_aware_v2_300_qwen25vl_3b_lora.yaml`
- `AGENTS.md`
- `/root/Workspace/VLM/项目文档/03_实验与训练报告/HistoryAwareSFTv2-300任务构建训练评测报告_20260528_0038.md`
- `docs/项目文档/03_实验与训练报告/HistoryAwareSFTv2-300任务构建训练评测报告_20260528_0038.md`
- `/root/Workspace/VLM/项目文档/文档索引_20260527_1411.md`
- `docs/项目文档/文档索引_20260527_1411.md`
- `docs/codex-worklog.md`

### 当前限制

1. 300 任务数据质量合格，但模型 test success_rate 只有 0.3333，不能直接视为已具备稳定 GUI 操作能力。
2. 失败主要是视觉坐标 grounding，不是 JSON 格式；典型问题是点击高度偏移，导致输入框、搜索结果或表格行没有被点中。
3. 每类训练任务实际只有几十个，覆盖不足，尤其 form/search/menu/table 的布局泛化还弱。

### 下一步计划

1. 扩到 1000-2000 个 local browser tasks。
2. 给 click 类动作加入元素内部坐标扰动增强，提升点击区域鲁棒性。
3. 增加失败恢复示范，让模型学习 verifier progress 没变化时如何重新定位。
4. 训练 2-3 epoch，并在 val/test 上跟踪成功率，而不是只看 train loss。
5. test success_rate 达到 0.6 左右后，再进入小规模 verifier-guided GRPO / on-policy RL。

## 2026-05-28 03:45 CST 更新：1000 任务增强 SFT v3 达到 test100 success_rate 0.82

本阶段按用户要求执行：扩到 1000 个本地 browser GUI-RL 任务，加入坐标扰动增强和失败恢复示范，训练本地 Qwen2.5-VL-3B history-aware LoRA，并在独立 test100 上做真实可执行环境评测。同时完成更大的 Qwen2.5-VL-7B-Instruct 下载到 `/root/models`。

### 完成内容

1. 构建 1000 个可 reset/step/verifier 的本地 browser GUI-RL 任务。
   - 输出：`outputs/browser_rl_task_suite_1000_20260528_0058`
   - split：train 800、val 100、test 100
   - family：form 220、search 220、menu 170、table 170、todo 80、choice 90、advanced 50

2. 构建增强 oracle rollout。
   - 脚本：`scripts/run_browser_rl_augmented_oracle.py`
   - 输出：`outputs/browser_rl_task_suite_1000_augmented_oracle_20260528_0104`
   - rollouts：2600
   - success_rate：0.9962
   - variants：center 1000、jitter 800、recovery 800
   - trainable_steps：7970
   - 已知问题：10 条 choice recovery 失败，已被 SFT 构建器剔除。

3. 构建 history-aware SFT v3 数据。
   - 数据目录：`/root/models/datasets/local_qwen_history_aware_sft_v3_1000_aug_20260528_0140`
   - rows：7942
   - train/val/test rows：7322/310/310
   - unique_tasks：1000
   - task_overlap：空

4. 训练本地 current model。
   - base：`/root/models/Qwen2.5-VL-3B-Instruct`
   - 配置：`configs/sft_local_qwen_history_aware_v3_1000_aug_qwen25vl_3b_lora.yaml`
   - output：`checkpoints/qwen25vl_3b_history_aware_sft_v3_1000_aug_lora`
   - best checkpoint：`checkpoint-900`
   - eval_loss：step300 0.2223、step600 0.1781、step900 0.1606
   - train_loss：0.2901

5. 在独立 test100 上评测本地模型。
   - 输出：`outputs/rollouts_local_qwen25vl_history_aware_v3_1000_aug_test100_20260528_0327`
   - success_rate：0.82
   - avg_steps：3.59
   - valid_json_rate：1.0
   - valid_action_rate：1.0
   - policy_error_rate：0.0
   - family 弱项：form 0.6364、choice 0.6667
   - family 强项：table 1.0、todo 1.0、search 0.8636、menu 0.8235

6. 完成质量校验和报告。
   - 校验输出：`outputs/browser_rl_v3_1000_aug_validation_20260528_0345`
   - valid_rollouts：2700/2700
   - error_count：0
   - 报告：`/root/Workspace/VLM/项目文档/03_实验与训练报告/HistoryAwareSFTv3-1000任务增强训练评测报告_20260528_0345.md`
   - 仓库副本：`docs/项目文档/03_实验与训练报告/HistoryAwareSFTv3-1000任务增强训练评测报告_20260528_0345.md`

7. 完成 Qwen2.5-VL-7B-Instruct 下载和本地配置验证。
   - 目标路径：`/root/models/Qwen2.5-VL-7B-Instruct`
   - 使用 HF、HF transfer、hf-mirror 与 `aria2c` 断点续传组合。
   - 截至 2026-05-28 04:34 CST 已完成 5/5 个 safetensors 分片，目录大小 17G。
   - `transformers.AutoConfig` 本地读取验证通过：`model_type=qwen2_5_vl`。

### 关键命令

```bash
python3 scripts/build_browser_rl_task_suite.py \
  --output-dir outputs/browser_rl_task_suite_1000_20260528_0058 \
  --timestamp 20260528_0058 \
  --counts-json '{"form":220,"search":220,"menu":170,"table":170,"todo":80,"choice":90,"advanced":50}'
```

```bash
python3 scripts/run_browser_rl_augmented_oracle.py \
  --tasks outputs/browser_rl_task_suite_1000_20260528_0058/all_tasks.jsonl \
  --output-dir outputs/browser_rl_task_suite_1000_augmented_oracle_20260528_0104 \
  --headless \
  --policy-prefix scripted_oracle_aug_v1 \
  --train-jitter-rollouts 1 \
  --train-recovery-rollouts 1 \
  --seed 42 \
  --reuse-context
```

```bash
python3 scripts/build_local_qwen_history_aware_sft.py \
  --input outputs/browser_rl_task_suite_1000_augmented_oracle_20260528_0104/rollouts.jsonl \
  --tasks outputs/browser_rl_task_suite_1000_20260528_0058/all_tasks.jsonl \
  --output-dir /root/models/datasets/local_qwen_history_aware_sft_v3_1000_aug_20260528_0140 \
  --respect-task-splits \
  --max-history 4 \
  --dataset-name local_qwen_history_aware_sft_train \
  --val-dataset-name local_qwen_history_aware_sft_val \
  --test-dataset-name local_qwen_history_aware_sft_test
```

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/root/Workspace/VLM/EviTool-VL/third_party/LlamaFactory/src:${PYTHONPATH:-} \
python3 -m llamafactory.cli train \
  configs/sft_local_qwen_history_aware_v3_1000_aug_qwen25vl_3b_lora.yaml
```

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_browser_rl_rollouts.py \
  --env-source local_smoke \
  --policy local_qwen \
  --tasks outputs/browser_rl_task_suite_1000_20260528_0058/test_tasks.jsonl \
  --output-dir outputs/rollouts_local_qwen25vl_history_aware_v3_1000_aug_test100_20260528_0327 \
  --limit 100 \
  --max-steps 6 \
  --headless \
  --local-model /root/models/Qwen2.5-VL-3B-Instruct \
  --local-adapter checkpoints/qwen25vl_3b_history_aware_sft_v3_1000_aug_lora \
  --local-load-in-4bit \
  --local-max-new-tokens 96 \
  --local-temperature 0 \
  --local-max-history 4 \
  --local-prompt-style full
```

### 修改文件

- `envs/browser_rl/playwright_env.py`
- `scripts/run_browser_rl_augmented_oracle.py`
- `scripts/build_local_qwen_history_aware_sft.py`
- `configs/sft_local_qwen_history_aware_v3_1000_aug_qwen25vl_3b_lora.yaml`
- `AGENTS.md`
- `/root/Workspace/VLM/项目文档/03_实验与训练报告/HistoryAwareSFTv3-1000任务增强训练评测报告_20260528_0345.md`
- `docs/项目文档/03_实验与训练报告/HistoryAwareSFTv3-1000任务增强训练评测报告_20260528_0345.md`
- `/root/Workspace/VLM/项目文档/文档索引_20260527_1411.md`
- `docs/项目文档/文档索引_20260527_1411.md`
- `docs/codex-worklog.md`

### 当前限制

1. 这次 0.82 是 SFT warmup 后的可执行环境评测，不是严格 on-policy RL。
2. form 和 choice 仍是主要瓶颈，分别为 0.6364 和 0.6667。
3. recovery 规则对 checkbox/radio 这类状态切换控件仍需专项修复。

### 下一步计划

1. 用 v3 LoRA 作为 current model，开始小规模 verifier-guided GRPO / on-policy RL。
2. 针对 form 和 choice 增加少量定向 SFT 修复数据。
3. 让当前模型自己 rollout，并把失败轨迹、重复点击、reward 无进展状态纳入 RL 训练。
4. 评估是否用 Qwen2.5-VL-7B-Instruct 做同样的 LoRA SFT/RL；优先先用 3B v3 LoRA 跑小规模 on-policy RL，确认算法闭环后再放大底模。

## 2026-05-28 10:41 CST 更新：输出阶段复盘详细版与简历面试版

按用户要求新增两份阶段总结，并同步到根项目文档和仓库文档：

- `/root/Workspace/VLM/项目文档/04_阶段总结/VLM-Agentic-RL阶段复盘详细版_20260528_1041.md`
- `/root/Workspace/VLM/项目文档/04_阶段总结/VLM-Agentic-RL简历面试版阶段总结_20260528_1041.md`
- `docs/项目文档/04_阶段总结/VLM-Agentic-RL阶段复盘详细版_20260528_1041.md`
- `docs/项目文档/04_阶段总结/VLM-Agentic-RL简历面试版阶段总结_20260528_1041.md`

详细版记录了候选框路线、路线调整、本地 browser GUI-RL 环境、1000 任务增强 SFT、test100 结果和下一步 on-policy RL 判断，并解释了关键技术名词。简历面试版整理为项目一句话、简历 bullet、面试讲述、量化指标、技术栈和高频问答。
