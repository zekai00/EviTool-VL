# eval_medium_600 SFT 前后评估报告

生成时间：2026-05-24  
代码状态：`239e099 Add SFT pipeline and split eval reporting` 已推送到 `origin/main`。本次评测过程中本地新增了一个 `eval/eval_baseline.py` 兼容补丁，用于绕过 AutoAWQ 与当前 transformers 的 LoRA 加载冲突；该补丁尚未提交。  
评测输出目录：`outputs/eval_medium_600/`

## 1. 数据与切分

- Eval 集：`/root/models/datasets/evitool_eval_medium/eval_medium_600.jsonl`
- 样本量：600
- 任务分布：ChartQA 150、DocVQA 150、AI2D 100、GUI grounding 200
- Train/Eval 隔离：`split_ok=true`
- 与 1k 训练集 source identity overlap：0
- source range violation：0
- semantic duplicate warning：2，属于泛化问题/相似问法预警，不是切分失败

## 2. 总体结果

### 2.1 Direct 模式

| 设置 | Text relaxed | Δ | Text exact | Δ | GUI pointing | Δ | GUI IoU@0.5 | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 3B direct | 83.00% -> 82.25% | -0.75 pp | 78.00% -> 77.00% | -1.00 pp | 21.50% -> 40.00% | +18.50 pp | 3.00% -> 6.50% | +3.50 pp |
| 4B direct | 85.50% -> 86.50% | +1.00 pp | 80.75% -> 83.00% | +2.25 pp | 1.00% -> 1.00% | +0.00 pp | 0.00% -> 0.00% | +0.00 pp |


### 2.2 Tool 模式

| 设置 | Text relaxed | Δ | GUI pointing | Δ | GUI IoU@0.5 | Δ | Evidence closed | Δ | Avg tools | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3B tool | 47.75% -> 72.00% | +24.25 pp | 10.00% -> 45.00% | +35.00 pp | 0.50% -> 3.00% | +2.50 pp | 17.83% -> 36.50% | +18.67 pp | 1.51 -> 0.42 | -1.09 |
| 4B tool | 83.50% -> 80.50% | -3.00 pp | 3.50% -> 11.00% | +7.50 pp | 1.50% -> 7.00% | +5.50 pp | 98.33% -> 98.83% | +0.50 pp | 1.87 -> 1.61 | -0.26 |


## 3. 分任务结果

| 设置 | 任务 | 主指标 | 辅助指标 | 其他 |
|---|---|---:|---:|---:|
| 3B direct | ChartQA | relaxed 77.33% -> 77.33% (+0.00 pp) | exact 65.33% -> 64.67% (-0.67 pp) | - |
| 3B direct | DocVQA | relaxed 86.67% -> 84.67% (-2.00 pp) | exact 85.33% -> 83.33% (-2.00 pp) | - |
| 3B direct | AI2D | relaxed 86.00% -> 86.00% (+0.00 pp) | exact 86.00% -> 86.00% (+0.00 pp) | - |
| 3B direct | GUI | pointing 21.50% -> 40.00% (+18.50 pp) | IoU@0.5 3.00% -> 6.50% (+3.50 pp) | mean IoU 0.06 -> 0.12 |
| 4B direct | ChartQA | relaxed 82.00% -> 82.67% (+0.67 pp) | exact 70.67% -> 74.00% (+3.33 pp) | - |
| 4B direct | DocVQA | relaxed 85.33% -> 84.67% (-0.67 pp) | exact 84.00% -> 84.00% (+0.00 pp) | - |
| 4B direct | AI2D | relaxed 91.00% -> 95.00% (+4.00 pp) | exact 91.00% -> 95.00% (+4.00 pp) | - |
| 4B direct | GUI | pointing 1.00% -> 1.00% (+0.00 pp) | IoU@0.5 0.00% -> 0.00% (+0.00 pp) | mean IoU 0.00 -> 0.00 |
| 3B tool | ChartQA | relaxed 50.00% -> 58.00% (+8.00 pp) | exact 32.00% -> 40.67% (+8.67 pp) | - |
| 3B tool | DocVQA | relaxed 52.67% -> 80.67% (+28.00 pp) | exact 49.33% -> 77.33% (+28.00 pp) | - |
| 3B tool | AI2D | relaxed 37.00% -> 80.00% (+43.00 pp) | exact 37.00% -> 80.00% (+43.00 pp) | - |
| 3B tool | GUI | pointing 10.00% -> 45.00% (+35.00 pp) | IoU@0.5 0.50% -> 3.00% (+2.50 pp) | mean IoU 0.02 -> 0.08 |
| 4B tool | ChartQA | relaxed 82.00% -> 76.00% (-6.00 pp) | exact 53.33% -> 56.00% (+2.67 pp) | - |
| 4B tool | DocVQA | relaxed 84.67% -> 80.00% (-4.67 pp) | exact 83.33% -> 76.67% (-6.67 pp) | - |
| 4B tool | AI2D | relaxed 84.00% -> 88.00% (+4.00 pp) | exact 84.00% -> 88.00% (+4.00 pp) | - |
| 4B tool | GUI | pointing 3.50% -> 11.00% (+7.50 pp) | IoU@0.5 1.50% -> 7.00% (+5.50 pp) | mean IoU 0.02 -> 0.06 |


## 4. 关键结论

1. `239e099` 已成功推送到 GitHub；本次评测结果完整落盘。
2. SFT 对 3B tool 的协议稳定性和 GUI pointing 有明显帮助：final parse 从 43.00% 到 92.67%，GUI pointing 从 10.00% 到 45.00%。
3. 3B tool 的文本 relaxed 从 47.75% 到 72.00%，提升很大，但仍低于 3B direct post 的 82.25%。
4. 3B tool post 的 avg tools 从 1.51 降到 0.42，说明 SFT 后模型学会了更稳定的 JSON/FINAL，但也学会了大量绕过工具。DocVQA 的 tool evidence closed 为 0.00%，这不是理想的证据型工具使用。
5. 4B direct post 有小幅收益：Text relaxed 从 85.50% 到 86.50%，AI2D relaxed 从 91.00% 到 95.00%。
6. 4B tool post 的 evidence closed 仍很高：98.83%，protocol error 为 0.00%，但 Text relaxed 从 83.50% 降到 80.50%，说明 1k SFT 对 4B 的文本工具链有轻微负迁移。
7. GUI 是当前最大短板。最好的 GUI pointing 是 3B tool post 的 45.00%，但对应 IoU@0.5 只有 3.00%。4B tool post 的 IoU@0.5 从 1.50% 到 7.00%，有提升，但绝对值仍只有 7.00%。
8. Evidence closed 与定位正确率不等价。4B tool post 的 GUI evidence closed 是 100.00%，但 GUI pointing 只有 11.00%，说明模型能引用工具证据，但不能可靠从候选中选对目标或输出紧框。

## 5. 当前问题与可能原因

### 5.1 工具链文本 QA 不稳定

- 现象：Direct 模式文本结果整体高于 Tool 模式，尤其 4B direct post text relaxed 为 86.50%，4B tool post 为 80.50%。
- 原因：当前工具 prompt 和 SFT 数据更偏协议学习，不足以保证 OCR/Chart measure 后的答案推理优于模型直接视觉能力。
- 原因：ChartQA 中 detect/measure/OCR 对轴、图例、单位、条形高度的结构化抽取还不够强，容易引入错误证据。
- 方案：把文本 QA 拆成 direct-answer SFT 与 evidence-tool SFT 两条路线；工具路线只保留确实需要证据的样本，并加入强制 OCR/局部证据位置。

### 5.2 3B SFT 后工具调用过少

- 现象：3B tool avg tools 从 1.51 降到 0.42；DocVQA avg tools 为 0.00。
- 原因：训练样本中可能包含大量直接 FINAL，模型学到了“先满足 JSON 格式和最终答案”，而不是“先取证再回答”。
- 方案：重建 SFT v2 时按任务控制 tool-first 比例。DocVQA/ChartQA/GUI grounding 应显式要求至少一次证据工具，AI2D 可区分简单常识题和图中证据题。

### 5.3 GUI 定位质量仍低

- 现象：3B tool post pointing 高于 direct，但 IoU@0.5 仅 3.00%；4B tool post IoU@0.5 也只有 7.00%。
- 原因：detect(ui) 给候选，但模型缺少稳定的候选重排/目标匹配能力，经常输出大框、错框或坐标尺度偏移。
- 原因：当前自由生成 bbox 对 GUI 不友好。模型需要在候选集中选择，而不是凭语言直接回归坐标。
- 方案：把 GUI 改成 candidate selection/reranking：detect 生成 top-K 候选，模型输出 candidate id 或 click point，再由程序返回 bbox。训练时加入 candidate id、候选文本、候选中心点、IoU 标签。

### 5.4 AI2D 有改善，但证据质量仍需分层

- 现象：4B direct AI2D relaxed 从 91.00% 到 95.00%；3B tool AI2D 从 37.00% 到 80.00%。
- 问题：3B tool post 的 AI2D evidence closed 只有 7.00%，很多样本仍是直接作答。
- 方案：为 AI2D 加细粒度证据标签：diagram label evidence、arrow/edge evidence、object-region evidence、option-text evidence、commonsense-only。只对前四类要求强证据。

## 6. 是否达到目标

- Text relaxed：4B direct post 达到 86.50%，已接近或超过 85%。但 tool post 只有 80.50%，3B tool post 只有 72.00%。
- GUI pointing：3B tool post 达到 45.00%，相对有进展，但仍不是论文级 GUI grounding。
- GUI IoU@0.5：最高为 4B tool post 的 7.00%，仍明显不足。
- Evidence closed：4B tool post 达到 98.83%，满足闭环指标；3B tool post 为 36.50%，未满足。
- Candidate Recall@30 / Oracle GT 注入比例：本次 eval_medium_600 摘要没有输出这两个候选生成指标，因此不能用本轮 SFT 结果直接判断是否达到 Recall@30 75%+ 或 Oracle GT 注入低于 25%。这两个指标仍需单独跑 candidate/evidence 构建评估。

## 7. 下一步计划

1. 固化本轮报告与输出：保留 `outputs/eval_medium_600/*.summary.json` 作为 SFT v1 的 medium 回归基线。
2. 修复并提交 LoRA 加载兼容补丁：`eval/eval_baseline.py` 当前本地有 AutoAWQ/transformers shim，建议单独提交，避免下次 adapter 评测失败。
3. 做 failure audit：抽样 3B tool post、4B tool post 各 50 条失败样本，按 `format/protocol`、`wrong evidence`、`bad OCR`、`bad candidate`、`bbox scale`、`reasoning error` 分类。
4. 重构 SFT v2 数据：按任务分层采样，文本 QA 分 direct 与 evidence 两种轨道；DocVQA/ChartQA 必须保存 evidence bbox/span；GUI 必须保存候选 id、候选 bbox、target bbox、候选 IoU 标签。
5. 增强 detect(ui)：多源候选融合、OCR text boxes、layout boxes、icon/connected components、large-container suppression、候选去重和 score calibration。
6. 引入 GUI reranker：先优化 Candidate Recall@30 和 Oracle GT 注入比例，再训练小型 candidate selector/reranker；目标是先让 Recall@30 过 75%，Oracle GT 注入低于 25%。
7. 重新训练：不要直接扩大数据量；先用 SFT v2 的 1k/2k 小规模验证工具使用率、evidence closed、GUI pointing/IoU 是否同步改善。
8. 再跑 eval_medium_600：必须同时报告 direct/tool、pre/post、3B/4B，并固定 `Text relaxed`、`GUI pointing`、`GUI IoU@0.5`、`evidence closed`、`avg tools`、`protocol error`。

## 8. 复现备注

- 本次使用两张 3090 并行跑：GPU0 跑 3B 链路，GPU1 跑 4B 链路。
- 3B post 第一次加载 adapter 失败，原因是 `autoawq` 依赖旧 transformers 符号 `PytorchGELUTanh`。已在 `eval/eval_baseline.py` 本地加入兼容 shim 后重跑成功。
- 当前工作区还有历史未跟踪文件：`eval/eval_bailian_direct.py`、`reports/bailian_direct_smoke.md`，本报告未处理这些文件。
