# eval_medium_600 SFT v2 评测报告

生成时间：2026-05-25 14:50 Asia/Shanghai
评测状态：已结束。`ps` 未发现 eval/model 推理进程，`nvidia-smi` 显示两张 3090 无运行进程。
数据：`/root/models/datasets/evitool_eval_medium/eval_medium_600.jsonl`，共 600 条，ChartQA 150、DocVQA 150、AI2D 100、GUI 200。

## 1. 文件位置

- 最新结果目录：`outputs/eval_medium_600_sft_v2/`
- 最新汇总表：`outputs/eval_medium_600_sft_v2/comparison.md`
- 本报告：`reports/eval_medium_600_sft_v2_report.md`
- 历史 SFT v1 报告：`reports/eval_medium_600_sft_report.md`

## 2. SFT v2 总体结果

| Run | Text relaxed | Text exact | GUI pointing | GUI IoU@0.5 | Evidence closed | Avg tools | Protocol error | Final parse | Avg latency(s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3B direct v2 | 81.75% | 77.50% | 20.00% | 4.50% | - | - | - | - | 1.87 |
| 3B tool v2 | 48.25% | 43.25% | 28.50% | 6.50% | 64.50% | 1.67 | 0.83% | 65.33% | 16.56 |
| 4B direct v2 | 85.50% | 80.50% | 1.00% | 0.00% | - | - | - | - | 1.76 |
| 4B tool v2 | 83.00% | 72.25% | 7.50% | 5.00% | 95.00% | 1.93 | 0.00% | 95.00% | 18.20 |

## 3. 关键结论

- Direct 文本 QA 仍是当前最强文本路径：4B direct v2 text relaxed 为 85.50%，3B direct v2 为 81.75%。
- 4B tool v2 文本恢复到 83.00%，但 GUI pointing 只有 7.50%，IoU@0.5 为 5.00%；证据闭环 95.00%，协议稳定。
- 3B tool v2 的工具调用恢复到 1.67 次/样本，协议错误降到 0.83%，但 final parse 只有 65.33%，文本 relaxed 只有 48.25%；GUI pointing 28.50%，IoU@0.5 6.50%。
- 相比 SFT v1，SFT v2 更像“强制工具使用/证据闭环”版本，但 3B 文本能力和 GUI pointing 退化明显；不建议直接把 v2 作为主线 checkpoint。
- 本轮 3B tool 推理写满 600/600，但原始汇总在 `tool_counts` 统计时因 `action.tool` 出现 dict 而崩溃；我已从 JSONL 重建 `tool_3b_v2.jsonl.summary.json`。源码里的健壮性修复仍需要单独提交。

## 4. OmniParser A/B 判断

- 需要做，但应该先做 candidate-level A/B，不要先跑完整 VLM eval。
- 当前已有候选基线：`reports/tool_candidate_source_ablation.md` 显示 current pipeline Recall@30 为 80.20%，Oracle GT 注入估计为 19.80%，已经达到中期候选目标。
- OmniParser 目前只做了无权重 smoke：`outputs/ablate_gui_candidates_omni_smoke.md`，结果显示权重未安装时 `omniparser` 为 0 candidates，`current_omniparser` 与 current 相同。
- 下一步应下载 OmniParser 权重后跑同一脚本的 500 条 ScreenSpot GUI trace A/B：比较 `current_ocr`、`omniparser`、`current_omniparser`。只有 Recall@30、Recall@1/3 或 Oracle 注入显著改善，才把 OmniParser 放进训练数据 pipeline。

建议顺序：先修 `eval_tool_baseline.py` 的 summary 统计 bug，然后跑 OmniParser candidate A/B；如果 A/B 没有显著收益，优先做 GUI candidate reranker/selector，而不是扩大 SFT v2 数据。
