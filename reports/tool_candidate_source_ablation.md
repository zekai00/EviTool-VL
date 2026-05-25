# Tool Candidate Source A/B 测试

- 数据：`/root/models/datasets/evitool_traces_1k/train_traces_1000.jsonl` 中的 ScreenSpot GUI trace，共 500 条。
- 评估：不跑 VLM，仅比较 detector 候选是否覆盖 GT。命中定义与 trace 构建一致：候选 bbox IoU >= 0.3 或候选中心点落入 GT bbox。
- 目的：判断现有 `detect(ui)` 各候选来源是否值得保留，并为后续 OmniParser/其他 detector 接入提供统一对比口径。

## 1. Pipeline A/B

| Variant | Recall@30 | Oracle GT 注入估计 | Avg Best IoU | Avg Hit Rank |
|---|---:|---:|---:|---:|
| `A_initial_no_ocr` | 62.20% | 37.80% | 0.2906 | 10.97 |
| `B_pipeline_with_ocr_fallback` | 80.20% | 19.80% | 0.3561 | 9.09 |

- OCR fallback 带来的 Recall@30 增益：18.00%。
- 当前完整 pipeline 已达到中期候选目标：Recall@30 80.20%，Oracle GT 注入估计 19.80%。

## 2. 候选源消融

| Source | Source-only Recall@30 | Remove-source Recall@30 | Removal Delta | Full 候选出现数 | 被选中来源数 |
|---|---:|---:|---:|---:|---:|
| `query_prior` | 5.40% | 74.80% | +5.40% | 2087 | 52 |
| `ocr_text` | 18.00% | 62.20% | +18.00% | 2482 | 90 |
| `text_expanded` | 51.60% | 59.40% | +20.80% | 6462 | 224 |
| `text_visual` | 21.20% | 78.40% | +1.80% | 1069 | 45 |
| `row_container` | 21.80% | 79.60% | +0.60% | 2270 | 56 |
| `ui_rect` | 26.80% | 74.80% | +5.40% | 1741 | 96 |
| `ui_edge` | 23.20% | 76.20% | +4.00% | 1498 | 81 |
| `icon_visual` | 17.00% | 75.00% | +5.20% | 1130 | 63 |
| `layout` | 20.00% | 77.20% | +3.00% | 1034 | 63 |

## 3. 结论

- `include_ocr`/OCR fallback 是当前最明确有效的增益来源，主要把初始 miss 的文本型目标拉回候选集。
- `query_prior` 对窗口控制、工具栏、颜色等 query 可预测位置有明显贡献，但容易产生大量高分先验框，后续需要 reranker 抑制误排。
- `text_expanded`、`row_container`、`ocr_text` 对文本按钮和列表项有价值，应保留。
- `ui_edge`、`ui_rect`、`icon_visual`、`layout` 单独能力有限，但作为召回补充仍可能覆盖非文本/图标类目标，不建议直接删除。
- OmniParser 的接入价值需要用同一脚本比较：`current_detect` vs `current_detect+OmniParser` vs `OmniParser_only`。只有 Recall@30 或 Oracle 注入显著改善，才进入训练数据构建。