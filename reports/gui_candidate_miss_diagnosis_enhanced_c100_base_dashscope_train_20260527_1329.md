# GUI Candidate Miss Diagnosis

- Data: `outputs/gui_candidate_rl_os_atlas_linux_2k_enhanced_c100_base_dashscope_20260527_1210/train.jsonl`
- Rows: 1577
- Low-IoU threshold: `0.3`

## Overall

| Metric | Value |
|---|---:|
| Avg candidates | 91.1249 |
| Oracle hit | 82.44% |
| Oracle pointing | 81.17% |
| Oracle IoU@0.5 | 55.10% |
| Avg oracle IoU | 0.5162 |
| Oracle miss rows | 277 |
| Low-IoU rows | 441 |

## By OS-Atlas Data Type

| Type | Rows | Avg Cand | Oracle Hit | IoU@0.5 | Avg IoU |
|---|---:|---:|---:|---:|---:|
| `push-button` | 563 | 89.7922 | 86.15% | 66.96% | 0.5735 |
| `menu` | 161 | 94.4907 | 89.44% | 43.48% | 0.5050 |
| `label` | 150 | 80.0267 | 80.67% | 26.00% | 0.3532 |
| `tree-item` | 112 | 98.6696 | 95.54% | 77.68% | 0.6971 |
| `link` | 93 | 81.0753 | 81.72% | 47.31% | 0.4740 |
| `page-tab` | 92 | 95.0978 | 90.22% | 64.13% | 0.5904 |
| `menu-item` | 88 | 99.3864 | 90.91% | 46.59% | 0.4893 |
| `table-cell` | 84 | 100.6190 | 5.95% | 0.00% | 0.0606 |
| `toggle-button` | 74 | 95.1216 | 93.24% | 75.68% | 0.6354 |
| `radio-button` | 52 | 96.8846 | 100.00% | 84.62% | 0.6625 |
| `list-item` | 35 | 87.2857 | 65.71% | 37.14% | 0.3630 |
| `check-box` | 26 | 91.7308 | 61.54% | 26.92% | 0.3450 |
| `check-menu-item` | 18 | 62.5556 | 94.44% | 83.33% | 0.7683 |
| `combo-box` | 17 | 100.0588 | 94.12% | 88.24% | 0.7575 |
| `icon` | 4 | 62.0000 | 0.00% | 0.00% | 0.0275 |
| `page-tab-list` | 3 | 99.0000 | 100.00% | 0.00% | 0.2011 |
| `scroll-bar` | 2 | 103.0000 | 0.00% | 0.00% | 0.0526 |
| `text` | 2 | 100.0000 | 100.00% | 100.00% | 0.6333 |
| `tree` | 1 | 101.0000 | 100.00% | 0.00% | 0.3216 |

## Oracle Hit Sources

| Source | Count |
|---|---:|
| `base_c60` | 811 |
| `dashscope_target_augment+gui-plus-2026-02-26` | 255 |
| `text_expanded` | 65 |
| `ocr_text+ui_edge+ui_rect` | 27 |
| `ui_edge+ui_rect` | 22 |
| `layout+ocr_text+ui_edge+ui_rect` | 21 |
| `layout+ocr_text+text_visual+ui_edge+ui_rect` | 17 |
| `row_container` | 13 |
| `icon_visual+ui_edge+ui_rect` | 10 |
| `text_visual` | 9 |
| `icon_visual+text_expanded` | 9 |
| `ocr_text+text_visual` | 8 |
| `layout+ocr_text+text_visual` | 6 |
| `ocr_text` | 5 |
| `layout+ui_edge+ui_rect` | 4 |
| `ocr_text+text_visual+ui_edge+ui_rect` | 3 |
| `icon_visual+ocr_text+ui_edge+ui_rect` | 3 |
| `icon_visual+layout+ocr_text+text_visual+ui_edge+ui_rect` | 2 |
| `ocr_text+text_visual+ui_rect` | 2 |
| `icon_visual+ocr_text+text_visual+ui_edge+ui_rect` | 2 |
| `icon_visual` | 2 |
| `icon_visual+layout+ui_edge+ui_rect` | 2 |
| `icon_visual+layout+ocr_text+ui_edge+ui_rect` | 1 |
| `ui_rect` | 1 |
