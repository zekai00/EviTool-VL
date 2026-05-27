# GUI Candidate Miss Diagnosis

- Data: `outputs/gui_candidate_rl_os_atlas_linux_2k_enhanced_c100_base_local_20260527_1210/val.jsonl`
- Rows: 423
- Low-IoU threshold: `0.3`

## Overall

| Metric | Value |
|---|---:|
| Avg candidates | 79.0686 |
| Oracle hit | 78.72% |
| Oracle pointing | 76.83% |
| Oracle IoU@0.5 | 54.85% |
| Avg oracle IoU | 0.5031 |
| Oracle miss rows | 90 |
| Low-IoU rows | 141 |

## By OS-Atlas Data Type

| Type | Rows | Avg Cand | Oracle Hit | IoU@0.5 | Avg IoU |
|---|---:|---:|---:|---:|---:|
| `push-button` | 181 | 78.4475 | 86.74% | 68.51% | 0.5901 |
| `link` | 74 | 66.8919 | 94.59% | 71.62% | 0.6508 |
| `menu` | 30 | 87.7333 | 80.00% | 20.00% | 0.3311 |
| `page-tab` | 25 | 85.3200 | 72.00% | 68.00% | 0.5636 |
| `table-cell` | 21 | 100.0000 | 0.00% | 0.00% | 0.0004 |
| `label` | 19 | 49.7895 | 84.21% | 10.53% | 0.3181 |
| `toggle-button` | 17 | 78.1765 | 88.24% | 64.71% | 0.5784 |
| `tree-item` | 13 | 100.0000 | 61.54% | 23.08% | 0.2893 |
| `radio-button` | 12 | 98.3333 | 100.00% | 75.00% | 0.5755 |
| `check-menu-item` | 5 | 100.0000 | 0.00% | 0.00% | 0.0027 |
| `list-item` | 5 | 82.0000 | 80.00% | 40.00% | 0.4483 |
| `menu-item` | 5 | 100.0000 | 20.00% | 0.00% | 0.0749 |
| `icon` | 4 | 68.5000 | 50.00% | 0.00% | 0.1089 |
| `radio-menu-item` | 4 | 100.0000 | 0.00% | 0.00% | 0.0629 |
| `check-box` | 3 | 56.0000 | 100.00% | 66.67% | 0.4996 |
| `combo-box` | 3 | 91.6667 | 66.67% | 66.67% | 0.5703 |
| `scroll-bar` | 1 | 100.0000 | 0.00% | 0.00% | 0.0086 |
| `text` | 1 | 50.0000 | 100.00% | 100.00% | 0.7430 |

## Oracle Hit Sources

| Source | Count |
|---|---:|
| `base_c60` | 243 |
| `text_expanded` | 16 |
| `ui_edge+ui_rect` | 11 |
| `ocr_text` | 10 |
| `ocr_text+ui_edge+ui_rect` | 9 |
| `layout+ocr_text+text_visual+ui_edge+ui_rect` | 5 |
| `layout+ocr_text+text_visual` | 5 |
| `layout+ocr_text+ui_edge+ui_rect` | 5 |
| `icon_visual` | 5 |
| `row_container` | 4 |
| `ocr_text+text_visual+ui_edge+ui_rect` | 4 |
| `icon_visual+ui_edge+ui_rect` | 4 |
| `text_visual` | 4 |
| `ocr_text+text_visual` | 4 |
| `layout+ui_edge+ui_rect` | 1 |
| `icon_visual+layout+ocr_text+ui_edge+ui_rect` | 1 |
| `icon_visual+layout+ocr_text+text_visual+ui_edge+ui_rect` | 1 |
| `icon_visual+text_expanded` | 1 |
