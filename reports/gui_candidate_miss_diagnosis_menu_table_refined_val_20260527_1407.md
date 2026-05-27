# GUI Candidate Miss Diagnosis

- Data: `outputs/gui_candidate_rl_os_atlas_linux_2k_enhanced_c100_base_dashscope_menu_table_refined_20260527_1407/val.jsonl`
- Rows: 423
- Low-IoU threshold: `0.3`

## Overall

| Metric | Value |
|---|---:|
| Avg candidates | 81.8983 |
| Oracle hit | 89.83% |
| Oracle pointing | 87.94% |
| Oracle IoU@0.5 | 69.50% |
| Avg oracle IoU | 0.6151 |
| Oracle miss rows | 43 |
| Low-IoU rows | 83 |

## By OS-Atlas Data Type

| Type | Rows | Avg Cand | Oracle Hit | IoU@0.5 | Avg IoU |
|---|---:|---:|---:|---:|---:|
| `push-button` | 181 | 78.9558 | 89.50% | 69.61% | 0.6038 |
| `link` | 74 | 67.4324 | 98.65% | 79.73% | 0.6905 |
| `menu` | 30 | 110.9000 | 100.00% | 90.00% | 0.8966 |
| `page-tab` | 25 | 85.5200 | 76.00% | 68.00% | 0.5681 |
| `table-cell` | 21 | 102.8571 | 71.43% | 71.43% | 0.7040 |
| `label` | 19 | 50.4737 | 84.21% | 52.63% | 0.4561 |
| `toggle-button` | 17 | 78.6471 | 94.12% | 64.71% | 0.5850 |
| `tree-item` | 13 | 100.9231 | 69.23% | 23.08% | 0.2904 |
| `radio-button` | 12 | 98.9167 | 100.00% | 75.00% | 0.5755 |
| `check-menu-item` | 5 | 123.8000 | 100.00% | 100.00% | 0.7986 |
| `list-item` | 5 | 82.6000 | 100.00% | 60.00% | 0.5262 |
| `menu-item` | 5 | 124.6000 | 100.00% | 80.00% | 0.5991 |
| `icon` | 4 | 68.7500 | 75.00% | 0.00% | 0.1120 |
| `radio-menu-item` | 4 | 103.0000 | 75.00% | 0.00% | 0.0922 |
| `check-box` | 3 | 57.6667 | 100.00% | 66.67% | 0.4996 |
| `combo-box` | 3 | 92.0000 | 66.67% | 66.67% | 0.5703 |
| `scroll-bar` | 1 | 101.0000 | 100.00% | 0.00% | 0.2302 |
| `text` | 1 | 50.0000 | 100.00% | 100.00% | 0.7430 |

## Oracle Hit Sources

| Source | Count |
|---|---:|
| `base_c60` | 230 |
| `dashscope_target_augment+gui-plus-2026-02-26` | 38 |
| `menu_instruction_prior` | 20 |
| `spreadsheet_cell_query` | 15 |
| `text_expanded` | 14 |
| `menu_row_expanded` | 12 |
| `ocr_text` | 8 |
| `ui_edge+ui_rect` | 7 |
| `icon_visual` | 5 |
| `row_container` | 4 |
| `layout+ocr_text+text_visual+ui_edge+ui_rect` | 4 |
| `ocr_text+text_visual` | 4 |
| `layout+ocr_text+text_visual` | 3 |
| `ocr_text+text_visual+ui_edge+ui_rect` | 3 |
| `icon_visual+ui_edge+ui_rect` | 3 |
| `layout+ocr_text+ui_edge+ui_rect` | 2 |
| `ocr_text+ui_edge+ui_rect` | 2 |
| `text_visual` | 2 |
| `layout+ui_edge+ui_rect` | 1 |
| `icon_visual+layout+ocr_text+ui_edge+ui_rect` | 1 |
| `icon_visual+layout+ocr_text+text_visual+ui_edge+ui_rect` | 1 |
| `icon_visual+text_expanded` | 1 |
