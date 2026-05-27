# GUI Candidate Miss Diagnosis

- Data: `outputs/gui_candidate_rl_os_atlas_linux_2k_c60/val.jsonl`
- Rows: 423
- Low-IoU threshold: `0.3`

## Overall

| Metric | Value |
|---|---:|
| Avg candidates | 39.4019 |
| Oracle hit | 62.88% |
| Oracle pointing | 61.70% |
| Oracle IoU@0.5 | 52.25% |
| Avg oracle IoU | 0.4426 |
| Oracle miss rows | 157 |
| Low-IoU rows | 182 |

## By OS-Atlas Data Type

| Type | Rows | Avg Cand | Oracle Hit | IoU@0.5 | Avg IoU |
|---|---:|---:|---:|---:|---:|
| `push-button` | 181 | 40.8453 | 76.80% | 66.85% | 0.5515 |
| `link` | 74 | 34.5676 | 75.68% | 67.57% | 0.5658 |
| `menu` | 30 | 41.7667 | 23.33% | 20.00% | 0.1937 |
| `page-tab` | 25 | 47.7200 | 72.00% | 64.00% | 0.5211 |
| `table-cell` | 21 | 32.8571 | 0.00% | 0.00% | 0.0000 |
| `label` | 19 | 32.2105 | 68.42% | 0.00% | 0.2044 |
| `toggle-button` | 17 | 34.6471 | 70.59% | 58.82% | 0.4959 |
| `tree-item` | 13 | 60.0000 | 23.08% | 23.08% | 0.1746 |
| `radio-button` | 12 | 20.6667 | 50.00% | 50.00% | 0.3463 |
| `check-menu-item` | 5 | 50.0000 | 0.00% | 0.00% | 0.0027 |
| `list-item` | 5 | 46.8000 | 80.00% | 80.00% | 0.6149 |
| `menu-item` | 5 | 50.0000 | 20.00% | 0.00% | 0.0749 |
| `icon` | 4 | 37.5000 | 25.00% | 0.00% | 0.0696 |
| `radio-menu-item` | 4 | 50.0000 | 0.00% | 0.00% | 0.0629 |
| `check-box` | 3 | 10.0000 | 100.00% | 66.67% | 0.4996 |
| `combo-box` | 3 | 51.0000 | 66.67% | 66.67% | 0.5703 |
| `scroll-bar` | 1 | 50.0000 | 0.00% | 0.00% | 0.0000 |
| `text` | 1 | 34.0000 | 100.00% | 100.00% | 0.7430 |

## Oracle Hit Sources

| Source | Count |
|---|---:|
| `omniparser` | 266 |
