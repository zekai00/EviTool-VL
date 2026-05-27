# Full-Aware GUI Candidate RL Dataset Report

## 构建配置

- Input dir: `outputs/gui_candidate_rl_os_atlas_linux_2k_enhanced_c100_base_dashscope_20260527_1210`
- Output dir: `outputs/gui_candidate_rl_os_atlas_linux_2k_enhanced_c100_base_dashscope_full_aware_20260527_1329`
- Max actions per row: `16`
- Train oracle gate: hit=`True`, min_iou=`0.05`
- Model score cache files: `0`
- Model-score coverage is allowed to be 0; in that case top-ranked detector mistakes are used as a deterministic current-policy proxy.

## Split 质量

| Split | Rows | Trainable | Avg candidates | Avg actions | Oracle hit | Oracle IoU@0.5 | Avg oracle IoU | Avg action reward std | Zero-std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 2000 | 1656 | 88.7125 | 15.9830 | 83.15% | 56.10% | 0.5208 | 0.2187 | 0.00% |
| train | 1294 | 1294 | 90.2666 | 15.9776 | 100.00% | 67.16% | 0.6178 | 0.2185 | 0.00% |
| val | 423 | 362 | 79.7187 | 15.9882 | 85.82% | 59.81% | 0.5379 | 0.2195 | 0.00% |
| val_oracle_hit | 362 | 362 | 77.2735 | 15.9862 | 100.00% | 69.89% | 0.6160 | 0.2195 | 0.00% |

## Full-Aware Action 角色分布（Train）

| Role | Count |
|---|---:|
| `current_model_proxy_false_positive` | 5176 |
| `detector_top_rank_wrong` | 5176 |
| `distant_false_positive` | 2158 |
| `high_iou_wrong` | 3882 |
| `high_reward_wrong` | 3882 |
| `near_center_wrong` | 3882 |
| `oracle_positive` | 1294 |
| `random_full_pool_negative` | 9272 |
| `text_similar_wrong` | 202 |

## 泄漏检查

- Train unique images: `79`
- Val unique images: `21`
- Train/Val image overlap: `0`

## 数据字段

- `candidates`: 完整候选池，保留所有 candidate bbox / rank / score。
- `candidate_rewards_v3`: 每个候选的几何 reward。
- `full_aware_action_ids`: 用于 CC-GRPO/listwise 的 full-aware action subset。
- `full_aware_action_roles`: 每个 action 的来源角色，例如 oracle、top-rank wrong、near-center wrong。
- `cc_action_ids`: 与 `full_aware_action_ids` 相同，保持旧训练脚本兼容。

## 判断

这版数据保留 full candidates，同时将训练 action set 对齐到 full-pool 错误分布；它比旧 `cc_action_ids` 更适合后续 full-candidate 评估。如果后续生成了全量 current-model score cache，可以用同一脚本重新构建，把 `current_model_proxy_false_positive` 替换成真实 `current_model_false_positive`。
