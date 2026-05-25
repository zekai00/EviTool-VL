# GUI Candidate-Selection RL Smoke

- Data: `/root/models/datasets/evitool_traces_1k/train_traces_1000.jsonl`
- Samples: 8
- Candidate source: OmniParser-only YOLO detector
- Max candidates: 30
- Action schema: `{"candidate_id": "cXX"}`
- Reward: format + valid candidate + pointing + IoU@0.5 + shaped IoU, with invalid/format penalties.

## Candidate Oracle

| Metric | Value |
|---|---:|
| Avg candidates | 25.8750 |
| Oracle hit rate | 100.00% |
| Oracle pointing | 100.00% |
| Oracle IoU@0.5 | 100.00% |
| Avg oracle IoU | 0.7806 |

## Policy Smoke

| Policy | Avg reward | Valid | Pointing | IoU@0.5 | Avg IoU | Avg rank |
|---|---:|---:|---:|---:|---:|---:|
| `top1` | 0.3884 | 100.00% | 12.50% | 12.50% | 0.1121 | 1.0000 |
| `oracle` | 1.0000 | 100.00% | 100.00% | 100.00% | 0.7806 | 9.3750 |
| `random` | 0.3000 | 100.00% | 0.00% | 0.00% | 0.0000 | 7.5000 |
| `invalid` | -0.2500 | 0.00% | 0.00% | 0.00% | 0.0000 | - |

## Decision

- This is a plumbing smoke, not model training. It validates the candidate action space and reward scale before wiring GRPO/PPO.
- If oracle hit/pointing is high enough, the next step is to connect a VLM policy that emits `candidate_id` and run a tiny GRPO step.
- If top1 is much worse than oracle, RL has a meaningful learning signal: selection matters and the action space is not already solved by detector score.
