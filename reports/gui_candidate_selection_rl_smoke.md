# GUI Candidate-Selection RL Smoke

- Data: `/root/models/datasets/evitool_traces_1k/train_traces_1000.jsonl`
- Samples: 500
- Candidate source: OmniParser-only YOLO detector
- Max candidates: 30
- Action schema: `{"candidate_id": "cXX"}`
- Reward: format + valid candidate + pointing + IoU@0.5 + shaped IoU, with invalid/format penalties.

## Candidate Oracle

| Metric | Value |
|---|---:|
| Avg candidates | 20.9540 |
| Oracle hit rate | 82.60% |
| Oracle pointing | 81.80% |
| Oracle IoU@0.5 | 69.60% |
| Avg oracle IoU | 0.5892 |

## Policy Smoke

| Policy | Avg reward | Valid | Pointing | IoU@0.5 | Avg IoU | Avg rank |
|---|---:|---:|---:|---:|---:|---:|
| `top1` | 0.3493 | 98.40% | 9.00% | 7.40% | 0.0664 | 1.0000 |
| `oracle` | 0.8274 | 98.40% | 81.80% | 69.60% | 0.5892 | 8.7480 |
| `random` | 0.3245 | 98.40% | 5.20% | 4.20% | 0.0369 | 10.9472 |
| `invalid` | -0.2500 | 0.00% | 0.00% | 0.00% | 0.0000 | - |

## Decision

- This is a plumbing smoke, not model training. It validates the candidate action space and reward scale before wiring GRPO/PPO.
- If oracle hit/pointing is high enough, the next step is to connect a VLM policy that emits `candidate_id` and run a tiny GRPO step.
- If top1 is much worse than oracle, RL has a meaningful learning signal: selection matters and the action space is not already solved by detector score.
