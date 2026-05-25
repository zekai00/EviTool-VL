# GUI Candidate Policy Eval

- Data: `outputs/gui_candidate_rl/val.jsonl`
- Policy: `model`
- Model: `/root/models/Qwen2.5-VL-3B-Instruct`
- Adapter: `outputs/gui_candidate_rl/grpo_3b_sftv2_20step`
- Samples: 100

| Metric | Value |
|---|---:|
| Avg reward | 0.6750 |
| Parseable | 100.00% |
| Valid candidate | 97.00% |
| Pointing | 58.00% |
| IoU@0.5 | 53.00% |
| Avg IoU | 0.4419 |
| Avg selected rank | 6.7835 |
| Avg latency(s) | 1.8628 |
