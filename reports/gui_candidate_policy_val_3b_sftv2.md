# GUI Candidate Policy Eval

- Data: `outputs/gui_candidate_rl/val.jsonl`
- Policy: `model`
- Model: `/root/models/Qwen2.5-VL-3B-Instruct`
- Adapter: `checkpoints/qwen25vl_3b_evitool_sft_v2_tool_gui_lora`
- Samples: 100

| Metric | Value |
|---|---:|
| Avg reward | 0.6750 |
| Parseable | 100.00% |
| Valid candidate | 97.00% |
| Pointing | 58.00% |
| IoU@0.5 | 53.00% |
| Avg IoU | 0.4413 |
| Avg selected rank | 6.5361 |
| Avg latency(s) | 2.0701 |
