# GUI Candidate Policy Eval

- Data: `outputs/gui_candidate_rl_smoke/all.jsonl`
- Policy: `model`
- Model: `/root/models/Qwen2.5-VL-3B-Instruct`
- Adapter: `checkpoints/qwen25vl_3b_evitool_sft_v2_tool_gui_lora`
- Samples: 8

| Metric | Value |
|---|---:|
| Avg reward | 0.5625 |
| Parseable | 100.00% |
| Valid candidate | 100.00% |
| Pointing | 37.50% |
| IoU@0.5 | 37.50% |
| Avg IoU | 0.3118 |
| Avg selected rank | 3.8750 |
| Avg latency(s) | 1.3652 |
