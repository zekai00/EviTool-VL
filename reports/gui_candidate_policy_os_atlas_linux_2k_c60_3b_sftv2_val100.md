# GUI Candidate Policy Eval

- Data: `outputs/gui_candidate_rl_os_atlas_linux_2k_c60/val.jsonl`
- Policy: `model`
- Model: `/root/models/Qwen2.5-VL-3B-Instruct`
- Adapter: `checkpoints/qwen25vl_3b_evitool_sft_v2_tool_gui_lora`
- Samples: 100

| Metric | Value |
|---|---:|
| Avg reward | 0.3693 |
| Parseable | 100.00% |
| Valid candidate | 96.00% |
| Pointing | 18.00% |
| IoU@0.5 | 7.00% |
| Avg IoU | 0.0908 |
| Avg selected rank | 3.7500 |
| Avg latency(s) | 1.4257 |
