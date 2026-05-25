# GUI Candidate Score Eval

- Data: `outputs/gui_candidate_rl_os_atlas_linux_2k_c60_cc_pref/val.jsonl`
- Model: `/root/models/Qwen2.5-VL-3B-Instruct`
- Adapter: `checkpoints/qwen25vl_3b_gui_candidate_c60_sft_warmup_lora`
- Use cc_action_ids: `True`
- Max actions: `all`
- Samples: 100

| Metric | Value |
|---|---:|
| Avg reward(v2 parser metric) | 0.2552 |
| Valid candidate | 100.00% |
| Pointing | 11.00% |
| IoU@0.5 | 9.00% |
| Avg IoU | 0.0807 |
| Avg selected rank | 2.2600 |
| Avg candidate count | 37.8000 |
| Avg latency(s) | 4.8177 |
