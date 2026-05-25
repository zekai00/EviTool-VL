# GUI Candidate Score Eval

- Data: `outputs/gui_candidate_rl_os_atlas_linux_2k_c60_cc_pref/val.jsonl`
- Model: `/root/models/Qwen2.5-VL-3B-Instruct`
- Adapter: `checkpoints/qwen25vl_3b_gui_candidate_c60_sft_warmup_lora`
- Use cc_action_ids: `False`
- Max actions: `all`
- Samples: 30

| Metric | Value |
|---|---:|
| Avg reward(v2 parser metric) | 0.2529 |
| Valid candidate | 100.00% |
| Pointing | 10.00% |
| IoU@0.5 | 6.67% |
| Avg IoU | 0.0689 |
| Avg selected rank | 2.5000 |
| Avg candidate count | 39.1333 |
| Avg latency(s) | 18.6052 |
