# GUI Candidate Score Eval

- Data: `outputs/gui_candidate_rl_os_atlas_linux_2k_c60_cc_pref/val.jsonl`
- Model: `/root/models/Qwen2.5-VL-3B-Instruct`
- Adapter: `outputs/gui_candidate_rl_os_atlas_linux_2k_c60_cc_pref/cc_grpo_ddp_g4_a8_100step`
- Use cc_action_ids: `False`
- Max actions: `all`
- Samples: 30

| Metric | Value |
|---|---:|
| Avg reward(v2 parser metric) | 0.2958 |
| Valid candidate | 100.00% |
| Pointing | 13.33% |
| IoU@0.5 | 6.67% |
| Avg IoU | 0.0747 |
| Avg selected rank | 16.1000 |
| Avg candidate count | 39.1333 |
| Avg latency(s) | 18.6490 |
