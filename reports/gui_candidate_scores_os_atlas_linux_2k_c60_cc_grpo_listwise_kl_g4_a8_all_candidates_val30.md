# GUI Candidate Score Eval

- Data: `outputs/gui_candidate_rl_os_atlas_linux_2k_c60_cc_pref/val.jsonl`
- Model: `/root/models/Qwen2.5-VL-3B-Instruct`
- Adapter: `outputs/gui_candidate_rl_os_atlas_linux_2k_c60_cc_pref/cc_grpo_listwise_kl_g4_a8_100step`
- Use cc_action_ids: `False`
- Max actions: `all`
- Samples: 30

| Metric | Value |
|---|---:|
| Avg reward(v2 parser metric) | 0.2471 |
| Valid candidate | 100.00% |
| Pointing | 6.67% |
| IoU@0.5 | 3.33% |
| Avg IoU | 0.0394 |
| Avg selected rank | 23.0000 |
| Avg candidate count | 39.1333 |
| Avg latency(s) | 18.5722 |
