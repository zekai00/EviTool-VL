# GUI Candidate Score Eval

- Data: `outputs/gui_candidate_rl_os_atlas_linux_2k_c60_cc_pref/val.jsonl`
- Model: `/root/models/Qwen2.5-VL-3B-Instruct`
- Adapter: `outputs/gui_candidate_rl_os_atlas_linux_2k_c60_cc_pref/cc_grpo_listwise_kl_g4_a8_100step`
- Use cc_action_ids: `True`
- Max actions: `all`
- Samples: 100

| Metric | Value |
|---|---:|
| Avg reward(v2 parser metric) | 0.3656 |
| Valid candidate | 100.00% |
| Pointing | 24.00% |
| IoU@0.5 | 18.00% |
| Avg IoU | 0.1572 |
| Avg selected rank | 23.2500 |
| Avg candidate count | 37.8000 |
| Avg latency(s) | 4.9724 |
