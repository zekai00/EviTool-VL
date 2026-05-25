# GUI Candidate Score Eval

- Data: `outputs/gui_candidate_rl_os_atlas_linux_2k_c60_cc_pref/val.jsonl`
- Model: `/root/models/Qwen2.5-VL-3B-Instruct`
- Adapter: `checkpoints/qwen25vl_3b_gui_candidate_c60_sft_warmup_lora`
- Use cc_action_ids: `True`
- Max actions: `3`
- Samples: 1

| Metric | Value |
|---|---:|
| Avg reward(v2 parser metric) | 0.9994 |
| Valid candidate | 100.00% |
| Pointing | 100.00% |
| IoU@0.5 | 100.00% |
| Avg IoU | 0.7473 |
| Avg selected rank | 8.0000 |
| Avg candidate count | 18.0000 |
| Avg latency(s) | 2.4715 |
