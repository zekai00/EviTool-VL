# GUI Candidate Policy Eval

- Data: `outputs/gui_candidate_rl_os_atlas_linux_2k_c60/val.jsonl`
- Policy: `model`
- Reward version: `v2`
- Model: `/root/models/Qwen2.5-VL-3B-Instruct`
- Adapter: `checkpoints/qwen25vl_3b_gui_candidate_c60_sft_warmup_lora`
- Samples: 100

| Metric | Value |
|---|---:|
| Avg reward | 0.2756 |
| Parseable | 100.00% |
| Valid candidate | 96.00% |
| Pointing | 19.00% |
| IoU@0.5 | 7.00% |
| Avg IoU | 0.0934 |
| Avg selected rank | 4.9583 |
| Avg latency(s) | 1.4426 |
