# GUI Candidate Policy Eval

- Data: `outputs/gui_candidate_rl_os_atlas_linux_2k_c60/val.jsonl`
- Policy: `model`
- Reward version: `v2`
- Model: `/root/models/Qwen2.5-VL-3B-Instruct`
- Adapter: `outputs/gui_candidate_rl_os_atlas_linux_2k_c60/grpo_3b_candidate_c60_sft_rewardv2_g2_t10_len16_100step`
- Samples: 100

| Metric | Value |
|---|---:|
| Avg reward | 0.2731 |
| Parseable | 100.00% |
| Valid candidate | 96.00% |
| Pointing | 18.00% |
| IoU@0.5 | 7.00% |
| Avg IoU | 0.0911 |
| Avg selected rank | 5.1354 |
| Avg latency(s) | 1.5554 |
