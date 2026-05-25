# GUI Candidate RL Current Status

Generated: 2026-05-25

## Fixed Data

- Data dir: `outputs/gui_candidate_rl/`
- Source: `/root/models/datasets/evitool_traces_1k/train_traces_1000.jsonl`
- Samples: 500 GUI rows
- Split: 400 train / 100 val
- Candidate source: OmniParser-only YOLO detector
- Overlay images: `outputs/gui_candidate_rl/overlays/`

| Split | Count | Avg Candidates | Oracle Pointing | Oracle IoU@0.5 |
|---|---:|---:|---:|---:|
| all | 500 | 20.954 | 81.80% | 69.60% |
| train | 400 | 20.948 | 81.00% | 68.50% |
| val | 100 | 20.980 | 85.00% | 74.00% |

## Val Baselines

| Policy | Avg Reward | Valid | Pointing | IoU@0.5 | Avg IoU | Avg Rank |
|---|---:|---:|---:|---:|---:|---:|
| top1 | 0.3294 | 97.00% | 7.00% | 6.00% | 0.0495 | 1.0000 |
| random | 0.3116 | 97.00% | 4.00% | 4.00% | 0.0360 | 10.7629 |
| 3B SFT v2 model | 0.6750 | 97.00% | 58.00% | 53.00% | 0.4413 | 6.5361 |
| oracle | 0.8432 | 97.00% | 85.00% | 74.00% | 0.6141 | 8.2887 |

## Decision

- Candidate-selection RL is now the right next experiment.
- The 3B SFT v2 adapter is a good warm start: it is much better than detector score top1, but still has a 27 pp pointing gap to oracle on val.
- Train single-step GRPO on `candidate_id` selection before attempting multi-turn tool RL.

## GRPO Smoke

One-step GRPO smoke completed successfully with:

- Output adapter: `outputs/gui_candidate_rl/grpo_3b_sftv2_smoke/`
- Train subset: 4 rows
- `num_generations`: 2
- `max_steps`: 1
- Reward mean: 0.6500
- Reward std: 0.4950
- Valid candidate: 100.00%
- Pointing: 50.00%
- IoU@0.5: 50.00%
- Grad norm: 2.6928

This validates the training loop only. It is not a meaningful trained checkpoint yet.

## 20-Step Trial

Command output adapter: `outputs/gui_candidate_rl/grpo_3b_sftv2_20step/`

Val result:

| Adapter | Avg Reward | Valid | Pointing | IoU@0.5 | Avg IoU | Avg Rank |
|---|---:|---:|---:|---:|---:|---:|
| SFT v2 warm start | 0.6750 | 97.00% | 58.00% | 53.00% | 0.4413 | 6.5361 |
| GRPO 20-step | 0.6750 | 97.00% | 58.00% | 53.00% | 0.4419 | 6.7835 |

Prediction diff against SFT v2: 2 / 100 val rows.

Conclusion: the GRPO loop works, but this configuration barely changes behavior. Training logs show many batches with `reward_std=0`, so GRPO has no useful within-group advantage on those batches. The next run should increase exploration before increasing duration.

## Commands

Dry-run reward/data plumbing:

```bash
python3 scripts/train_gui_candidate_grpo.py --dry-run --limit 8 --eval-limit 4
```

One-step GRPO smoke:

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/train_gui_candidate_grpo.py \
  --model /root/models/Qwen2.5-VL-3B-Instruct \
  --adapter checkpoints/qwen25vl_3b_evitool_sft_v2_tool_gui_lora \
  --train-data outputs/gui_candidate_rl/train.jsonl \
  --eval-data outputs/gui_candidate_rl/val.jsonl \
  --output-dir outputs/gui_candidate_rl/grpo_3b_sftv2_smoke \
  --limit 4 \
  --eval-limit 2 \
  --num-generations 2 \
  --max-steps 1 \
  --learning-rate 1e-6
```

Next exploratory run:

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/train_gui_candidate_grpo.py \
  --model /root/models/Qwen2.5-VL-3B-Instruct \
  --adapter checkpoints/qwen25vl_3b_evitool_sft_v2_tool_gui_lora \
  --train-data outputs/gui_candidate_rl/train.jsonl \
  --eval-data outputs/gui_candidate_rl/val.jsonl \
  --output-dir outputs/gui_candidate_rl/grpo_3b_sftv2_g4_t12_100step \
  --limit 160 \
  --eval-limit 32 \
  --num-generations 4 \
  --max-steps 100 \
  --learning-rate 1e-6 \
  --temperature 1.2 \
  --top-p 0.95
```
