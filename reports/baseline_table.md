# Baseline Table

Dataset: `data/eval_mini/eval_mini_100.jsonl`
Setting: direct answer, greedy decoding, `max_new_tokens=128`

| Run | Model | Setting | Samples | Text Exact | Text Relaxed | ChartQA Relaxed | DocVQA Relaxed | AI2D Acc | ScreenSpot BBox Parse | ScreenSpot IoU@0.5 | ScreenSpot Pointing | Empty Output | Avg Latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 | Qwen2.5-VL-3B-Instruct | direct answer | 100 | 77.33% | 81.33% | 63.33% | 92.00% | 95.00% | 32.00% | 0.00% | 12.00% | 0.00% | 0.915s |
| B3 | Qwen3-VL-4B-Instruct | direct answer | 100 | 72.00% | 76.00% | 66.67% | 80.00% | 85.00% | 100.00% | 0.00% | 8.00% | 0.00% | 0.889s |

Prediction files:

- `outputs/baseline_3b_direct_eval_mini_100.jsonl`
- `outputs/baseline_3b_direct_eval_mini_100.jsonl.summary.json`
- `outputs/baseline_4b_direct_eval_mini_100.jsonl`
- `outputs/baseline_4b_direct_eval_mini_100.jsonl.summary.json`

Smoke files:

- `outputs/smoke_3b_direct.jsonl`
- `outputs/smoke_4b_direct.jsonl`

Notes:

- `Text Exact` and `Text Relaxed` are computed over ChartQA, DocVQA, and AI2D only.
- ChartQA relaxed numeric match allows 3% relative error.
- AI2D accepts either the correct option letter or the correct option text.
- ScreenSpot evaluates parsed bbox IoU@0.5 and whether the predicted bbox center falls inside the GT bbox.
