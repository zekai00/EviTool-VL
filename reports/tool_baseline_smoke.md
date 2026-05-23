# Prompt-Only Tool-Use Smoke Results

Dataset: first sample per task from `eval_mini_100`.

| Model | Samples | Text Relaxed | ScreenSpot Pointing | Final Parse | Parse Error | Protocol Error | Avg Tools | Tool Success | Evidence Closed | Tool Counts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen2.5-VL-3B | 4 | 66.67% | 100.00% | 75.00% | 25.00% | 75.00% | 1.00 | 100.00% | 0.00% | `{"detect": 3, "click": 1}` |
| Qwen3-VL-4B | 4 | 33.33% | 0.00% | 100.00% | 0.00% | 0.00% | 1.75 | 100.00% | 100.00% | `{"detect": 3, "ocr": 1, "zoom": 2, "click": 1}` |

Output files:

- `outputs/tool_smoke_3b_eval_mini_v4.jsonl`
- `outputs/tool_smoke_3b_eval_mini_v4.jsonl.summary.json`
- `outputs/tool_smoke_4b_eval_mini.jsonl`
- `outputs/tool_smoke_4b_eval_mini.jsonl.summary.json`

Interpretation:

- Qwen3-VL-4B follows the action-observation-final protocol much better in this smoke test.
- Qwen2.5-VL-3B often answers directly despite prompt instructions, so protocol error rate is high.
- This confirms why SFT data should explicitly train the JSON tool protocol and evidence citations.
