# Prompt-Only Tool-Use Full Eval

Dataset: `data/eval_mini/eval_mini_100.jsonl`

Tool protocol: prompt-only JSON action-observation-final loop. The model does
not use native function calling; `eval/eval_tool_baseline.py` parses text JSON,
executes local visual tools, appends observations, and scores the final answer.

Run configuration:

- `max_tool_steps=2`
- `max_new_tokens=384`
- no forced retry for direct final
- no evidence repair pass

Output files:

- `outputs/tool_3b_eval_mini_100.jsonl`
- `outputs/tool_3b_eval_mini_100.jsonl.summary.json`
- `outputs/tool_4b_eval_mini_100.jsonl`
- `outputs/tool_4b_eval_mini_100.jsonl.summary.json`

## Main Results

| Model | Text Relaxed | ChartQA | DocVQA | AI2D | ScreenSpot IoU@0.5 | ScreenSpot Pointing | Final Parse | Protocol Error | Avg Tools | Tool Success | Evidence Closed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-VL-3B | 56.00% | 36.67% | 72.00% | 65.00% | 12.00% | 56.00% | 93.00% | 73.00% | 1.45 | 86.21% | 18.00% |
| Qwen3-VL-4B | 77.33% | 73.33% | 80.00% | 80.00% | 12.00% | 24.00% | 100.00% | 0.00% | 1.86 | 94.09% | 100.00% |

## Direct Baseline Comparison

Previous direct-answer baseline:

| Model | Text Relaxed | ScreenSpot IoU@0.5 | ScreenSpot Pointing |
|---|---:|---:|---:|
| Qwen2.5-VL-3B direct | 81.33% | 0.00% | 12.00% |
| Qwen3-VL-4B direct | 76.00% | 0.00% | 8.00% |

Prompt-only tool use changes the failure shape:

- Qwen2.5-VL-3B loses text QA accuracy but improves GUI pointing from 12% to 56%.
- Qwen3-VL-4B roughly preserves/improves text QA and improves GUI pointing from 8% to 24%.
- Both models improve ScreenSpot IoU@0.5 from 0% to 12%, but grounding is still weak.

## Tool Behavior

Qwen2.5-VL-3B:

- total tool calls: 145
- tool counts: `{"measure": 39, "count": 1, "detect": 49, "ocr": 28, "click": 28}`
- protocol error rate: 73%
- evidence reference rate: 20%
- main issue: often answers directly or omits evidence ids, despite the prompt requiring tool-first behavior.

Qwen3-VL-4B:

- total tool calls: 186
- tool counts: `{"detect": 78, "ocr": 64, "zoom": 15, "measure": 1, "inspect": 10, "final": 3, "crop": 5, "click": 10}`
- protocol error rate: 0%
- evidence reference rate: 100%
- main issue: protocol is stable, but tool choice and bbox localization are still not strong enough.

## Interpretation

The loop is working: model action JSON is parsed, tool observations are appended,
final JSON is scored, and protocol/tool/evidence statistics are produced.

The most important next step is not adding more prompt text. The results show
that prompt-only tool use changes behavior but is unreliable, especially for
Qwen2.5-VL-3B. The next useful work should be to build supervised traces:

- convert correct direct-answer and tool-assisted examples into
  action-observation-final training samples
- teach a fixed tool schema, especially `detect`, `ocr`, `zoom`, `measure`, and
  virtual `click`
- add negative examples for invalid tools such as `count` or `final` as tool
  names
- strengthen GUI grounding with tool-generated candidate boxes and point-in-box
  reward/filters
