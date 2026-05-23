# Baseline Findings

Dataset: `data/eval_mini/eval_mini_100.jsonl`
Setting: direct answer, no visual tools, no evidence trace.

## Main Problems

1. GUI grounding is the biggest failure mode.
   - Qwen2.5-VL-3B: ScreenSpot IoU@0.5 = 0.00%, pointing = 12.00%.
   - Qwen3-VL-4B: ScreenSpot IoU@0.5 = 0.00%, pointing = 8.00%.
   - The models can sometimes output bbox-like text, but the boxes are not reliably aligned to the target UI element.

2. ChartQA still has local evidence and numeric reading errors.
   - 3B ChartQA relaxed accuracy = 63.33%.
   - 4B ChartQA relaxed accuracy = 66.67%.
   - Many errors look like the model reads or estimates nearby values without zooming/cropping the relevant region.

3. Direct answer gives no verifiable evidence.
   - Even when an answer is correct, the output does not show which image region, OCR span, or visual cue supports it.
   - This makes answer-only baseline weak for debugging and unsuitable for evidence-grounded reasoning.

4. Qwen3-VL-4B is not automatically better in this first direct-answer setup.
   - 4B has better bbox parse behavior on ScreenSpot, but still poor location accuracy.
   - 3B scored higher on this small DocVQA/AI2D sample. This is not a final model ranking, but it shows why controlled baselines are necessary.

## Why Visual Tools Are The Next Step

The direct baselines fail exactly where EviTool-VL is supposed to help:

- crop/zoom should reduce ChartQA and DocVQA local reading errors.
- OCR should expose explicit text spans instead of relying on hidden model perception.
- detect/mark/measure should turn GUI and chart localization into structured evidence.
- evidence ids should make each final answer auditable.

## Current Target

Implement a local tool layer that is independent from training:

- deterministic JSON input/output
- reusable from prompt-only baseline, SFT data construction, GRPO rewards, and demo
- safe fallback when optional OCR/detection backends are unavailable
