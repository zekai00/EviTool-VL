# Bailian Direct Model Smoke Eval

Date: 2026-05-23

This smoke eval checks whether Bailian-hosted Qwen3.5/Qwen3.6 multimodal models can be used as stronger direct-answer baselines or teacher models for EviTool-VL.

Setting:

- API: Bailian OpenAI-compatible `/chat/completions`
- Script: `eval/eval_bailian_direct.py`
- Dataset: `data/eval_mini/eval_mini_100.jsonl`
- Sample: `--sample-per-task 2`, 8 total examples: ChartQA 2, DocVQA 2, AI2D 2, ScreenSpot-v2 2
- Prompt: direct final answer only, no local visual tool loop
- `max_tokens`: 128

| Model | Text Relaxed | ChartQA | DocVQA | AI2D | ScreenSpot BBox Parse | ScreenSpot IoU@0.5 | ScreenSpot Pointing | Avg Latency | Total Tokens | Output |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3.5-flash | 83.33% | 100.00% | 50.00% | 100.00% | 100.00% | 0.00% | 0.00% | 4.566s | 10,391 | `outputs/bailian_qwen35_flash_direct_sample8.jsonl` |
| qwen3.5-27b | 83.33% | 100.00% | 50.00% | 100.00% | 100.00% | 0.00% | 0.00% | 1.083s | 6,409 | `outputs/bailian_qwen35_27b_direct_sample8.jsonl` |
| qwen3.5-plus | 83.33% | 100.00% | 50.00% | 100.00% | 100.00% | 0.00% | 0.00% | 11.307s | 10,553 | `outputs/bailian_qwen35_plus_direct_sample8.jsonl` |
| qwen3.5-122b-a10b | 66.67% | 100.00% | 50.00% | 50.00% | 100.00% | 0.00% | 0.00% | 1.097s | 6,415 | `outputs/bailian_qwen35_122b_a10b_direct_sample8.jsonl` |
| qwen3.6-plus | 83.33% | 100.00% | 50.00% | 100.00% | 100.00% | 0.00% | 0.00% | 11.183s | 10,664 | `outputs/bailian_qwen36_plus_direct_sample8.jsonl` |

## Interpretation

- The current evaluation data is already built from public datasets: ChartQA, DocVQA, AI2D, and ScreenSpot-v2. We do not need to hand-write the base visual QA data from scratch.
- What public datasets do not provide is the project-specific `action -> observation -> final` trace using EviTool-VL local tools such as `crop`, `zoom`, `ocr`, `detect`, `measure`, and `click`.
- Stronger Bailian models can answer simple text/chart/diagram cases directly, but direct bbox generation still fails on the two ScreenSpot samples. All tested models produced parseable boxes or points, but none hit IoU@0.5 or pointing accuracy.
- This supports using stronger models as teachers for candidate reasoning and tool-plan generation, then using deterministic local tools and ground-truth checks to filter traces. It does not support replacing the evidence-bound training data with direct-answer labels only.

## Recommended Next Step

Use public datasets as the raw task pool, then generate tool traces semi-automatically:

1. Run teacher models on selected public-dataset samples with the EviTool action schema.
2. Execute proposed local tools and attach real observations/evidence ids.
3. Filter by answer correctness, JSON validity, evidence closure, and grounding checks.
4. Keep high-confidence traces for SFT and use failed/near-miss traces as hard negatives for GRPO reward design.
