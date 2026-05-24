# SFT 1K Results

Date: 2026-05-24

## Summary

This report freezes the first EviTool-VL 1K SFT experiment. The run is useful as a diagnostic baseline, but it should not be treated as the final SFT recipe.

Main conclusion:

- `Qwen2.5-VL-3B` benefits from SFT in direct mode, especially GUI grounding, but the first tool-use adapter learns to answer directly too often.
- `Qwen3-VL-4B` already had stable prompt-only tool use before SFT; the mixed SFT adapter slightly improves GUI IoU but hurts text/tool accuracy and evidence closure.
- The next SFT round should split direct/tool adapters and keep train/eval source examples separated.

## Data

Training trace dataset:

- Trace path: `/root/models/datasets/evitool_traces_1k/train_traces_1000.jsonl`
- SFT conversion path: `/root/models/datasets/evitool_sft_lf_1k/evitool_sft_1k.json`
- Converted samples: `867`
- Trace samples: `754`
- Direct retention samples: `113`

Converted SFT source counts:

| Task | Count |
|---|---:|
| GUI Grounding | 401 |
| DocVQA | 213 |
| ChartQA | 92 |
| AI2D | 48 |

The first SFT conversion used strong evidence traces plus 15% direct retention samples. This helped direct/text retention, but likely caused the 3B tool adapter to produce direct final answers without tool calls.

## Training

| Model | Adapter | Epoch | Train loss | Runtime |
|---|---|---:|---:|---:|
| Qwen2.5-VL-3B-Instruct | `checkpoints/qwen25vl_3b_evitool_sft_1k_lora` | 1.0 | 0.6115 | 1669s |
| Qwen3-VL-4B-Instruct | `checkpoints/qwen3vl_4b_evitool_sft_1k_lora` | 1.0 | 0.9753 | 2280s |

Training completed successfully and produced LoRA adapter checkpoints for both models.

## Evaluation Results

Evaluation set: `data/eval_mini/eval_mini_100.jsonl`

| Run | Text relaxed | GUI pointing | GUI IoU@0.5 | Evidence closed | Avg tools | Protocol error | Final parse |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3B pre-SFT direct | 81.33% | 12.00% | 0.00% | - | - | - | - |
| 3B pre-SFT tool | 56.00% | 56.00% | 12.00% | 18.00% | 1.45 | 73.00% | 93.00% |
| 3B post-SFT direct | 81.33% | 28.00% | 8.00% | - | - | - | - |
| 3B post-SFT tool | 74.67% | 28.00% | 8.00% | 30.00% | 0.34 | 62.00% | 93.00% |
| 4B pre-SFT direct | 76.00% | 8.00% | 0.00% | - | - | - | - |
| 4B pre-SFT tool | 77.33% | 24.00% | 12.00% | 100.00% | 1.86 | 0.00% | 100.00% |
| 4B post-SFT direct | 80.00% | 8.00% | 0.00% | - | - | - | - |
| 4B post-SFT tool | 68.00% | 20.00% | 16.00% | 96.00% | 1.70 | 0.00% | 96.00% |

## Interpretation

### 3B

Positive:

- Direct GUI grounding improved: pointing `12% -> 28%`, IoU@0.5 `0% -> 8%`.
- Direct text relaxed stayed stable at `81.33%`.
- Tool text relaxed improved from `56.00%` to `74.67%`.

Negative:

- Tool-use behavior regressed: avg tool calls dropped from `1.45` to `0.34`.
- 68/100 post-SFT tool eval samples used zero tools.
- GUI tool metrics fell from `56%` pointing and `12%` IoU@0.5 to `28%` and `8%`.

Likely cause: mixed tool traces and direct retention taught 3B that direct final answers are acceptable in tool mode.

### 4B

Positive:

- Direct text relaxed improved from `76.00%` to `80.00%`.
- Tool GUI IoU@0.5 improved slightly from `12.00%` to `16.00%`.

Negative:

- Tool text relaxed dropped from `77.33%` to `68.00%`.
- Evidence closed dropped from `100.00%` to `96.00%`.
- Final parse dropped from `100.00%` to `96.00%`.

Likely cause: 4B already had strong prompt-only protocol adherence, so mixed SFT perturbed an already-good tool policy.

## Train/Eval Split Policy

The previous mini evaluation set may overlap with the first-N training trace construction. Going forward, eval sets must be source-separated from SFT trace data.

Current training trace source ranges are treated as:

| Task | Training source range |
|---|---:|
| ScreenSpot | first 500 |
| DocVQA | first 250 |
| ChartQA | first 200 |
| AI2D | first 50 |

A source-separated medium evaluation set has been built:

- Path: `/root/models/datasets/evitool_eval_medium/eval_medium_600.jsonl`
- Summary: `/root/models/datasets/evitool_eval_medium/summary.json`
- Split check: `/root/models/datasets/evitool_eval_medium/split_check.json`

Eval medium composition:

| Task | Count | Offset |
|---|---:|---:|
| ChartQA | 150 | 200 |
| DocVQA | 150 | 250 |
| AI2D | 100 | 50 |
| ScreenSpot | 200 | 500 |

Split check result:

| Check | Result |
|---|---:|
| Source identity overlap | 0 |
| Source range violations | 0 |
| Semantic duplicate warnings | 2 |
| Split OK | true |

Semantic duplicate warnings are generic question/answer duplicates across distinct source examples, not source example leakage.

## Next Steps

1. Run `3B post-SFT tool` with `--force-tool-first` to verify whether the adapter can recover if direct final answers are blocked.
2. Train a separate 3B tool-only adapter with no direct retention and stricter first-action prompts.
3. Train a 4B GUI-targeted adapter only on GUI strong traces with lower learning rate (`3e-5` to `5e-5`).
4. Evaluate all future adapters on `eval_medium_600`, not only `eval_mini_100`.
5. Continue data/tool improvements for GUI reranking, ChartQA derived numeric evidence, and AI2D label-object linking.

## Reproducibility Assets

Committed pipeline files should include:

- `datasets/prepare_sft_data.py`
- `datasets/build_eval_mini.py`
- `scripts/run_sft_experiment.sh`
- `scripts/eval_sft_adapters.sh`
- `scripts/compare_sft_metrics.py`
- `scripts/check_train_eval_split.py`
- `configs/sft_qwen25vl_3b_lora.yaml`
- `configs/sft_qwen3vl_4b_lora.yaml`
- `eval/eval_baseline.py`
- `eval/eval_tool_baseline.py`
- `reports/sft_1k_results.md`
