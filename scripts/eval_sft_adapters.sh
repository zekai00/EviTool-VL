#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_DIR="${OUT_DIR:-outputs/sft_eval}"
DATA="${DATA:-data/eval_mini/eval_mini_100.jsonl}"
IMAGE_ROOT="${IMAGE_ROOT:-data/eval_mini}"
MAX_TOOL_STEPS="${MAX_TOOL_STEPS:-2}"
mkdir -p "$OUT_DIR"

python3 eval/eval_baseline.py \
  --model /root/models/Qwen2.5-VL-3B-Instruct \
  --adapter checkpoints/qwen25vl_3b_evitool_sft_1k_lora \
  --data "$DATA" \
  --image-root "$IMAGE_ROOT" \
  --output "$OUT_DIR/direct_3b_sft.jsonl" \
  --max-new-tokens 128

python3 eval/eval_tool_baseline.py \
  --model /root/models/Qwen2.5-VL-3B-Instruct \
  --adapter checkpoints/qwen25vl_3b_evitool_sft_1k_lora \
  --data "$DATA" \
  --image-root "$IMAGE_ROOT" \
  --output "$OUT_DIR/tool_3b_sft.jsonl" \
  --max-tool-steps "$MAX_TOOL_STEPS" \
  --max-new-tokens 384 \
  --repair-missing-evidence

python3 eval/eval_baseline.py \
  --model /root/models/Qwen3-VL-4B-Instruct \
  --adapter checkpoints/qwen3vl_4b_evitool_sft_1k_lora \
  --data "$DATA" \
  --image-root "$IMAGE_ROOT" \
  --output "$OUT_DIR/direct_4b_sft.jsonl" \
  --max-new-tokens 128

python3 eval/eval_tool_baseline.py \
  --model /root/models/Qwen3-VL-4B-Instruct \
  --adapter checkpoints/qwen3vl_4b_evitool_sft_1k_lora \
  --data "$DATA" \
  --image-root "$IMAGE_ROOT" \
  --output "$OUT_DIR/tool_4b_sft.jsonl" \
  --max-tool-steps "$MAX_TOOL_STEPS" \
  --max-new-tokens 384 \
  --repair-missing-evidence

python3 scripts/compare_sft_metrics.py \
  --summary outputs/baseline_3b_direct_eval_mini_100.jsonl.summary.json \
  --summary outputs/tool_3b_eval_mini_100.jsonl.summary.json \
  --summary outputs/baseline_4b_direct_eval_mini_100.jsonl.summary.json \
  --summary outputs/tool_4b_eval_mini_100.jsonl.summary.json \
  --summary "$OUT_DIR/direct_3b_sft.jsonl.summary.json" \
  --summary "$OUT_DIR/tool_3b_sft.jsonl.summary.json" \
  --summary "$OUT_DIR/direct_4b_sft.jsonl.summary.json" \
  --summary "$OUT_DIR/tool_4b_sft.jsonl.summary.json" \
  --output "$OUT_DIR/comparison.md"
