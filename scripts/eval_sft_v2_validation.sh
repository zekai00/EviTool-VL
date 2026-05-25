#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
OUT_DIR="${OUT_DIR:-outputs/eval_medium_600_sft_v2}"
DATA="${DATA:-/root/models/datasets/evitool_eval_medium/eval_medium_600.jsonl}"
IMAGE_ROOT="${IMAGE_ROOT:-/root/models/datasets/evitool_eval_medium}"
MAX_TOOL_STEPS="${MAX_TOOL_STEPS:-2}"
MAX_NEW_TOKENS_DIRECT="${MAX_NEW_TOKENS_DIRECT:-128}"
MAX_NEW_TOKENS_TOOL="${MAX_NEW_TOKENS_TOOL:-384}"

mkdir -p "$OUT_DIR"

run_model_eval() {
  local gpu="$1"
  local label="$2"
  local model="$3"
  local adapter="$4"

  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" eval/eval_baseline.py \
    --model "$model" \
    --adapter "$adapter" \
    --data "$DATA" \
    --image-root "$IMAGE_ROOT" \
    --output "$OUT_DIR/direct_${label}_v2.jsonl" \
    --max-new-tokens "$MAX_NEW_TOKENS_DIRECT"

  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" eval/eval_tool_baseline.py \
    --model "$model" \
    --adapter "$adapter" \
    --data "$DATA" \
    --image-root "$IMAGE_ROOT" \
    --output "$OUT_DIR/tool_${label}_v2.jsonl" \
    --max-tool-steps "$MAX_TOOL_STEPS" \
    --max-new-tokens "$MAX_NEW_TOKENS_TOOL" \
    --repair-missing-evidence
}

run_model_eval \
  0 \
  3b \
  /root/models/Qwen2.5-VL-3B-Instruct \
  checkpoints/qwen25vl_3b_evitool_sft_v2_tool_gui_lora \
  > "$OUT_DIR/eval_3b.log" 2>&1 &
pid_3b=$!

run_model_eval \
  1 \
  4b \
  /root/models/Qwen3-VL-4B-Instruct \
  checkpoints/qwen3vl_4b_evitool_sft_v2_tool_gui_lora \
  > "$OUT_DIR/eval_4b.log" 2>&1 &
pid_4b=$!

status_3b=0
status_4b=0
wait "$pid_3b" || status_3b=$?
wait "$pid_4b" || status_4b=$?

if [[ "$status_3b" != "0" || "$status_4b" != "0" ]]; then
  echo "[ERROR] eval failed: 3B=$status_3b 4B=$status_4b" >&2
  exit 1
fi

"$PYTHON_BIN" scripts/compare_sft_metrics.py \
  --summary outputs/eval_medium_600/direct_3b_pre.jsonl.summary.json \
  --summary outputs/eval_medium_600/tool_3b_pre.jsonl.summary.json \
  --summary outputs/eval_medium_600/direct_4b_pre.jsonl.summary.json \
  --summary outputs/eval_medium_600/tool_4b_pre.jsonl.summary.json \
  --summary outputs/eval_medium_600/direct_3b_post.jsonl.summary.json \
  --summary outputs/eval_medium_600/tool_3b_post.jsonl.summary.json \
  --summary outputs/eval_medium_600/direct_4b_post.jsonl.summary.json \
  --summary outputs/eval_medium_600/tool_4b_post.jsonl.summary.json \
  --summary "$OUT_DIR/direct_3b_v2.jsonl.summary.json" \
  --summary "$OUT_DIR/tool_3b_v2.jsonl.summary.json" \
  --summary "$OUT_DIR/direct_4b_v2.jsonl.summary.json" \
  --summary "$OUT_DIR/tool_4b_v2.jsonl.summary.json" \
  --output "$OUT_DIR/comparison.md"

echo "[INFO] wrote $OUT_DIR/comparison.md"
