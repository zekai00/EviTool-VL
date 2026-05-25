#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/llama/bin/python}"
LOG_DIR="${LOG_DIR:-outputs/sft_v2_validation}"
mkdir -p "$LOG_DIR"

export PYTHONPATH="$ROOT/third_party/LlamaFactory/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

echo "[INFO] ROOT=$ROOT"
echo "[INFO] PYTHON_BIN=$PYTHON_BIN"
"$PYTHON_BIN" -V
"$PYTHON_BIN" -m llamafactory.cli version

"$PYTHON_BIN" datasets/prepare_sft_v2_data.py \
  --input /root/models/datasets/evitool_traces_1k/train_traces_1000.jsonl \
  --eval /root/models/datasets/evitool_eval_medium/eval_medium_600.jsonl \
  --output-dir /root/models/datasets/evitool_sft_v2 \
  --report /root/Workspace/VLM/sftv2_dataset_report.md

CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" -m llamafactory.cli train \
  configs/sft_v2_qwen25vl_3b_tool_gui_lora.yaml \
  > "$LOG_DIR/train_3b_tool_gui.log" 2>&1 &
pid_3b=$!

CUDA_VISIBLE_DEVICES=1 "$PYTHON_BIN" -m llamafactory.cli train \
  configs/sft_v2_qwen3vl_4b_tool_gui_lora.yaml \
  > "$LOG_DIR/train_4b_tool_gui.log" 2>&1 &
pid_4b=$!

echo "[INFO] started 3B pid=$pid_3b log=$LOG_DIR/train_3b_tool_gui.log"
echo "[INFO] started 4B pid=$pid_4b log=$LOG_DIR/train_4b_tool_gui.log"

status_3b=0
status_4b=0
wait "$pid_3b" || status_3b=$?
wait "$pid_4b" || status_4b=$?

echo "[INFO] 3B exit status=$status_3b"
echo "[INFO] 4B exit status=$status_4b"

if [[ "$status_3b" != "0" || "$status_4b" != "0" ]]; then
  exit 1
fi
