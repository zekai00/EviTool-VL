#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TRACE_INPUT="${TRACE_INPUT:-/root/models/datasets/evitool_traces_1k/train_traces_1000.jsonl}"
SFT_DATA_DIR="${SFT_DATA_DIR:-/root/models/datasets/evitool_sft_lf_1k}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "[INFO] ROOT=$ROOT"
echo "[INFO] Python=$(which python)"
python --version

python - <<'PY'
import sys
print("[INFO] executable:", sys.executable)
if sys.version_info < (3, 11):
    raise SystemExit("[ERROR] 当前环境 Python < 3.11，不能运行新版 LlamaFactory")
PY

python datasets/prepare_sft_data.py \
  --input "$TRACE_INPUT" \
  --output-dir "$SFT_DATA_DIR" \
  --dataset-name evitool_sft_1k \
  --direct-retention-ratio 0.15

export PYTHONPATH="$ROOT/third_party/LlamaFactory/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"

python -m llamafactory.cli train configs/sft_qwen25vl_3b_lora.yaml
python -m llamafactory.cli train configs/sft_qwen3vl_4b_lora.yaml