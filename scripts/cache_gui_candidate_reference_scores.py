#!/usr/bin/env python3
"""Cache reference candidate log-probabilities for KL-regularized CC-GRPO.

Keeping a frozen reference VLM on each 3090 alongside the trainable model is too
memory-heavy.  This script computes reference log-prob scores once and stores
them in the JSONL rows, so CC-GRPO can add a KL penalty without loading a second
model during training.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.candidate_constrained import build_action_ids, build_cc_prompt, compute_candidate_completion_logprobs

BASELINE_PATH = PROJECT_ROOT / "eval" / "eval_baseline.py"
spec = importlib.util.spec_from_file_location("eval_baseline", BASELINE_PATH)
baseline = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(baseline)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="/root/models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--image-max-pixels", type=int, default=262144)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-actions", type=int, default=8)
    parser.add_argument("--max-hard-negatives", type=int, default=6)
    parser.add_argument("--score-batch-size", type=int, default=8)
    parser.add_argument("--reference-key", default="reference_logprobs")
    parser.add_argument("--use-overlay", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_jsonl(path: str | Path, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def image_path_for_row(row: dict[str, Any], use_overlay: bool) -> Path:
    image_field = row.get("overlay_image") if use_overlay and row.get("overlay_image") else row.get("image_path") or row.get("image")
    path = Path(str(image_field))
    return path if path.is_absolute() else PROJECT_ROOT / path


def score_chunks(
    model: Any,
    processor: Any,
    row: dict[str, Any],
    action_ids: list[str],
    *,
    use_overlay: bool,
    chunk_size: int,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    prompt = build_cc_prompt(row, row.get("candidates") or [])
    image_path = image_path_for_row(row, use_overlay)
    for start in range(0, len(action_ids), max(1, chunk_size)):
        chunk_ids = action_ids[start : start + max(1, chunk_size)]
        # Cache only candidate-level completion scores.  CC-GRPO later converts
        # these scores to a reference distribution with log_softmax, so the JSONL
        # stays compact and independent of the training-time policy temperature.
        with torch.inference_mode():
            chunk_scores = compute_candidate_completion_logprobs(
                model,
                processor,
                image_path,
                prompt,
                chunk_ids,
                require_grad=False,
            )
        for candidate_id, score in zip(chunk_ids, chunk_scores.detach().cpu().tolist(), strict=False):
            scores[candidate_id] = round(float(score), 6)
    return scores


def main() -> int:
    args = parse_args()
    rows = read_jsonl(args.input, args.limit)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True, max_pixels=args.image_max_pixels)
    if getattr(processor, "tokenizer", None) is not None:
        processor.tokenizer.padding_side = "right"
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model = baseline.load_model(args.model, args.adapter)
    model.eval()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as out:
        for index, row in enumerate(rows, start=1):
            action_ids = [str(candidate_id) for candidate_id in row.get("cc_action_ids") or []]
            if not action_ids:
                action_ids = build_action_ids(
                    row,
                    row.get("candidates") or [],
                    max_actions=args.max_actions,
                    max_hard=args.max_hard_negatives,
                    seed=args.seed,
                )
            action_ids = action_ids[: args.max_actions]
            enriched = dict(row)
            enriched[args.reference_key] = score_chunks(
                model,
                processor,
                row,
                action_ids,
                use_overlay=args.use_overlay,
                chunk_size=args.score_batch_size,
            )
            out.write(json.dumps(enriched, ensure_ascii=False) + "\n")
            out.flush()
            print(f"[{index}/{len(rows)}] {row.get('id')} cached {len(enriched[args.reference_key])} scores", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
