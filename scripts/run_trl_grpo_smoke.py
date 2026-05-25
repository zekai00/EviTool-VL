#!/usr/bin/env python3
"""Prepare a TRL GRPO smoke run for Qwen2.5-VL-3B.

This script intentionally separates two checks:
1. --dry-run: validate dataset formatting and reward code without loading a model.
2. actual train(): load TRL/Transformers and run a tiny single-turn GRPO plumbing test.

Full tool-interactive RL should use `rollout_func` or `environment_factory` after
the current eval jobs finish. This smoke is only a safe first training-loop check.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TRL_ROOT = PROJECT_ROOT / "third_party" / "trl"
if TRL_ROOT.exists() and str(TRL_ROOT) not in sys.path:
    sys.path.insert(0, str(TRL_ROOT))

from rl.rewards import grpo_reward, score_rollout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter", default="checkpoints/qwen25vl_3b_evitool_sft_v2_tool_gui_lora")
    parser.add_argument("--data", default="/root/models/datasets/evitool_eval_medium/eval_medium_600.jsonl")
    parser.add_argument("--image-root", default="/root/models/datasets/evitool_eval_medium")
    parser.add_argument("--output-dir", default="outputs/rl_smoke/grpo_qwen25vl_3b")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    rows = []
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        rows.append(row)
        if len(rows) >= limit:
            break
    return rows


def make_prompt(row: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": (
                "Answer the visual task with a concise final answer. "
                "For this GRPO plumbing smoke, return plain answer text only.\n\n"
                f"Question: {row.get('question')}"
            ),
        }
    ]


def main() -> int:
    args = parse_args()
    rows = load_rows(Path(args.data), args.limit)
    if args.dry_run:
        checks = []
        for row in rows[: min(4, len(rows))]:
            rollout = {
                "prediction": str(row.get("answer_bbox") if row.get("task_type") == "gui_grounding" else row.get("answer")),
                "final_parseable": True,
                "tool_call_count": 0,
                "tool_success_count": 0,
                "parse_errors": [],
                "protocol_errors": [],
                "referenced_evidence_ids": [],
                "missing_evidence_ids": [],
            }
            checks.append({"id": row.get("id"), "task_type": row.get("task_type"), "reward": score_rollout(row, rollout)})
        print(json.dumps({"dry_run": True, "count": len(rows), "checks": checks}, ensure_ascii=False, indent=2))
        return 0

    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer
    from transformers import AutoProcessor
    import importlib.util

    baseline_path = PROJECT_ROOT / "eval" / "eval_baseline.py"
    spec = importlib.util.spec_from_file_location("eval_baseline", baseline_path)
    baseline = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(baseline)

    # Use text-only prompts for the first training-loop smoke. The interactive
    # visual rollout is prepared in rl/rollout_env.py and should be wired next.
    dataset = Dataset.from_list([{"prompt": make_prompt(row), "row": row, **row} for row in rows])
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=1,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=args.num_generations,
        max_prompt_length=1024,
        max_completion_length=128,
        learning_rate=1e-6,
        bf16=True,
        logging_steps=1,
        save_steps=0,
        report_to=[],
        model_init_kwargs={"torch_dtype": "bfloat16", "trust_remote_code": True},
    )
    if args.adapter:
        model = baseline.load_model(args.model, args.adapter)
        peft_config = None
    else:
        model = args.model
        peft_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM")
    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        reward_funcs=grpo_reward,
        processing_class=processor,
        peft_config=peft_config,
    )
    trainer.train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
