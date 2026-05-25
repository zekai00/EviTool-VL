#!/usr/bin/env python3
"""Train a single-step GUI candidate selector with TRL GRPO."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TRL_ROOT = PROJECT_ROOT / "third_party" / "trl"
if TRL_ROOT.exists() and str(TRL_ROOT) not in sys.path:
    sys.path.insert(0, str(TRL_ROOT))

from rl.gui_candidate_env import build_candidate_prompt, score_candidate_action

BASELINE_PATH = PROJECT_ROOT / "eval" / "eval_baseline.py"
spec = importlib.util.spec_from_file_location("eval_baseline", BASELINE_PATH)
baseline = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(baseline)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter", default="checkpoints/qwen25vl_3b_evitool_sft_v2_tool_gui_lora")
    parser.add_argument("--train-data", default="outputs/gui_candidate_rl/train.jsonl")
    parser.add_argument("--eval-data", default="outputs/gui_candidate_rl/val.jsonl")
    parser.add_argument("--output-dir", default="outputs/gui_candidate_rl/grpo_3b_sftv2")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--per-device-train-batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--max-completion-length", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def make_prompt(row: dict[str, Any]) -> list[dict[str, str]]:
    text = (
        f"{build_candidate_prompt(row, row.get('candidates') or [])}\n\n"
        "The image is annotated with candidate ids. Select the candidate that best matches the instruction. "
        "Return only JSON. Do not explain."
    )
    return [{"role": "user", "content": text}]


def load_dataset_rows(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    examples = []
    for row in load_jsonl(path, limit):
        overlay = row.get("overlay_image")
        if not overlay:
            continue
        examples.append(
            {
                "id": row.get("id"),
                "prompt": make_prompt(row),
                "image": Image.open(overlay).convert("RGB"),
                "answer_bbox": row.get("answer_bbox"),
                "candidates": row.get("candidates") or [],
            }
        )
    return examples


def _completion_text(completion: Any) -> str:
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            content = last.get("content", "")
            if isinstance(content, list):
                return "\n".join(str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content)
            return str(content)
    return str(completion)


def candidate_reward(completions: list[Any], answer_bbox: list[Any], candidates: list[Any], **kwargs: Any) -> list[float]:
    rewards: list[float] = []
    valid_values: list[float] = []
    pointing_values: list[float] = []
    iou50_values: list[float] = []
    for completion, gt_bbox, cand_list in zip(completions, answer_bbox, candidates, strict=False):
        row = {"answer_bbox": gt_bbox}
        reward = score_candidate_action(row, cand_list or [], _completion_text(completion))
        rewards.append(float(reward["total"]))
        valid_values.append(float(reward["valid"]))
        pointing_values.append(float(reward["pointing"]))
        iou50_values.append(float(reward["iou_50"]))
    log_metric = kwargs.get("log_metric")
    if callable(log_metric) and rewards:
        log_metric("candidate_reward/valid", sum(valid_values) / len(valid_values))
        log_metric("candidate_reward/pointing", sum(pointing_values) / len(pointing_values))
        log_metric("candidate_reward/iou50", sum(iou50_values) / len(iou50_values))
    return rewards


def load_trainable_model(model_name_or_path: str, adapter_name_or_path: str | None):
    model = baseline.load_model(model_name_or_path, None)
    if adapter_name_or_path:
        def patch_autoawq_transformers_compat() -> None:
            import transformers.activations as activations

            if not hasattr(activations, "PytorchGELUTanh"):
                if hasattr(activations, "PytorchGELUTanhActivation"):
                    activations.PytorchGELUTanh = activations.PytorchGELUTanhActivation
                else:
                    activations.PytorchGELUTanh = activations.GELUActivation

        patch_autoawq_transformers_compat()
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_name_or_path, is_trainable=True)
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    return model


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    train_examples = load_dataset_rows(args.train_data, args.limit)
    eval_examples = load_dataset_rows(args.eval_data, args.eval_limit) if args.eval_data else []

    if args.dry_run:
        rows = load_jsonl(args.train_data, min(4, args.limit or 4))
        checks = []
        for row in rows:
            oracle_id = row.get("oracle_candidate_id") or "c00"
            for label, completion in [("top1", '{"candidate_id":"c00"}'), ("oracle", json.dumps({"candidate_id": oracle_id}))]:
                reward = score_candidate_action({"answer_bbox": row.get("answer_bbox")}, row.get("candidates") or [], completion)
                checks.append({"id": row.get("id"), "policy": label, "completion": completion, "reward": reward})
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "train_examples": len(train_examples),
                    "eval_examples": len(eval_examples),
                    "checks": checks,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    if getattr(processor, "tokenizer", None) is not None:
        processor.tokenizer.padding_side = "left"
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

    per_device_train_batch_size = args.per_device_train_batch_size or args.num_generations
    config = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=1,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        beta=0.0,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="no",
        save_steps=0,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=args.seed,
    )
    if args.adapter:
        model = load_trainable_model(args.model, args.adapter)
        peft_config = None
    else:
        model = args.model
        peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )

    trainer = GRPOTrainer(
        model=model,
        args=config,
        reward_funcs=candidate_reward,
        train_dataset=Dataset.from_list(train_examples),
        eval_dataset=Dataset.from_list(eval_examples) if eval_examples else None,
        processing_class=processor,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
