#!/usr/bin/env python3
"""Train GUI candidate selection with Candidate-Constrained GRPO.

This is a task-specific GRPO variant.  Instead of sampling arbitrary text, each
prompt samples from a finite candidate-id action set.  The model still produces
scores through teacher-forced VLM log probabilities, but RL happens over legal
GUI candidates, which removes JSON-format noise from the optimization target.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.candidate_constrained import (
    build_action_ids,
    build_cc_prompt,
    candidate_reward_table,
    compute_candidate_completion_logprobs,
    rank_balanced_indices,
)

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except Exception:  # pragma: no cover
    Qwen2_5_VLForConditionalGeneration = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter", default="checkpoints/qwen25vl_3b_gui_candidate_c60_sft_warmup_lora")
    parser.add_argument("--image-max-pixels", type=int, default=262144)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--eval-data", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--num-generations", type=int, default=8, help="Candidate actions sampled per prompt/group.")
    parser.add_argument("--max-actions", type=int, default=12, help="Maximum legal candidate ids scored per prompt.")
    parser.add_argument("--max-hard-negatives", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--policy-temperature", type=float, default=1.2)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--kl-coef", type=float, default=0.0, help="KL penalty against precomputed reference logprobs.")
    parser.add_argument("--reference-logprobs-key", default="reference_logprobs")
    parser.add_argument(
        "--rank-balanced-sampler",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Interleave early/middle/late oracle-rank rows before DDP sharding.",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-metrics", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-overlay", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Disabled by default because PEFT LoRA + DDP + reentrant checkpointing "
            "can mark the same LoRA parameter ready twice during backward."
        ),
    )
    parser.add_argument(
        "--ddp-find-unused-parameters",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep false for this LoRA path after smoke tests confirmed all trainable parameters are used.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def patch_autoawq_transformers_compat() -> None:
    """Keep PEFT adapter loading compatible with older AutoAWQ import paths."""
    import transformers.activations as activations

    if not hasattr(activations, "PytorchGELUTanh"):
        if hasattr(activations, "PytorchGELUTanhActivation"):
            activations.PytorchGELUTanh = activations.PytorchGELUTanhActivation
        else:
            activations.PytorchGELUTanh = activations.GELUActivation


def init_distributed() -> tuple[bool, int, int, int, torch.device]:
    """Initialize torchrun/NCCL DDP if environment variables are present."""
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
        return True, rank, local_rank, world_size, device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return False, 0, 0, 1, device


def cleanup_distributed(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


def log(rank: int, message: str) -> None:
    if is_main(rank):
        print(message, flush=True)


def load_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


class CandidateRows(Dataset):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def collate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return rows


def load_trainable_model(
    model_name_or_path: str,
    adapter_name_or_path: str | None,
    device: torch.device,
    *,
    gradient_checkpointing: bool,
) -> torch.nn.Module:
    """Load one full model replica on the local DDP device.

    We intentionally do not use `device_map="auto"` here.  In DDP each process
    owns exactly one GPU and a full LoRA-trainable model replica; gradients are
    synchronized by NCCL all-reduce after backward.
    """
    if Qwen2_5_VLForConditionalGeneration is None:
        raise RuntimeError("Current transformers does not expose Qwen2_5_VLForConditionalGeneration.")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name_or_path,
        dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    if adapter_name_or_path:
        patch_autoawq_transformers_compat()
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_name_or_path, is_trainable=True)
        model.to(device)
    model.config.use_cache = False
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        try:
            # Non-reentrant checkpointing is important for DDP + PEFT LoRA.
            # Reentrant checkpointing can trigger "parameter marked ready twice"
            # because LoRA parameters participate in repeated checkpointed
            # subgraphs during the same backward pass.
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    return model


def trainable_parameter_count(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return trainable, total


def prepare_row_actions(row: dict[str, Any], args: argparse.Namespace) -> tuple[list[str], dict[str, float]]:
    candidates = row.get("candidates") or []
    rewards = {str(k): float(v) for k, v in (row.get("candidate_rewards_v3") or {}).items()}
    if not rewards:
        rewards = candidate_reward_table(row, candidates)
    action_ids = [str(candidate_id) for candidate_id in row.get("cc_action_ids") or []]
    if not action_ids:
        action_ids = build_action_ids(
            row,
            candidates,
            max_actions=args.max_actions,
            max_hard=args.max_hard_negatives,
            seed=args.seed,
        )
    action_ids = [candidate_id for candidate_id in action_ids if candidate_id in rewards][: args.max_actions]
    return action_ids, rewards


def image_path_for_row(row: dict[str, Any], use_overlay: bool) -> Path:
    image_field = row.get("overlay_image") if use_overlay and row.get("overlay_image") else row.get("image_path") or row.get("image")
    path = Path(str(image_field))
    return path if path.is_absolute() else PROJECT_ROOT / path


def reduce_metrics(metrics: dict[str, float], device: torch.device, world_size: int) -> dict[str, float]:
    if world_size == 1:
        return metrics
    keys = sorted(metrics)
    values = torch.tensor([float(metrics[key]) for key in keys], device=device)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values = values / world_size
    return {key: float(value) for key, value in zip(keys, values.tolist(), strict=False)}


def train_one_batch(
    model: torch.nn.Module,
    processor: Any,
    batch_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    losses: list[torch.Tensor] = []
    metric_sums: dict[str, float] = {
        "reward_mean": 0.0,
        "reward_std": 0.0,
        "frac_reward_zero_std": 0.0,
        "entropy": 0.0,
        "oracle_sample_rate": 0.0,
        "top1_bias_rate": 0.0,
        "best_action_reward": 0.0,
        "kl_ref": 0.0,
        "kl_ref_rows": 0.0,
        "valid_rows": 0.0,
    }

    for row in batch_rows:
        action_ids, reward_lookup = prepare_row_actions(row, args)
        if len(action_ids) < 2:
            continue
        prompt = build_cc_prompt(row, row.get("candidates") or [])
        log_scores = compute_candidate_completion_logprobs(
            model,
            processor,
            image_path_for_row(row, args.use_overlay),
            prompt,
            action_ids,
            require_grad=True,
        )
        logits = log_scores / max(args.policy_temperature, 1e-6)
        logpi = F.log_softmax(logits, dim=0)
        probs = logpi.exp()
        kl_ref = torch.zeros((), device=device)
        ref_lookup = row.get(args.reference_logprobs_key) or {}
        if args.kl_coef > 0 and all(candidate_id in ref_lookup for candidate_id in action_ids):
            ref_scores = torch.tensor([float(ref_lookup[candidate_id]) for candidate_id in action_ids], device=device)
            ref_logpi = F.log_softmax(ref_scores / max(args.policy_temperature, 1e-6), dim=0)
            # Forward KL under the current policy.  The reference distribution is
            # precomputed offline so training does not need a second frozen VLM
            # replica on each 3090.
            kl_ref = (probs * (logpi - ref_logpi)).sum()

        # Sampling is detached: the gradient comes from the selected log-probs,
        # while actions are drawn from the current candidate-constrained policy.
        sample_count = min(args.num_generations, max(2, len(action_ids)))
        with torch.no_grad():
            sampled_idx = torch.multinomial(probs.detach(), sample_count, replacement=True)
            sampled_ids = [action_ids[int(index)] for index in sampled_idx.tolist()]
            rewards = torch.tensor([float(reward_lookup[candidate_id]) for candidate_id in sampled_ids], device=device)
            reward_mean = rewards.mean()
            reward_std = rewards.std(unbiased=False)
            if float(reward_std.item()) <= 1e-8:
                advantages = torch.zeros_like(rewards)
                zero_std = 1.0
            else:
                advantages = (rewards - reward_mean) / reward_std.clamp_min(1e-8)
                zero_std = 0.0

        selected_logpi = logpi[sampled_idx]
        old_selected_logpi = selected_logpi.detach()
        ratio = torch.exp(selected_logpi - old_selected_logpi)
        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * advantages
        policy_loss = -torch.minimum(unclipped, clipped).mean()
        entropy = -(probs * logpi).sum()
        losses.append(policy_loss + args.kl_coef * kl_ref - args.entropy_coef * entropy)

        oracle_id = str(row.get("oracle_candidate_id") or "")
        rank1_id = None
        for candidate in row.get("candidates") or []:
            if int(candidate.get("rank") or 0) == 1:
                rank1_id = str(candidate.get("candidate_id"))
                break
        metric_sums["reward_mean"] += float(reward_mean.item())
        metric_sums["reward_std"] += float(reward_std.item())
        metric_sums["frac_reward_zero_std"] += zero_std
        metric_sums["entropy"] += float(entropy.detach().item())
        metric_sums["oracle_sample_rate"] += sum(float(candidate_id == oracle_id) for candidate_id in sampled_ids) / len(sampled_ids)
        metric_sums["top1_bias_rate"] += sum(float(candidate_id == rank1_id) for candidate_id in sampled_ids) / len(sampled_ids)
        metric_sums["best_action_reward"] += max(float(reward_lookup[candidate_id]) for candidate_id in action_ids)
        metric_sums["kl_ref"] += float(kl_ref.detach().item())
        metric_sums["kl_ref_rows"] += float(args.kl_coef > 0 and all(candidate_id in ref_lookup for candidate_id in action_ids))
        metric_sums["valid_rows"] += 1.0

    if not losses:
        return torch.zeros((), device=device, requires_grad=True), metric_sums
    loss = torch.stack(losses).mean()
    denom = max(metric_sums["valid_rows"], 1.0)
    metrics = {key: value / denom for key, value in metric_sums.items() if key != "valid_rows"}
    metrics["valid_rows"] = metric_sums["valid_rows"]
    metrics["loss"] = float(loss.detach().item())
    return loss, metrics


def main() -> int:
    args = parse_args()
    distributed, rank, local_rank, world_size, device = init_distributed()
    seed = args.seed + rank
    random.seed(seed)
    torch.manual_seed(seed)

    rows = load_jsonl(args.train_data, args.max_train_rows)
    rows = [row for row in rows if row.get("overlay_image") and row.get("candidates")]
    if args.dry_run:
        log(rank, json.dumps({"dry_run": True, "rows": len(rows), "first_id": rows[0].get("id") if rows else None}, ensure_ascii=False, indent=2))
        cleanup_distributed(distributed)
        return 0

    if not rows:
        raise RuntimeError("No train rows with overlay/candidates were loaded.")

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True, max_pixels=args.image_max_pixels)
    if getattr(processor, "tokenizer", None) is not None:
        processor.tokenizer.padding_side = "right"
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = load_trainable_model(args.model, args.adapter, device, gradient_checkpointing=args.gradient_checkpointing)
    trainable, total = trainable_parameter_count(model)
    log(rank, f"DDP: distributed={distributed} world_size={world_size} local_rank={local_rank}")
    log(rank, f"Trainable parameters: {trainable:,} / {total:,}")
    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=args.ddp_find_unused_parameters,
        )

    optimizer = torch.optim.AdamW((param for param in model.parameters() if param.requires_grad), lr=args.learning_rate)
    metrics_history: list[dict[str, Any]] = []
    global_step = 0
    epoch = 0
    model.train()
    while global_step < args.max_steps:
        if args.rank_balanced_sampler:
            order = rank_balanced_indices(rows, seed=args.seed, epoch=epoch)
            epoch_rows = [rows[index] for index in order]
            if distributed:
                epoch_rows = epoch_rows[rank::world_size]
            loader = DataLoader(
                CandidateRows(epoch_rows),
                batch_size=args.per_device_batch_size,
                shuffle=False,
                collate_fn=collate_rows,
                num_workers=0,
                drop_last=False,
            )
        else:
            dataset = CandidateRows(rows)
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed) if distributed else None
            if sampler is not None:
                sampler.set_epoch(epoch)
            loader = DataLoader(
                dataset,
                batch_size=args.per_device_batch_size,
                sampler=sampler,
                shuffle=sampler is None,
                collate_fn=collate_rows,
                num_workers=0,
                drop_last=False,
            )
        for batch_rows in loader:
            if global_step >= args.max_steps:
                break
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = train_one_batch(model, processor, batch_rows, args, device)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_([param for param in model.parameters() if param.requires_grad], args.grad_clip)
            optimizer.step()

            reduced = reduce_metrics(metrics, device, world_size)
            global_step += 1
            reduced.update(
                {
                    "step": global_step,
                    "epoch": epoch,
                    "lr": args.learning_rate,
                    "grad_norm": float(grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm),
                }
            )
            if is_main(rank):
                metrics_history.append(reduced)
                if global_step % args.logging_steps == 0:
                    print(json.dumps(reduced, ensure_ascii=False), flush=True)
        epoch += 1

    if distributed:
        dist.barrier()
    if is_main(rank):
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = model.module if isinstance(model, DDP) else model
        unwrapped.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
        metrics_path = Path(args.save_metrics) if args.save_metrics else output_dir / "train_metrics.jsonl"
        with metrics_path.open("w", encoding="utf-8") as f:
            for item in metrics_history:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        summary = {
            "method": "Candidate-Constrained GRPO",
            "distributed": distributed,
            "world_size": world_size,
            "ddp_backend": "nccl" if distributed else None,
            "train_data": args.train_data,
            "output_dir": args.output_dir,
            "max_steps": args.max_steps,
            "num_generations": args.num_generations,
            "max_actions": args.max_actions,
            "learning_rate": args.learning_rate,
            "image_max_pixels": args.image_max_pixels,
            "policy_temperature": args.policy_temperature,
            "entropy_coef": args.entropy_coef,
            "kl_coef": args.kl_coef,
            "reference_logprobs_key": args.reference_logprobs_key,
            "rank_balanced_sampler": args.rank_balanced_sampler,
            "gradient_checkpointing": args.gradient_checkpointing,
            "ddp_find_unused_parameters": args.ddp_find_unused_parameters,
            "metrics_tail": metrics_history[-5:],
        }
        (output_dir / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    cleanup_distributed(distributed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
