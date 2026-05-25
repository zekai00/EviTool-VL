#!/usr/bin/env python3
"""Candidate-level listwise warmup for GUI candidate selection.

This supervised warmup trains the model to rank a fixed candidate action set
according to reward_v3 before CC-GRPO.  It is deliberately candidate-level:
there is no free-form decoding and no JSON-format objective.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.candidate_constrained import build_cc_prompt, compute_candidate_completion_logprobs, rank_balanced_indices
from scripts.train_gui_candidate_cc_grpo import (
    CandidateRows,
    cleanup_distributed,
    collate_rows,
    image_path_for_row,
    init_distributed,
    is_main,
    load_jsonl,
    load_trainable_model,
    log,
    prepare_row_actions,
    reduce_metrics,
    trainable_parameter_count,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter", default="checkpoints/qwen25vl_3b_gui_candidate_c60_sft_warmup_lora")
    parser.add_argument("--image-max-pixels", type=int, default=262144)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--max-actions", type=int, default=8)
    parser.add_argument("--max-hard-negatives", type=int, default=6)
    parser.add_argument("--target-temperature", type=float, default=0.20)
    parser.add_argument("--policy-temperature", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-overlay", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rank-balanced-sampler", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ddp-find-unused-parameters", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def train_one_batch(
    model: torch.nn.Module,
    processor: Any,
    batch_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    losses: list[torch.Tensor] = []
    metric_sums = {
        "ce_loss": 0.0,
        "expected_reward": 0.0,
        "oracle_prob": 0.0,
        "top1_prob": 0.0,
        "policy_entropy": 0.0,
        "target_entropy": 0.0,
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
        logpi = F.log_softmax(log_scores / max(args.policy_temperature, 1e-6), dim=0)
        probs = logpi.exp()

        rewards = torch.tensor([float(reward_lookup[candidate_id]) for candidate_id in action_ids], device=device)
        # The target is a soft listwise distribution, not a one-hot oracle id.
        # This keeps near-correct GUI boxes useful during warmup and avoids
        # teaching the model that all non-oracle candidates are equally bad.
        target = F.softmax(rewards / max(args.target_temperature, 1e-6), dim=0).detach()
        ce_loss = -(target * logpi).sum()
        policy_entropy = -(probs * logpi).sum()
        loss = ce_loss - args.entropy_coef * policy_entropy
        losses.append(loss)

        oracle_id = str(row.get("oracle_candidate_id") or "")
        rank1_id = None
        for candidate in row.get("candidates") or []:
            if int(candidate.get("rank") or 0) == 1:
                rank1_id = str(candidate.get("candidate_id"))
                break
        oracle_index = action_ids.index(oracle_id) if oracle_id in action_ids else None
        top1_index = action_ids.index(rank1_id) if rank1_id in action_ids else None
        metric_sums["ce_loss"] += float(ce_loss.detach().item())
        metric_sums["expected_reward"] += float((probs.detach() * rewards).sum().item())
        metric_sums["oracle_prob"] += float(probs.detach()[oracle_index].item()) if oracle_index is not None else 0.0
        metric_sums["top1_prob"] += float(probs.detach()[top1_index].item()) if top1_index is not None else 0.0
        metric_sums["policy_entropy"] += float(policy_entropy.detach().item())
        metric_sums["target_entropy"] += float((-(target * target.clamp_min(1e-8).log()).sum()).item())
        metric_sums["valid_rows"] += 1.0

    if not losses:
        return torch.zeros((), device=device, requires_grad=True), metric_sums
    loss = torch.stack(losses).mean()
    denom = max(metric_sums["valid_rows"], 1.0)
    metrics = {key: value / denom for key, value in metric_sums.items() if key != "valid_rows"}
    metrics["valid_rows"] = metric_sums["valid_rows"]
    metrics["loss"] = float(loss.detach().item())
    return loss, metrics


def make_loader(rows: list[dict[str, Any]], args: argparse.Namespace, rank: int, world_size: int, epoch: int, distributed: bool) -> DataLoader:
    if args.rank_balanced_sampler:
        # Short RL/listwise runs are sensitive to which rank buckets appear
        # early.  We build one deterministic global order, then shard it by DDP
        # rank, so both GPUs see different rows from the same balanced sequence.
        order = rank_balanced_indices(rows, seed=args.seed, epoch=epoch)
        epoch_rows = [rows[index] for index in order]
        if distributed:
            epoch_rows = epoch_rows[rank::world_size]
        return DataLoader(
            CandidateRows(epoch_rows),
            batch_size=args.per_device_batch_size,
            shuffle=False,
            collate_fn=collate_rows,
            num_workers=0,
            drop_last=False,
        )

    dataset = CandidateRows(rows)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed) if distributed else None
    if sampler is not None:
        sampler.set_epoch(epoch)
    return DataLoader(
        dataset,
        batch_size=args.per_device_batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        collate_fn=collate_rows,
        num_workers=0,
        drop_last=False,
    )


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
    log(rank, f"Listwise DDP: distributed={distributed} world_size={world_size} local_rank={local_rank}")
    log(rank, f"Trainable parameters: {trainable:,} / {total:,}")
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=args.ddp_find_unused_parameters)

    optimizer = torch.optim.AdamW((param for param in model.parameters() if param.requires_grad), lr=args.learning_rate)
    metrics_history: list[dict[str, Any]] = []
    global_step = 0
    epoch = 0
    model.train()
    while global_step < args.max_steps:
        loader = make_loader(rows, args, rank, world_size, epoch, distributed)
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
            reduced.update({"step": global_step, "epoch": epoch, "lr": args.learning_rate, "grad_norm": float(grad_norm.detach().item())})
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
        with (output_dir / "train_metrics.jsonl").open("w", encoding="utf-8") as f:
            for item in metrics_history:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        summary = {
            "method": "candidate-level listwise warmup",
            "distributed": distributed,
            "world_size": world_size,
            "ddp_backend": "nccl" if distributed else None,
            "train_data": args.train_data,
            "output_dir": args.output_dir,
            "max_steps": args.max_steps,
            "max_actions": args.max_actions,
            "target_temperature": args.target_temperature,
            "policy_temperature": args.policy_temperature,
            "rank_balanced_sampler": args.rank_balanced_sampler,
            "metrics_tail": metrics_history[-5:],
        }
        (output_dir / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    cleanup_distributed(distributed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
