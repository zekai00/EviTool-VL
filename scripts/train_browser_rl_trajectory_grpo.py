#!/usr/bin/env python3
"""Trajectory-level GRPO training for BrowserRL.

This is intentionally separate from the existing step-wise GRPO script.  A
group here is one task with several complete trajectories sampled from the
same initial state.  The policy loss compares complete trajectories instead of
single next actions.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.browser_rl import load_tasks
from envs.browser_rl.local_qwen_policy import local_qwen_prompt
from scripts.train_browser_rl_onpolicy_grpo import (
    completion_logprob,
    infer_input_device,
    load_sft_replay_rows,
    load_trainable_qwen,
    parameter_counts,
    read_jsonl,
    sample_replay_count,
    sft_replay_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BrowserRL trajectory-level GRPO.")
    parser.add_argument("--groups", required=True, help="trajectory_groups.jsonl produced by collect_browser_rl_trajectory_grpo_probe.py")
    parser.add_argument(
        "--tasks",
        default="/root/datasets/browser_rl/task_suites/browser_rl_task_suite_2000_20260528_1344/train_tasks.jsonl",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="/root/models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--max-groups", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-7)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--kl-beta", type=float, default=0.01)
    parser.add_argument("--min-reward-std", type=float, default=1e-6)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--logprob-reduction", choices=["mean", "sum"], default="mean")
    parser.add_argument("--trajectory-logprob-reduction", choices=["sum", "mean"], default="sum")
    parser.add_argument("--prompt-style", default="full", choices=["full", "sft_minimal"])
    parser.add_argument("--max-history", type=int, default=4)
    parser.add_argument("--image-max-pixels", type=int, default=262144)
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--torch-dtype", default="bf16", choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    parser.add_argument("--seed", type=int, default=82)
    parser.add_argument("--replay-sft-jsonl", default=None)
    parser.add_argument("--replay-loss-weight", type=float, default=0.05)
    parser.add_argument("--replay-ratio", type=float, default=0.25)
    parser.add_argument("--replay-max-rows", type=int, default=2048)
    parser.add_argument("--skip-old-logprob-cache", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = load_trainable_groups(args)
    if not groups:
        summary = {"trained": False, "reason": "no_trainable_groups", "groups": 0}
        (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    task_map = {task.task_id: task for task in load_tasks(args.tasks, limit=None)}
    processor, model = load_trainable_qwen(args)
    optimizer = torch.optim.AdamW([param for param in model.parameters() if param.requires_grad], lr=args.learning_rate)
    trainable, total = parameter_counts(model)

    if not args.skip_old_logprob_cache:
        model.eval()
        fill_old_logprobs(model, processor, groups, task_map, args)
    model.train()

    replay_rows = load_sft_replay_rows(args)
    replay_rng = random.Random(args.seed + 3001)
    losses: list[float] = []
    policy_losses: list[float] = []
    clip_losses: list[float] = []
    approx_kls: list[float] = []
    replay_losses: list[float] = []
    reward_means: list[float] = []
    reward_stds: list[float] = []
    trainable_group_count = 0
    micro_steps = 0
    optimizer_steps = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        random.Random(args.seed + epoch).shuffle(groups)
        for group in groups:
            metrics = trajectory_group_backward(model, processor, group, task_map, args)
            if metrics is None:
                continue
            trainable_group_count += 1
            loss_scalar = metrics["policy_loss"]
            replay_count = sample_replay_count(args.replay_ratio, replay_rng) if replay_rows and args.replay_loss_weight > 0 else 0
            replay_loss_value: torch.Tensor | None = None
            if replay_count > 0:
                sampled = []
                for _ in range(replay_count):
                    sampled.append(sft_replay_loss(model, processor, replay_rows[replay_rng.randrange(len(replay_rows))], args))
                replay_loss_value = torch.stack(sampled).mean()
                replay_scalar = float(replay_loss_value.detach().cpu().item())
                loss_scalar += float(args.replay_loss_weight) * replay_scalar
                replay_losses.append(replay_scalar)
                (float(args.replay_loss_weight) * replay_loss_value / max(1, args.gradient_accumulation_steps)).backward()
            losses.append(loss_scalar)
            policy_losses.append(metrics["policy_loss"])
            clip_losses.append(metrics["clip_loss"])
            approx_kls.append(metrics["approx_kl"])
            reward_means.append(metrics["reward_mean"])
            reward_stds.append(metrics["reward_std"])
            micro_steps += 1
            if micro_steps % max(1, args.gradient_accumulation_steps) == 0:
                step_optimizer(model, optimizer, args)
                optimizer_steps += 1
            if trainable_group_count % 5 == 0:
                print(
                    json.dumps(
                        {
                            "trained_groups": trainable_group_count,
                            "optimizer_steps": optimizer_steps,
                            "last_loss": loss_scalar,
                            "last_policy_loss": metrics["policy_loss"],
                            "last_approx_kl": metrics["approx_kl"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    if micro_steps % max(1, args.gradient_accumulation_steps) != 0:
        step_optimizer(model, optimizer, args)
        optimizer_steps += 1

    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    write_jsonl(output_dir / "train_groups_with_old_logprobs.jsonl", groups)
    summary = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "trained": True,
        "algorithm": "trajectory_level_grpo_clip_approx_kl",
        "groups_input": args.groups,
        "trainable_groups_loaded": len(groups),
        "trainable_groups_used": trainable_group_count,
        "micro_steps": micro_steps,
        "optimizer_steps": optimizer_steps,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "clip_epsilon": args.clip_epsilon,
        "kl_beta": args.kl_beta,
        "trajectory_logprob_reduction": args.trajectory_logprob_reduction,
        "logprob_reduction": args.logprob_reduction,
        "adapter_dir": str(adapter_dir),
        "base_model": args.model,
        "start_adapter": args.adapter,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_ratio": trainable / max(1, total),
        "loss_mean": mean(losses),
        "loss_last": losses[-1] if losses else None,
        "policy_loss_mean": mean(policy_losses),
        "clip_loss_mean": mean(clip_losses),
        "approx_kl_mean": mean(approx_kls),
        "reward_mean": mean(reward_means),
        "reward_std_mean": mean(reward_stds),
        "replay_rows": len(replay_rows),
        "replay_ratio": args.replay_ratio,
        "replay_loss_weight": args.replay_loss_weight,
        "replay_loss_mean": mean(replay_losses) if replay_losses else None,
        "note": "old_logprob is precomputed from the starting adapter; KL is an approximate sampled-action KL proxy 0.5*(logp_new-logp_old)^2.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


def load_trainable_groups(args: argparse.Namespace) -> list[dict[str, Any]]:
    groups = [group for group in read_jsonl(Path(args.groups)) if group.get("trainable")]
    groups = [group for group in groups if float(group.get("reward_std", 0.0)) > args.min_reward_std]
    if args.max_groups and len(groups) > args.max_groups:
        groups = groups[: args.max_groups]
    return groups


def fill_old_logprobs(model: Any, processor: Any, groups: list[dict[str, Any]], task_map: dict[str, Any], args: argparse.Namespace) -> None:
    device = infer_input_device(model)
    for group_index, group in enumerate(groups):
        for traj in group.get("trajectories", []):
            with torch.no_grad():
                logp = trajectory_logprob(model, processor, group, traj, task_map, args)
            traj["old_logprob"] = float(logp.detach().to(device).cpu().item())
        if (group_index + 1) % 5 == 0:
            print(json.dumps({"old_logprob_cached_groups": group_index + 1, "total": len(groups)}, ensure_ascii=False), flush=True)


def trajectory_group_backward(
    model: Any,
    processor: Any,
    group: dict[str, Any],
    task_map: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, float] | None:
    trajectories = list(group.get("trajectories") or [])
    rewards = torch.tensor([float(traj.get("reward", 0.0)) for traj in trajectories], device=infer_input_device(model))
    if rewards.numel() < 2:
        return None
    reward_std = rewards.std(unbiased=False)
    if float(reward_std.detach().cpu().item()) <= args.min_reward_std:
        return None
    advantages = (rewards - rewards.mean()) / reward_std.clamp_min(1e-8)
    grad_scale = 1.0 / (len(trajectories) * max(1, args.gradient_accumulation_steps))
    policy_loss_values = []
    clip_loss_values = []
    approx_kl_values = []
    for traj, advantage in zip(trajectories, advantages, strict=False):
        if "old_logprob" not in traj:
            raise RuntimeError(f"missing old_logprob in {group.get('group_id')} {traj.get('trajectory_id')}")
        logp = trajectory_logprob(model, processor, group, traj, task_map, args)
        old_logp = torch.tensor(float(traj["old_logprob"]), device=logp.device, dtype=logp.dtype)
        adv = advantage.detach().to(device=logp.device, dtype=logp.dtype)
        log_ratio = logp - old_logp
        ratio = torch.exp(log_ratio.clamp(min=-20.0, max=20.0))
        clipped_ratio = torch.clamp(ratio, 1.0 - float(args.clip_epsilon), 1.0 + float(args.clip_epsilon))
        objective = torch.minimum(ratio * adv, clipped_ratio * adv)
        clip_loss = -objective
        approx_kl = 0.5 * (log_ratio**2)
        policy_loss = clip_loss + float(args.kl_beta) * approx_kl
        if policy_loss.requires_grad:
            (policy_loss * grad_scale).backward()
        policy_loss_values.append(float(policy_loss.detach().cpu().item()))
        clip_loss_values.append(float(clip_loss.detach().cpu().item()))
        approx_kl_values.append(float(approx_kl.detach().cpu().item()))
        del logp, old_logp, adv, log_ratio, ratio, clipped_ratio, objective, clip_loss, approx_kl, policy_loss
    return {
        "policy_loss": mean(policy_loss_values) or 0.0,
        "clip_loss": mean(clip_loss_values) or 0.0,
        "approx_kl": mean(approx_kl_values) or 0.0,
        "reward_mean": float(rewards.mean().detach().cpu().item()),
        "reward_std": float(reward_std.detach().cpu().item()),
    }


def trajectory_logprob(model: Any, processor: Any, group: dict[str, Any], traj: dict[str, Any], task_map: dict[str, Any], args: argparse.Namespace) -> torch.Tensor:
    task_id = str(group.get("task_id"))
    task = task_map.get(task_id)
    if task is None:
        raise KeyError(f"task not found for group: {task_id}")
    step_logps = []
    history: list[dict[str, Any]] = []
    for step in traj.get("steps", []):
        obs = {
            "task_id": task_id,
            "goal": group.get("goal") or task.goal,
            "screenshot": step.get("screenshot"),
            "viewport": list(task.viewport),
            "step": step.get("t"),
            "history": list(history),
            "action_space": task.action_space,
            "max_steps": task.max_steps,
        }
        prompt = local_qwen_prompt(obs, max_history=args.max_history, style=args.prompt_style)
        completion = str((step.get("policy_info") or {}).get("raw_text") or json.dumps(step.get("action") or {}, ensure_ascii=False, separators=(",", ":")))
        step_logps.append(
            completion_logprob(
                model,
                processor,
                image_path=resolve_path(step.get("screenshot")),
                prompt=prompt,
                completion=completion,
                system_prompt=args.system_prompt,
                reduction=args.logprob_reduction,
            )
        )
        history.append(
            {
                "step": int(step.get("t", 0)) + 1,
                "action": step.get("action"),
                "exec_status": step.get("exec_status"),
                "exec_error": step.get("exec_error"),
                "verifier": step.get("verifier"),
            }
        )
    if not step_logps:
        return torch.zeros((), device=infer_input_device(model))
    stacked = torch.stack(step_logps)
    if args.trajectory_logprob_reduction == "mean":
        return stacked.mean()
    return stacked.sum()


def step_optimizer(model: Any, optimizer: torch.optim.Optimizer, args: argparse.Namespace) -> None:
    if args.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_([param for param in model.parameters() if param.requires_grad], args.grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def resolve_path(value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else PROJECT_ROOT / path


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


if __name__ == "__main__":
    raise SystemExit(main())
