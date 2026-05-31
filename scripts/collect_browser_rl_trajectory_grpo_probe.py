#!/usr/bin/env python3
"""Collect trajectory-level GRPO probe groups for BrowserRL.

This script does not train.  It answers one question before we commit to a
full trajectory-level RL implementation: if the current model samples K full
trajectories for the same task, how many task groups have non-zero reward
variance and are therefore trainable?
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.browser_rl import LocalQwenPolicy, PlaywrightBrowserEnv, load_tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe trajectory-level GRPO trainability for BrowserRL.")
    parser.add_argument(
        "--tasks",
        default="/root/datasets/browser_rl/task_suites/browser_rl_task_suite_2000_20260528_1344/train_tasks.jsonl",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="/root/models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter", default="outputs/onpolicy_browser_rl_grpo_table_advanced_repair_289tg_safe_20260529_0101/adapter")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shuffle-tasks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-families", default=None)
    parser.add_argument("--include-templates", default=None)
    parser.add_argument("--num-trajectories", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--sampling-temperature", type=float, default=1.0)
    parser.add_argument("--max-history", type=int, default=4)
    parser.add_argument("--prompt-style", default="full", choices=["full", "sft_minimal"])
    parser.add_argument("--image-max-pixels", type=int, default=262144)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--torch-dtype", default="bf16", choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-reward-std", type=float, default=1e-6)
    parser.add_argument("--invalid-penalty", type=float, default=0.2)
    parser.add_argument("--exec-error-penalty", type=float, default=0.2)
    parser.add_argument("--step-cost", type=float, default=0.0)
    parser.add_argument("--success-bonus", type=float, default=0.0)
    parser.add_argument("--stream-flush-every", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    groups_path = output_dir / "trajectory_groups.jsonl"
    summary_path = output_dir / "summary.json"
    tasks = select_tasks(args)
    policy = LocalQwenPolicy(
        model_path=args.model,
        adapter_path=args.adapter,
        load_in_4bit=args.load_in_4bit,
        torch_dtype=args.torch_dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.sampling_temperature,
        max_history=args.max_history,
        prompt_style=args.prompt_style,
        system_prompt=args.system_prompt,
        image_max_pixels=args.image_max_pixels,
    )

    groups: list[dict[str, Any]] = []
    with PlaywrightBrowserEnv(output_dir / "artifacts", headless=args.headless, reuse_context=True) as env:
        with groups_path.open("w", encoding="utf-8") as f:
            for task_index, task in enumerate(tasks):
                group = collect_task_group(env, policy, task, task_index, args)
                groups.append(group)
                f.write(json.dumps(group, ensure_ascii=False, separators=(",", ":")) + "\n")
                if args.stream_flush_every and (task_index + 1) % args.stream_flush_every == 0:
                    f.flush()
                    summary_path.write_text(
                        json.dumps(summarize(groups, args, started, completed=False), ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                print(
                    json.dumps(
                        {
                            "task_id": task.task_id,
                            "template": task.template,
                            "reward_mean": group["reward_mean"],
                            "reward_std": group["reward_std"],
                            "success_rate": group["success_rate"],
                            "trainable": group["trainable"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    summary = summarize(groups, args, started, completed=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def select_tasks(args: argparse.Namespace) -> list[Any]:
    tasks = load_tasks(args.tasks, limit=None)
    families = parse_csv_set(args.include_families)
    templates = parse_csv_set(args.include_templates)
    if families:
        tasks = [task for task in tasks if str((task.metadata or {}).get("family") or task.template) in families]
    if templates:
        tasks = [task for task in tasks if str(task.template) in templates]
    if args.shuffle_tasks:
        random.Random(args.seed).shuffle(tasks)
    if args.offset:
        tasks = tasks[args.offset :]
    if args.num_shards > 1:
        tasks = [task for index, task in enumerate(tasks) if index % args.num_shards == args.shard_index]
    return tasks[: args.limit]


def collect_task_group(env: PlaywrightBrowserEnv, policy: LocalQwenPolicy, task: Any, task_index: int, args: argparse.Namespace) -> dict[str, Any]:
    trajectories = []
    for traj_index in range(args.num_trajectories):
        trajectories.append(run_one_trajectory(env, policy, task, task_index, traj_index, args))
    rewards = [float(row["reward"]) for row in trajectories]
    reward_mean = sum(rewards) / max(1, len(rewards))
    reward_std = statistics.pstdev(rewards) if len(rewards) >= 2 else 0.0
    success_rate = sum(1 for row in trajectories if row.get("success")) / max(1, len(trajectories))
    return {
        "group_id": f"{task.task_id}_traj_g{task_index:05d}",
        "task_id": task.task_id,
        "goal": task.goal,
        "template": task.template,
        "family": (task.metadata or {}).get("family") or task.template,
        "split": task.split,
        "num_trajectories": len(trajectories),
        "max_steps": min(int(task.max_steps), args.max_steps),
        "reward_mean": reward_mean,
        "reward_std": reward_std,
        "success_rate": success_rate,
        "trainable": reward_std > args.min_reward_std and len(trajectories) >= 2,
        "all_zero_reward": all(abs(value) <= 1e-12 for value in rewards),
        "unique_action_sequences": len({row.get("action_sequence_key", "") for row in trajectories}),
        "trajectories": trajectories,
    }


def run_one_trajectory(
    env: PlaywrightBrowserEnv,
    policy: LocalQwenPolicy,
    task: Any,
    task_index: int,
    traj_index: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    env.screenshot_prefix = f"traj_{task.task_id}_{task_index:05d}_{traj_index:02d}"
    obs, _ = env.reset(task)
    total_reward = 0.0
    total_env_reward = 0.0
    steps = []
    success = False
    terminated = False
    truncated = False
    max_steps = min(int(task.max_steps), args.max_steps)
    for step_index in range(max_steps):
        result = policy.act(obs)
        next_obs, env_reward, terminated, truncated, info = env.step(result.action)
        shaped = shape_step_reward(float(env_reward), info, result.info, args)
        total_reward += shaped
        total_env_reward += float(env_reward)
        success = bool((info.get("verifier") or {}).get("success"))
        steps.append(
            {
                "t": step_index,
                "screenshot": obs.get("screenshot"),
                "action": result.action,
                "policy_info": result.info,
                "env_reward": float(env_reward),
                "reward": shaped,
                "exec_status": info.get("exec_status"),
                "exec_error": info.get("exec_error"),
                "valid_action": info.get("valid_action"),
                "verifier": info.get("verifier"),
                "terminated": terminated,
                "truncated": truncated,
            }
        )
        obs = next_obs
        if terminated or truncated:
            break
    return {
        "trajectory_id": f"{task.task_id}_traj{traj_index:02d}",
        "reward": total_reward,
        "env_reward": total_env_reward,
        "success": success,
        "terminated": terminated,
        "truncated": truncated,
        "num_steps": len(steps),
        "action_sequence_key": action_sequence_key(steps),
        "steps": steps,
    }


def shape_step_reward(env_reward: float, info: dict[str, Any], policy_info: dict[str, Any], args: argparse.Namespace) -> float:
    reward = env_reward - float(args.step_cost)
    if bool((info.get("verifier") or {}).get("success")):
        reward += float(args.success_bonus)
    if policy_info.get("valid_json") is False or policy_info.get("valid_action") is False:
        reward -= float(args.invalid_penalty)
    if info.get("exec_status") != "ok":
        reward -= float(args.exec_error_penalty)
    return reward


def action_sequence_key(steps: list[dict[str, Any]]) -> str:
    actions = [step.get("action") or {} for step in steps]
    return "|".join(json.dumps(action, ensure_ascii=False, sort_keys=True, separators=(",", ":")) for action in actions)


def summarize(groups: list[dict[str, Any]], args: argparse.Namespace, started: float, *, completed: bool) -> dict[str, Any]:
    trainable = [group for group in groups if group.get("trainable")]
    rewards = [float(traj.get("reward", 0.0)) for group in groups for traj in group.get("trajectories", [])]
    trajectories = [traj for group in groups for traj in group.get("trajectories", [])]
    family_counts = Counter(str(group.get("family") or "unknown") for group in groups)
    family_trainable = Counter(str(group.get("family") or "unknown") for group in trainable)
    template_counts = Counter(str(group.get("template") or "unknown") for group in groups)
    template_trainable = Counter(str(group.get("template") or "unknown") for group in trainable)
    return {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "completed": completed,
        "wall_time_sec": time.time() - started,
        "tasks_path": args.tasks,
        "model": args.model,
        "adapter": args.adapter,
        "groups": len(groups),
        "trajectories": len(trajectories),
        "num_trajectories_per_group": args.num_trajectories,
        "trainable_groups": len(trainable),
        "trainable_group_rate": len(trainable) / max(1, len(groups)),
        "all_zero_reward_groups": sum(1 for group in groups if group.get("all_zero_reward")),
        "rollout_success_rate": sum(1 for traj in trajectories if traj.get("success")) / max(1, len(trajectories)),
        "reward_mean": sum(rewards) / max(1, len(rewards)),
        "reward_std": statistics.pstdev(rewards) if len(rewards) >= 2 else 0.0,
        "avg_group_reward_std": sum(float(group.get("reward_std", 0.0)) for group in groups) / max(1, len(groups)),
        "avg_unique_action_sequences": sum(int(group.get("unique_action_sequences", 0)) for group in groups) / max(1, len(groups)),
        "groups_by_family": dict(family_counts),
        "trainable_groups_by_family": dict(family_trainable),
        "groups_by_template": dict(template_counts),
        "trainable_groups_by_template": dict(template_trainable),
    }


def parse_csv_set(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


if __name__ == "__main__":
    raise SystemExit(main())
