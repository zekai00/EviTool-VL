#!/usr/bin/env python3
"""Manually play one local browser-RL task by typing direct-action JSON."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.browser_rl import PlaywrightBrowserEnv, load_tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="outputs/browser_rl_smoke_tasks_20260527_1w/all_tasks.jsonl")
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--index", type=int, default=1, help="1-based task index when --task-id is not set.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--list", action="store_true", help="List tasks and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = load_tasks(args.tasks)
    if args.list:
        for idx, task in enumerate(tasks, start=1):
            print(f"{idx:02d}\t{task.task_id}\t{task.template}\t{task.goal}")
        return
    if args.task_id:
        task = next((item for item in tasks if item.task_id == args.task_id), None)
        if task is None:
            raise SystemExit(f"task not found: {args.task_id}")
    else:
        task = tasks[args.index - 1]
    out_dir = Path(args.output_dir or f"outputs/browser_rl_manual_{task.task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    print(f"Task: {task.task_id}")
    print(f"Goal: {task.goal}")
    print(f"Max steps: {task.max_steps}")
    print("Example actions:")
    print('  {"action":"click","x":360,"y":194}')
    print('  {"action":"type","text":"user_1"}')
    print('  {"action":"press","key":"Enter"}')
    print('  {"action":"finish"}')
    print("Coordinates use 0-1000 normalized screenshot space. Type q to quit.")
    with PlaywrightBrowserEnv(out_dir, headless=args.headless) as env:
        obs, _ = env.reset(task)
        print(f"Initial screenshot: {obs['screenshot']}")
        total_reward = 0.0
        for _ in range(task.max_steps):
            text = input("action> ").strip()
            if text.lower() in {"q", "quit", "exit"}:
                break
            if not text:
                continue
            obs, reward, terminated, truncated, info = env.step(text)
            total_reward += reward
            print(json.dumps({
                "reward": reward,
                "total_reward": total_reward,
                "terminated": terminated,
                "truncated": truncated,
                "exec_status": info["exec_status"],
                "exec_error": info["exec_error"],
                "verifier": info["verifier"],
                "screenshot": obs["screenshot"],
            }, ensure_ascii=False, indent=2))
            if terminated or truncated:
                break
    print(f"Artifacts: {out_dir}")


if __name__ == "__main__":
    main()
