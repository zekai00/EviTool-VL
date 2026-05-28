#!/usr/bin/env python3
"""Run scripted oracle rollouts on browser-RL smoke tasks."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.browser_rl import PlaywrightBrowserEnv, load_tasks
from envs.browser_rl.recorder import RolloutRecorder, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--policy-version", default="scripted_oracle")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    tasks = load_tasks(args.tasks, limit=args.limit)
    rollouts: list[dict[str, object]] = []
    sft_rows: list[dict[str, object]] = []
    with PlaywrightBrowserEnv(output_dir / "artifacts", headless=args.headless) as env:
        for task in tasks:
            recorder = RolloutRecorder(output_dir=output_dir / "artifacts", policy_version=args.policy_version)
            recorder.start(f"{args.policy_version}_{task.task_id}", task.task_id, task.goal)
            obs, _ = env.reset(task)
            total_reward = 0.0
            terminated = False
            truncated = False
            final_info = {}
            for oracle_action in task.oracle_actions:
                before_screenshot = obs["screenshot"]
                obs, reward, terminated, truncated, info = env.step(oracle_action)
                total_reward += reward
                step = {
                    "t": info["step"] - 1,
                    "screenshot": before_screenshot,
                    "goal": task.goal,
                    "oracle_input": oracle_action,
                    "action": info["action"],
                    "exec_status": info["exec_status"],
                    "reward_step": reward,
                    "verifier": info["verifier"],
                }
                recorder.add_step(step)
                sft_rows.append(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": "<image>\n任务：" + task.goal + "\n请输出下一步 GUI action JSON。",
                            },
                            {"role": "assistant", "content": json.dumps(info["action"], ensure_ascii=False)},
                        ],
                        "images": [before_screenshot],
                        "task_id": task.task_id,
                        "step": step["t"],
                    }
                )
                final_info = info
                if terminated or truncated:
                    break
            success = bool((final_info.get("verifier") or {}).get("success"))
            rollouts.append(recorder.finish(success=success, total_reward=total_reward, final_info=final_info))
    write_jsonl(output_dir / "rollouts.jsonl", rollouts)
    (output_dir / "sft_messages.json").write_text(json.dumps(sft_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    success_values = [1.0 if row["success"] else 0.0 for row in rollouts]
    steps = [int(row["num_steps"]) for row in rollouts]
    summary = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tasks": len(tasks),
        "rollouts": len(rollouts),
        "success_rate": sum(success_values) / max(1, len(success_values)),
        "avg_steps": statistics.mean(steps) if steps else 0.0,
        "sft_rows": len(sft_rows),
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
