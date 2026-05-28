#!/usr/bin/env python3
"""Run center, jittered, and recovery scripted-oracle browser rollouts.

Jittered rollouts click random points inside the target element instead of only
the center point. Recovery rollouts first execute one near-miss click, mark that
bad step as not trainable, then demonstrate the correct recovery action with the
bad action present in history.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.browser_rl import PlaywrightBrowserEnv, load_tasks
from envs.browser_rl.actions import pixels_to_normalized
from envs.browser_rl.recorder import RolloutRecorder, write_jsonl
from envs.browser_rl.task_spec import BrowserTaskSpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policy-prefix", default="scripted_oracle_aug_v1")
    parser.add_argument("--train-jitter-rollouts", type=int, default=2)
    parser.add_argument("--train-recovery-rollouts", type=int, default=1)
    parser.add_argument("--reuse-context", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--jitter-low", type=float, default=0.2, help="Lower fractional bound inside element bbox.")
    parser.add_argument("--jitter-high", type=float, default=0.8, help="Upper fractional bound inside element bbox.")
    parser.add_argument("--no-center", action="store_true", help="Do not write center oracle rollouts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(args.tasks, limit=args.limit)
    rng = random.Random(args.seed)
    rollouts: list[dict[str, Any]] = []
    variant_counter: Counter[str] = Counter()
    split_counter: Counter[str] = Counter()
    family_counter: Counter[str] = Counter()

    with PlaywrightBrowserEnv(output_dir / "artifacts", headless=args.headless, reuse_context=args.reuse_context) as env:
        for task in tasks:
            split = str(task.split or "train")
            split_counter[split] += 1
            family_counter[str(task.metadata.get("family", task.template))] += 1
            if not args.no_center:
                row = run_one_rollout(
                    env,
                    task,
                    rng=rng,
                    policy_version=f"{args.policy_prefix}_center",
                    variant="center",
                    jitter_clicks=False,
                    recovery=False,
                    jitter_low=args.jitter_low,
                    jitter_high=args.jitter_high,
                    replica=0,
                )
                rollouts.append(row)
                variant_counter["center"] += 1

            if split != "train":
                continue

            for replica in range(args.train_jitter_rollouts):
                row = run_one_rollout(
                    env,
                    task,
                    rng=rng,
                    policy_version=f"{args.policy_prefix}_jitter",
                    variant="jitter",
                    jitter_clicks=True,
                    recovery=False,
                    jitter_low=args.jitter_low,
                    jitter_high=args.jitter_high,
                    replica=replica,
                )
                rollouts.append(row)
                variant_counter["jitter"] += 1

            for replica in range(args.train_recovery_rollouts):
                row = run_one_rollout(
                    env,
                    task,
                    rng=rng,
                    policy_version=f"{args.policy_prefix}_recovery",
                    variant="recovery",
                    jitter_clicks=True,
                    recovery=True,
                    jitter_low=args.jitter_low,
                    jitter_high=args.jitter_high,
                    replica=replica,
                )
                rollouts.append(row)
                variant_counter["recovery"] += 1

    write_jsonl(output_dir / "rollouts.jsonl", rollouts)
    success_values = [1.0 if row.get("success") else 0.0 for row in rollouts]
    steps = [int(row.get("num_steps") or 0) for row in rollouts]
    trainable_steps = sum(1 for row in rollouts for step in row.get("trajectory", []) if step.get("sft_include", True))
    skipped_steps = sum(1 for row in rollouts for step in row.get("trajectory", []) if not step.get("sft_include", True))
    summary = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tasks": len(tasks),
        "rollouts": len(rollouts),
        "success_rate": sum(success_values) / max(1, len(success_values)),
        "avg_steps": statistics.mean(steps) if steps else 0.0,
        "trainable_steps": trainable_steps,
        "non_trainable_failure_steps": skipped_steps,
        "variants": dict(variant_counter),
        "task_splits": dict(split_counter),
        "families": dict(family_counter),
        "args": vars(args),
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def run_one_rollout(
    env: PlaywrightBrowserEnv,
    task: BrowserTaskSpec,
    *,
    rng: random.Random,
    policy_version: str,
    variant: str,
    jitter_clicks: bool,
    recovery: bool,
    jitter_low: float,
    jitter_high: float,
    replica: int,
) -> dict[str, Any]:
    recorder = RolloutRecorder(output_dir=env.output_dir, policy_version=policy_version)
    rollout_id = f"{policy_version}_{task.task_id}_r{replica}"
    recorder.start(rollout_id, task.task_id, task.goal)
    env.screenshot_prefix = rollout_id
    obs, _ = env.reset(task)
    total_reward = 0.0
    terminated = False
    truncated = False
    final_info: dict[str, Any] = {}
    recovery_before_index = choose_recovery_index(task, rng) if recovery else None

    for oracle_index, oracle_action in enumerate(task.oracle_actions):
        if recovery_before_index == oracle_index:
            bad_action = near_miss_click(env, oracle_action, task.viewport, rng)
            before_screenshot = obs["screenshot"]
            obs, reward, terminated, truncated, info = env.step(bad_action)
            total_reward += reward
            recorder.add_step(
                rollout_step(
                    task,
                    before_screenshot,
                    info,
                    reward,
                    oracle_action=oracle_action,
                    variant=variant,
                    oracle_index=oracle_index,
                    sft_include=False,
                    augmentation={
                        "kind": "failure_recovery_context",
                        "failure_action": bad_action,
                        "note": "Near-miss action is context only; do not train it as the target action.",
                    },
                )
            )
            final_info = info
            if terminated or truncated:
                break

        action = materialize_oracle_action(
            env,
            oracle_action,
            task.viewport,
            rng,
            jitter=jitter_clicks,
            jitter_low=jitter_low,
            jitter_high=jitter_high,
        )
        before_screenshot = obs["screenshot"]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        recorder.add_step(
            rollout_step(
                task,
                before_screenshot,
                info,
                reward,
                oracle_action=oracle_action,
                variant=variant,
                oracle_index=oracle_index,
                sft_include=True,
                augmentation={
                    "kind": "recovery_target" if recovery_before_index == oracle_index else "coordinate_jitter" if jitter_clicks else "center",
                    "replica": replica,
                },
            )
        )
        final_info = info
        if terminated or truncated:
            break

    success = bool((final_info.get("verifier") or {}).get("success"))
    return recorder.finish(success=success, total_reward=total_reward, final_info=final_info)


def choose_recovery_index(task: BrowserTaskSpec, rng: random.Random) -> int | None:
    indices = [
        index
        for index, action in enumerate(task.oracle_actions)
        if str(action.get("action") or "") in {"click", "double_click"} and action.get("selector")
    ]
    if not indices:
        return None
    return rng.choice(indices)


def materialize_oracle_action(
    env: PlaywrightBrowserEnv,
    oracle_action: dict[str, Any],
    viewport: tuple[int, int],
    rng: random.Random,
    *,
    jitter: bool,
    jitter_low: float,
    jitter_high: float,
) -> dict[str, Any]:
    name = str(oracle_action.get("action") or "")
    selector = oracle_action.get("selector")
    if name in {"click", "double_click"} and selector and jitter:
        return jittered_click(env, str(selector), viewport, rng, action=name, low=jitter_low, high=jitter_high)
    return dict(oracle_action)


def jittered_click(
    env: PlaywrightBrowserEnv,
    selector: str,
    viewport: tuple[int, int],
    rng: random.Random,
    *,
    action: str,
    low: float,
    high: float,
) -> dict[str, Any]:
    box = selector_box(env, selector)
    low = max(0.05, min(0.95, low))
    high = max(low, min(0.95, high))
    px = box["x"] + box["width"] * rng.uniform(low, high)
    py = box["y"] + box["height"] * rng.uniform(low, high)
    x, y = pixels_to_normalized(px, py, viewport)
    return {"action": action, "x": float(x), "y": float(y)}


def near_miss_click(
    env: PlaywrightBrowserEnv,
    oracle_action: dict[str, Any],
    viewport: tuple[int, int],
    rng: random.Random,
) -> dict[str, Any]:
    selector = oracle_action.get("selector")
    if not selector:
        return {"action": "click", "x": 20.0, "y": 20.0}
    box = selector_box(env, str(selector))
    width, height = viewport
    candidates = [
        (box["x"] - 18, box["y"] + box["height"] / 2),
        (box["x"] + box["width"] + 18, box["y"] + box["height"] / 2),
        (box["x"] + box["width"] / 2, box["y"] - 18),
        (box["x"] + box["width"] / 2, box["y"] + box["height"] + 18),
        (24, 24),
    ]
    rng.shuffle(candidates)
    for px, py in candidates:
        if 1 <= px <= width - 1 and 1 <= py <= height - 1 and not point_in_box(px, py, box):
            x, y = pixels_to_normalized(px, py, viewport)
            return {"action": "click", "x": float(x), "y": float(y)}
    return {"action": "click", "x": 20.0, "y": 20.0}


def selector_box(env: PlaywrightBrowserEnv, selector: str) -> dict[str, float]:
    locator = env.page.locator(selector).first
    locator.wait_for(state="visible", timeout=5000)
    box = locator.bounding_box()
    if box is None:
        raise RuntimeError(f"selector has no bounding box: {selector}")
    return {name: float(box[name]) for name in ("x", "y", "width", "height")}


def point_in_box(px: float, py: float, box: dict[str, float]) -> bool:
    return box["x"] <= px <= box["x"] + box["width"] and box["y"] <= py <= box["y"] + box["height"]


def rollout_step(
    task: BrowserTaskSpec,
    before_screenshot: str,
    info: dict[str, Any],
    reward: float,
    *,
    oracle_action: dict[str, Any],
    variant: str,
    oracle_index: int,
    sft_include: bool,
    augmentation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "t": info["step"] - 1,
        "screenshot": before_screenshot,
        "goal": task.goal,
        "oracle_input": oracle_action,
        "action": info["action"],
        "exec_status": info["exec_status"],
        "exec_error": info.get("exec_error"),
        "reward_step": reward,
        "verifier": info["verifier"],
        "sft_include": sft_include,
        "augmentation": {
            "variant": variant,
            "oracle_index": oracle_index,
            **augmentation,
        },
    }


if __name__ == "__main__":
    main()
