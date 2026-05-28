#!/usr/bin/env python3
"""Run browser-RL rollouts with local or BrowserGym task sources.

Rollout means one full attempt at a task: observe screenshot, choose an action,
execute it, receive reward, and repeat until success/failure/step limit.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.browser_rl import BrowserGymMiniwobAdapter, LocalQwenPolicy, PlaywrightBrowserEnv, QwenDashScopePolicy, load_tasks
from envs.browser_rl.recorder import RolloutRecorder, write_jsonl


DEFAULT_MINIWOB_TASKS = [
    "browsergym/miniwob.click-button",
    "browsergym/miniwob.click-checkboxes",
    "browsergym/miniwob.enter-text",
    "browsergym/miniwob.choose-list",
    "browsergym/miniwob.focus-text",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-source", choices=["local_smoke", "browsergym_miniwob"], required=True)
    parser.add_argument(
        "--policy",
        choices=["scripted_oracle", "random", "noop", "qwen_dashscope", "local_qwen"],
        default="scripted_oracle",
    )
    parser.add_argument("--tasks", default="outputs/browser_rl_smoke_tasks_20260527_1w/all_tasks.jsonl")
    parser.add_argument("--browsergym-tasks", nargs="*", default=DEFAULT_MINIWOB_TASKS)
    parser.add_argument("--miniwob-url", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--qwen-models",
        default="qwen3.6-flash-2026-04-16,qwen3.7-max-2026-05-20,qwen3.7-max",
        help="Comma-separated DashScope Qwen teacher/baseline model fallback list; this is not the local trainable current model.",
    )
    parser.add_argument("--qwen-base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--qwen-api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--qwen-env-file", default=".env")
    parser.add_argument("--qwen-temperature", type=float, default=0.0)
    parser.add_argument("--qwen-max-tokens", type=int, default=256)
    parser.add_argument("--qwen-timeout", type=float, default=120.0)
    parser.add_argument("--qwen-retries", type=int, default=0)
    parser.add_argument("--qwen-max-history", type=int, default=4)
    parser.add_argument("--local-model", default="/root/models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--local-adapter", default=None)
    parser.add_argument("--local-device-map", default="auto")
    parser.add_argument("--local-load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-torch-dtype", default="auto", choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    parser.add_argument("--local-max-new-tokens", type=int, default=128)
    parser.add_argument("--local-temperature", type=float, default=0.0)
    parser.add_argument("--local-max-history", type=int, default=4)
    parser.add_argument("--local-prompt-style", default="sft_minimal", choices=["sft_minimal", "full"])
    parser.add_argument("--local-system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--local-image-max-pixels", type=int, default=262144)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_policy = build_model_policy(args)
    if args.env_source == "local_smoke":
        rollouts, sft_rows = run_local(args, rng, model_policy)
    else:
        rollouts, sft_rows = run_browsergym(args, rng, model_policy)
    write_jsonl(output_dir / "rollouts.jsonl", rollouts)
    (output_dir / "sft_messages.json").write_text(json.dumps(sft_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    success_values = [1.0 if row.get("success") else 0.0 for row in rollouts]
    steps = [int(row.get("num_steps") or 0) for row in rollouts]
    valid_rates = [valid_action_rate(row) for row in rollouts]
    summary = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "env_source": args.env_source,
        "policy": args.policy,
        "rollouts": len(rollouts),
        "success_rate": sum(success_values) / max(1, len(success_values)),
        "avg_steps": statistics.mean(steps) if steps else 0.0,
        "avg_valid_action_rate": statistics.mean(valid_rates) if valid_rates else 0.0,
        "qwen_valid_json_rate": qwen_metric_rate(rollouts, "valid_json"),
        "qwen_valid_action_rate": qwen_metric_rate(rollouts, "valid_action"),
        "qwen_error_rate": qwen_error_rate(rollouts),
        "qwen_models": split_models(args.qwen_models) if args.policy == "qwen_dashscope" else [],
        "policy_valid_json_rate": policy_metric_rate(rollouts, args.policy, "valid_json"),
        "policy_valid_action_rate": policy_metric_rate(rollouts, args.policy, "valid_action"),
        "policy_error_rate": policy_error_rate(rollouts, args.policy),
        "local_model": args.local_model if args.policy == "local_qwen" else None,
        "local_adapter": args.local_adapter if args.policy == "local_qwen" else None,
        "local_load_in_4bit": args.local_load_in_4bit if args.policy == "local_qwen" else None,
        "sft_rows": len(sft_rows),
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


ModelPolicy = QwenDashScopePolicy | LocalQwenPolicy


def build_model_policy(args: argparse.Namespace) -> ModelPolicy | None:
    if args.policy == "qwen_dashscope":
        return build_qwen_policy(args)
    if args.policy == "local_qwen":
        return build_local_qwen_policy(args)
    return None


def build_qwen_policy(args: argparse.Namespace) -> QwenDashScopePolicy:
    return QwenDashScopePolicy(
        models=split_models(args.qwen_models),
        base_url=args.qwen_base_url,
        api_key_env=args.qwen_api_key_env,
        env_file=args.qwen_env_file,
        temperature=args.qwen_temperature,
        max_tokens=args.qwen_max_tokens,
        timeout=args.qwen_timeout,
        retries=args.qwen_retries,
        max_history=args.qwen_max_history,
    )


def build_local_qwen_policy(args: argparse.Namespace) -> LocalQwenPolicy:
    return LocalQwenPolicy(
        model_path=args.local_model,
        adapter_path=args.local_adapter,
        device_map=args.local_device_map,
        load_in_4bit=args.local_load_in_4bit,
        torch_dtype=args.local_torch_dtype,
        max_new_tokens=args.local_max_new_tokens,
        temperature=args.local_temperature,
        max_history=args.local_max_history,
        prompt_style=args.local_prompt_style,
        system_prompt=args.local_system_prompt,
        image_max_pixels=args.local_image_max_pixels,
    )


def split_models(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def run_local(
    args: argparse.Namespace,
    rng: random.Random,
    model_policy: ModelPolicy | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tasks = load_tasks(args.tasks, limit=args.limit)
    rollouts: list[dict[str, Any]] = []
    sft_rows: list[dict[str, Any]] = []
    with PlaywrightBrowserEnv(Path(args.output_dir) / "artifacts", headless=args.headless) as env:
        for task in tasks:
            recorder = RolloutRecorder(Path(args.output_dir) / "artifacts", policy_version=args.policy)
            recorder.start(f"{args.policy}_{task.task_id}", task.task_id, task.goal)
            obs, _ = env.reset(task)
            total_reward = 0.0
            final_info: dict[str, Any] = {}
            for step_index in range(min(task.max_steps, args.max_steps)):
                action, policy_info = choose_local_action(args.policy, task, step_index, rng, obs, model_policy)
                before_screenshot = obs["screenshot"]
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                step = rollout_step(task.goal, before_screenshot, action, info, reward, step_index, policy_info)
                recorder.add_step(step)
                if args.policy == "scripted_oracle":
                    sft_rows.append(sft_row(task.task_id, step_index, task.goal, before_screenshot, info["action"]))
                final_info = info
                if terminated or truncated:
                    break
            success = bool((final_info.get("verifier") or {}).get("success"))
            rollouts.append(recorder.finish(success=success, total_reward=total_reward, final_info=final_info))
    return rollouts, sft_rows


def run_browsergym(
    args: argparse.Namespace,
    rng: random.Random,
    model_policy: ModelPolicy | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    task_ids = args.browsergym_tasks[: args.limit] if args.limit is not None else list(args.browsergym_tasks)
    miniwob_url = args.miniwob_url or default_miniwob_url()
    rollouts: list[dict[str, Any]] = []
    with BrowserGymMiniwobAdapter(Path(args.output_dir) / "artifacts", miniwob_url=miniwob_url, headless=args.headless) as adapter:
        for task_id in task_ids:
            recorder = RolloutRecorder(Path(args.output_dir) / "artifacts", policy_version=args.policy)
            obs, _ = adapter.reset(task_id, seed=args.seed)
            recorder.start(f"{args.policy}_{task_id.replace('/', '_')}", task_id, obs["goal"])
            total_reward = 0.0
            final_info: dict[str, Any] = {}
            for step_index in range(args.max_steps):
                action, policy_info = choose_browsergym_action(args.policy, rng, obs, model_policy)
                before_screenshot = obs["screenshot"]
                obs, reward, terminated, truncated, info = adapter.step(action)
                total_reward += reward
                recorder.add_step(rollout_step(obs["goal"], before_screenshot, action, info, reward, step_index, policy_info))
                final_info = info
                if terminated or truncated:
                    break
            success = bool((final_info.get("verifier") or {}).get("success"))
            rollouts.append(recorder.finish(success=success, total_reward=total_reward, final_info=final_info))
    return rollouts, []


def choose_local_action(
    policy: str,
    task: Any,
    step_index: int,
    rng: random.Random,
    observation: dict[str, Any],
    model_policy: ModelPolicy | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if policy in {"qwen_dashscope", "local_qwen"}:
        if model_policy is None:
            raise RuntimeError(f"{policy} policy is not initialized")
        result = model_policy.act(observation)
        return result.action, result.info
    if policy == "scripted_oracle":
        if step_index < len(task.oracle_actions):
            return task.oracle_actions[step_index], {"policy": policy}
        return {"action": "finish"}, {"policy": policy}
    action = choose_random_action(rng) if policy == "random" else {"action": "wait"}
    return action, {"policy": policy}


def choose_browsergym_action(
    policy: str,
    rng: random.Random,
    observation: dict[str, Any],
    model_policy: ModelPolicy | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if policy in {"qwen_dashscope", "local_qwen"}:
        if model_policy is None:
            raise RuntimeError(f"{policy} policy is not initialized")
        result = model_policy.act(observation)
        return result.action, result.info
    action = choose_random_action(rng) if policy == "random" else {"action": "wait"}
    return action, {"policy": policy}


def choose_random_action(rng: random.Random) -> dict[str, Any]:
    choice = rng.choice(["click", "wait", "press", "type", "scroll"])
    if choice == "click":
        return {"action": "click", "x": rng.randint(50, 950), "y": rng.randint(50, 950)}
    if choice == "press":
        return {"action": "press", "key": rng.choice(["Enter", "Tab", "Escape"])}
    if choice == "type":
        return {"action": "type", "text": rng.choice(["test", "hello", "1"])}
    if choice == "scroll":
        return {"action": "scroll", "dy": rng.choice([-250, 250])}
    return {"action": "wait"}


def rollout_step(
    goal: str,
    screenshot: str,
    action: dict[str, Any],
    info: dict[str, Any],
    reward: float,
    step_index: int,
    policy_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "t": step_index,
        "screenshot": screenshot,
        "goal": goal,
        "policy_input": {"goal": goal, "screenshot": screenshot},
        "policy_output": action,
        "policy_info": policy_info or {},
        "action": info.get("action"),
        "exec_status": info.get("exec_status"),
        "exec_error": info.get("exec_error"),
        "reward_step": reward,
        "verifier": info.get("verifier"),
        "browsergym_action": info.get("browsergym_action"),
    }


def sft_row(task_id: str, step_index: int, goal: str, screenshot: str, action: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": "<image>\n任务：" + goal + "\n请输出下一步 GUI action JSON。"},
            {"role": "assistant", "content": json.dumps(action, ensure_ascii=False)},
        ],
        "images": [screenshot],
        "task_id": task_id,
        "step": step_index,
    }


def valid_action_rate(row: dict[str, Any]) -> float:
    steps = row.get("trajectory") or []
    if not steps:
        return 0.0
    return sum(1 for step in steps if step.get("exec_status") == "ok") / len(steps)


def qwen_metric_rate(rollouts: list[dict[str, Any]], key: str) -> float | None:
    return policy_metric_rate(rollouts, "qwen_dashscope", key)


def policy_metric_rate(rollouts: list[dict[str, Any]], policy: str, key: str) -> float | None:
    values: list[bool] = []
    for row in rollouts:
        for step in row.get("trajectory") or []:
            policy_info = step.get("policy_info") or {}
            if policy_info.get("policy") == policy and key in policy_info:
                values.append(bool(policy_info.get(key)))
    if not values:
        return None
    return sum(1 for value in values if value) / len(values)


def qwen_error_rate(rollouts: list[dict[str, Any]]) -> float | None:
    return policy_error_rate(rollouts, "qwen_dashscope")


def policy_error_rate(rollouts: list[dict[str, Any]], policy: str) -> float | None:
    values: list[bool] = []
    for row in rollouts:
        for step in row.get("trajectory") or []:
            policy_info = step.get("policy_info") or {}
            if policy_info.get("policy") == policy:
                values.append(bool(policy_info.get("error")))
    if not values:
        return None
    return sum(1 for value in values if value) / len(values)


def default_miniwob_url() -> str | None:
    path = Path("/root/models/datasets/miniwob-plusplus-main/miniwob/html/miniwob")
    if path.exists():
        return f"file://{path}/"
    return None


if __name__ == "__main__":
    main()
