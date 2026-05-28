#!/usr/bin/env python3
"""Validate browser GUI-RL rollout JSONL files.

Schema validator means a data-format checker. It verifies that rollout records
contain the fields required by SFT/RL pipelines: screenshots, actions, rewards,
verifier results, terminal status, and policy metadata when available.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.browser_rl.actions import parse_action
from envs.browser_rl.recorder import write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, help="rollouts.jsonl path; can be repeated.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--max-errors", type=int, default=200)
    parser.add_argument("--write-valid-rollouts", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(args.repo_root).resolve()

    all_rows: list[dict[str, Any]] = []
    all_errors: list[dict[str, Any]] = []
    valid_rows: list[dict[str, Any]] = []
    per_file: list[dict[str, Any]] = []

    for input_path in [Path(path) for path in args.input]:
        rows, read_errors = read_jsonl(input_path)
        all_errors.extend(read_errors)
        file_errors: list[dict[str, Any]] = list(read_errors)
        file_valid = 0
        for index, row in enumerate(rows):
            errors = validate_rollout(row, input_path=input_path, row_index=index, repo_root=repo_root)
            if errors:
                file_errors.extend(errors)
                all_errors.extend(errors)
            else:
                file_valid += 1
                valid_rows.append(row)
            all_rows.append(row)
        per_file.append(
            {
                "input": str(input_path),
                "rows": len(rows),
                "valid_rows": file_valid,
                "error_count": len(file_errors),
            }
        )

    summary = build_summary(all_rows, all_errors, per_file)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(output_dir / "errors.jsonl", all_errors[: args.max_errors])
    if args.write_valid_rollouts:
        write_jsonl(output_dir / "valid_rollouts.jsonl", valid_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    if not path.exists():
        return [], [error_record(path, None, None, "file_missing", f"input file does not exist: {path}")]
    with path.open(encoding="utf-8") as f:
        for line_index, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError("row is not a JSON object")
                rows.append(payload)
            except Exception as exc:
                errors.append(error_record(path, None, None, "json_parse_error", f"line {line_index}: {type(exc).__name__}: {exc}"))
    return rows, errors


def validate_rollout(row: dict[str, Any], *, input_path: Path, row_index: int, repo_root: Path) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    rollout_id = string_or_none(row.get("rollout_id"))
    task_id = string_or_none(row.get("task_id"))
    trajectory = row.get("trajectory")

    for field in ("rollout_id", "policy_version", "task_id", "goal"):
        if not string_or_none(row.get(field)):
            errors.append(error_record(input_path, row_index, rollout_id, "missing_top_level_field", field))
    if not isinstance(trajectory, list):
        errors.append(error_record(input_path, row_index, rollout_id, "invalid_trajectory", "trajectory must be a list"))
        return errors
    if row.get("num_steps") != len(trajectory):
        errors.append(
            error_record(input_path, row_index, rollout_id, "num_steps_mismatch", f"num_steps={row.get('num_steps')} len={len(trajectory)}")
        )
    if not isinstance(row.get("success"), bool):
        errors.append(error_record(input_path, row_index, rollout_id, "invalid_success", "success must be bool"))
    if not is_number(row.get("total_reward")):
        errors.append(error_record(input_path, row_index, rollout_id, "invalid_total_reward", "total_reward must be numeric"))

    final_info = row.get("final_info")
    if final_info is not None and not isinstance(final_info, dict):
        errors.append(error_record(input_path, row_index, rollout_id, "invalid_final_info", "final_info must be object"))
    if isinstance(final_info, dict):
        final_verifier = final_info.get("verifier") or {}
        if isinstance(final_verifier, dict) and "success" in final_verifier and isinstance(row.get("success"), bool):
            if bool(final_verifier.get("success")) != bool(row.get("success")):
                errors.append(error_record(input_path, row_index, rollout_id, "success_mismatch", "top-level success differs from final verifier"))

    for step_index, step in enumerate(trajectory):
        errors.extend(validate_step(step, input_path=input_path, row_index=row_index, rollout_id=rollout_id, step_index=step_index, repo_root=repo_root))
    return errors


def validate_step(
    step: Any,
    *,
    input_path: Path,
    row_index: int,
    rollout_id: str | None,
    step_index: int,
    repo_root: Path,
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    if not isinstance(step, dict):
        return [error_record(input_path, row_index, rollout_id, "invalid_step", f"step {step_index} is not an object", step_index)]
    if step.get("t") != step_index:
        errors.append(error_record(input_path, row_index, rollout_id, "step_index_mismatch", f"t={step.get('t')} expected={step_index}", step_index))
    screenshot = string_or_none(step.get("screenshot"))
    if not screenshot:
        errors.append(error_record(input_path, row_index, rollout_id, "missing_screenshot", "step screenshot is missing", step_index))
    elif not resolve_path(screenshot, repo_root).exists():
        errors.append(error_record(input_path, row_index, rollout_id, "screenshot_missing_on_disk", screenshot, step_index))
    action = step.get("action")
    if not isinstance(action, dict):
        errors.append(error_record(input_path, row_index, rollout_id, "invalid_action_field", "action must be object", step_index))
    else:
        parsed = parse_action(action)
        if not parsed.ok:
            errors.append(error_record(input_path, row_index, rollout_id, "action_parse_failed", str(parsed.error), step_index))
    if step.get("exec_status") != "ok":
        errors.append(error_record(input_path, row_index, rollout_id, "exec_status_not_ok", str(step.get("exec_status")), step_index))
    if not is_number(step.get("reward_step")):
        errors.append(error_record(input_path, row_index, rollout_id, "invalid_reward_step", "reward_step must be numeric", step_index))
    verifier = step.get("verifier")
    if not isinstance(verifier, dict):
        errors.append(error_record(input_path, row_index, rollout_id, "invalid_verifier", "verifier must be object", step_index))
    else:
        if "success" in verifier and not isinstance(verifier.get("success"), bool):
            errors.append(error_record(input_path, row_index, rollout_id, "invalid_verifier_success", "verifier.success must be bool", step_index))
        if "reward" in verifier and not is_number(verifier.get("reward")):
            errors.append(error_record(input_path, row_index, rollout_id, "invalid_verifier_reward", "verifier.reward must be numeric", step_index))
    policy_info = step.get("policy_info")
    if policy_info is not None and not isinstance(policy_info, dict):
        errors.append(error_record(input_path, row_index, rollout_id, "invalid_policy_info", "policy_info must be object", step_index))
    return errors


def build_summary(rows: list[dict[str, Any]], errors: list[dict[str, Any]], per_file: list[dict[str, Any]]) -> dict[str, Any]:
    step_count = sum(len(row.get("trajectory") or []) for row in rows)
    success_values = [1.0 if row.get("success") else 0.0 for row in rows]
    valid_exec_steps = 0
    action_parse_steps = 0
    screenshot_steps = 0
    qwen_steps = 0
    qwen_valid_json = 0
    qwen_valid_action = 0
    rewards: list[float] = []
    policy_counter: Counter[str] = Counter()
    task_counter: Counter[str] = Counter()

    for row in rows:
        policy_counter[str(row.get("policy_version") or "unknown")] += 1
        task_counter[str(row.get("task_id") or "unknown")] += 1
        for step in row.get("trajectory") or []:
            if not isinstance(step, dict):
                continue
            if step.get("exec_status") == "ok":
                valid_exec_steps += 1
            action = step.get("action")
            if isinstance(action, dict) and parse_action(action).ok:
                action_parse_steps += 1
            screenshot = step.get("screenshot")
            if isinstance(screenshot, str) and screenshot:
                screenshot_steps += 1
            if is_number(step.get("reward_step")):
                rewards.append(float(step["reward_step"]))
            policy_info = step.get("policy_info") or {}
            if isinstance(policy_info, dict) and policy_info.get("policy") == "qwen_dashscope":
                qwen_steps += 1
                qwen_valid_json += 1 if policy_info.get("valid_json") else 0
                qwen_valid_action += 1 if policy_info.get("valid_action") else 0

    error_counter = Counter(str(item.get("code") or "unknown") for item in errors)
    return {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "files": per_file,
        "rollouts": len(rows),
        "valid_rollouts": max(0, len(rows) - len({(item.get("input"), item.get("row_index")) for item in errors if item.get("row_index") is not None})),
        "error_count": len(errors),
        "error_codes": dict(error_counter.most_common()),
        "success_rate": sum(success_values) / max(1, len(success_values)),
        "steps": step_count,
        "exec_ok_step_rate": valid_exec_steps / max(1, step_count),
        "action_parse_step_rate": action_parse_steps / max(1, step_count),
        "screenshot_field_rate": screenshot_steps / max(1, step_count),
        "qwen_steps": qwen_steps,
        "qwen_valid_json_rate": qwen_valid_json / max(1, qwen_steps) if qwen_steps else None,
        "qwen_valid_action_rate": qwen_valid_action / max(1, qwen_steps) if qwen_steps else None,
        "reward_step_mean": statistics.mean(rewards) if rewards else None,
        "policy_versions": dict(policy_counter.most_common()),
        "unique_tasks": len(task_counter),
    }


def resolve_path(path: str, repo_root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return repo_root / candidate


def error_record(input_path: Path, row_index: int | None, rollout_id: str | None, code: str, message: str, step_index: int | None = None) -> dict[str, Any]:
    return {
        "input": str(input_path),
        "row_index": row_index,
        "rollout_id": rollout_id,
        "step_index": step_index,
        "code": code,
        "message": message,
    }


def string_or_none(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


if __name__ == "__main__":
    main()
