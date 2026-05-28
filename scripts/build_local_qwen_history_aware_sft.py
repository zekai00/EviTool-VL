#!/usr/bin/env python3
"""Build history-aware local Qwen trajectory SFT data from GUI rollouts.

History-aware SFT means each training sample includes the current screenshot,
task goal, current step, prior actions, verifier progress, action space, and
then the expert next action. This matches the `local_qwen --local-prompt-style
full` inference path and is intended as the warmup before on-policy RL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.browser_rl.actions import SUPPORTED_ACTIONS, parse_action
from envs.browser_rl.qwen_policy import build_prompt
from envs.browser_rl.task_spec import BrowserTaskSpec


DEFAULT_VIEWPORT = [1280, 720]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, help="rollouts.jsonl path; can be repeated.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tasks", action="append", default=[], help="task jsonl path; can be repeated.")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--require-success", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-policies", default="", help="Comma-separated policy versions to include; empty means all.")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-history", type=int, default=4)
    parser.add_argument("--absolute-image-paths", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dedupe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--split-by-task", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--respect-task-splits", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dataset-name", default="local_qwen_history_aware_sft_train")
    parser.add_argument("--val-dataset-name", default="local_qwen_history_aware_sft_val")
    parser.add_argument("--test-dataset-name", default="local_qwen_history_aware_sft_test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    include_policies = {item.strip() for item in args.include_policies.split(",") if item.strip()}
    task_map = load_task_map([Path(path) for path in args.tasks])
    rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    seen: set[str] = set()

    for input_path in [Path(path) for path in args.input]:
        for rollout in read_jsonl(input_path):
            extracted, local_skipped = extract_rows(
                rollout,
                input_path=input_path,
                repo_root=repo_root,
                task_map=task_map,
                require_success=args.require_success,
                include_policies=include_policies,
                max_history=args.max_history,
                absolute_image_paths=args.absolute_image_paths,
            )
            skipped.update(local_skipped)
            for row in extracted:
                key = row_key(row)
                if args.dedupe and key in seen:
                    skipped["deduped"] += 1
                    continue
                seen.add(key)
                rows.append(row)

    if args.respect_task_splits:
        train_rows, val_rows, test_rows = split_rows_by_declared_split(rows)
    else:
        train_rows, val_rows = split_rows(rows, val_ratio=args.val_ratio, seed=args.seed, split_by_task=args.split_by_task)
        test_rows = []
    write_outputs(
        output_dir=output_dir,
        rows=rows,
        train_rows=train_rows,
        val_rows=val_rows,
        test_rows=test_rows,
        dataset_name=args.dataset_name,
        val_dataset_name=args.val_dataset_name,
        test_dataset_name=args.test_dataset_name,
    )
    summary = build_summary(
        rows,
        train_rows,
        val_rows,
        test_rows,
        skipped=skipped,
        split_by_task=args.split_by_task,
        respect_task_splits=args.respect_task_splits,
        max_history=args.max_history,
        task_map_size=len(task_map),
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def load_task_map(paths: list[Path]) -> dict[str, BrowserTaskSpec]:
    tasks: dict[str, BrowserTaskSpec] = {}
    for path in paths:
        if not path.exists():
            continue
        for item in read_jsonl(path):
            try:
                task = BrowserTaskSpec.from_dict(item)
            except Exception:
                continue
            tasks[task.task_id] = task
    return tasks


def extract_rows(
    rollout: dict[str, Any],
    *,
    input_path: Path,
    repo_root: Path,
    task_map: dict[str, BrowserTaskSpec],
    require_success: bool,
    include_policies: set[str],
    max_history: int,
    absolute_image_paths: bool,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    skipped: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    policy_version = str(rollout.get("policy_version") or "")
    if include_policies and policy_version not in include_policies:
        skipped["policy_filtered"] += 1
        return rows, skipped
    if require_success and not rollout.get("success"):
        skipped["rollout_not_success"] += 1
        return rows, skipped

    task_id = str(rollout.get("task_id") or "")
    task = task_map.get(task_id)
    goal = str(rollout.get("goal") or (task.goal if task else ""))
    max_steps = task.max_steps if task else rollout.get("num_steps")
    viewport = list(task.viewport) if task else list(DEFAULT_VIEWPORT)
    action_space = task.action_space if task else sorted(SUPPORTED_ACTIONS)
    task_split = task.split if task else str(rollout.get("split") or "train")
    rollout_id = str(rollout.get("rollout_id") or "")
    history: list[dict[str, Any]] = []

    for index, step in enumerate(rollout.get("trajectory") or []):
        if not isinstance(step, dict):
            skipped["invalid_step"] += 1
            continue
        if step.get("sft_include") is False:
            skipped["sft_excluded_step"] += 1
            history.append(history_item(step, index=index))
            continue
        row = build_row(
            step=step,
            index=index,
            input_path=input_path,
            repo_root=repo_root,
            task_id=task_id,
            rollout_id=rollout_id,
            goal=goal,
            policy_version=policy_version,
            viewport=viewport,
            max_steps=max_steps,
            action_space=action_space,
            task_split=task_split,
            history=history,
            max_history=max_history,
            absolute_image_paths=absolute_image_paths,
        )
        if row is None:
            skipped["step_not_usable"] += 1
        else:
            rows.append(row)
        history.append(history_item(step, index=index))
    return rows, skipped


def build_row(
    *,
    step: dict[str, Any],
    index: int,
    input_path: Path,
    repo_root: Path,
    task_id: str,
    rollout_id: str,
    goal: str,
    policy_version: str,
    viewport: list[int],
    max_steps: Any,
    action_space: list[str],
    task_split: str,
    history: list[dict[str, Any]],
    max_history: int,
    absolute_image_paths: bool,
) -> dict[str, Any] | None:
    if step.get("exec_status") != "ok":
        return None
    action = step.get("action")
    if not isinstance(action, dict):
        return None
    parsed = parse_action(action)
    if not parsed.ok or parsed.action is None:
        return None
    screenshot = str(step.get("screenshot") or "")
    screenshot_path = resolve_path(screenshot, repo_root)
    if not screenshot_path.exists():
        return None
    image_value = str(screenshot_path) if absolute_image_paths else screenshot
    step_index = int(step.get("t") if step.get("t") is not None else index)
    observation = {
        "task_id": task_id,
        "goal": goal,
        "screenshot": image_value,
        "viewport": viewport,
        "step": step_index,
        "history": list(history),
        "action_space": action_space,
        "max_steps": max_steps,
    }
    prompt = build_prompt(observation, max_history=max_history)
    assistant_action = parsed.action.to_json()
    return {
        "messages": [
            {
                "role": "user",
                "content": "<image>\n" + prompt,
            },
            {
                "role": "assistant",
                "content": json.dumps(assistant_action, ensure_ascii=False, separators=(",", ":")),
            },
        ],
        "images": [image_value],
        "task_id": task_id,
        "rollout_id": rollout_id,
        "step": step_index,
        "history_len": min(len(history), max_history),
        "source_rollouts": str(input_path),
        "policy_version": policy_version,
        "task_split": task_split,
        "prompt_style": "history_aware_full",
        "reward_step": step.get("reward_step"),
        "target_action": assistant_action,
    }


def history_item(step: dict[str, Any], *, index: int) -> dict[str, Any]:
    step_index = int(step.get("t") if step.get("t") is not None else index)
    return {
        "step": step_index + 1,
        "action": step.get("action"),
        "exec_status": step.get("exec_status"),
        "exec_error": step.get("exec_error"),
        "verifier": step.get("verifier"),
    }


def resolve_path(path: str, repo_root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return repo_root / candidate


def row_key(row: dict[str, Any]) -> str:
    payload = {
        "image": row.get("images"),
        "user": (row.get("messages") or [{}])[0].get("content"),
        "assistant": (row.get("messages") or [{}, {}])[1].get("content"),
        "task_id": row.get("task_id"),
        "step": row.get("step"),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def split_rows(
    rows: list[dict[str, Any]],
    *,
    val_ratio: float,
    seed: int,
    split_by_task: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(rows, key=stable_sort_key)
    rng = random.Random(seed)
    if not split_by_task:
        shuffled = list(ordered)
        rng.shuffle(shuffled)
        val_count = val_size(len(shuffled), val_ratio)
        return shuffled[val_count:], shuffled[:val_count]

    task_ids = sorted({str(row.get("task_id") or "") for row in ordered})
    rng.shuffle(task_ids)
    val_task_count = val_size(len(task_ids), val_ratio)
    val_tasks = set(task_ids[:val_task_count])
    train_rows = [row for row in ordered if str(row.get("task_id") or "") not in val_tasks]
    val_rows = [row for row in ordered if str(row.get("task_id") or "") in val_tasks]
    if not train_rows and val_rows:
        train_rows, val_rows = val_rows, []
    return train_rows, val_rows


def split_rows_by_declared_split(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(rows, key=stable_sort_key)
    train_rows = [row for row in ordered if str(row.get("task_split") or "train") == "train"]
    val_rows = [row for row in ordered if str(row.get("task_split") or "train") == "val"]
    test_rows = [row for row in ordered if str(row.get("task_split") or "train") == "test"]
    other_rows = [row for row in ordered if str(row.get("task_split") or "train") not in {"train", "val", "test"}]
    train_rows.extend(other_rows)
    return train_rows, val_rows, test_rows


def val_size(total: int, ratio: float) -> int:
    if total <= 1 or ratio <= 0:
        return 0
    return min(max(1, int(round(total * ratio))), total - 1)


def stable_sort_key(row: dict[str, Any]) -> str:
    return hashlib.sha1(
        json.dumps(
            [row.get("task_id"), row.get("rollout_id"), row.get("step"), row.get("policy_version")],
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def write_outputs(
    *,
    output_dir: Path,
    rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    dataset_name: str,
    val_dataset_name: str,
    test_dataset_name: str,
) -> None:
    write_json(output_dir / "messages.json", rows)
    write_jsonl(output_dir / "messages.jsonl", rows)
    train_file = f"{dataset_name}.json"
    val_file = f"{val_dataset_name}.json"
    test_file = f"{test_dataset_name}.json"
    write_json(output_dir / train_file, train_rows)
    write_json(output_dir / val_file, val_rows)
    write_json(output_dir / test_file, test_rows)
    write_jsonl(output_dir / f"{dataset_name}.jsonl", train_rows)
    write_jsonl(output_dir / f"{val_dataset_name}.jsonl", val_rows)
    write_jsonl(output_dir / f"{test_dataset_name}.jsonl", test_rows)
    write_dataset_info(output_dir / "dataset_info.json", dataset_name, train_file, val_dataset_name, val_file, test_dataset_name, test_file)


def build_summary(
    rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    *,
    skipped: Counter[str],
    split_by_task: bool,
    respect_task_splits: bool,
    max_history: int,
    task_map_size: int,
) -> dict[str, Any]:
    policy_counter = Counter(str(row.get("policy_version") or "unknown") for row in rows)
    task_counter = Counter(str(row.get("task_id") or "unknown") for row in rows)
    source_counter = Counter(str(row.get("source_rollouts") or "unknown") for row in rows)
    action_counter = Counter(str((row.get("target_action") or {}).get("action") or "unknown") for row in rows)
    history_counter = Counter(str(row.get("history_len")) for row in rows)
    train_tasks = {str(row.get("task_id") or "") for row in train_rows}
    val_tasks = {str(row.get("task_id") or "") for row in val_rows}
    test_tasks = {str(row.get("task_id") or "") for row in test_rows}
    rows_by_task: dict[str, int] = defaultdict(int)
    for row in rows:
        rows_by_task[str(row.get("task_id") or "unknown")] += 1
    return {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rows": len(rows),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "test_rows": len(test_rows),
        "unique_tasks": len(task_counter),
        "train_tasks": len(train_tasks),
        "val_tasks": len(val_tasks),
        "test_tasks": len(test_tasks),
        "task_overlap": sorted((train_tasks & val_tasks) | (train_tasks & test_tasks) | (val_tasks & test_tasks)),
        "split_by_task": split_by_task,
        "respect_task_splits": respect_task_splits,
        "max_history": max_history,
        "task_map_size": task_map_size,
        "policy_versions": dict(policy_counter.most_common()),
        "action_distribution": dict(action_counter.most_common()),
        "history_len_distribution": dict(history_counter.most_common()),
        "rows_by_task_top20": dict(Counter(rows_by_task).most_common(20)),
        "sources": dict(source_counter.most_common()),
        "skipped": dict(skipped.most_common()),
        "format": "messages_with_images_history_aware_action_json",
    }


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_dataset_info(
    path: Path,
    dataset_name: str,
    train_file: str,
    val_dataset_name: str,
    val_file: str,
    test_dataset_name: str,
    test_file: str,
) -> None:
    def entry(file_name: str) -> dict[str, Any]:
        return {
            "file_name": file_name,
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"},
        }

    payload = {
        dataset_name: entry(train_file),
        val_dataset_name: entry(val_file),
        test_dataset_name: entry(test_file),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
