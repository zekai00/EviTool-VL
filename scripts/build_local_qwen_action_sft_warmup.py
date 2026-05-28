#!/usr/bin/env python3
"""Build local Qwen action-JSON SFT warmup data from validated rollouts.

SFT means supervised fine-tuning. This script extracts successful GUI action
steps from rollout records so a local Qwen2.5-VL/Qwen3-VL checkpoint can first
learn to emit valid direct GUI action JSON before on-policy RL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.browser_rl.actions import parse_action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, help="rollouts.jsonl path; can be repeated.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--require-success", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-policies", default="", help="Comma-separated policy versions to include; empty means all.")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--absolute-image-paths", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dedupe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dataset-name", default="local_qwen_action_sft_train")
    parser.add_argument("--val-dataset-name", default="local_qwen_action_sft_val")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    include_policies = {item.strip() for item in args.include_policies.split(",") if item.strip()}
    rows = []
    skipped: Counter[str] = Counter()
    seen: set[str] = set()

    for input_path in [Path(path) for path in args.input]:
        for rollout in read_jsonl(input_path):
            extracted, local_skipped = extract_rows(
                rollout,
                input_path=input_path,
                repo_root=repo_root,
                require_success=args.require_success,
                include_policies=include_policies,
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

    rng = random.Random(args.seed)
    rows.sort(key=lambda item: stable_sort_key(item))
    rng.shuffle(rows)
    val_count = int(round(len(rows) * args.val_ratio)) if rows else 0
    val_count = min(max(0, val_count), max(0, len(rows) - 1)) if len(rows) > 1 else 0
    val_rows = rows[:val_count]
    train_rows = rows[val_count:]

    write_json(output_dir / "messages.json", rows)
    write_jsonl(output_dir / "messages.jsonl", rows)
    train_file = f"{args.dataset_name}.json"
    val_file = f"{args.val_dataset_name}.json"
    write_json(output_dir / train_file, train_rows)
    write_json(output_dir / val_file, val_rows)
    write_jsonl(output_dir / f"{args.dataset_name}.jsonl", train_rows)
    write_jsonl(output_dir / f"{args.val_dataset_name}.jsonl", val_rows)
    write_dataset_info(output_dir / "dataset_info.json", args.dataset_name, train_file, args.val_dataset_name, val_file)
    summary = build_summary(rows, train_rows, val_rows, skipped)
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


def extract_rows(
    rollout: dict[str, Any],
    *,
    input_path: Path,
    repo_root: Path,
    require_success: bool,
    include_policies: set[str],
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
    goal = str(rollout.get("goal") or "")
    task_id = str(rollout.get("task_id") or "")
    rollout_id = str(rollout.get("rollout_id") or "")
    for step in rollout.get("trajectory") or []:
        if not isinstance(step, dict):
            skipped["invalid_step"] += 1
            continue
        if step.get("exec_status") != "ok":
            skipped["exec_not_ok"] += 1
            continue
        action = step.get("action")
        if not isinstance(action, dict):
            skipped["missing_action"] += 1
            continue
        parsed = parse_action(action)
        if not parsed.ok or parsed.action is None:
            skipped["action_parse_failed"] += 1
            continue
        screenshot = str(step.get("screenshot") or "")
        screenshot_path = resolve_path(screenshot, repo_root)
        if not screenshot_path.exists():
            skipped["screenshot_missing"] += 1
            continue
        image_value = str(screenshot_path) if absolute_image_paths else screenshot
        assistant_action = parsed.action.to_json()
        rows.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "<image>\n任务：" + goal + "\n请输出下一步 GUI action JSON。",
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(assistant_action, ensure_ascii=False, separators=(",", ":")),
                    },
                ],
                "images": [image_value],
                "task_id": task_id,
                "rollout_id": rollout_id,
                "step": step.get("t"),
                "source_rollouts": str(input_path),
                "policy_version": policy_version,
                "success": bool(rollout.get("success")),
                "reward_step": step.get("reward_step"),
            }
        )
    return rows, skipped


def resolve_path(path: str, repo_root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return repo_root / candidate


def row_key(row: dict[str, Any]) -> str:
    payload = {
        "image": row.get("images"),
        "assistant": (row.get("messages") or [{}, {}])[1].get("content"),
        "task_id": row.get("task_id"),
        "step": row.get("step"),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def stable_sort_key(row: dict[str, Any]) -> str:
    return hashlib.sha1(
        json.dumps(
            [row.get("task_id"), row.get("rollout_id"), row.get("step"), row.get("policy_version")],
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def build_summary(rows: list[dict[str, Any]], train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]], skipped: Counter[str]) -> dict[str, Any]:
    policy_counter = Counter(str(row.get("policy_version") or "unknown") for row in rows)
    task_counter = Counter(str(row.get("task_id") or "unknown") for row in rows)
    source_counter = Counter(str(row.get("source_rollouts") or "unknown") for row in rows)
    return {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rows": len(rows),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "unique_tasks": len(task_counter),
        "policy_versions": dict(policy_counter.most_common()),
        "sources": dict(source_counter.most_common()),
        "skipped": dict(skipped.most_common()),
        "format": "messages_with_images_action_json",
    }


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_dataset_info(path: Path, dataset_name: str, train_file: str, val_dataset_name: str, val_file: str) -> None:
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
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
