#!/usr/bin/env python3
"""Run CUA-Gym mock_web setup/reward scripts in a weak local sandbox.

Sandbox means container-internal weak isolation here: copied task files, a
minimal subprocess environment, timeout/resource limits, and no API keys.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.browser_rl.cua_gym_sandbox import CuaGymMockWebSandbox, MockWebStateServer, write_sandbox_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--tasks-jsonl",
        default="outputs/cua_gym_mock_web_inspection_20260527_2125/mock_web_tasks.jsonl",
    )
    parser.add_argument(
        "--bundle-root",
        default="outputs/cua_gym_mock_web_inspection_20260527_2125/bundles",
    )
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--timeout-sec", type=int, default=45)
    parser.add_argument("--cpu-seconds", type=int, default=20)
    parser.add_argument("--memory-mb", type=int, default=2048)
    parser.add_argument("--file-mb", type=int, default=128)
    parser.add_argument("--host", default="127.0.0.1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    rows = load_rows(Path(args.tasks_jsonl))
    selected = select_rows(rows, args.task_id, args.limit)
    sandbox = CuaGymMockWebSandbox(
        output_dir,
        timeout_sec=args.timeout_sec,
        cpu_seconds=args.cpu_seconds,
        memory_mb=args.memory_mb,
        file_mb=args.file_mb,
    )
    results = []
    with MockWebStateServer(host=args.host) as server:
        for row in selected:
            task_id = str(row["task_id"])
            bundle_dir = Path(args.bundle_root) / task_id
            if not bundle_dir.exists():
                raise FileNotFoundError(f"bundle dir not found: {bundle_dir}")
            results.append(sandbox.run_task(row, bundle_dir, server))
    summary = write_sandbox_summary(output_dir, results)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def select_rows(rows: list[dict[str, Any]], task_ids: list[str], limit: int) -> list[dict[str, Any]]:
    if task_ids:
        wanted = set(task_ids)
        selected = [row for row in rows if str(row.get("task_id")) in wanted]
        missing = wanted - {str(row.get("task_id")) for row in selected}
        if missing:
            raise ValueError(f"task ids not found in tasks jsonl: {sorted(missing)}")
        return selected
    return rows[:limit]


if __name__ == "__main__":
    main()
