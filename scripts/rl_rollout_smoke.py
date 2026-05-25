#!/usr/bin/env python3
"""Dry-run the EviTool RL rollout environment without loading a VLM."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.rollout_env import EviToolRolloutEnvironment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/root/models/datasets/evitool_eval_medium/eval_medium_600.jsonl")
    parser.add_argument("--image-root", default="/root/models/datasets/evitool_eval_medium")
    parser.add_argument("--output", default="outputs/rl_smoke/rollout_dryrun.jsonl")
    parser.add_argument("--limit", type=int, default=4)
    return parser.parse_args()


def load_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    rows = []
    seen_tasks = set()
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        task = row.get("task_type")
        if task in seen_tasks and len(rows) < limit:
            continue
        rows.append(row)
        seen_tasks.add(task)
        if len(rows) >= limit:
            break
    return rows


def scripted_actions(row: dict[str, Any]) -> list[dict[str, Any]]:
    task = row.get("task_type")
    if task == "gui_grounding":
        instruction = str((row.get("meta") or {}).get("instruction") or row.get("question") or "")
        return [
            {"tool": "detect", "args": {"mode": "ui", "query": instruction, "max_results": 10, "min_area": 20}},
            {"tool": "click", "args": {"bbox": row.get("answer_bbox"), "label": instruction}},
        ]
    if task in {"doc_qa", "chart_qa"}:
        return [{"tool": "ocr", "args": {"engine": "easyocr", "languages": ["en"], "max_regions": 20}}]
    return [{"tool": "detect", "args": {"mode": "diagram", "max_results": 10, "min_area": 20}}]


def final_answer(row: dict[str, Any]) -> Any:
    return row.get("answer_bbox") if row.get("task_type") == "gui_grounding" else row.get("answer")


def main() -> int:
    args = parse_args()
    rows = load_rows(Path(args.data), args.limit)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    env = EviToolRolloutEnvironment(args.image_root)
    results = []
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            env.reset(row=row)
            result = env.run_scripted(scripted_actions(row), final_answer(row))
            compact = {
                "id": row.get("id"),
                "task_type": row.get("task_type"),
                "tool_call_count": result["tool_call_count"],
                "evidence_closed": result["evidence_closed"],
                "reward": result["reward"],
            }
            f.write(json.dumps(compact, ensure_ascii=False) + "\n")
            results.append(compact)
            print(json.dumps(compact, ensure_ascii=False), flush=True)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
