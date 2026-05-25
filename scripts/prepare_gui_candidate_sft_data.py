#!/usr/bin/env python3
"""Prepare candidate-selection SFT data from fixed GUI candidate-RL rows.

GRPO needs within-prompt exploration to learn.  In the previous run the model
already produced valid JSON, but mostly defaulted to early candidates.  This SFT
warmup teaches the exact candidate-id schema used by GRPO before RL starts:

    user: overlay image + instruction + candidate list
    assistant: {"candidate_id": "cXX"}

The target candidate is the oracle candidate computed from the fixed candidate
set, so no free-form bbox regression is introduced.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.gui_candidate_env import build_candidate_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--val-data", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", default="gui_candidate_sft_train")
    parser.add_argument("--val-dataset-name", default="gui_candidate_sft_val")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--require-oracle-hit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-overlay", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def read_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def keep_row(row: dict[str, Any], *, require_oracle_hit: bool, require_overlay: bool) -> bool:
    if require_overlay and not row.get("overlay_image"):
        return False
    if not row.get("oracle_candidate_id"):
        return False
    if require_oracle_hit and not bool((row.get("oracle_metrics") or {}).get("hit")):
        return False
    return True


def to_lf_item(row: dict[str, Any]) -> dict[str, Any]:
    prompt = (
        "<image>GUI candidate selection task. Select the candidate_id that best matches the instruction. "
        "Return only JSON in this exact schema: {\"candidate_id\":\"c00\"}. Do not explain.\n\n"
        f"{build_candidate_prompt(row, row.get('candidates') or [])}"
    )
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": compact_json({"candidate_id": row["oracle_candidate_id"]})},
        ],
        "images": [str((PROJECT_ROOT / str(row.get("overlay_image"))).resolve())],
        "meta": {
            "id": row.get("id"),
            "source": "gui_candidate_sft",
            "oracle_candidate_id": row.get("oracle_candidate_id"),
            "oracle_metrics": row.get("oracle_metrics"),
            "candidate_count": row.get("candidate_count"),
        },
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_items(path: str | Path, limit: int | None, args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = read_jsonl(path, limit)
    kept = [row for row in rows if keep_row(row, require_oracle_hit=args.require_oracle_hit, require_overlay=args.require_overlay)]
    return rows, [to_lf_item(row) for row in kept]


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)

    train_rows, train_items = build_items(args.train_data, args.limit, args)
    val_rows: list[dict[str, Any]] = []
    val_items: list[dict[str, Any]] = []
    if args.val_data:
        val_rows, val_items = build_items(args.val_data, args.val_limit, args)

    train_file = f"{args.dataset_name}.json"
    write_json(output_dir / train_file, train_items)
    dataset_info: dict[str, Any] = {
        args.dataset_name: {
            "file_name": train_file,
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"},
        }
    }
    if val_items:
        val_file = f"{args.val_dataset_name}.json"
        write_json(output_dir / val_file, val_items)
        dataset_info[args.val_dataset_name] = {
            "file_name": val_file,
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"},
        }
    write_json(output_dir / "dataset_info.json", dataset_info)

    summary = {
        "train_input": args.train_data,
        "val_input": args.val_data,
        "output_dir": str(output_dir),
        "dataset_name": args.dataset_name,
        "val_dataset_name": args.val_dataset_name if val_items else None,
        "require_oracle_hit": args.require_oracle_hit,
        "require_overlay": args.require_overlay,
        "train_rows": len(train_rows),
        "train_items": len(train_items),
        "val_rows": len(val_rows),
        "val_items": len(val_items),
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
