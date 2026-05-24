#!/usr/bin/env python3
"""Convert EviTool trace JSONL into LlamaFactory multimodal SFT data.

The trace builder stores explicit user/action/tool/final turns. LlamaFactory's
ShareGPT multimodal format only needs user/assistant turns, so tool
observations are serialized as user feedback turns.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/root/models/datasets/evitool_traces_1k/train_traces_1000.jsonl")
    parser.add_argument("--output-dir", default="/root/models/datasets/evitool_sft_lf_1k")
    parser.add_argument("--dataset-name", default="evitool_sft_1k")
    parser.add_argument("--direct-retention-ratio", type=float, default=0.15)
    parser.add_argument("--include-moderate-ai2d", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-observation-chars", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dump_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def quality(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("quality") or {}


def is_strong_trace(row: dict[str, Any], include_moderate_ai2d: bool) -> bool:
    task = row.get("task_type")
    q = quality(row)
    if task == "gui_grounding":
        return not q.get("oracle_gt_injected", False) and bool(q.get("candidate_recall_at_30", False))
    if task in {"doc_qa", "chart_qa"}:
        return bool(q.get("answer_evidence_position_found", False) or q.get("answer_in_ocr", False))
    if task == "science_diagram_qa":
        if q.get("answer_option_evidence_position_found", False):
            return True
        return include_moderate_ai2d and bool(q.get("diagram_structure_evidence", False))
    return bool(q.get("strong_evidence", False))


def compact_json(obj: Any, max_chars: int) -> str:
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except json.JSONDecodeError:
            pass
    text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 32] + "...<truncated>"


def image_path_for_row(row: dict[str, Any], input_path: Path) -> str:
    image = str(row["image"])
    if image.startswith("/"):
        return image
    dataset_root = input_path.parent
    return str((dataset_root / image).resolve())


def trace_to_lf_item(row: dict[str, Any], input_path: Path, max_observation_chars: int) -> dict[str, Any]:
    source_messages = row.get("messages") or []
    if not source_messages:
        raise ValueError(f"row {row.get('id')} has no messages")

    first_user = source_messages[0]
    messages: list[dict[str, str]] = [
        {
            "role": "user",
            "content": "<image>" + str(first_user.get("content", "")),
        }
    ]
    for msg in source_messages[1:]:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "assistant":
            messages.append({"role": "assistant", "content": str(content)})
        elif role == "tool":
            messages.append(
                {
                    "role": "user",
                    "content": "Tool observation:\n" + compact_json(content, max_observation_chars),
                }
            )
        elif role == "user":
            messages.append({"role": "user", "content": str(content)})

    return {
        "messages": messages,
        "images": [image_path_for_row(row, input_path)],
        "meta": {
            "id": row.get("id"),
            "task_type": row.get("task_type"),
            "source": "evitool_trace",
            "strong_evidence": bool(quality(row).get("strong_evidence", False)),
        },
    }


def direct_retention_item(row: dict[str, Any], input_path: Path) -> dict[str, Any]:
    answer = row.get("answer")
    if row.get("task_type") == "science_diagram_qa":
        final = row.get("final") or {}
        answer = final.get("answer", answer)
    elif isinstance(row.get("answers"), list) and row["answers"]:
        answer = row["answers"][0]
    return {
        "messages": [
            {
                "role": "user",
                "content": "<image>Answer the question based on the image. Return only the final answer.\n\n"
                f"Question: {row.get('question')}",
            },
            {"role": "assistant", "content": str(answer)},
        ],
        "images": [image_path_for_row(row, input_path)],
        "meta": {
            "id": row.get("id"),
            "task_type": row.get("task_type"),
            "source": "direct_retention",
        },
    }


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(input_path)
    selected = [row for row in rows if is_strong_trace(row, args.include_moderate_ai2d)]
    trace_items = [trace_to_lf_item(row, input_path, args.max_observation_chars) for row in selected]

    text_rows = [row for row in selected if row.get("task_type") != "gui_grounding"]
    retention_count = int(round(len(trace_items) * args.direct_retention_ratio))
    retention_source = random.sample(text_rows, min(retention_count, len(text_rows))) if retention_count > 0 else []
    retention_items = [direct_retention_item(row, input_path) for row in retention_source]

    items = trace_items + retention_items
    random.shuffle(items)

    data_file = output_dir / f"{args.dataset_name}.json"
    info_file = output_dir / "dataset_info.json"
    summary_file = output_dir / "summary.json"

    dump_json(data_file, items)
    dump_json(
        info_file,
        {
            args.dataset_name: {
                "file_name": data_file.name,
                "formatting": "sharegpt",
                "columns": {"messages": "messages", "images": "images"},
                "tags": {
                    "role_tag": "role",
                    "content_tag": "content",
                    "user_tag": "user",
                    "assistant_tag": "assistant",
                },
            }
        },
    )
    dump_json(
        summary_file,
        {
            "input": str(input_path),
            "dataset_name": args.dataset_name,
            "output": str(data_file),
            "count": len(items),
            "trace_count": len(trace_items),
            "direct_retention_count": len(retention_items),
            "source_task_counts": dict(Counter(row.get("task_type") for row in selected)),
            "item_source_counts": dict(Counter(item["meta"]["source"] for item in items)),
        },
    )
    print(json.dumps(json.loads(summary_file.read_text(encoding="utf-8")), ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
