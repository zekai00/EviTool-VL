#!/usr/bin/env python3
"""Build fixed GUI candidate-selection data with overlay images."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.gui_candidate_env import (
    draw_candidate_overlay,
    generate_omniparser_candidates,
    load_gui_rows,
    oracle_candidate,
    summarize_candidate_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/root/models/datasets/evitool_traces_1k/train_traces_1000.jsonl")
    parser.add_argument("--image-root", default="/root/models/datasets/evitool_traces_1k")
    parser.add_argument("--output-dir", default="outputs/gui_candidate_rl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=30)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--omniparser-root", default="third_party/OmniParser")
    parser.add_argument("--omniparser-weights-dir", default="third_party/OmniParser/weights")
    return parser.parse_args()


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    overlay_dir = output_dir / "overlays"
    rows = load_gui_rows(args.data, args.limit)
    records: list[dict[str, Any]] = []

    for idx, row in enumerate(rows, start=1):
        image_path = Path(args.image_root) / str(row["image"])
        candidates, meta = generate_omniparser_candidates(
            image_path,
            max_candidates=args.max_candidates,
            omniparser_root=args.omniparser_root,
            omniparser_weights_dir=args.omniparser_weights_dir,
        )
        oracle, oracle_metrics = oracle_candidate(row, candidates)
        raw_id = str(row.get("id") or f"{idx:04d}")
        stem = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in raw_id)
        overlay_path = overlay_dir / f"{stem}.png"
        if candidates:
            draw_candidate_overlay(image_path, candidates, overlay_path, max_candidates=args.max_candidates)
        record = {
            "id": row.get("id"),
            "task_type": row.get("task_type"),
            "image": row.get("image"),
            "image_path": str(image_path),
            "overlay_image": str(overlay_path) if candidates else None,
            "question": row.get("question"),
            "instruction": (row.get("meta") or {}).get("instruction") or row.get("question"),
            "answer_bbox": row.get("answer_bbox"),
            "candidate_count": len(candidates),
            "candidate_meta": meta,
            "oracle_candidate_id": oracle.get("candidate_id") if oracle else None,
            "oracle_bbox": oracle.get("bbox") if oracle else None,
            "oracle_rank": oracle.get("rank") if oracle else None,
            "oracle_metrics": oracle_metrics,
            "candidates": candidates,
        }
        records.append(record)
        if idx % 25 == 0 or idx == len(rows):
            print(f"[{idx}/{len(rows)}] built", flush=True)

    rng = random.Random(args.seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    val_count = int(round(len(shuffled) * args.val_ratio))
    val_records = shuffled[:val_count]
    train_records = shuffled[val_count:]

    write_jsonl(output_dir / "all.jsonl", records)
    write_jsonl(output_dir / "train.jsonl", train_records)
    write_jsonl(output_dir / "val.jsonl", val_records)
    summary = {
        "data": args.data,
        "image_root": args.image_root,
        "max_candidates": args.max_candidates,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "all": summarize_candidate_records(records),
        "train": summarize_candidate_records(train_records),
        "val": summarize_candidate_records(val_records),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
