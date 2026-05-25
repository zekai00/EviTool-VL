#!/usr/bin/env python3
"""Build fixed GUI candidate-selection data with overlay images."""

from __future__ import annotations

import argparse
import copy
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
    parser.add_argument(
        "--group-split-key",
        default=None,
        help=(
            "Optional key used to split train/val by group instead of by row. "
            "Use `image` for datasets with many targets per screenshot, so the "
            "same screenshot cannot appear in both train and val."
        ),
    )
    parser.add_argument("--omniparser-root", default="third_party/OmniParser")
    parser.add_argument("--omniparser-weights-dir", default="third_party/OmniParser/weights")
    return parser.parse_args()


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def split_records(
    records: list[dict[str, Any]],
    *,
    val_ratio: float,
    seed: int,
    group_key: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split records into train/val, optionally keeping groups intact.

    Candidate-RL datasets often contain several target elements for one
    screenshot.  If those rows are split independently, validation can include
    the exact screenshot seen during training with only a different instruction.
    Grouped splitting avoids that leakage while preserving the old row-level
    behavior when `group_key` is not provided.
    """
    rng = random.Random(seed)
    if not group_key:
        shuffled = list(records)
        rng.shuffle(shuffled)
        val_count = int(round(len(shuffled) * val_ratio))
        return shuffled[val_count:], shuffled[:val_count]

    groups: dict[str, list[dict[str, Any]]] = {}
    missing_group_records: list[dict[str, Any]] = []
    for record in records:
        value = record.get(group_key)
        if value is None:
            missing_group_records.append(record)
            continue
        groups.setdefault(str(value), []).append(record)

    group_items = list(groups.items())
    rng.shuffle(group_items)
    target_val_count = int(round(len(records) * val_ratio))
    val_records: list[dict[str, Any]] = []
    train_records: list[dict[str, Any]] = []
    for _, group_records in group_items:
        # Greedily fill validation by whole groups.  This may miss the requested
        # ratio by a small amount, but it keeps screenshot-level independence.
        if len(val_records) < target_val_count:
            val_records.extend(group_records)
        else:
            train_records.extend(group_records)
    train_records.extend(missing_group_records)
    return train_records, val_records


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    overlay_dir = output_dir / "overlays"
    rows = load_gui_rows(args.data, args.limit)
    records: list[dict[str, Any]] = []
    candidate_cache: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]] = {}
    candidate_cache_hits = 0

    for idx, row in enumerate(rows, start=1):
        image_path = Path(args.image_root) / str(row["image"])
        image_cache_key = str(image_path.resolve())
        cached = candidate_cache.get(image_cache_key)
        if cached is None:
            # OmniParser candidates depend only on the screenshot, not on the
            # natural-language instruction.  OS-Atlas often has several target
            # elements per screenshot, so caching avoids repeatedly running the
            # detector on identical pixels.
            candidates, meta = generate_omniparser_candidates(
                image_path,
                max_candidates=args.max_candidates,
                omniparser_root=args.omniparser_root,
                omniparser_weights_dir=args.omniparser_weights_dir,
            )
            candidate_cache[image_cache_key] = (copy.deepcopy(candidates), copy.deepcopy(meta))
        else:
            candidate_cache_hits += 1
            candidates, meta = copy.deepcopy(cached)
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

    train_records, val_records = split_records(
        records,
        val_ratio=args.val_ratio,
        seed=args.seed,
        group_key=args.group_split_key,
    )

    write_jsonl(output_dir / "all.jsonl", records)
    write_jsonl(output_dir / "train.jsonl", train_records)
    write_jsonl(output_dir / "val.jsonl", val_records)
    summary = {
        "data": args.data,
        "image_root": args.image_root,
        "max_candidates": args.max_candidates,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "group_split_key": args.group_split_key,
        "unique_candidate_images": len(candidate_cache),
        "candidate_cache_hits": candidate_cache_hits,
        "all": summarize_candidate_records(records),
        "train": summarize_candidate_records(train_records),
        "val": summarize_candidate_records(val_records),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
