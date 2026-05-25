#!/usr/bin/env python3
"""Filter GUI candidate-RL rows into a higher-signal training split.

The first OS-Atlas GRPO run used all rows whose candidate list was non-empty.
Many of those rows are not useful for candidate-selection RL: the correct UI
target is not covered by any generated candidate, so every sampled candidate
gets nearly the same low reward.  GRPO then sees zero within-group reward
variance and produces no gradient.

This script keeps validation unchanged by default, but filters train rows to
examples whose oracle candidate has enough geometric overlap or pointing signal.
That gives the policy a learnable target before we evaluate on the harder full
validation set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Directory containing train.jsonl/val.jsonl/all.jsonl.")
    parser.add_argument("--output-dir", required=True, help="Directory for filtered train plus copied val/all files.")
    parser.add_argument("--min-oracle-iou", type=float, default=0.05)
    parser.add_argument("--require-hit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-iou50", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-overlay", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-candidates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--filter-val",
        action="store_true",
        help="Also filter val.jsonl. Default keeps val full for honest evaluation.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def keep_row(row: dict[str, Any], args: argparse.Namespace) -> bool:
    metrics = row.get("oracle_metrics") or {}
    if args.require_overlay and not row.get("overlay_image"):
        return False
    if args.require_candidates and int(row.get("candidate_count") or 0) <= 0:
        return False
    if args.require_iou50 and not bool(metrics.get("iou_50")):
        return False
    if args.require_hit and not bool(metrics.get("hit")):
        return False
    if float(metrics.get("iou") or 0.0) < args.min_oracle_iou:
        return False
    return True


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    metrics = [row.get("oracle_metrics") or {} for row in rows]
    return {
        "count": len(rows),
        "avg_candidates": mean([float(row.get("candidate_count") or 0) for row in rows]),
        "oracle_hit_rate": mean([float(bool(metric.get("hit"))) for metric in metrics]),
        "oracle_pointing_rate": mean([float(bool(metric.get("pointing"))) for metric in metrics]),
        "oracle_iou50_rate": mean([float(bool(metric.get("iou_50"))) for metric in metrics]),
        "avg_oracle_iou": mean([float(metric.get("iou") or 0.0) for metric in metrics]),
    }


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    train = read_jsonl(input_dir / "train.jsonl")
    val = read_jsonl(input_dir / "val.jsonl")
    all_rows = read_jsonl(input_dir / "all.jsonl")

    filtered_train = [row for row in train if keep_row(row, args)]
    filtered_val = [row for row in val if keep_row(row, args)] if args.filter_val else val

    write_jsonl(output_dir / "train.jsonl", filtered_train)
    write_jsonl(output_dir / "val.jsonl", filtered_val)
    write_jsonl(output_dir / "all.jsonl", filtered_train + filtered_val)

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "criteria": {
            "min_oracle_iou": args.min_oracle_iou,
            "require_hit": args.require_hit,
            "require_iou50": args.require_iou50,
            "require_overlay": args.require_overlay,
            "require_candidates": args.require_candidates,
            "filter_val": args.filter_val,
        },
        "input": {
            "all": summarize(all_rows),
            "train": summarize(train),
            "val": summarize(val),
        },
        "output": {
            "all": summarize(filtered_train + filtered_val),
            "train": summarize(filtered_train),
            "val": summarize(filtered_val),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
