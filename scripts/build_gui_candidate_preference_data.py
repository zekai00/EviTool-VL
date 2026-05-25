#!/usr/bin/env python3
"""Build candidate-constrained preference data for GUI RL.

The output keeps the original candidate list, but adds per-candidate reward_v3
and a compact `cc_action_ids` set containing the oracle plus hard negatives.
This is the data layer that makes CC-GRPO different from free-form text GRPO:
each optimization group samples from real candidate ids, not arbitrary strings.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.candidate_constrained import (
    action_reward_std,
    build_action_ids,
    candidate_reward_table,
    rank_bucket,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--val-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-actions", type=int, default=12)
    parser.add_argument("--max-hard-negatives", type=int, default=8)
    parser.add_argument("--min-action-reward-std", type=float, default=1e-6)
    parser.add_argument("--require-oracle-hit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-overlay", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
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
    if args.require_overlay and not row.get("overlay_image"):
        return False
    if not row.get("candidates"):
        return False
    if args.require_oracle_hit and not bool((row.get("oracle_metrics") or {}).get("hit")):
        return False
    return True


def enrich_row(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
    """Attach reward table, action subset, oracle rank bucket, and std stats."""
    if not keep_row(row, args):
        return None
    candidates = row.get("candidates") or []
    rewards = candidate_reward_table(row, candidates)
    action_ids = build_action_ids(
        row,
        candidates,
        max_actions=args.max_actions,
        max_hard=args.max_hard_negatives,
        seed=args.seed,
    )
    if len(action_ids) < 2:
        return None
    std = action_reward_std(action_ids, rewards)
    if std < args.min_action_reward_std:
        return None

    oracle_id = row.get("oracle_candidate_id")
    oracle_rank = None
    for candidate in candidates:
        if candidate.get("candidate_id") == oracle_id:
            oracle_rank = int(candidate.get("rank") or 0)
            break

    enriched = dict(row)
    enriched["candidate_rewards_v3"] = rewards
    enriched["cc_action_ids"] = action_ids
    enriched["cc_hard_negative_ids"] = [candidate_id for candidate_id in action_ids if candidate_id != oracle_id]
    enriched["cc_reward_std"] = round(std, 6)
    enriched["oracle_rank"] = oracle_rank
    enriched["oracle_rank_bucket"] = rank_bucket(oracle_rank)
    return enriched


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "avg_actions": None,
            "avg_reward_std": None,
            "zero_reward_std_rate": None,
            "rank_buckets": {},
        }
    stds = [float(row.get("cc_reward_std") or 0.0) for row in rows]
    buckets: dict[str, int] = {}
    for row in rows:
        bucket = str(row.get("oracle_rank_bucket") or "unknown")
        buckets[bucket] = buckets.get(bucket, 0) + 1
    return {
        "count": len(rows),
        "avg_actions": statistics.mean([len(row.get("cc_action_ids") or []) for row in rows]),
        "avg_reward_std": statistics.mean(stds),
        "min_reward_std": min(stds),
        "max_reward_std": max(stds),
        "zero_reward_std_rate": sum(float(std <= 1e-9) for std in stds) / len(stds),
        "rank_buckets": buckets,
    }


def build_split(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        item = enrich_row(row, args)
        if item is not None:
            enriched.append(item)
    return enriched


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    train_rows = read_jsonl(args.train_data)
    val_rows = read_jsonl(args.val_data)
    train = build_split(train_rows, args)
    val = build_split(val_rows, args)

    write_jsonl(output_dir / "train.jsonl", train)
    write_jsonl(output_dir / "val.jsonl", val)
    summary = {
        "train_input": args.train_data,
        "val_input": args.val_data,
        "output_dir": str(output_dir),
        "max_actions": args.max_actions,
        "max_hard_negatives": args.max_hard_negatives,
        "min_action_reward_std": args.min_action_reward_std,
        "require_oracle_hit": args.require_oracle_hit,
        "require_overlay": args.require_overlay,
        "input": {
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
        },
        "output": {
            "train": summarize(train),
            "val": summarize(val),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
