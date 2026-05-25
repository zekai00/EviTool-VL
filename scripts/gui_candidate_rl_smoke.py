#!/usr/bin/env python3
"""Smoke-test GUI candidate-selection RL data, actions, and rewards."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.gui_candidate_env import (
    build_candidate_prompt,
    generate_omniparser_candidates,
    load_gui_rows,
    oracle_candidate,
    policy_action,
    score_candidate_action,
)

POLICIES = ("top1", "oracle", "random", "invalid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/root/models/datasets/evitool_traces_1k/train_traces_1000.jsonl")
    parser.add_argument("--image-root", default="/root/models/datasets/evitool_traces_1k")
    parser.add_argument("--output", default="outputs/rl_smoke/gui_candidate_selection_smoke.jsonl")
    parser.add_argument("--report", default="reports/gui_candidate_selection_rl_smoke.md")
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--max-candidates", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--omniparser-root", default="third_party/OmniParser")
    parser.add_argument("--omniparser-weights-dir", default="third_party/OmniParser/weights")
    return parser.parse_args()


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def pct(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.2f}%"


def num(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_counts = [float(r["candidate_count"]) for r in results]
    oracle_hits = [float(r["oracle_metrics"]["hit"]) for r in results]
    oracle_iou50 = [float(r["oracle_metrics"]["iou_50"]) for r in results]
    oracle_pointing = [float(r["oracle_metrics"]["pointing"]) for r in results]
    oracle_ious = [float(r["oracle_metrics"]["iou"]) for r in results]
    summary: dict[str, Any] = {
        "count": len(results),
        "avg_candidates": mean(candidate_counts),
        "oracle_hit_rate": mean(oracle_hits),
        "oracle_pointing_rate": mean(oracle_pointing),
        "oracle_iou50_rate": mean(oracle_iou50),
        "avg_oracle_iou": mean(oracle_ious),
        "policies": {},
    }
    for policy in POLICIES:
        rewards = [r["policy_rewards"][policy] for r in results]
        summary["policies"][policy] = {
            "avg_total_reward": mean([float(x["total"]) for x in rewards]),
            "valid_rate": mean([float(x["valid"]) for x in rewards]),
            "pointing_rate": mean([float(x["pointing"]) for x in rewards]),
            "iou50_rate": mean([float(x["iou_50"]) for x in rewards]),
            "avg_iou": mean([float(x["metrics"]["iou"]) for x in rewards]),
            "avg_selected_rank": mean([float(x["selected_rank"]) for x in rewards if x.get("selected_rank") is not None]),
        }
    return summary


def write_report(path: Path, args: argparse.Namespace, summary: dict[str, Any]) -> None:
    lines = [
        "# GUI Candidate-Selection RL Smoke",
        "",
        f"- Data: `{args.data}`",
        f"- Samples: {summary['count']}",
        f"- Candidate source: OmniParser-only YOLO detector",
        f"- Max candidates: {args.max_candidates}",
        "- Action schema: `{\"candidate_id\": \"cXX\"}`",
        "- Reward: format + valid candidate + pointing + IoU@0.5 + shaped IoU, with invalid/format penalties.",
        "",
        "## Candidate Oracle",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Avg candidates | {num(summary['avg_candidates'])} |",
        f"| Oracle hit rate | {pct(summary['oracle_hit_rate'])} |",
        f"| Oracle pointing | {pct(summary['oracle_pointing_rate'])} |",
        f"| Oracle IoU@0.5 | {pct(summary['oracle_iou50_rate'])} |",
        f"| Avg oracle IoU | {num(summary['avg_oracle_iou'])} |",
        "",
        "## Policy Smoke",
        "",
        "| Policy | Avg reward | Valid | Pointing | IoU@0.5 | Avg IoU | Avg rank |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in POLICIES:
        item = summary["policies"][policy]
        lines.append(
            f"| `{policy}` | {num(item['avg_total_reward'])} | {pct(item['valid_rate'])} | "
            f"{pct(item['pointing_rate'])} | {pct(item['iou50_rate'])} | {num(item['avg_iou'])} | {num(item['avg_selected_rank'])} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- This is a plumbing smoke, not model training. It validates the candidate action space and reward scale before wiring GRPO/PPO.",
            "- If oracle hit/pointing is high enough, the next step is to connect a VLM policy that emits `candidate_id` and run a tiny GRPO step.",
            "- If top1 is much worse than oracle, RL has a meaningful learning signal: selection matters and the action space is not already solved by detector score.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    rows = load_gui_rows(args.data, args.limit)
    rng = random.Random(args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    started = time.time()

    with out.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(rows, start=1):
            image_path = Path(args.image_root) / str(row["image"])
            candidates, meta = generate_omniparser_candidates(
                image_path,
                max_candidates=args.max_candidates,
                omniparser_root=args.omniparser_root,
                omniparser_weights_dir=args.omniparser_weights_dir,
            )
            oracle, oracle_metrics = oracle_candidate(row, candidates)
            policy_rewards: dict[str, Any] = {}
            policy_actions: dict[str, Any] = {}
            for policy in POLICIES:
                action = policy_action(policy, row, candidates, rng)
                reward = score_candidate_action(row, candidates, action)
                policy_actions[policy] = action
                policy_rewards[policy] = reward
            record = {
                "id": row.get("id"),
                "image": row.get("image"),
                "instruction": (row.get("meta") or {}).get("instruction") or row.get("question"),
                "answer_bbox": row.get("answer_bbox"),
                "candidate_count": len(candidates),
                "candidate_meta": meta,
                "oracle_candidate_id": oracle.get("candidate_id") if oracle else None,
                "oracle_bbox": oracle.get("bbox") if oracle else None,
                "oracle_rank": oracle.get("rank") if oracle else None,
                "oracle_metrics": oracle_metrics,
                "policy_actions": policy_actions,
                "policy_rewards": policy_rewards,
                "prompt": build_candidate_prompt(row, candidates),
                "candidates": candidates,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            results.append(record)
            if idx % 16 == 0:
                print(f"[{idx}/{len(rows)}] processed", flush=True)

    summary = summarize(results)
    summary["elapsed_sec"] = round(time.time() - started, 4)
    summary_path = out.with_suffix(out.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(Path(args.report), args, summary)
    print(f"wrote {out}")
    print(f"wrote {summary_path}")
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
