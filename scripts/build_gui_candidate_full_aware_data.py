#!/usr/bin/env python3
"""Build full-aware candidate-constrained GUI RL data.

The previous CC-GRPO preference data used a compact `cc_action_ids` subset:
oracle plus hand-crafted hard negatives.  That made training stable, but it
created a distribution gap against full-candidate evaluation.  This builder
keeps the full candidate list in every row and builds an action subset that is
aware of the full pool: top detector mistakes, high-IoU/near-center misses,
distant false positives, and optionally current-model high-score mistakes from
score-cache JSONL files.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.candidate_constrained import (
    action_reward_std,
    candidate_reward_v3,
    rank_bucket,
)
from rl.gui_candidate_env import candidate_metrics, center_distance_score, oracle_candidate


ROLE_ORACLE = "oracle_positive"
ROLE_NO_HIT_BEST = "best_available_no_oracle_hit"
ROLE_MODEL_FP = "current_model_false_positive"
ROLE_MODEL_PROXY_FP = "current_model_proxy_false_positive"
ROLE_TOP_RANK_WRONG = "detector_top_rank_wrong"
ROLE_HIGH_IOU_WRONG = "high_iou_wrong"
ROLE_NEAR_CENTER_WRONG = "near_center_wrong"
ROLE_HIGH_REWARD_WRONG = "high_reward_wrong"
ROLE_TEXT_SIMILAR_WRONG = "text_similar_wrong"
ROLE_DISTANT_FP = "distant_false_positive"
ROLE_RANDOM_NEG = "random_full_pool_negative"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="outputs/gui_candidate_rl_os_atlas_linux_2k_c60")
    parser.add_argument("--output-dir", default="outputs/gui_candidate_rl_os_atlas_linux_2k_c60_full_aware")
    parser.add_argument("--report", default="reports/gui_candidate_full_aware_dataset_os_atlas_linux_2k_c60.md")
    parser.add_argument("--max-actions", type=int, default=16)
    parser.add_argument("--min-train-oracle-iou", type=float, default=0.05)
    parser.add_argument("--min-action-reward-std", type=float, default=1e-6)
    parser.add_argument("--require-train-oracle-hit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--model-score-jsonl",
        action="append",
        default=[],
        help=(
            "Optional score-cache/eval JSONL. Each row should contain id and "
            "scores={candidate_id: logprob}. High-scored low-reward candidates "
            "will be mined as current-model false positives."
        ),
    )
    parser.add_argument("--model-fp-limit", type=int, default=4)
    parser.add_argument("--top-rank-limit", type=int, default=4)
    parser.add_argument("--high-iou-limit", type=int, default=3)
    parser.add_argument("--near-center-limit", type=int, default=3)
    parser.add_argument("--high-reward-limit", type=int, default=3)
    parser.add_argument("--distant-limit", type=int, default=2)
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


def load_model_score_cache(paths: list[str]) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    """Load optional current-policy candidate scores.

    `scripts/eval_gui_candidate_scores.py` already writes the needed shape:
    `{"id": ..., "scores": {"c00": -1.2, ...}, "candidate_id": "c00"}`.
    The builder can also run without this cache; in that case it records that
    current-model false positives were approximated by top-ranked detector
    mistakes, which matches the observed c00/c01 shortcut in recent evals.
    """
    score_cache: dict[str, dict[str, float]] = {}
    selected_cache: dict[str, str] = {}
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                row_id = str(item.get("id") or "")
                scores = item.get("scores") or {}
                if not row_id or not isinstance(scores, dict):
                    continue
                score_cache[row_id] = {str(k): float(v) for k, v in scores.items()}
                if item.get("candidate_id"):
                    selected_cache[row_id] = str(item["candidate_id"])
    return score_cache, selected_cache


def candidate_by_id(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(candidate["candidate_id"]): candidate for candidate in row.get("candidates") or []}


def reward_table(row: dict[str, Any]) -> dict[str, float]:
    table: dict[str, float] = {}
    for candidate in row.get("candidates") or []:
        table[str(candidate["candidate_id"])] = float(candidate_reward_v3(row, candidate)["total"])
    return table


def metric_table(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(candidate["candidate_id"]): candidate_metrics(row, candidate)
        for candidate in row.get("candidates") or []
    }


def ensure_oracle(row: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    """Return the candidate id that represents the best available candidate."""
    by_id = candidate_by_id(row)
    oracle_id = row.get("oracle_candidate_id")
    metrics = row.get("oracle_metrics") or {}
    if oracle_id and oracle_id in by_id:
        return str(oracle_id), metrics
    oracle, oracle_metrics = oracle_candidate(row, row.get("candidates") or [])
    return (str(oracle["candidate_id"]), oracle_metrics) if oracle else (None, oracle_metrics)


def is_oracle_hit(metrics: dict[str, Any], min_iou: float) -> bool:
    return bool(metrics.get("hit")) and float(metrics.get("iou") or 0.0) >= min_iou


def text_tokens(value: str) -> set[str]:
    return {token for token in "".join(ch.lower() if ch.isalnum() else " " for ch in value).split() if len(token) >= 2}


def build_full_aware_actions(
    row: dict[str, Any],
    *,
    args: argparse.Namespace,
    scores: dict[str, float] | None,
) -> tuple[list[str], dict[str, list[str]], dict[str, Any]]:
    candidates = list(row.get("candidates") or [])
    by_id = candidate_by_id(row)
    rewards = reward_table(row)
    metrics_by_id = metric_table(row)
    oracle_id, oracle_metrics = ensure_oracle(row)
    action_ids: list[str] = []
    roles: dict[str, list[str]] = {}

    def add(candidate_id: str | None, role: str) -> None:
        if not candidate_id or candidate_id not in by_id:
            return
        roles.setdefault(candidate_id, [])
        if role not in roles[candidate_id]:
            roles[candidate_id].append(role)
        if candidate_id not in action_ids and len(action_ids) < args.max_actions:
            action_ids.append(candidate_id)

    positive_available = bool(oracle_id and is_oracle_hit(oracle_metrics, args.min_train_oracle_iou))
    add(oracle_id, ROLE_ORACLE if positive_available else ROLE_NO_HIT_BEST)

    wrong_candidates = [candidate for candidate in candidates if str(candidate["candidate_id"]) != oracle_id]

    # Current-model mistakes are the most important full-aware ingredient.  When
    # a score cache is present, use true model high-score low-reward candidates;
    # otherwise use top-ranked detector mistakes as a deterministic proxy for
    # the observed c00/c01 shortcut.
    model_fp_count = 0
    if scores:
        for candidate_id, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True):
            if candidate_id == oracle_id or candidate_id not in by_id:
                continue
            metric = metrics_by_id.get(candidate_id) or {}
            if metric.get("hit") or rewards.get(candidate_id, 0.0) >= 0.35:
                continue
            add(candidate_id, ROLE_MODEL_FP)
            model_fp_count += 1
            if model_fp_count >= args.model_fp_limit:
                break
    if model_fp_count == 0:
        for candidate in sorted(wrong_candidates, key=lambda item: int(item.get("rank") or 9999)):
            candidate_id = str(candidate["candidate_id"])
            metric = metrics_by_id[candidate_id]
            if metric.get("hit"):
                continue
            add(candidate_id, ROLE_MODEL_PROXY_FP)
            model_fp_count += 1
            if model_fp_count >= args.model_fp_limit:
                break

    for candidate in sorted(wrong_candidates, key=lambda item: int(item.get("rank") or 9999)):
        candidate_id = str(candidate["candidate_id"])
        if not metrics_by_id[candidate_id].get("hit"):
            add(candidate_id, ROLE_TOP_RANK_WRONG)
        if sum(ROLE_TOP_RANK_WRONG in role_list for role_list in roles.values()) >= args.top_rank_limit:
            break

    for candidate in sorted(wrong_candidates, key=lambda item: float(metrics_by_id[str(item["candidate_id"])]["iou"]), reverse=True):
        candidate_id = str(candidate["candidate_id"])
        metric = metrics_by_id[candidate_id]
        if not metric.get("iou_50"):
            add(candidate_id, ROLE_HIGH_IOU_WRONG)
        if sum(ROLE_HIGH_IOU_WRONG in role_list for role_list in roles.values()) >= args.high_iou_limit:
            break

    for candidate in sorted(wrong_candidates, key=lambda item: center_distance_score(row.get("answer_bbox"), item), reverse=True):
        candidate_id = str(candidate["candidate_id"])
        if not metrics_by_id[candidate_id].get("hit"):
            add(candidate_id, ROLE_NEAR_CENTER_WRONG)
        if sum(ROLE_NEAR_CENTER_WRONG in role_list for role_list in roles.values()) >= args.near_center_limit:
            break

    for candidate in sorted(wrong_candidates, key=lambda item: rewards.get(str(item["candidate_id"]), 0.0), reverse=True):
        candidate_id = str(candidate["candidate_id"])
        if not metrics_by_id[candidate_id].get("hit"):
            add(candidate_id, ROLE_HIGH_REWARD_WRONG)
        if sum(ROLE_HIGH_REWARD_WRONG in role_list for role_list in roles.values()) >= args.high_reward_limit:
            break

    instruction_tokens = text_tokens(str(row.get("instruction") or row.get("question") or ""))
    if instruction_tokens:
        for candidate in wrong_candidates:
            candidate_id = str(candidate["candidate_id"])
            cand_tokens = text_tokens(str(candidate.get("text") or candidate.get("label") or ""))
            if cand_tokens and instruction_tokens.intersection(cand_tokens) and not metrics_by_id[candidate_id].get("hit"):
                add(candidate_id, ROLE_TEXT_SIMILAR_WRONG)

    low_reward = sorted(
        wrong_candidates,
        key=lambda item: (rewards.get(str(item["candidate_id"]), 0.0), -int(item.get("rank") or 0)),
    )
    distant_added = 0
    for candidate in low_reward:
        candidate_id = str(candidate["candidate_id"])
        if center_distance_score(row.get("answer_bbox"), candidate) <= 0.05:
            add(candidate_id, ROLE_DISTANT_FP)
            distant_added += 1
        if distant_added >= args.distant_limit:
            break

    rng = random.Random(f"full-aware:{args.seed}:{row.get('id')}")
    shuffled = wrong_candidates[:]
    rng.shuffle(shuffled)
    for candidate in shuffled:
        if len(action_ids) >= args.max_actions:
            break
        candidate_id = str(candidate["candidate_id"])
        if not metrics_by_id[candidate_id].get("hit"):
            add(candidate_id, ROLE_RANDOM_NEG)

    # If all useful negatives were duplicates, fill with any remaining full-pool
    # candidate so every trainable row can still expose a compact full-aware set.
    for candidate in candidates:
        if len(action_ids) >= args.max_actions:
            break
        add(str(candidate["candidate_id"]), ROLE_RANDOM_NEG)

    action_rewards = {candidate_id: rewards[candidate_id] for candidate_id in action_ids if candidate_id in rewards}
    std = action_reward_std(action_ids, rewards) if len(action_ids) >= 2 else 0.0
    meta = {
        "oracle_id": oracle_id,
        "oracle_metrics": oracle_metrics,
        "positive_available": positive_available,
        "model_score_available": bool(scores),
        "model_false_positive_count": model_fp_count,
        "action_reward_std": std,
        "action_reward_min": min(action_rewards.values()) if action_rewards else None,
        "action_reward_max": max(action_rewards.values()) if action_rewards else None,
    }
    return action_ids, roles, meta


def enrich_row(
    row: dict[str, Any],
    *,
    args: argparse.Namespace,
    score_cache: dict[str, dict[str, float]],
    split: str,
) -> dict[str, Any]:
    row_id = str(row.get("id") or "")
    rewards = reward_table(row)
    action_ids, roles, meta = build_full_aware_actions(row, args=args, scores=score_cache.get(row_id))
    oracle_id = meta["oracle_id"]
    oracle_rank = None
    for candidate in row.get("candidates") or []:
        if str(candidate.get("candidate_id")) == oracle_id:
            oracle_rank = int(candidate.get("rank") or 0)
            break
    trainable = (
        bool(row.get("overlay_image"))
        and bool(row.get("candidates"))
        and bool(meta["positive_available"])
        and len(action_ids) >= 2
        and float(meta["action_reward_std"]) >= args.min_action_reward_std
    )
    enriched = dict(row)
    enriched["split"] = split
    enriched["candidate_rewards_v3"] = rewards
    enriched["full_aware_action_ids"] = action_ids
    enriched["full_aware_action_roles"] = roles
    enriched["full_aware_action_reward_std"] = round(float(meta["action_reward_std"]), 6)
    enriched["full_aware_trainable"] = trainable
    enriched["model_score_available"] = bool(meta["model_score_available"])
    enriched["oracle_candidate_id"] = oracle_id
    enriched["oracle_metrics"] = meta["oracle_metrics"]
    enriched["oracle_rank"] = oracle_rank
    enriched["oracle_rank_bucket"] = rank_bucket(oracle_rank)
    enriched["positive_ids"] = [oracle_id] if meta["positive_available"] and oracle_id else []
    enriched["hard_negative_ids"] = [candidate_id for candidate_id in action_ids if candidate_id not in enriched["positive_ids"]]
    # Keep compatibility with existing CC-GRPO/listwise scripts.
    enriched["cc_action_ids"] = action_ids
    enriched["cc_hard_negative_ids"] = enriched["hard_negative_ids"]
    enriched["cc_reward_std"] = enriched["full_aware_action_reward_std"]
    return enriched


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(values: list[float]) -> float | None:
        return statistics.mean(values) if values else None

    oracle_metrics = [row.get("oracle_metrics") or {} for row in rows]
    trainable_rows = [row for row in rows if row.get("full_aware_trainable")]
    role_counter: Counter[str] = Counter()
    for row in rows:
        for role_list in (row.get("full_aware_action_roles") or {}).values():
            role_counter.update(role_list)
    rank_buckets: Counter[str] = Counter(str(row.get("oracle_rank_bucket") or "unknown") for row in rows)
    stds = [float(row.get("full_aware_action_reward_std") or 0.0) for row in trainable_rows]
    return {
        "count": len(rows),
        "trainable_count": len(trainable_rows),
        "avg_candidates": mean([float(row.get("candidate_count") or len(row.get("candidates") or [])) for row in rows]),
        "avg_actions": mean([float(len(row.get("full_aware_action_ids") or [])) for row in rows]),
        "avg_trainable_actions": mean([float(len(row.get("full_aware_action_ids") or [])) for row in trainable_rows]),
        "oracle_hit_rate": mean([float(bool(metric.get("hit"))) for metric in oracle_metrics]),
        "oracle_pointing_rate": mean([float(bool(metric.get("pointing"))) for metric in oracle_metrics]),
        "oracle_iou50_rate": mean([float(bool(metric.get("iou_50"))) for metric in oracle_metrics]),
        "avg_oracle_iou": mean([float(metric.get("iou") or 0.0) for metric in oracle_metrics]),
        "avg_action_reward_std": mean(stds),
        "min_action_reward_std": min(stds) if stds else None,
        "zero_std_rate": mean([float(std <= 1e-9) for std in stds]),
        "model_score_coverage": mean([float(bool(row.get("model_score_available"))) for row in rows]),
        "rank_buckets": dict(rank_buckets),
        "role_counts": dict(role_counter),
    }


def split_trainable(rows: list[dict[str, Any]], *, args: argparse.Namespace) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        metrics = row.get("oracle_metrics") or {}
        if args.require_train_oracle_hit and not bool(metrics.get("hit")):
            continue
        if float(metrics.get("iou") or 0.0) < args.min_train_oracle_iou:
            continue
        if not row.get("full_aware_trainable"):
            continue
        output.append(row)
    return output


def pct(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.2f}%"


def num(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def table_row(label: str, summary: dict[str, Any]) -> str:
    return (
        f"| {label} | {summary['count']} | {summary['trainable_count']} | "
        f"{num(summary['avg_candidates'])} | {num(summary['avg_actions'])} | "
        f"{pct(summary['oracle_hit_rate'])} | {pct(summary['oracle_iou50_rate'])} | "
        f"{num(summary['avg_oracle_iou'])} | {num(summary['avg_action_reward_std'])} | "
        f"{pct(summary['zero_std_rate'])} |"
    )


def write_report(
    path: Path,
    *,
    args: argparse.Namespace,
    summary: dict[str, Any],
    train_images: set[str],
    val_images: set[str],
) -> None:
    role_counts = summary["output"]["train"]["role_counts"]
    role_lines = ["| Role | Count |", "|---|---:|"]
    for role, count in sorted(role_counts.items()):
        role_lines.append(f"| `{role}` | {count} |")

    overlap = sorted(train_images.intersection(val_images))
    lines = [
        "# Full-Aware GUI Candidate RL Dataset Report",
        "",
        "## 构建配置",
        "",
        f"- Input dir: `{args.input_dir}`",
        f"- Output dir: `{args.output_dir}`",
        f"- Max actions per row: `{args.max_actions}`",
        f"- Train oracle gate: hit=`{args.require_train_oracle_hit}`, min_iou=`{args.min_train_oracle_iou}`",
        f"- Model score cache files: `{len(args.model_score_jsonl)}`",
        f"- Model-score coverage is allowed to be 0; in that case top-ranked detector mistakes are used as a deterministic current-policy proxy.",
        "",
        "## Split 质量",
        "",
        "| Split | Rows | Trainable | Avg candidates | Avg actions | Oracle hit | Oracle IoU@0.5 | Avg oracle IoU | Avg action reward std | Zero-std |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        table_row("all", summary["output"]["all"]),
        table_row("train", summary["output"]["train"]),
        table_row("val", summary["output"]["val"]),
        table_row("val_oracle_hit", summary["output"]["val_oracle_hit"]),
        "",
        "## Full-Aware Action 角色分布（Train）",
        "",
        *role_lines,
        "",
        "## 泄漏检查",
        "",
        f"- Train unique images: `{len(train_images)}`",
        f"- Val unique images: `{len(val_images)}`",
        f"- Train/Val image overlap: `{len(overlap)}`",
        "",
        "## 数据字段",
        "",
        "- `candidates`: 完整候选池，保留所有 candidate bbox / rank / score。",
        "- `candidate_rewards_v3`: 每个候选的几何 reward。",
        "- `full_aware_action_ids`: 用于 CC-GRPO/listwise 的 full-aware action subset。",
        "- `full_aware_action_roles`: 每个 action 的来源角色，例如 oracle、top-rank wrong、near-center wrong。",
        "- `cc_action_ids`: 与 `full_aware_action_ids` 相同，保持旧训练脚本兼容。",
        "",
        "## 判断",
        "",
        "这版数据保留 full candidates，同时将训练 action set 对齐到 full-pool 错误分布；"
        "它比旧 `cc_action_ids` 更适合后续 full-candidate 评估。"
        "如果后续生成了全量 current-model score cache，可以用同一脚本重新构建，"
        "把 `current_model_proxy_false_positive` 替换成真实 `current_model_false_positive`。",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    score_cache, _ = load_model_score_cache(args.model_score_jsonl)

    train_source = read_jsonl(input_dir / "train.jsonl")
    val_source = read_jsonl(input_dir / "val.jsonl")
    all_source = read_jsonl(input_dir / "all.jsonl") if (input_dir / "all.jsonl").exists() else train_source + val_source

    train_ids = {str(row.get("id")) for row in train_source}
    val_ids = {str(row.get("id")) for row in val_source}
    all_rows = []
    for row in all_source:
        row_id = str(row.get("id"))
        split = "train" if row_id in train_ids else "val" if row_id in val_ids else "unknown"
        all_rows.append(enrich_row(row, args=args, score_cache=score_cache, split=split))

    train_all = [row for row in all_rows if row.get("split") == "train"]
    val_all = [row for row in all_rows if row.get("split") == "val"]
    train = split_trainable(train_all, args=args)
    val_oracle_hit = split_trainable(val_all, args=args)

    write_jsonl(output_dir / "all.jsonl", all_rows)
    write_jsonl(output_dir / "train.jsonl", train)
    write_jsonl(output_dir / "val.jsonl", val_all)
    write_jsonl(output_dir / "val_oracle_hit.jsonl", val_oracle_hit)

    train_images = {str(row.get("image")) for row in train_all}
    val_images = {str(row.get("image")) for row in val_all}
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "max_actions": args.max_actions,
        "min_train_oracle_iou": args.min_train_oracle_iou,
        "require_train_oracle_hit": args.require_train_oracle_hit,
        "model_score_files": args.model_score_jsonl,
        "input": {
            "all_rows": len(all_source),
            "train_rows": len(train_source),
            "val_rows": len(val_source),
            "model_score_rows": len(score_cache),
        },
        "output": {
            "all": summarize(all_rows),
            "train": summarize(train),
            "val": summarize(val_all),
            "val_oracle_hit": summarize(val_oracle_hit),
        },
        "leakage": {
            "train_unique_images": len(train_images),
            "val_unique_images": len(val_images),
            "train_val_image_overlap": len(train_images.intersection(val_images)),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(Path(args.report), args=args, summary=summary, train_images=train_images, val_images=val_images)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
