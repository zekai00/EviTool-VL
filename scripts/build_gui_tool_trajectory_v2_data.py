#!/usr/bin/env python3
"""Build GUI tool-call trajectory v2 data.

V2 focuses on candidate selection rather than bbox copying:

- split/cap by image to reduce same-screen memorization;
- sample 8-16 local shuffled candidates per trace;
- hide original rank/score and remap candidate ids per observation;
- train `detect -> select_candidate -> final` trajectories.

No teacher model is required. The builder uses ground-truth boxes and the
existing full-aware candidate pool to create auditable oracle trajectories.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.common import bbox_center, bbox_iou, point_in_bbox
from tools.runner import run_tool


DEFAULT_INPUT_DIR = (
    "outputs/gui_candidate_rl_os_atlas_linux_2k_enhanced_c100_base_dashscope_"
    "menu_table_refined_full_aware_20260527_1407"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--sft-output-dir", default=None)
    parser.add_argument("--doc-report", default=None)
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-candidates", type=int, default=12)
    parser.add_argument("--min-candidates", type=int, default=8)
    parser.add_argument("--max-candidates", type=int, default=16)
    parser.add_argument("--min-positive-iou", type=float, default=0.5)
    parser.add_argument("--train-max-per-image", type=int, default=3)
    parser.add_argument("--eval-max-per-image", type=int, default=1)
    parser.add_argument("--val-image-count", type=int, default=10)
    parser.add_argument("--include-rank-fields", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tool-out-dir", default="outputs/gui_tool_trajectory_v2_artifacts")
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


def write_json(path: Path, data: Any, indent: int | None = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=indent), encoding="utf-8")


def candidate_by_id(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(candidate.get("candidate_id")): candidate for candidate in row.get("candidates") or []}


def norm_bbox(bbox: list[Any]) -> list[int]:
    return [int(round(float(v))) for v in bbox]


def center_distance(answer_bbox: list[Any], candidate: dict[str, Any]) -> float:
    ax, ay = bbox_center(answer_bbox)
    cx, cy = bbox_center(candidate["bbox"])
    width = max(1.0, float(max(answer_bbox[2], candidate["bbox"][2]) - min(answer_bbox[0], candidate["bbox"][0])))
    height = max(1.0, float(max(answer_bbox[3], candidate["bbox"][3]) - min(answer_bbox[1], candidate["bbox"][1])))
    return ((ax - cx) / width) ** 2 + ((ay - cy) / height) ** 2


def tokens(text: Any) -> set[str]:
    return {part for part in "".join(ch.lower() if ch.isalnum() else " " for ch in str(text)).split() if len(part) >= 2}


def text_overlap_score(row: dict[str, Any], candidate: dict[str, Any]) -> int:
    query_tokens = tokens(row.get("instruction") or row.get("question") or "")
    candidate_tokens = tokens(candidate.get("text") or candidate.get("label") or "")
    return len(query_tokens & candidate_tokens)


def candidate_iou(row: dict[str, Any], candidate: dict[str, Any]) -> float:
    return bbox_iou(row["answer_bbox"], candidate["bbox"])


def is_clean_positive(row: dict[str, Any], min_positive_iou: float) -> bool:
    oracle_id = str(row.get("oracle_candidate_id") or "")
    by_id = candidate_by_id(row)
    if oracle_id not in by_id:
        return False
    return candidate_iou(row, by_id[oracle_id]) >= min_positive_iou and bool((row.get("oracle_metrics") or {}).get("hit"))


def is_unambiguous_negative(row: dict[str, Any], candidate: dict[str, Any], positive_id: str) -> bool:
    if str(candidate.get("candidate_id")) == positive_id:
        return False
    iou = candidate_iou(row, candidate)
    if iou >= 0.5:
        return False
    if point_in_bbox(bbox_center(candidate["bbox"]), row["answer_bbox"]):
        return False
    return True


def add_unique(target: list[dict[str, Any]], candidates: list[dict[str, Any]], limit: int | None = None) -> None:
    seen = {str(item.get("candidate_id")) for item in target}
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id"))
        if candidate_id in seen:
            continue
        target.append(candidate)
        seen.add(candidate_id)
        if limit is not None and len(target) >= limit:
            return


def sample_candidates(row: dict[str, Any], args: argparse.Namespace, rng: random.Random) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_id = candidate_by_id(row)
    positive_id = str(row["oracle_candidate_id"])
    positive = by_id[positive_id]
    wrong = [candidate for candidate in row.get("candidates") or [] if is_unambiguous_negative(row, candidate, positive_id)]

    selected: list[dict[str, Any]] = [positive]
    role_map: dict[str, list[str]] = {positive_id: ["oracle_positive"]}

    def mark(items: list[dict[str, Any]], role: str, n: int) -> None:
        before = {str(item.get("candidate_id")) for item in selected}
        add_unique(selected, items, limit=min(args.max_candidates, len(selected) + n))
        for item in selected:
            candidate_id = str(item.get("candidate_id"))
            if candidate_id not in before and candidate_id != positive_id:
                role_map.setdefault(candidate_id, []).append(role)

    top_rank = sorted(wrong, key=lambda item: int(item.get("rank") or 9999))
    mark(top_rank, "top_rank_negative", 4)

    near_iou = sorted(
        [item for item in wrong if candidate_iou(row, item) > 0.02],
        key=lambda item: candidate_iou(row, item),
        reverse=True,
    )
    mark(near_iou, "near_iou_negative", 3)

    text_like = sorted(
        [item for item in wrong if text_overlap_score(row, item) > 0],
        key=lambda item: text_overlap_score(row, item),
        reverse=True,
    )
    mark(text_like, "text_similar_negative", 3)

    near_center = sorted(wrong, key=lambda item: center_distance(row["answer_bbox"], item))
    mark(near_center, "near_center_negative", 3)

    far = [item for item in wrong if candidate_iou(row, item) == 0.0]
    rng.shuffle(far)
    mark(far, "random_far_negative", args.max_candidates)

    if len(selected) < args.min_candidates:
        return [], {"skip_reason": "too_few_candidates", "selected_count": len(selected)}

    if len(selected) > args.target_candidates:
        keep_positive = [positive]
        negatives = [item for item in selected if str(item.get("candidate_id")) != positive_id]
        rng.shuffle(negatives)
        selected = keep_positive + negatives[: args.target_candidates - 1]

    rng.shuffle(selected)
    local_candidates: list[dict[str, Any]] = []
    original_to_local: dict[str, str] = {}
    local_to_original: dict[str, str] = {}
    for index, candidate in enumerate(selected, start=1):
        original_id = str(candidate.get("candidate_id"))
        local_id = f"cand_{index:02d}"
        original_to_local[original_id] = local_id
        local_to_original[local_id] = original_id
        item = {
            "candidate_id": local_id,
            "bbox": norm_bbox(candidate["bbox"]),
            "label": candidate.get("label") or "",
            "source": candidate.get("source") or "",
        }
        if candidate.get("text"):
            item["text"] = str(candidate.get("text"))
        if args.include_rank_fields:
            item["rank"] = candidate.get("rank")
            item["score"] = candidate.get("score")
        local_candidates.append(item)

    positive_local_id = original_to_local[positive_id]
    meta = {
        "positive_original_id": positive_id,
        "positive_local_id": positive_local_id,
        "local_to_original": local_to_original,
        "candidate_roles": {original_to_local[k]: v for k, v in role_map.items() if k in original_to_local},
        "candidate_count": len(local_candidates),
        "positive_iou": round(candidate_iou(row, positive), 6),
    }
    return local_candidates, meta


def choose_rows_by_image(rows: list[dict[str, Any]], max_per_image: int, rng: random.Random) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["image_path"])].append(row)
    selected: list[dict[str, Any]] = []
    for image_path in sorted(grouped):
        image_rows = list(grouped[image_path])
        rng.shuffle(image_rows)
        selected.extend(image_rows[:max_per_image])
    return selected


def split_val_test_by_image(rows: list[dict[str, Any]], val_image_count: int, rng: random.Random) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["image_path"])].append(row)
    images = sorted(grouped)
    rng.shuffle(images)
    val_images = set(images[: min(val_image_count, len(images))])
    val_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    for image_path in images:
        image_rows = list(grouped[image_path])
        rng.shuffle(image_rows)
        if image_path in val_images:
            val_rows.extend(image_rows[:1])
        else:
            test_rows.extend(image_rows[:1])
    return val_rows, test_rows


def assistant_action(thought: str, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"thought": thought, "action": {"tool": tool, "args": args}}


def make_trajectory(row: dict[str, Any], candidates: list[dict[str, Any]], cand_meta: dict[str, Any], args: argparse.Namespace, split: str) -> dict[str, Any]:
    instruction = str(row.get("instruction") or row.get("question") or "")
    image_path = str(row["image_path"])
    positive_local_id = cand_meta["positive_local_id"]
    positive_candidate = next(item for item in candidates if item["candidate_id"] == positive_local_id)
    selected_bbox = positive_candidate["bbox"]

    user_prompt = (
        "<image>You are EviTool-VL, a GUI tool-use agent.\n"
        "Locate the UI target described by the instruction. Use local candidate ids from the latest detect observation. "
        "Candidate ids are local to each observation and are not ranked.\n\n"
        f"Instruction: {instruction}\n"
        "Allowed tools: detect(mode, query, max_results), select_candidate(candidate_id).\n"
        "Return exactly one valid JSON object for each assistant turn."
    )
    detect_action = assistant_action(
        "Find GUI candidates that may match the instruction.",
        "detect",
        {"mode": "ui", "query": instruction, "max_results": len(candidates)},
    )
    detect_observation = {
        "evidence_id": "ev_001",
        "tool": "detect",
        "image": row.get("image"),
        "bbox": None,
        "ok": True,
        "error": None,
        "content": {
            "mode": "ui",
            "query": instruction,
            "source": "v2_local_shuffled_candidate_subset",
            "candidate_id_scope": "local_per_observation",
            "candidate_count": len(candidates),
            "candidates": candidates,
        },
        "artifacts": {},
    }
    select_action = assistant_action(
        f"Select {positive_local_id}, the candidate that matches the instruction.",
        "select_candidate",
        {"candidate_id": positive_local_id},
    )
    select_observation = run_tool(
        image_path,
        {
            "tool": "select_candidate",
            "args": {"candidate_id": positive_local_id, "bbox": selected_bbox, "label": instruction},
        },
        out_dir=args.tool_out_dir,
        evidence_id="ev_002",
    )
    final = {
        "reasoning": [
            {
                "step": f"Candidate {positive_local_id} matches the requested GUI target.",
                "evidence": ["ev_001", "ev_002"],
            }
        ],
        "answer": {"type": "candidate", "candidate_id": positive_local_id, "bbox": selected_bbox},
    }
    trajectory = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": detect_action},
        {"role": "tool", "content": detect_observation},
        {"role": "assistant", "content": select_action},
        {"role": "tool", "content": select_observation},
        {"role": "assistant", "content": final},
    ]
    final_iou = bbox_iou(selected_bbox, row["answer_bbox"])
    return {
        "id": f"{row['id']}_trajv2",
        "source": "os_atlas_linux_2k_refined_full_aware",
        "split": split,
        "task_type": "gui_grounding",
        "image": row.get("image"),
        "image_path": image_path,
        "instruction": instruction,
        "question": row.get("question"),
        "answer_bbox": norm_bbox(row["answer_bbox"]),
        "final_answer": final["answer"],
        "trajectory": trajectory,
        "tool_observations": [detect_observation, select_observation],
        "reward_signals": {"answer": 1.0, "evidence": 1.0, "format": 1.0, "tool": 1.0, "total": 1.0},
        "quality": {
            "template": "detect_select_candidate_v2",
            "candidate_count": len(candidates),
            "positive_local_id": positive_local_id,
            "positive_original_id": cand_meta["positive_original_id"],
            "positive_iou": cand_meta["positive_iou"],
            "final_iou": round(final_iou, 6),
            "final_iou_50": final_iou >= 0.5,
            "final_pointing": point_in_bbox(bbox_center(selected_bbox), row["answer_bbox"]),
            "evidence_closed": True,
            "tool_success": bool(select_observation.get("ok")),
            "uses_local_shuffled_ids": True,
            "rank_fields_exposed": args.include_rank_fields,
        },
        "metadata": {
            "candidate_roles": cand_meta["candidate_roles"],
            "local_to_original": cand_meta["local_to_original"],
            "oracle_metrics": row.get("oracle_metrics"),
            "oracle_rank": row.get("oracle_rank"),
            "source_split": row.get("split"),
        },
    }


def to_sft_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for turn in row["trajectory"]:
        role = turn["role"]
        content = turn["content"]
        if role == "user":
            messages.append({"role": "user", "content": content})
        elif role == "assistant":
            messages.append({"role": "assistant", "content": json.dumps(content, ensure_ascii=False)})
        elif role == "tool":
            messages.append({"role": "user", "content": "Tool observation:\n" + json.dumps(content, ensure_ascii=False)})
        else:
            raise ValueError(f"unsupported role: {role}")
    return messages


def build_split(rows: list[dict[str, Any]], split: str, args: argparse.Namespace, rng: random.Random) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    rl_rows: list[dict[str, Any]] = []
    sft_rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    for row in rows:
        if not is_clean_positive(row, args.min_positive_iou):
            skipped["not_clean_positive"] += 1
            continue
        row_rng = random.Random(f"{args.seed}:{split}:{row['id']}")
        candidates, cand_meta = sample_candidates(row, args, row_rng)
        if not candidates:
            skipped[cand_meta.get("skip_reason", "candidate_sampling_failed")] += 1
            continue
        rl_row = make_trajectory(row, candidates, cand_meta, args, split)
        if not rl_row["quality"]["tool_success"]:
            skipped["tool_failed"] += 1
            continue
        rl_rows.append(rl_row)
        sft_rows.append(
            {
                "messages": to_sft_messages(rl_row),
                "images": [rl_row["image_path"]],
                "meta": {
                    "id": rl_row["id"],
                    "split": split,
                    "template": rl_row["quality"]["template"],
                    "positive_local_id": rl_row["quality"]["positive_local_id"],
                    "positive_original_id": rl_row["quality"]["positive_original_id"],
                },
            }
        )
    return rl_rows, sft_rows, skipped


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    image_paths = [row["image_path"] for row in rows]
    candidate_counts = [row["quality"]["candidate_count"] for row in rows]
    ious = [row["quality"]["final_iou"] for row in rows]
    return {
        "count": len(rows),
        "unique_images": len(set(image_paths)),
        "max_rows_per_image": max(Counter(image_paths).values()),
        "avg_rows_per_image": len(rows) / max(1, len(set(image_paths))),
        "avg_candidates": statistics.mean(candidate_counts),
        "min_candidates": min(candidate_counts),
        "max_candidates": max(candidate_counts),
        "final_iou_50_rate": sum(row["quality"]["final_iou_50"] for row in rows) / len(rows),
        "final_pointing_rate": sum(row["quality"]["final_pointing"] for row in rows) / len(rows),
        "evidence_closed_rate": sum(row["quality"]["evidence_closed"] for row in rows) / len(rows),
        "tool_success_rate": sum(row["quality"]["tool_success"] for row in rows) / len(rows),
        "avg_final_iou": statistics.mean(ious),
        "local_id_rate": sum(row["quality"]["uses_local_shuffled_ids"] for row in rows) / len(rows),
        "rank_fields_exposed_rate": sum(row["quality"]["rank_fields_exposed"] for row in rows) / len(rows),
    }


def summarize(all_rl: dict[str, list[dict[str, Any]]], skipped: dict[str, Counter[str]], args: argparse.Namespace) -> dict[str, Any]:
    summary = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S CST"),
        "builder": "build_gui_tool_trajectory_v2_data.py",
        "settings": {
            "seed": args.seed,
            "target_candidates": args.target_candidates,
            "min_candidates": args.min_candidates,
            "max_candidates": args.max_candidates,
            "min_positive_iou": args.min_positive_iou,
            "train_max_per_image": args.train_max_per_image,
            "eval_max_per_image": args.eval_max_per_image,
            "val_image_count": args.val_image_count,
            "include_rank_fields": args.include_rank_fields,
            "teacher_model_used": False,
        },
        "splits": {split: summarize_rows(rows) for split, rows in all_rl.items()},
        "skipped": {split: dict(counter) for split, counter in skipped.items()},
    }
    split_images = {split: {row["image_path"] for row in rows} for split, rows in all_rl.items()}
    summary["split_check"] = {
        "train_val_image_overlap": len(split_images.get("train", set()) & split_images.get("val", set())),
        "train_test_image_overlap": len(split_images.get("train", set()) & split_images.get("test", set())),
        "val_test_image_overlap": len(split_images.get("val", set()) & split_images.get("test", set())),
    }
    all_rows = [row for rows in all_rl.values() for row in rows]
    summary["overall"] = summarize_rows(all_rows)
    return summary


def make_report(summary: dict[str, Any], output_dir: Path, sft_output_dir: Path, args: argparse.Namespace) -> str:
    lines = [
        "# GUI Tool-Call Trajectory v2 数据集构建报告",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M CST')}",
        f"- 输入 full-aware 数据：`{Path(args.input_dir).resolve()}`",
        f"- RL/TRL 输出目录：`{output_dir}`",
        f"- SFT 输出目录：`{sft_output_dir}`",
        "- 是否调用大模型：否，本版完全使用本地 GT/candidate/IoU 规则构建。",
        "",
        "## 构建规则",
        "",
        f"- train 每张图最多 `{args.train_max_per_image}` 条轨迹。",
        f"- val/test 每张图最多 `{args.eval_max_per_image}` 条轨迹。",
        f"- 每条 observation 采样 `{args.min_candidates}-{args.max_candidates}` 个候选，目标约 `{args.target_candidates}` 个。",
        "- 候选 id 在每条 observation 内重新随机编号为 `cand_01...cand_N`。",
        "- observation 不暴露原始 rank/score，降低 top-rank shortcut。",
        "- action 使用 `select_candidate(candidate_id)`，不再直接训练 `click(bbox)`。",
        "",
        "## 指标",
        "",
        "| split | rows | unique images | max/image | avg candidates | IoU@0.5 | pointing | evidence |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ["train", "val", "test", "overall"]:
        item = summary["overall"] if split == "overall" else summary["splits"].get(split, {"count": 0})
        if not item.get("count"):
            continue
        lines.append(
            "| {split} | {count} | {imgs} | {max_img} | {cand:.2f} | {iou:.2%} | {point:.2%} | {ev:.2%} |".format(
                split=split,
                count=item["count"],
                imgs=item["unique_images"],
                max_img=item["max_rows_per_image"],
                cand=item["avg_candidates"],
                iou=item["final_iou_50_rate"],
                point=item["final_pointing_rate"],
                ev=item["evidence_closed_rate"],
            )
        )
    lines.extend(
        [
            "",
            "## 输出文件",
            "",
            f"- RL train：`{output_dir / 'train_rl.jsonl'}`",
            f"- RL val：`{output_dir / 'val_rl.jsonl'}`",
            f"- RL test：`{output_dir / 'test_rl.jsonl'}`",
            f"- SFT train：`{sft_output_dir / 'gui_tool_trajectory_v2_train_sft.json'}`",
            f"- SFT val：`{sft_output_dir / 'gui_tool_trajectory_v2_val_sft.json'}`",
            f"- SFT test：`{sft_output_dir / 'gui_tool_trajectory_v2_test_sft.json'}`",
            "",
            "## 当前限制",
            "",
            "- 本版仍受限于当前 OS-Atlas Linux 2k 子集，unique image 数量不大。",
            "- 它适合验证 v2 格式和训练候选选择，不应被当成最终大规模 agentic RL 数据。",
            "- 下一步应扩充更多 unique screenshots 后复用同一构建器。",
            "",
            "## 跳过统计",
            "",
        ]
    )
    for split in ["train", "val", "test"]:
        lines.append(f"- `{split}` skipped: `{summary.get('skipped', {}).get(split, {})}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M")
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir or f"outputs/gui_tool_trajectory_v2_os_atlas_linux_2k_{timestamp}")
    sft_output_dir = Path(args.sft_output_dir or f"/root/models/datasets/gui_tool_trajectory_v2_os_atlas_linux_2k_{timestamp}")
    doc_report = Path(
        args.doc_report
        or f"/root/Workspace/VLM/项目文档/02_指标与数据/GUI工具调用轨迹v2数据集构建报告_{timestamp}.md"
    )
    args.tool_out_dir = str(Path(args.tool_out_dir) / timestamp)

    rng = random.Random(args.seed)
    train_source = [row for row in read_jsonl(input_dir / "train.jsonl") if is_clean_positive(row, args.min_positive_iou)]
    val_source = [row for row in read_jsonl(input_dir / "val.jsonl") if is_clean_positive(row, args.min_positive_iou)]
    train_candidates = choose_rows_by_image(train_source, args.train_max_per_image, rng)
    val_rows_source, test_rows_source = split_val_test_by_image(val_source, args.val_image_count, rng)
    val_candidates = choose_rows_by_image(val_rows_source, args.eval_max_per_image, rng)
    test_candidates = choose_rows_by_image(test_rows_source, args.eval_max_per_image, rng)

    all_source = {"train": train_candidates, "val": val_candidates, "test": test_candidates}
    all_rl: dict[str, list[dict[str, Any]]] = {}
    all_sft: dict[str, list[dict[str, Any]]] = {}
    skipped: dict[str, Counter[str]] = {}
    for split, rows in all_source.items():
        split_rng = random.Random(f"{args.seed}:{split}")
        rl_rows, sft_rows, split_skipped = build_split(rows, split, args, split_rng)
        all_rl[split] = rl_rows
        all_sft[split] = sft_rows
        skipped[split] = split_skipped

    summary = summarize(all_rl, skipped, args)
    summary["input_dir"] = str(input_dir.resolve())
    summary["output_dir"] = str(output_dir.resolve())
    summary["sft_output_dir"] = str(sft_output_dir.resolve())
    summary["source_counts"] = {
        "train_clean_positive": len(train_source),
        "val_clean_positive": len(val_source),
        "train_after_image_cap": len(train_candidates),
        "val_source_after_image_split": len(val_rows_source),
        "test_source_after_image_split": len(test_rows_source),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    sft_output_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        write_jsonl(output_dir / f"{split}_rl.jsonl", all_rl[split])
        write_json(sft_output_dir / f"gui_tool_trajectory_v2_{split}_sft.json", all_sft[split])
    write_json(output_dir / "summary.json", summary)
    write_json(sft_output_dir / "summary.json", summary)
    write_json(
        sft_output_dir / "dataset_info.json",
        {
            f"gui_tool_trajectory_v2_{split}_sft": {
                "file_name": f"gui_tool_trajectory_v2_{split}_sft.json",
                "formatting": "sharegpt",
                "columns": {"messages": "messages", "images": "images"},
                "tags": {
                    "role_tag": "role",
                    "content_tag": "content",
                    "user_tag": "user",
                    "assistant_tag": "assistant",
                },
            }
            for split in ["train", "val", "test"]
        },
    )
    report = make_report(summary, output_dir.resolve(), sft_output_dir.resolve(), args)
    (output_dir / "DATASET_REPORT.md").write_text(report, encoding="utf-8")
    doc_report.parent.mkdir(parents=True, exist_ok=True)
    doc_report.write_text(report, encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
