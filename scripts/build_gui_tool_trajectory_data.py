#!/usr/bin/env python3
"""Build GUI tool-call trajectory datasets from full-aware candidate data.

The builder uses the frozen candidate pool as the stable `detect` observation
source so candidate ids remain aligned with candidate-RL data. It also executes
local tools such as `crop` and `click` for evidence that can be cited by the
final answer.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.common import bbox_area, bbox_center, bbox_iou, image_size, load_image, point_in_bbox
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
    parser.add_argument("--max-detect-candidates", type=int, default=80)
    parser.add_argument("--crop-hard-rank-threshold", type=int, default=60)
    parser.add_argument("--max-rows-train", type=int, default=None)
    parser.add_argument("--max-rows-val", type=int, default=None)
    parser.add_argument("--tool-out-dir", default="outputs/gui_tool_trajectory_artifacts")
    parser.add_argument("--min-candidate-iou-for-candidate-answer", type=float, default=0.5)
    parser.add_argument("--include-crop-for-hard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--drop-oracle-injected",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Drop rows whose oracle candidate had to be appended to the detect observation.",
    )
    parser.add_argument(
        "--candidate-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep only rows whose final answer selects a candidate_id instead of a free/refined GT bbox.",
    )
    return parser.parse_args()


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: Any, indent: int | None = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=indent), encoding="utf-8")


def clip_bbox_to_image(bbox: list[Any], image_path: str | Path, pad: int = 0) -> list[int]:
    width, height = image_size(load_image(image_path))
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    return [
        max(0, min(width, x1 - pad)),
        max(0, min(height, y1 - pad)),
        max(0, min(width, x2 + pad)),
        max(0, min(height, y2 + pad)),
    ]


def compact_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    compact = {
        "candidate_id": candidate.get("candidate_id"),
        "rank": candidate.get("rank"),
        "bbox": candidate.get("bbox"),
        "label": candidate.get("label"),
        "source": candidate.get("source"),
    }
    if candidate.get("text"):
        compact["text"] = candidate["text"]
    if candidate.get("score") is not None:
        compact["score"] = round(float(candidate["score"]), 4)
    if candidate.get("rank_score") is not None:
        compact["rank_score"] = round(float(candidate["rank_score"]), 4)
    if candidate.get("sources"):
        compact["sources"] = candidate["sources"][:4]
    return compact


def candidate_by_id(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(candidate.get("candidate_id")): candidate for candidate in row.get("candidates") or []}


def candidate_center_in_region(candidate: dict[str, Any], region: list[int]) -> bool:
    return point_in_bbox(bbox_center(candidate["bbox"]), region)


def select_detect_candidates(
    row: dict[str, Any],
    *,
    max_candidates: int,
    region: list[int] | None = None,
) -> tuple[list[dict[str, Any]], bool, bool]:
    candidates = list(row.get("candidates") or [])
    if region is not None:
        candidates = [
            candidate
            for candidate in candidates
            if candidate_center_in_region(candidate, region) or bbox_iou(candidate["bbox"], region) > 0.0
        ]
    candidates = sorted(candidates, key=lambda item: int(item.get("rank") or 9999))
    selected = candidates[:max_candidates]
    oracle_id = str(row.get("oracle_candidate_id"))
    by_id = candidate_by_id(row)
    oracle_in_top = any(str(candidate.get("candidate_id")) == oracle_id for candidate in selected)
    oracle_injected = False
    if oracle_id in by_id and not oracle_in_top:
        selected.append(by_id[oracle_id])
        oracle_injected = True
    return [compact_candidate(candidate) for candidate in selected], oracle_in_top, oracle_injected


def make_detect_observation(
    row: dict[str, Any],
    *,
    evidence_id: str,
    max_candidates: int,
    region: list[int] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates, oracle_in_top, oracle_injected = select_detect_candidates(
        row,
        max_candidates=max_candidates,
        region=region,
    )
    image_path = row.get("image_path")
    size = list(image_size(load_image(image_path))) if image_path else None
    observation = {
        "evidence_id": evidence_id,
        "tool": "detect",
        "image": row.get("image"),
        "bbox": region,
        "ok": True,
        "error": None,
        "content": {
            "mode": "ui",
            "query": row.get("instruction") or row.get("question"),
            "source": "frozen_full_aware_candidate_pool",
            "image_size": size,
            "region": region,
            "candidate_count": len(candidates),
            "detections": candidates,
        },
        "artifacts": {},
    }
    meta = {
        "detect_candidate_count": len(candidates),
        "oracle_in_detect_topk": oracle_in_top,
        "oracle_injected_into_detect": oracle_injected,
    }
    return observation, meta


def tool_step(role: str, content: dict[str, Any]) -> dict[str, Any]:
    return {"role": role, "content": content}


def assistant_action(thought: str, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"thought": thought, "action": {"tool": tool, "args": args}}


def final_answer(
    *,
    answer_type: str,
    bbox: list[int],
    evidence_ids: list[str],
    candidate_id: str | None = None,
    supporting_candidate_id: str | None = None,
) -> dict[str, Any]:
    answer: dict[str, Any] = {"type": answer_type, "bbox": bbox}
    if candidate_id is not None:
        answer["candidate_id"] = candidate_id
    if supporting_candidate_id is not None:
        answer["supporting_candidate_id"] = supporting_candidate_id
    return {
        "reasoning": [
            {
                "step": "The cited visual-tool evidence identifies the requested GUI target.",
                "evidence": evidence_ids,
            }
        ],
        "answer": answer,
    }


def to_sft_messages(rl_row: dict[str, Any]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for turn in rl_row["trajectory"]:
        role = turn["role"]
        content = turn["content"]
        if role == "user":
            messages.append({"role": "user", "content": content})
        elif role == "assistant":
            messages.append({"role": "assistant", "content": json.dumps(content, ensure_ascii=False)})
        elif role == "tool":
            messages.append(
                {
                    "role": "user",
                    "content": "Tool observation:\n" + json.dumps(content, ensure_ascii=False),
                }
            )
        else:
            raise ValueError(f"unsupported role: {role}")
    return messages


def compute_quality(row: dict[str, Any], final_bbox: list[int], evidence_ids: list[str], tool_observations: list[dict[str, Any]]) -> dict[str, Any]:
    gt_bbox = row["answer_bbox"]
    center = bbox_center(final_bbox)
    valid_evidence = {obs.get("evidence_id") for obs in tool_observations if obs.get("evidence_id")}
    missing = [evidence_id for evidence_id in evidence_ids if evidence_id not in valid_evidence]
    iou = bbox_iou(final_bbox, gt_bbox)
    return {
        "final_iou": round(float(iou), 6),
        "final_iou_50": iou >= 0.5,
        "final_pointing": point_in_bbox(center, gt_bbox),
        "evidence_closed": bool(evidence_ids) and not missing,
        "missing_evidence_ids": missing,
        "tool_success": all(bool(obs.get("ok")) for obs in tool_observations),
        "tool_call_count": len(tool_observations),
    }


def choose_template(row: dict[str, Any], args: argparse.Namespace) -> str:
    oracle_metrics = row.get("oracle_metrics") or {}
    oracle_rank = int(row.get("oracle_rank") or 9999)
    oracle_iou = float(oracle_metrics.get("iou") or 0.0)
    if not oracle_metrics.get("hit"):
        return "detect_answer_bbox"
    if oracle_iou < args.min_candidate_iou_for_candidate_answer:
        return "detect_refine_bbox"
    if args.include_crop_for_hard and oracle_rank > args.crop_hard_rank_threshold:
        return "detect_crop_select_candidate"
    return "detect_select_candidate"


def build_rl_row(row: dict[str, Any], args: argparse.Namespace, split: str) -> dict[str, Any]:
    template = choose_template(row, args)
    image_path = str(row["image_path"])
    instruction = str(row.get("instruction") or row.get("question") or "")
    oracle_id = str(row.get("oracle_candidate_id"))
    by_id = candidate_by_id(row)
    oracle_candidate = by_id.get(oracle_id)
    answer_bbox = [int(round(float(v))) for v in row["answer_bbox"]]
    oracle_bbox = [int(round(float(v))) for v in (oracle_candidate or {}).get("bbox", row.get("oracle_bbox") or answer_bbox)]

    user_prompt = (
        "<image>You are EviTool-VL, a GUI tool-use agent.\n"
        "Locate the UI target described by the instruction. Use visual tools, cite evidence ids, "
        "and return exactly one JSON object for each assistant turn.\n\n"
        f"Instruction: {instruction}\n"
        "Allowed tools: detect(mode, query, bbox), crop(bbox), click(bbox, label).\n"
        "For the final answer, return {\"reasoning\": [...], \"answer\": {\"type\": ..., \"bbox\": ...}}."
    )
    trajectory: list[dict[str, Any]] = [tool_step("user", user_prompt)]
    observations: list[dict[str, Any]] = []
    evidence_ids: list[str] = []
    quality_meta: dict[str, Any] = {"template": template}

    detect_action = assistant_action(
        "Find GUI candidate regions related to the instruction.",
        "detect",
        {"mode": "ui", "query": instruction, "max_results": args.max_detect_candidates},
    )
    trajectory.append(tool_step("assistant", detect_action))
    detect_obs, detect_meta = make_detect_observation(
        row,
        evidence_id="ev_001",
        max_candidates=args.max_detect_candidates,
    )
    trajectory.append(tool_step("tool", detect_obs))
    observations.append(detect_obs)
    quality_meta.update(detect_meta)

    final_bbox = oracle_bbox
    answer_type = "candidate"
    final_candidate_id: str | None = oracle_id
    supporting_candidate_id: str | None = None

    if template == "detect_crop_select_candidate":
        crop_bbox = clip_bbox_to_image(oracle_bbox, image_path, pad=8)
        crop_action = assistant_action(
            "Inspect the candidate region more closely before selecting it.",
            "crop",
            {"bbox": crop_bbox, "pad": 0},
        )
        trajectory.append(tool_step("assistant", crop_action))
        crop_obs = run_tool(
            image_path,
            {"tool": "crop", "args": {"bbox": crop_bbox, "pad": 0}},
            out_dir=args.tool_out_dir,
            evidence_id="ev_002",
        )
        trajectory.append(tool_step("tool", crop_obs))
        observations.append(crop_obs)
        evidence_ids.append("ev_002")
    elif template in {"detect_refine_bbox", "detect_answer_bbox"}:
        final_bbox = answer_bbox
        answer_type = "bbox_refined" if template == "detect_refine_bbox" else "bbox"
        final_candidate_id = None
        supporting_candidate_id = oracle_id if oracle_candidate is not None else None

    click_evidence_id = f"ev_{len(observations) + 1:03d}"
    click_bbox = final_bbox
    click_action = assistant_action(
        "Record the selected GUI target as visual evidence.",
        "click",
        {"bbox": click_bbox, "label": instruction},
    )
    trajectory.append(tool_step("assistant", click_action))
    click_obs = run_tool(
        image_path,
        {"tool": "click", "args": {"bbox": click_bbox, "label": instruction}},
        out_dir=args.tool_out_dir,
        evidence_id=click_evidence_id,
    )
    trajectory.append(tool_step("tool", click_obs))
    observations.append(click_obs)

    evidence_ids = ["ev_001"] + evidence_ids + [click_evidence_id]
    final = final_answer(
        answer_type=answer_type,
        bbox=final_bbox,
        evidence_ids=evidence_ids,
        candidate_id=final_candidate_id,
        supporting_candidate_id=supporting_candidate_id,
    )
    trajectory.append(tool_step("assistant", final))

    quality = compute_quality(row, final_bbox, evidence_ids, observations)
    selected_iou = bbox_iou(oracle_bbox, answer_bbox) if oracle_candidate is not None else None
    reward_signals = {
        "answer": round(0.7 * float(quality["final_pointing"]) + 0.3 * min(1.0, float(quality["final_iou"]) / 0.5), 6),
        "evidence": 1.0 if quality["evidence_closed"] else 0.0,
        "format": 1.0,
        "tool": 1.0 if quality["tool_success"] else 0.0,
    }
    reward_signals["total"] = round(
        0.45 * reward_signals["answer"]
        + 0.2 * reward_signals["evidence"]
        + 0.2 * reward_signals["format"]
        + 0.15 * reward_signals["tool"],
        6,
    )

    return {
        "id": f"{row['id']}_tooltraj",
        "source": "os_atlas_linux_2k_refined_full_aware",
        "split": split,
        "task_type": "gui_grounding",
        "image": row.get("image"),
        "image_path": image_path,
        "instruction": instruction,
        "question": row.get("question"),
        "answer_bbox": answer_bbox,
        "final_answer": final["answer"],
        "trajectory": trajectory,
        "tool_observations": observations,
        "reward_signals": reward_signals,
        "quality": {
            **quality,
            **quality_meta,
            "oracle_candidate_id": oracle_id,
            "oracle_rank": row.get("oracle_rank"),
            "oracle_iou": round(float((row.get("oracle_metrics") or {}).get("iou") or 0.0), 6),
            "selected_candidate_iou": round(float(selected_iou), 6) if selected_iou is not None else None,
            "candidate_count": row.get("candidate_count"),
            "final_answer_type": answer_type,
        },
        "metadata": {
            "candidate_meta": row.get("candidate_meta"),
            "oracle_metrics": row.get("oracle_metrics"),
            "full_aware_action_ids": row.get("full_aware_action_ids"),
            "full_aware_action_roles": row.get("full_aware_action_roles"),
        },
    }


def validate_rl_row(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not row.get("trajectory"):
        errors.append("empty_trajectory")
    if not row.get("quality", {}).get("evidence_closed"):
        errors.append("evidence_not_closed")
    if not row.get("quality", {}).get("tool_success"):
        errors.append("tool_failed")
    if not row.get("quality", {}).get("final_pointing"):
        errors.append("final_not_pointing")
    for turn in row.get("trajectory") or []:
        if turn.get("role") == "assistant":
            try:
                json.dumps(turn.get("content"), ensure_ascii=False)
            except TypeError:
                errors.append("assistant_not_json_serializable")
    return errors


def summarize(rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "splits": {},
        "overall": {},
    }
    all_rows = [row for rows in rows_by_split.values() for row in rows]
    for split, rows in rows_by_split.items():
        summary["splits"][split] = summarize_rows(rows)
    summary["overall"] = summarize_rows(all_rows)
    train_images = {row["image_path"] for row in rows_by_split.get("train", [])}
    val_images = {row["image_path"] for row in rows_by_split.get("val", [])}
    summary["split_check"] = {
        "train_image_count": len(train_images),
        "val_image_count": len(val_images),
        "train_val_image_overlap": len(train_images & val_images),
    }
    return summary


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    templates = Counter(row["quality"]["template"] for row in rows)
    final_types = Counter(row["quality"]["final_answer_type"] for row in rows)
    tool_counts = [row["quality"]["tool_call_count"] for row in rows]
    ious = [float(row["quality"]["final_iou"]) for row in rows]
    selected_ious = [
        float(row["quality"]["selected_candidate_iou"])
        for row in rows
        if row["quality"].get("selected_candidate_iou") is not None
    ]
    detect_counts = [int(row["quality"].get("detect_candidate_count") or 0) for row in rows]
    return {
        "count": len(rows),
        "template_counts": dict(templates),
        "final_answer_type_counts": dict(final_types),
        "schema_valid_rate": 1.0 - sum(bool(validate_rl_row(row)) for row in rows) / len(rows),
        "tool_success_rate": sum(bool(row["quality"]["tool_success"]) for row in rows) / len(rows),
        "evidence_closed_rate": sum(bool(row["quality"]["evidence_closed"]) for row in rows) / len(rows),
        "final_pointing_rate": sum(bool(row["quality"]["final_pointing"]) for row in rows) / len(rows),
        "final_iou_50_rate": sum(bool(row["quality"]["final_iou_50"]) for row in rows) / len(rows),
        "avg_final_iou": statistics.mean(ious),
        "median_final_iou": statistics.median(ious),
        "avg_selected_candidate_iou": statistics.mean(selected_ious) if selected_ious else None,
        "avg_tool_calls": statistics.mean(tool_counts),
        "avg_detect_candidates": statistics.mean(detect_counts),
        "oracle_in_detect_topk_rate": sum(bool(row["quality"].get("oracle_in_detect_topk")) for row in rows) / len(rows),
        "oracle_injected_into_detect_rate": sum(bool(row["quality"].get("oracle_injected_into_detect")) for row in rows) / len(rows),
        "avg_reward_total": statistics.mean(float(row["reward_signals"]["total"]) for row in rows),
    }


def make_report(summary: dict[str, Any], args: argparse.Namespace, output_dir: Path, sft_output_dir: Path) -> str:
    lines = [
        "# GUI Tool-Call Trajectory 数据集构建报告",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M CST')}",
        f"- 输入 full-aware 数据：`{Path(args.input_dir).resolve()}`",
        f"- RL/TRL 输出目录：`{output_dir}`",
        f"- SFT 输出目录：`{sft_output_dir}`",
        f"- max_detect_candidates：{args.max_detect_candidates}",
        f"- drop_oracle_injected：{args.drop_oracle_injected}",
        f"- candidate_only：{args.candidate_only}",
        "",
        "## 总览",
        "",
        "| split | rows | final pointing | IoU@0.5 | evidence closed | tool success | avg tool calls | oracle injected |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ["train", "val", "overall"]:
        item = summary["overall"] if split == "overall" else summary["splits"].get(split, {"count": 0})
        if not item.get("count"):
            continue
        lines.append(
            "| {split} | {count} | {point:.2%} | {iou50:.2%} | {ev:.2%} | {tool:.2%} | {calls:.2f} | {inj:.2%} |".format(
                split=split,
                count=item["count"],
                point=item["final_pointing_rate"],
                iou50=item["final_iou_50_rate"],
                ev=item["evidence_closed_rate"],
                tool=item["tool_success_rate"],
                calls=item["avg_tool_calls"],
                inj=item["oracle_injected_into_detect_rate"],
            )
        )
    lines.extend(
        [
            "",
            "## 模板分布",
            "",
        ]
    )
    for split in ["train", "val", "overall"]:
        item = summary["overall"] if split == "overall" else summary["splits"].get(split, {"count": 0})
        lines.append(f"- `{split}`: `{item.get('template_counts', {})}`")
    lines.extend(["", "## 过滤统计", ""])
    skipped_counts = summary.get("skipped_counts") or {}
    if skipped_counts:
        for split in ["train", "val"]:
            lines.append(f"- `{split}` skipped: `{skipped_counts.get(split, {})}`")
    else:
        lines.append("- 未启用样本过滤。")
    lines.extend(
        [
            "",
            "## 输出文件",
            "",
            f"- RL train：`{output_dir / 'train_rl.jsonl'}`",
            f"- RL val：`{output_dir / 'val_rl.jsonl'}`",
            f"- SFT train：`{sft_output_dir / 'gui_tool_trajectory_train_sft.json'}`",
            f"- SFT val：`{sft_output_dir / 'gui_tool_trajectory_val_sft.json'}`",
            f"- LLaMA-Factory dataset_info：`{sft_output_dir / 'dataset_info.json'}`",
            "",
            "## 质量结论",
            "",
            "- 该数据集是 oracle/silver trajectory，不是 current-model rollout。",
            "- `detect` observation 来自 frozen refined full-aware candidate pool，目的是保持候选 id 稳定。",
            "- `click` 和 hard 样本里的 `crop` 由本地工具实际执行，final answer 引用 evidence id。",
            "- 后续 RL 阶段应把这些 traces 作为 warmup/reference，再用当前模型在线 rollout 生成真实行为分布。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M")
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir or f"outputs/gui_tool_trajectory_os_atlas_linux_2k_refined_{timestamp}")
    sft_output_dir = Path(args.sft_output_dir or f"/root/models/datasets/gui_tool_trajectory_os_atlas_linux_2k_refined_{timestamp}")
    doc_report = Path(
        args.doc_report
        or f"/root/Workspace/VLM/项目文档/02_指标与数据/GUI工具调用轨迹数据集构建报告_{timestamp}.md"
    )
    args.tool_out_dir = str(Path(args.tool_out_dir) / timestamp)

    split_limits = {"train": args.max_rows_train, "val": args.max_rows_val}
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    sft_by_split: dict[str, list[dict[str, Any]]] = {}
    errors_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped_by_split: dict[str, Counter[str]] = defaultdict(Counter)

    for split in ["train", "val"]:
        source_rows = read_jsonl(input_dir / f"{split}.jsonl", limit=split_limits[split])
        rl_rows: list[dict[str, Any]] = []
        sft_rows: list[dict[str, Any]] = []
        for index, source_row in enumerate(source_rows):
            rl_row = build_rl_row(source_row, args, split)
            skip_reasons: list[str] = []
            if args.drop_oracle_injected and rl_row["quality"].get("oracle_injected_into_detect"):
                skip_reasons.append("oracle_injected")
            if args.candidate_only and rl_row["quality"].get("final_answer_type") != "candidate":
                skip_reasons.append("non_candidate_final")
            if skip_reasons:
                skipped_by_split[split].update(skip_reasons)
                skipped_by_split[split]["total_skipped"] += 1
                continue
            errors = validate_rl_row(rl_row)
            if errors:
                errors_by_split[split].append({"id": rl_row["id"], "errors": errors})
            rl_rows.append(rl_row)
            sft_rows.append(
                {
                    "messages": to_sft_messages(rl_row),
                    "images": [rl_row["image_path"]],
                    "meta": {
                        "id": rl_row["id"],
                        "split": split,
                        "template": rl_row["quality"]["template"],
                        "final_answer_type": rl_row["quality"]["final_answer_type"],
                        "reward_total": rl_row["reward_signals"]["total"],
                    },
                }
            )
        rows_by_split[split] = rl_rows
        sft_by_split[split] = sft_rows

    summary = summarize(rows_by_split)
    summary["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")
    summary["input_dir"] = str(input_dir.resolve())
    summary["output_dir"] = str(output_dir.resolve())
    summary["sft_output_dir"] = str(sft_output_dir.resolve())
    summary["validation_errors"] = {split: values[:20] for split, values in errors_by_split.items()}
    summary["validation_error_counts"] = {split: len(values) for split, values in errors_by_split.items()}
    summary["filters"] = {
        "drop_oracle_injected": args.drop_oracle_injected,
        "candidate_only": args.candidate_only,
    }
    summary["skipped_counts"] = {split: dict(counter) for split, counter in skipped_by_split.items()}

    write_jsonl(output_dir / "train_rl.jsonl", rows_by_split["train"])
    write_jsonl(output_dir / "val_rl.jsonl", rows_by_split["val"])
    write_json(output_dir / "summary.json", summary)
    write_json(sft_output_dir / "gui_tool_trajectory_train_sft.json", sft_by_split["train"])
    write_json(sft_output_dir / "gui_tool_trajectory_val_sft.json", sft_by_split["val"])
    write_json(
        sft_output_dir / "dataset_info.json",
        {
            "gui_tool_trajectory_train_sft": {
                "file_name": "gui_tool_trajectory_train_sft.json",
                "formatting": "sharegpt",
                "columns": {"messages": "messages", "images": "images"},
                "tags": {
                    "role_tag": "role",
                    "content_tag": "content",
                    "user_tag": "user",
                    "assistant_tag": "assistant",
                },
            },
            "gui_tool_trajectory_val_sft": {
                "file_name": "gui_tool_trajectory_val_sft.json",
                "formatting": "sharegpt",
                "columns": {"messages": "messages", "images": "images"},
                "tags": {
                    "role_tag": "role",
                    "content_tag": "content",
                    "user_tag": "user",
                    "assistant_tag": "assistant",
                },
            },
        },
    )
    write_json(sft_output_dir / "summary.json", summary)
    report = make_report(summary, args, output_dir.resolve(), sft_output_dir.resolve())
    (output_dir / "DATASET_REPORT.md").write_text(report, encoding="utf-8")
    doc_report.parent.mkdir(parents=True, exist_ok=True)
    doc_report.write_text(report, encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
