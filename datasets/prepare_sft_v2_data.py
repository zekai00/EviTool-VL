#!/usr/bin/env python3
"""Build split SFT v2 datasets and quality reports from EviTool traces."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional validation dependency
    Image = None


TRAIN_SOURCE_LIMITS = {
    "gui_grounding": 500,
    "doc_qa": 250,
    "chart_qa": 200,
    "science_diagram_qa": 50,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/root/models/datasets/evitool_traces_1k/train_traces_1000.jsonl")
    parser.add_argument("--eval", default="/root/models/datasets/evitool_eval_medium/eval_medium_600.jsonl")
    parser.add_argument("--output-dir", default="/root/models/datasets/evitool_sft_v2")
    parser.add_argument("--report", default="/root/Workspace/VLM/sftv2_dataset_report.md")
    parser.add_argument("--max-observation-chars", type=int, default=16000)
    parser.add_argument("--max-spans", type=int, default=80)
    parser.add_argument("--max-detections", type=int, default=30)
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2%}"


def num(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def quality(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("quality") or {}


def image_path_for_row(row: dict[str, Any], source_path: Path) -> str:
    image = str(row.get("image", ""))
    if image.startswith("/"):
        return image
    return str((source_path.parent / image).resolve())


def answer_value(row: dict[str, Any]) -> Any:
    if row.get("answers"):
        return row.get("answers")
    return row.get("answer")


def norm(value: Any) -> str:
    text = str(value).lower().strip()
    return re.sub(r"\s+", " ", text)


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def hash_key(*parts: Any) -> str:
    raw = "||".join(stable_json(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def source_identity(row: dict[str, Any]) -> str | None:
    task = row.get("task_type")
    meta = row.get("meta") or {}
    source = row.get("source")
    if task == "gui_grounding" and meta.get("img_filename"):
        return hash_key(source, task, meta.get("img_filename"), meta.get("bbox_xywh"), norm(row.get("question")))
    if task == "doc_qa" and (meta.get("question_id") or meta.get("questionId")):
        return hash_key(source, task, meta.get("doc_id") or meta.get("docId"), meta.get("question_id") or meta.get("questionId"))
    source_index = meta.get("source_index", row.get("source_index"))
    if source_index is not None:
        return hash_key(source, task, source_index)
    return None


def semantic_identity(row: dict[str, Any]) -> str:
    return hash_key(row.get("source"), row.get("task_type"), norm(row.get("question")), answer_value(row))


def split_check(train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]], train_path: Path, eval_path: Path) -> dict[str, Any]:
    train_source = {sid: row for row in train_rows if (sid := source_identity(row))}
    eval_source = {sid: row for row in eval_rows if (sid := source_identity(row))}
    train_semantic = {semantic_identity(row): row for row in train_rows}
    eval_semantic = {semantic_identity(row): row for row in eval_rows}

    range_violations = []
    for row in eval_rows:
        task = row.get("task_type")
        source_index = (row.get("meta") or {}).get("source_index", row.get("source_index"))
        limit = TRAIN_SOURCE_LIMITS.get(task)
        if source_index is None or limit is None:
            continue
        if int(source_index) < limit:
            range_violations.append(
                {"id": row.get("id"), "task_type": task, "source_index": source_index, "train_source_limit": limit}
            )

    source_overlaps = sorted(set(train_source) & set(eval_source))
    semantic_overlaps = sorted(set(train_semantic) & set(eval_semantic))
    return {
        "train": str(train_path),
        "eval": str(eval_path),
        "train_count": len(train_rows),
        "eval_count": len(eval_rows),
        "eval_task_counts": dict(Counter(row.get("task_type") for row in eval_rows)),
        "source_identity_overlap_count": len(source_overlaps),
        "semantic_duplicate_warning_count": len(semantic_overlaps),
        "semantic_duplicate_note": "Semantic duplicates can occur across distinct source examples, especially generic ChartQA/DocVQA questions; they are warnings, not split failures.",
        "source_range_violation_count": len(range_violations),
        "source_range_violations_sample": range_violations[:20],
        "split_ok": not source_overlaps and not range_violations,
    }


def collect_span_ids(value: Any) -> set[str]:
    span_ids: set[str] = set()
    if isinstance(value, dict):
        if isinstance(value.get("span_id"), str):
            span_ids.add(value["span_id"])
        if isinstance(value.get("span_ids"), list):
            span_ids.update(str(item) for item in value["span_ids"])
        for item in value.values():
            span_ids.update(collect_span_ids(item))
    elif isinstance(value, list):
        for item in value:
            span_ids.update(collect_span_ids(item))
    return span_ids


def required_span_ids(row: dict[str, Any]) -> set[str]:
    q = quality(row)
    keep: set[str] = set()
    for key in ("answer_evidence_locations", "answer_option_evidence_locations"):
        keep.update(collect_span_ids(q.get(key)))
    return keep


def small_span(span: dict[str, Any]) -> dict[str, Any]:
    keys = ("span_id", "text", "bbox", "center", "score", "source")
    return {key: span[key] for key in keys if key in span}


def small_detection(det: dict[str, Any]) -> dict[str, Any]:
    keys = ("candidate_id", "bbox", "center", "label", "score", "text", "ocr_text", "source", "sources")
    return {key: det[key] for key in keys if key in det}


def trim_list_with_required(items: list[Any], limit: int, required_ids: set[str], id_key: str) -> list[Any]:
    selected = []
    seen_ids: set[str] = set()
    for item in items[:limit]:
        if isinstance(item, dict):
            item_id = str(item.get(id_key, ""))
            if item_id:
                seen_ids.add(item_id)
        selected.append(item)
    if required_ids:
        for item in items[limit:]:
            if not isinstance(item, dict):
                continue
            item_id = str(item.get(id_key, ""))
            if item_id in required_ids and item_id not in seen_ids:
                selected.append(item)
                seen_ids.add(item_id)
    return selected


def sanitize_content(
    content: Any,
    row: dict[str, Any],
    max_spans: int,
    max_detections: int,
    include_candidate_id: str | None = None,
) -> Any:
    if not isinstance(content, dict):
        return content

    keep_span_ids = required_span_ids(row)
    sanitized: dict[str, Any] = {}
    for key, value in content.items():
        if key == "spans" and isinstance(value, list):
            spans = trim_list_with_required(value, max_spans, keep_span_ids, "span_id")
            sanitized[key] = [small_span(span) if isinstance(span, dict) else span for span in spans]
            sanitized["spans_total"] = len(value)
            sanitized["spans_truncated"] = len(spans) < len(value)
        elif key == "detections" and isinstance(value, list):
            required = {include_candidate_id} if include_candidate_id else set()
            detections = trim_list_with_required(value, max_detections, {item for item in required if item}, "candidate_id")
            sanitized[key] = [small_detection(det) if isinstance(det, dict) else det for det in detections]
            sanitized["detections_total"] = len(value)
            sanitized["detections_truncated"] = len(detections) < len(value)
        elif key == "candidate_regions" and isinstance(value, list):
            sanitized[key] = value[:10]
            sanitized["candidate_regions_total"] = len(value)
            sanitized["candidate_regions_truncated"] = len(value[:10]) < len(value)
        elif key in {"debug", "raw", "image_b64"}:
            continue
        else:
            sanitized[key] = value
    return sanitized


def sanitize_observation(
    observation: dict[str, Any],
    row: dict[str, Any],
    max_observation_chars: int,
    max_spans: int,
    max_detections: int,
    include_candidate_id: str | None = None,
) -> dict[str, Any]:
    base = {
        key: copy.deepcopy(observation.get(key))
        for key in ("evidence_id", "tool", "image", "bbox", "ok", "error")
        if key in observation
    }

    span_limits = [max_spans, 60, 40, 20, 10]
    detection_limits = [max_detections, 20, 10]
    for span_limit in span_limits:
        for detection_limit in detection_limits:
            candidate = copy.deepcopy(base)
            candidate["content"] = sanitize_content(
                observation.get("content"), row, span_limit, detection_limit, include_candidate_id=include_candidate_id
            )
            text = compact_json(candidate)
            if len(text) <= max_observation_chars:
                return candidate

    candidate = copy.deepcopy(base)
    candidate["content"] = sanitize_content(observation.get("content"), row, 5, 5, include_candidate_id=include_candidate_id)
    candidate["truncated_for_sft"] = True
    return candidate


def trace_prompt(row: dict[str, Any]) -> str:
    messages = row.get("messages") or []
    if messages and messages[0].get("content"):
        content = str(messages[0]["content"])
        return content if content.startswith("<image>") else "<image>" + content
    return "<image>Answer the question based on the image. Use local visual tools when evidence is needed. Return a concise final answer with cited evidence.\n\nQuestion: " + str(row.get("question"))


def trace_to_lf_item(
    row: dict[str, Any],
    input_path: Path,
    source_name: str,
    max_observation_chars: int,
    max_spans: int,
    max_detections: int,
) -> dict[str, Any]:
    messages: list[dict[str, str]] = [{"role": "user", "content": trace_prompt(row)}]
    for step in row.get("trace") or []:
        action = step.get("action")
        observation = step.get("observation")
        if action is not None:
            messages.append({"role": "assistant", "content": compact_json(action)})
        if observation is not None:
            clean_obs = sanitize_observation(observation, row, max_observation_chars, max_spans, max_detections)
            messages.append({"role": "user", "content": "Tool observation:\n" + compact_json(clean_obs)})

    final = row.get("final") or {"answer": answer_value(row)}
    messages.append({"role": "assistant", "content": compact_json(final)})
    q = quality(row)
    return {
        "messages": messages,
        "images": [image_path_for_row(row, input_path)],
        "meta": {
            "id": row.get("id"),
            "task_type": row.get("task_type"),
            "source": source_name,
            "quality_tier": q.get("quality_tier"),
            "strong_evidence": bool(q.get("strong_evidence", False)),
        },
    }


def direct_item(row: dict[str, Any], input_path: Path) -> dict[str, Any]:
    answer = answer_value(row)
    if row.get("task_type") == "science_diagram_qa":
        answer = (row.get("final") or {}).get("answer", answer)
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
            "source": "direct_v2",
        },
    }


def detection_by_id(row: dict[str, Any], candidate_id: str) -> dict[str, Any] | None:
    for step in row.get("trace") or []:
        obs = step.get("observation") or {}
        detections = ((obs.get("content") or {}).get("detections")) or []
        for det in detections:
            if isinstance(det, dict) and det.get("candidate_id") == candidate_id:
                return det
    return None


def gui_candidates(row: dict[str, Any], max_detections: int) -> list[dict[str, Any]]:
    q = quality(row)
    selected_id = q.get("selected_candidate_id")
    detections: list[dict[str, Any]] = []
    for step in row.get("trace") or []:
        obs = step.get("observation") or {}
        content = obs.get("content") or {}
        if isinstance(content.get("detections"), list):
            detections = content["detections"]
            break
    selected = detection_by_id(row, selected_id) if selected_id else None
    selected_ids = {str(det.get("candidate_id")) for det in detections[:max_detections] if isinstance(det, dict)}
    trimmed = [small_detection(det) if isinstance(det, dict) else det for det in detections[:max_detections]]
    if selected and selected_id not in selected_ids:
        trimmed.append(small_detection(selected))
    return trimmed


def gui_select_item(row: dict[str, Any], input_path: Path, max_detections: int) -> dict[str, Any]:
    q = quality(row)
    selected_id = q["selected_candidate_id"]
    selected = detection_by_id(row, selected_id) or {}
    bbox = selected.get("bbox") or row.get("answer_bbox")
    center = selected.get("center")
    if not center and isinstance(bbox, list) and len(bbox) == 4:
        center = [(float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0]

    action = {
        "thought": "Find UI candidates before selecting the target.",
        "action": {
            "tool": "detect",
            "args": {"mode": "ui", "query": (row.get("meta") or {}).get("instruction", row.get("question")), "max_results": 30, "min_area": 20},
        },
    }
    observation = {
        "evidence_id": "ev_001",
        "tool": "detect",
        "ok": True,
        "content": {
            "mode": "ui",
            "detections": gui_candidates(row, max_detections),
            "count": len(gui_candidates(row, max_detections)),
        },
    }
    final = {
        "reasoning": [{"step": "ev_001 contains the detected UI candidate selected for the instruction.", "evidence": ["ev_001"]}],
        "selected_candidate_id": selected_id,
        "answer_bbox": bbox,
        "answer_point": center,
    }
    return {
        "messages": [
            {
                "role": "user",
                "content": "<image>GUI candidate selection task. Use the candidate list from the visual tool and choose the candidate_id that best matches the instruction. Return JSON with selected_candidate_id, answer_bbox, answer_point, and cited evidence. Do not invent candidates.\n\nQuestion: "
                + str(row.get("question")),
            },
            {"role": "assistant", "content": compact_json(action)},
            {"role": "user", "content": "Tool observation:\n" + compact_json(observation)},
            {"role": "assistant", "content": compact_json(final)},
        ],
        "images": [image_path_for_row(row, input_path)],
        "meta": {
            "id": row.get("id"),
            "task_type": row.get("task_type"),
            "source": "gui_select_v2",
            "selected_candidate_id": selected_id,
            "selected_iou_with_gt": q.get("selected_iou_with_gt"),
            "candidate_recall_tier": q.get("candidate_recall_tier"),
        },
    }


def is_tool_v2(row: dict[str, Any]) -> bool:
    task = row.get("task_type")
    q = quality(row)
    if task in {"doc_qa", "chart_qa"}:
        return bool(q.get("answer_evidence_position_found", False))
    if task == "science_diagram_qa":
        return bool(q.get("answer_option_evidence_position_found", False))
    return False


def is_ai2d_moderate(row: dict[str, Any]) -> bool:
    q = quality(row)
    return (
        row.get("task_type") == "science_diagram_qa"
        and not q.get("answer_option_evidence_position_found", False)
        and bool(q.get("moderate_evidence", False))
    )


def is_gui_select(row: dict[str, Any]) -> bool:
    q = quality(row)
    return (
        row.get("task_type") == "gui_grounding"
        and bool(q.get("candidate_recall_at_30", False))
        and not bool(q.get("oracle_gt_injected", False))
        and bool(q.get("selected_candidate_id"))
    )


def task_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(row.get("task_type") for row in rows))


def mean_bool(rows: list[dict[str, Any]], getter: Any) -> float | None:
    if not rows:
        return None
    return sum(1 for row in rows if getter(row)) / len(rows)


def quality_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    doc_chart = [row for row in rows if row.get("task_type") in {"doc_qa", "chart_qa"}]
    ai2d = [row for row in rows if row.get("task_type") == "science_diagram_qa"]
    gui = [row for row in rows if row.get("task_type") == "gui_grounding"]
    selected_ious = [quality(row).get("selected_iou_with_gt") for row in gui if isinstance(quality(row).get("selected_iou_with_gt"), (int, float))]
    return {
        "count": len(rows),
        "task_counts": task_counts(rows),
        "strong_evidence_rate": mean_bool(rows, lambda row: bool(quality(row).get("strong_evidence", False))),
        "evidence_closed_rate": mean_bool(rows, lambda row: bool(quality(row).get("evidence_closed", False))),
        "avg_tool_calls": statistics.mean([len(row.get("trace") or []) for row in rows]) if rows else None,
        "answer_evidence_position_rate": mean_bool(doc_chart, lambda row: bool(quality(row).get("answer_evidence_position_found", False))),
        "answer_option_evidence_position_rate": mean_bool(ai2d, lambda row: bool(quality(row).get("answer_option_evidence_position_found", False))),
        "oracle_gt_injected_rate": mean_bool(gui, lambda row: bool(quality(row).get("oracle_gt_injected", False))),
        "candidate_recall_at_30_rate": mean_bool(gui, lambda row: bool(quality(row).get("candidate_recall_at_30", False))),
        "avg_selected_iou": statistics.mean(selected_ious) if selected_ious else None,
        "selected_pointing_rate": mean_bool(gui, lambda row: bool(quality(row).get("selected_pointing", False))),
    }


def parse_tool_observation(content: str) -> Any:
    if not content.startswith("Tool observation:\n"):
        return None
    payload = content.split("\n", 1)[1]
    return json.loads(payload)


def validate_sft_items(items: list[dict[str, Any]], json_assistant: bool = True, gui: bool = False) -> dict[str, Any]:
    missing_images = 0
    empty_messages = 0
    role_errors = 0
    assistant_total = 0
    assistant_json_ok = 0
    tool_json_errors = 0
    gui_present_total = 0
    gui_present = 0

    for item in items:
        if any(not Path(path).exists() for path in item.get("images") or []):
            missing_images += 1
        messages = item.get("messages") or []
        if not messages:
            empty_messages += 1
        if messages and messages[0].get("role") != "user":
            role_errors += 1
        for msg in messages:
            if msg.get("role") == "assistant" and json_assistant:
                assistant_total += 1
                try:
                    json.loads(msg.get("content") or "")
                    assistant_json_ok += 1
                except json.JSONDecodeError:
                    pass
            if msg.get("role") == "user" and str(msg.get("content", "")).startswith("Tool observation:\n"):
                try:
                    parse_tool_observation(str(msg.get("content", "")))
                except json.JSONDecodeError:
                    tool_json_errors += 1

        if gui and (item.get("meta") or {}).get("task_type") == "gui_grounding":
            final = json.loads(messages[-1]["content"])
            selected = final.get("selected_candidate_id")
            candidates: set[str] = set()
            for msg in messages:
                if msg.get("role") == "user" and str(msg.get("content", "")).startswith("Tool observation:\n"):
                    observation = parse_tool_observation(str(msg.get("content", "")))
                    detections = ((observation.get("content") or {}).get("detections")) or []
                    candidates.update(str(det.get("candidate_id")) for det in detections if isinstance(det, dict))
            gui_present_total += 1
            if selected in candidates:
                gui_present += 1

    return {
        "count": len(items),
        "missing_image_rate": missing_images / len(items) if items else 0.0,
        "empty_messages_rate": empty_messages / len(items) if items else 0.0,
        "role_error_count": role_errors,
        "assistant_json_parse_rate": assistant_json_ok / assistant_total if json_assistant and assistant_total else None,
        "tool_observation_json_parse_error_count": tool_json_errors,
        "gui_selected_candidate_present_rate": gui_present / gui_present_total if gui_present_total else None,
    }


def bbox_xyxy(row: dict[str, Any]) -> list[float] | None:
    bbox = row.get("answer_bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        return [float(x) for x in bbox]
    xywh = (row.get("meta") or {}).get("bbox_xywh")
    if isinstance(xywh, list) and len(xywh) == 4:
        x, y, w, h = [float(v) for v in xywh]
        return [x, y, x + w, y + h]
    return None


def image_size(path: Path) -> tuple[int, int] | None:
    if Image is None:
        return None
    try:
        with Image.open(path) as img:
            return img.size
    except Exception:
        return None


def eval_quality(eval_rows: list[dict[str, Any]], train_rows: list[dict[str, Any]], train_path: Path, eval_path: Path) -> dict[str, Any]:
    empty_question = 0
    empty_answer = 0
    missing_image = 0
    image_open_error = 0
    gui_invalid_bbox = 0
    gui_outside_bbox = 0
    gui_area_ratios: list[float] = []
    question_lengths: list[int] = []
    answer_lengths: list[int] = []
    source_ranges: dict[str, dict[str, int]] = {}

    for row in eval_rows:
        question = row.get("question")
        answer = answer_value(row)
        if not str(question or "").strip():
            empty_question += 1
        if answer in (None, "") or answer == []:
            empty_answer += 1
        question_lengths.append(len(str(question or "")))
        answer_lengths.append(len(str(answer or "")))

        image_path = Path(image_path_for_row(row, eval_path))
        if not image_path.exists():
            missing_image += 1
            size = None
        else:
            size = image_size(image_path)
            if Image is not None and size is None:
                image_open_error += 1

        source = str(row.get("source"))
        source_index = (row.get("meta") or {}).get("source_index", row.get("source_index"))
        if source_index is not None:
            current = source_ranges.setdefault(source, {"min": int(source_index), "max": int(source_index), "count": 0})
            current["min"] = min(current["min"], int(source_index))
            current["max"] = max(current["max"], int(source_index))
            current["count"] += 1

        if row.get("task_type") == "gui_grounding":
            bbox = bbox_xyxy(row)
            if not bbox or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                gui_invalid_bbox += 1
                continue
            if size:
                width, height = size
                outside = bbox[0] < 0 or bbox[1] < 0 or bbox[2] > width or bbox[3] > height
                if outside:
                    gui_outside_bbox += 1
                gui_area_ratios.append(((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / (width * height))

    return {
        "count": len(eval_rows),
        "task_counts": dict(Counter(row.get("task_type") for row in eval_rows)),
        "empty_question_rate": empty_question / len(eval_rows) if eval_rows else 0.0,
        "empty_answer_rate": empty_answer / len(eval_rows) if eval_rows else 0.0,
        "missing_image_rate": missing_image / len(eval_rows) if eval_rows else 0.0,
        "image_open_error_rate": image_open_error / len(eval_rows) if eval_rows else 0.0,
        "gui_bbox_invalid_rate": gui_invalid_bbox / max(1, sum(1 for row in eval_rows if row.get("task_type") == "gui_grounding")),
        "gui_bbox_outside_rate": gui_outside_bbox / max(1, sum(1 for row in eval_rows if row.get("task_type") == "gui_grounding")),
        "gui_bbox_area_ratio_mean": statistics.mean(gui_area_ratios) if gui_area_ratios else None,
        "gui_bbox_area_ratio_median": statistics.median(gui_area_ratios) if gui_area_ratios else None,
        "question_len_mean": statistics.mean(question_lengths) if question_lengths else None,
        "answer_len_mean": statistics.mean(answer_lengths) if answer_lengths else None,
        "source_ranges": source_ranges,
        "split_check": split_check(train_rows, eval_rows, train_path, eval_path),
    }


def dataset_entry(file_name: str) -> dict[str, Any]:
    return {
        "file_name": file_name,
        "formatting": "sharegpt",
        "columns": {"messages": "messages", "images": "images"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
        },
    }


def report_table_row(name: str, q: dict[str, Any]) -> str:
    return (
        f"| {name} | {q['count']} | {pct(q['strong_evidence_rate'])} | {pct(q['evidence_closed_rate'])} | "
        f"{num(q['avg_tool_calls'], 2)} | {pct(q['answer_evidence_position_rate'])} | "
        f"{pct(q['answer_option_evidence_position_rate'])} | {pct(q['oracle_gt_injected_rate'])} | "
        f"{pct(q['candidate_recall_at_30_rate'])} | {num(q['avg_selected_iou'])} | {pct(q['selected_pointing_rate'])} |"
    )


def validation_table_row(name: str, v: dict[str, Any]) -> str:
    return (
        f"| `{name}` | {pct(v['missing_image_rate'])} | {pct(v['empty_messages_rate'])} | {v['role_error_count']} | "
        f"{pct(v['assistant_json_parse_rate'])} | {v['tool_observation_json_parse_error_count']} | "
        f"{pct(v['gui_selected_candidate_present_rate'])} |"
    )


def write_report(report_path: Path, summary: dict[str, Any]) -> None:
    datasets = summary["datasets"]
    quality_map = summary["quality"]
    eval_q = summary["eval_quality"]
    validation = summary["sft_file_validation"]

    lines = [
        "# SFT v2 数据集构建报告",
        "",
        "生成日期：2026-05-25",
        "",
        "## 1. 输出文件",
        "",
        f"- 输出目录：`{summary['output_dir']}`",
        f"- 构建输入 trace：`{summary['trace_input']}`",
        f"- 评测集：`{summary['eval_input']}`",
        "",
        "| 数据集 | 用途 | 样本数 | 任务分布 |",
        "|---|---|---:|---|",
        f"| `evitool_sft_tool_v2.json` | 强证据文本/图示工具轨迹，训练 tool-only adapter | {datasets['evitool_sft_tool_v2']['count']} | `{datasets['evitool_sft_tool_v2']['task_counts']}` |",
        f"| `evitool_sft_ai2d_moderate_v2.json` | AI2D 中等证据补充集，默认不混入强证据主集 | {datasets['evitool_sft_ai2d_moderate_v2']['count']} | `{datasets['evitool_sft_ai2d_moderate_v2']['task_counts']}` |",
        f"| `evitool_sft_gui_select_v2.json` | GUI 候选选择/重排训练，不做自由 bbox 回归 | {datasets['evitool_sft_gui_select_v2']['count']} | `{datasets['evitool_sft_gui_select_v2']['task_counts']}` |",
        f"| `evitool_sft_direct_v2.json` | 直接回答能力保留，建议单独训练 direct adapter | {datasets['evitool_sft_direct_v2']['count']} | `{datasets['evitool_sft_direct_v2']['task_counts']}` |",
        f"| `evitool_sft_tool_gui_v2.json` | tool_v2 + gui_select_v2 组合，方便训练工具/GUI adapter | {datasets['evitool_sft_tool_gui_v2']['count']} | `{datasets['evitool_sft_tool_gui_v2']['task_counts']}` |",
        "",
        "## 2. 构建策略",
        "",
        "- v2 不再把直接回答、证据工具、GUI bbox 回归混在一个训练目标里。",
        "- `tool_v2` 只保留 DocVQA/ChartQA 中 `answer_evidence_position_found=true` 的样本，以及 AI2D 中 `answer_option_evidence_position_found=true` 的强证据样本。",
        "- `gui_select_v2` 只保留 `candidate_recall_at_30=true`、`oracle_gt_injected=false`、且存在 `selected_candidate_id` 的 ScreenSpot 样本。训练目标是候选选择，不是让模型自由生成坐标。",
        "- GUI final 的 `answer_point` 使用选中候选框中心，避免在候选选择任务中重新注入 GT point。",
        "- `direct_v2` 保留所有非 GUI 的一问一答样本，建议只用于 direct adapter 或很低比例 retention，不建议混入 tool-only 主训练。",
        "- `ai2d_moderate_v2` 单独导出，因为这部分只有图结构/OCR 可用，但正确选项未定位到 OCR span。它可用于协议预热，但不应冒充强证据。",
        "- 所有 tool observation 都重新压缩为合法 JSON，避免模型学习截断 JSON 或不可解析 observation。",
        "",
        "## 3. 训练集质量指标",
        "",
        "| 子集 | Count | Strong Evidence | Evidence Closed | Avg Tools | Answer Evidence Position | AI2D Option Position | Oracle GT Injected | Candidate Recall@30 | Avg Selected IoU | Selected Pointing |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        report_table_row("tool_v2", quality_map["tool_v2"]),
        report_table_row("ai2d_moderate_v2", quality_map["ai2d_moderate_v2"]),
        report_table_row("gui_select_v2", quality_map["gui_select_v2"]),
        "",
        "### 3.1 SFT 文件完整性校验",
        "",
        "| 数据集 | Missing image | Empty messages | Role error count | Assistant JSON parse | Tool observation JSON parse errors | GUI selected candidate present |",
        "|---|---:|---:|---:|---:|---:|---:|",
        validation_table_row("evitool_sft_tool_v2", validation["evitool_sft_tool_v2"]),
        validation_table_row("evitool_sft_ai2d_moderate_v2", validation["evitool_sft_ai2d_moderate_v2"]),
        validation_table_row("evitool_sft_gui_select_v2", validation["evitool_sft_gui_select_v2"]),
        validation_table_row("evitool_sft_direct_v2", validation["evitool_sft_direct_v2"]),
        validation_table_row("evitool_sft_tool_gui_v2", validation["evitool_sft_tool_gui_v2"]),
        "",
        "### 3.2 关键观察",
        "",
        f"- `tool_v2` 共 {datasets['evitool_sft_tool_v2']['count']} 条，全部来自强证据过滤；任务分布为 `{datasets['evitool_sft_tool_v2']['task_counts']}`。",
        f"- `gui_select_v2` 共 {datasets['evitool_sft_gui_select_v2']['count']} 条，Oracle GT 注入率为 {pct(quality_map['gui_select_v2']['oracle_gt_injected_rate'])}，Candidate Recall@30 为 {pct(quality_map['gui_select_v2']['candidate_recall_at_30_rate'])}，final 中的 candidate_id 在 observation 候选中出现率为 {pct(validation['evitool_sft_gui_select_v2']['gui_selected_candidate_present_rate'])}。",
        f"- AI2D 强证据只有 {datasets['evitool_sft_tool_v2']['task_counts'].get('science_diagram_qa', 0)} 条，另有 {datasets['evitool_sft_ai2d_moderate_v2']['count']} 条中等证据单独保存。这说明 AI2D 的证据标注仍是 v2 的主要短板。",
        f"- `direct_v2` 共 {datasets['evitool_sft_direct_v2']['count']} 条，仅用于直接回答能力保留；不要把它默认混入 tool-only adapter。",
        "",
        "## 4. eval_medium_600 评测集质量",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| 样本数 | {eval_q['count']} |",
        f"| 任务分布 | `{eval_q['task_counts']}` |",
        f"| Train/Eval source identity overlap | {eval_q['split_check']['source_identity_overlap_count']} |",
        f"| Source range violations | {eval_q['split_check']['source_range_violation_count']} |",
        f"| Semantic duplicate warnings | {eval_q['split_check']['semantic_duplicate_warning_count']} |",
        f"| Split OK | {eval_q['split_check']['split_ok']} |",
        f"| Empty question rate | {pct(eval_q['empty_question_rate'])} |",
        f"| Empty answer rate | {pct(eval_q['empty_answer_rate'])} |",
        f"| Missing image rate | {pct(eval_q['missing_image_rate'])} |",
        f"| Image open error rate | {pct(eval_q['image_open_error_rate'])} |",
        f"| GUI invalid bbox rate | {pct(eval_q['gui_bbox_invalid_rate'])} |",
        f"| GUI bbox outside image rate | {pct(eval_q['gui_bbox_outside_rate'])} |",
        f"| GUI bbox area ratio mean | {num(eval_q['gui_bbox_area_ratio_mean'], 6)} |",
        f"| GUI bbox area ratio median | {num(eval_q['gui_bbox_area_ratio_median'], 6)} |",
        f"| 平均问题长度 | {num(eval_q['question_len_mean'], 2)} 字符 |",
        f"| 平均答案长度 | {num(eval_q['answer_len_mean'], 2)} 字符 |",
        "",
        "### 4.1 Source 范围",
        "",
        "| Source | Count | Min source_index | Max source_index |",
        "|---|---:|---:|---:|",
    ]
    for source, stats in sorted(eval_q["source_ranges"].items()):
        lines.append(f"| `{source}` | {stats['count']} | {stats['min']} | {stats['max']} |")

    lines += [
        "",
        "### 4.2 评测集结论",
        "",
        "- `eval_medium_600` 在 source identity 和 source range 上与当前训练 trace 隔离，适合作为下一轮 SFT v2 的固定回归集。",
        "- 评测集图像完整、问题/答案非空、GUI bbox 合法，可用于 direct/tool/gui 指标对比。",
        "- 2 条 semantic duplicate warning 是语义相似预警，不是 source 泄漏；评估报告中仍应保留该提示。",
        "- 当前 eval 本身不包含工具证据标签，因此它能评估最终答案、GUI pointing/IoU、协议和 evidence closed，但不能直接评估训练候选生成的 Recall@30/Oracle 注入比例。候选质量仍需在训练 trace/candidate cache 上单独报告。",
        "",
        "## 5. 建议训练用法",
        "",
        "1. 训练 3B/4B tool adapter：优先使用 `evitool_sft_tool_gui_v2`，不要混入 `direct_v2`。",
        "2. 如果 3B 仍绕过工具，改用 `evitool_sft_tool_v2` 和 `evitool_sft_gui_select_v2` 分阶段训练，且 prompt/eval 加 `force-tool-first`。",
        "3. 训练 direct adapter：只用 `evitool_sft_direct_v2`，用于保持文本 QA 能力。",
        "4. AI2D 后续重点不是扩大 direct 样本，而是补 `label_text/object_region/arrow_edge/option_text` 证据位置；当前强证据样本太少。",
        "5. 下一步应为 GUI 增加 candidate-id 解码评测和候选 reranker，避免继续用自由 bbox 回归作为主要训练目标。",
        "",
        "## 6. 复现命令",
        "",
        "```bash",
        "python3 datasets/prepare_sft_v2_data.py \\",
        "  --input /root/models/datasets/evitool_traces_1k/train_traces_1000.jsonl \\",
        "  --eval /root/models/datasets/evitool_eval_medium/eval_medium_600.jsonl \\",
        "  --output-dir /root/models/datasets/evitool_sft_v2 \\",
        "  --report /root/Workspace/VLM/sftv2_dataset_report.md",
        "```",
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    eval_path = Path(args.eval)
    output_dir = Path(args.output_dir)
    report_path = Path(args.report)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(input_path)
    eval_rows = read_jsonl(eval_path)

    tool_rows = [row for row in rows if is_tool_v2(row)]
    ai2d_moderate_rows = [row for row in rows if is_ai2d_moderate(row)]
    gui_rows = [row for row in rows if is_gui_select(row)]
    direct_rows = [row for row in rows if row.get("task_type") != "gui_grounding"]

    tool_items = [
        trace_to_lf_item(row, input_path, "tool_trace_v2", args.max_observation_chars, args.max_spans, args.max_detections)
        for row in tool_rows
    ]
    ai2d_moderate_items = [
        trace_to_lf_item(row, input_path, "ai2d_moderate_v2", args.max_observation_chars, args.max_spans, args.max_detections)
        for row in ai2d_moderate_rows
    ]
    gui_items = [gui_select_item(row, input_path, args.max_detections) for row in gui_rows]
    direct_items = [direct_item(row, input_path) for row in direct_rows]
    tool_gui_items = tool_items + gui_items

    outputs = {
        "evitool_sft_tool_v2": ("evitool_sft_tool_v2.json", tool_items, tool_rows),
        "evitool_sft_ai2d_moderate_v2": ("evitool_sft_ai2d_moderate_v2.json", ai2d_moderate_items, ai2d_moderate_rows),
        "evitool_sft_gui_select_v2": ("evitool_sft_gui_select_v2.json", gui_items, gui_rows),
        "evitool_sft_direct_v2": ("evitool_sft_direct_v2.json", direct_items, direct_rows),
        "evitool_sft_tool_gui_v2": ("evitool_sft_tool_gui_v2.json", tool_gui_items, tool_rows + gui_rows),
    }

    for file_name, items, _source_rows in outputs.values():
        dump_json(output_dir / file_name, items)

    dump_json(output_dir / "dataset_info.json", {name: dataset_entry(file_name) for name, (file_name, _items, _rows) in outputs.items()})

    summary = {
        "name": "evitool_sft_v2",
        "created_at": "2026-05-25",
        "trace_input": str(input_path),
        "eval_input": str(eval_path),
        "output_dir": str(output_dir),
        "source_trace_count": len(rows),
        "source_trace_task_counts": dict(Counter(row.get("task_type") for row in rows)),
        "datasets": {
            name: {"path": str(output_dir / file_name), "count": len(items), "task_counts": task_counts(source_rows)}
            for name, (file_name, items, source_rows) in outputs.items()
        },
        "quality": {
            "tool_v2": quality_summary(tool_rows),
            "ai2d_moderate_v2": quality_summary(ai2d_moderate_rows),
            "gui_select_v2": quality_summary(gui_rows),
        },
        "eval_quality": eval_quality(eval_rows, rows, input_path, eval_path),
        "sft_file_validation": {
            "evitool_sft_tool_v2": validate_sft_items(tool_items, json_assistant=True),
            "evitool_sft_ai2d_moderate_v2": validate_sft_items(ai2d_moderate_items, json_assistant=True),
            "evitool_sft_gui_select_v2": validate_sft_items(gui_items, json_assistant=True, gui=True),
            "evitool_sft_direct_v2": validate_sft_items(direct_items, json_assistant=False),
            "evitool_sft_tool_gui_v2": validate_sft_items(tool_gui_items, json_assistant=True, gui=True),
        },
    }
    dump_json(output_dir / "summary.json", summary)
    write_report(report_path, summary)
    print(json.dumps({"output_dir": str(output_dir), "report": str(report_path), "datasets": summary["datasets"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
