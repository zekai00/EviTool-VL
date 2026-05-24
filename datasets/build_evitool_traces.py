#!/usr/bin/env python3
"""Build EviTool action-observation-final traces from public VLM datasets.

The builder favors deterministic, auditable traces over teacher-model prose:
public datasets provide images/questions/answers/boxes, local tools provide
observations, and rule checks summarize trace quality.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

# huggingface_hub reads HF_ENDPOINT during import in some code paths.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from datasets import load_dataset
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.common import bbox_center, bbox_iou, point_in_bbox
from tools.runner import run_tool


DEFAULT_OUTPUT_DIR = "/root/models/datasets/evitool_traces_1k"
DEFAULT_ENDPOINT = "https://hf-mirror.com"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", DEFAULT_ENDPOINT))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--screenspot", type=int, default=500)
    parser.add_argument("--docvqa", type=int, default=250)
    parser.add_argument("--chartqa", type=int, default=200)
    parser.add_argument("--ai2d", type=int, default=50)
    parser.add_argument("--max-scan", type=int, default=5000)
    parser.add_argument("--ocr-engine", default="easyocr")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def save_image(image: Image.Image, image_path: Path) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    if image.mode not in {"RGB", "RGBA", "L"}:
        image = image.convert("RGB")
    image.save(image_path)


def first_text(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value)


def take_stream(stream: Iterable[dict[str, Any]], limit: int, max_scan: int) -> Iterable[tuple[int, dict[str, Any]]]:
    count = 0
    for idx, example in enumerate(stream):
        if idx >= max_scan or count >= limit:
            break
        count += 1
        yield idx, example


def xywh_to_xyxy(bbox: list[Any]) -> list[int]:
    x, y, w, h = [int(round(float(v))) for v in bbox]
    return [x, y, x + w, y + h]


def norm_text(text: Any) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[\s\"'`]+|[\s\"'`]+$", "", text)
    text = re.sub(r"[。.!?,;:，；：]+$", "", text)
    return text


def norm_compact(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", norm_text(text))


def extract_number(text: Any) -> float | None:
    matches = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(text).replace(",", ""))
    if not matches:
        return None
    try:
        return float(matches[0])
    except ValueError:
        return None


def answer_match_type(answer: Any, text: str, rel_tol: float = 0.03) -> str | None:
    answer_norm = norm_compact(answer)
    text_norm = norm_compact(text)
    if answer_norm and answer_norm in text_norm:
        return "compact_text"
    answer_num = extract_number(answer)
    if answer_num is None:
        return None
    for match in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text.replace(",", "")):
        try:
            value = float(match)
        except ValueError:
            continue
        if abs(value - answer_num) / max(abs(answer_num), 1.0) <= rel_tol:
            return "numeric_tolerance"
    return None


def answer_in_text(answer: Any, text: str, rel_tol: float = 0.03) -> bool:
    return answer_match_type(answer, text, rel_tol=rel_tol) is not None


def union_bboxes(bboxes: list[list[Any]]) -> list[int] | None:
    valid = [bbox for bbox in bboxes if isinstance(bbox, list) and len(bbox) == 4]
    if not valid:
        return None
    return [
        int(min(float(bbox[0]) for bbox in valid)),
        int(min(float(bbox[1]) for bbox in valid)),
        int(max(float(bbox[2]) for bbox in valid)),
        int(max(float(bbox[3]) for bbox in valid)),
    ]


def add_ocr_span_fields(observation: dict[str, Any]) -> None:
    spans = observation.get("content", {}).get("spans")
    if not isinstance(spans, list):
        return
    for idx, span in enumerate(spans, start=1):
        span["span_id"] = f"ocr_{idx:03d}"
        bbox = span.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            span["center"] = [round(float(x), 3) for x in bbox_center(bbox)]


def find_answer_evidence_spans(
    answers: list[Any],
    spans: list[dict[str, Any]],
    evidence_id: str,
    max_window: int = 5,
    max_matches: int = 6,
) -> list[dict[str, Any]]:
    locations: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()

    def add_location(answer: Any, match_type: str, selected: list[dict[str, Any]]) -> None:
        span_ids = [str(span.get("span_id") or f"ocr_{idx + 1:03d}") for idx, span in selected]
        key = (str(answer), tuple(span_ids))
        if key in seen:
            return
        bbox = union_bboxes([span.get("bbox") for _, span in selected])
        if bbox is None:
            return
        seen.add(key)
        locations.append({
            "evidence_id": evidence_id,
            "answer": str(answer),
            "match_type": match_type,
            "span_ids": span_ids,
            "texts": [str(span.get("text", "")) for _, span in selected],
            "bbox": bbox,
        })

    indexed_spans = [(idx, span) for idx, span in enumerate(spans) if str(span.get("text", "")).strip()]
    for answer in answers:
        before = len(locations)
        for idx, span in indexed_spans:
            match_type = answer_match_type(answer, str(span.get("text", "")))
            if match_type:
                add_location(answer, match_type, [(idx, span)])
                if len(locations) >= max_matches:
                    return locations
        if len(locations) > before:
            continue
        for window in range(2, max_window + 1):
            if len(indexed_spans) < window:
                break
            for start in range(0, len(indexed_spans) - window + 1):
                selected = indexed_spans[start:start + window]
                combined = " ".join(str(span.get("text", "")) for _, span in selected)
                match_type = answer_match_type(answer, combined)
                if match_type:
                    add_location(answer, f"{match_type}_window", selected)
                    if len(locations) >= max_matches:
                        return locations
            if len(locations) > before:
                break
    return locations


def safe_observation(observation: dict[str, Any], image_rel: str) -> dict[str, Any]:
    obs = json.loads(json.dumps(observation, ensure_ascii=False))
    obs["image"] = image_rel
    obs.pop("elapsed_ms", None)
    return obs


def run_local_tool(image_path: Path, image_rel: str, action: dict[str, Any], evidence_id: str) -> dict[str, Any]:
    obs = run_tool(image_path, action, evidence_id=evidence_id)
    return safe_observation(obs, image_rel)


def add_candidate_fields(observation: dict[str, Any]) -> None:
    detections = observation.get("content", {}).get("detections")
    if not isinstance(detections, list):
        return
    for idx, item in enumerate(detections, start=1):
        bbox = item.get("bbox")
        item["candidate_id"] = f"cand_{idx:03d}"
        if isinstance(bbox, list) and len(bbox) == 4:
            item["center"] = [round(float(x), 3) for x in bbox_center(bbox)]


def make_user_prompt(row: dict[str, Any]) -> str:
    task = row["task_type"]
    if task == "gui_grounding":
        return (
            "Locate the UI element described by the instruction. Use visual tools when needed. "
            "Return a click point or bounding box with cited evidence.\n\n"
            f"Question: {row['question']}"
        )
    return (
        "Answer the question based on the image. Use local visual tools when evidence is needed. "
        "Return a concise final answer with cited evidence.\n\n"
        f"Question: {row['question']}"
    )


def action_turn(thought: str, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"thought": thought, "action": {"tool": tool, "args": args}}


def final_turn(step: str, evidence: list[str], answer: Any) -> dict[str, Any]:
    return {"reasoning": [{"step": step, "evidence": evidence}], "answer": answer}


def make_record(
    *,
    row: dict[str, Any],
    image_rel: str,
    trace: list[dict[str, Any]],
    final: dict[str, Any],
    quality: dict[str, Any],
) -> dict[str, Any]:
    messages = [{"role": "user", "image": image_rel, "content": make_user_prompt(row)}]
    for item in trace:
        messages.append({"role": "assistant", "content": json.dumps(item["action"], ensure_ascii=False)})
        messages.append({"role": "tool", "content": json.dumps(item["observation"], ensure_ascii=False)})
    messages.append({"role": "assistant", "content": json.dumps(final, ensure_ascii=False)})
    return {
        **row,
        "image": image_rel,
        "messages": messages,
        "trace": trace,
        "final": final,
        "quality": quality,
    }


def best_gui_candidate(detections: list[dict[str, Any]], gt_bbox: list[int]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    best = None
    best_score = -1.0
    recall = {1: False, 3: False, 5: False, 10: False, 30: False}
    for idx, det in enumerate(detections, start=1):
        bbox = det.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        iou = bbox_iou(bbox, gt_bbox)
        center_hit = point_in_bbox(bbox_center(bbox), gt_bbox)
        hit = iou >= 0.3 or center_hit
        for k in recall:
            if idx <= k and hit:
                recall[k] = True
        score = iou + (0.5 if center_hit else 0.0)
        if score > best_score:
            best_score = score
            best = {**det, "iou_with_gt": iou, "center_in_gt": center_hit, "rank": idx, "hit": hit}
    return best, {f"candidate_recall_at_{k}": recall[k] for k in recall}


def candidate_recall_tier(recall: dict[str, Any]) -> str:
    for k in (1, 3, 5, 10, 30):
        if recall.get(f"candidate_recall_at_{k}"):
            return f"top_{k}"
    return "miss"


def build_screenspot(out_dir: Path, limit: int, max_scan: int) -> list[dict[str, Any]]:
    ds = load_dataset("lmms-lab/ScreenSpot-v2", split="train", streaming=True)
    records = []
    for out_idx, (src_idx, ex) in enumerate(take_stream(ds, limit, max_scan)):
        image_rel = f"images/screenspot/{out_idx:04d}.png"
        image_path = out_dir / image_rel
        save_image(ex["image"], image_path)
        gt_bbox = xywh_to_xyxy(ex.get("bbox", [0, 0, 0, 0]))
        center = [round(float(x), 3) for x in bbox_center(gt_bbox)]
        instruction = str(ex.get("instruction", ""))
        row = {
            "id": f"screenspot_{out_idx:04d}",
            "source": "lmms-lab/ScreenSpot-v2",
            "source_index": src_idx,
            "task_type": "gui_grounding",
            "question": f"Locate the UI element for this instruction: {instruction}.",
            "answer": center,
            "answer_bbox": gt_bbox,
            "meta": {
                "instruction": instruction,
                "img_filename": ex.get("img_filename"),
                "data_type": ex.get("data_type"),
                "data_source": ex.get("data_source"),
                "bbox_xywh": ex.get("bbox"),
            },
        }

        detect_action = action_turn("Find UI element candidates before selecting the target.", "detect", {"mode": "ui", "query": instruction, "max_results": 30, "min_area": 20})
        detect_obs = run_local_tool(image_path, image_rel, detect_action["action"], "ev_001")
        add_candidate_fields(detect_obs)
        detections = detect_obs.get("content", {}).get("detections") or []
        best, recall = best_gui_candidate(detections, gt_bbox)

        if best and best.get("hit"):
            selected_bbox = [int(round(float(x))) for x in best["bbox"]]
            selected_source = "detected_candidate"
            selected_candidate_id = best.get("candidate_id")
        else:
            selected_bbox = gt_bbox
            selected_source = "oracle_gt_bbox"
            selected_candidate_id = None

        click_args = {"bbox": selected_bbox, "label": instruction}
        click_action = action_turn("Record the selected UI target as a click evidence.", "click", click_args)
        click_obs = run_local_tool(image_path, image_rel, click_action["action"], "ev_002")
        final = final_turn(
            "ev_001 lists UI candidates and ev_002 records the selected target for the requested instruction.",
            ["ev_001", "ev_002"],
            center,
        )
        recall_tier = candidate_recall_tier(recall)
        detect_candidate_hit = bool(best and best.get("hit"))
        oracle_gt_injected = selected_source == "oracle_gt_bbox"
        quality_tags = [
            f"candidate_recall_{recall_tier}",
            "detected_candidate" if detect_candidate_hit else "candidate_miss",
            "oracle_gt_injected" if oracle_gt_injected else "no_oracle_gt",
        ]
        quality = {
            "evidence_closed": True,
            "strong_evidence": detect_candidate_hit,
            "tool_sequence": ["detect", "click"],
            "quality_tier": f"detected_{recall_tier}" if detect_candidate_hit else "oracle_gt_injected",
            "quality_tags": quality_tags,
            "selected_source": selected_source,
            "selected_candidate_id": selected_candidate_id,
            "selected_candidate_sources": best.get("sources") if best else None,
            "selected_iou_with_gt": bbox_iou(selected_bbox, gt_bbox),
            "selected_pointing": point_in_bbox(bbox_center(selected_bbox), gt_bbox),
            "final_pointing": point_in_bbox(center, gt_bbox),
            "candidate_count": len(detections),
            "candidate_pool_size": detect_obs.get("content", {}).get("candidate_pool_size"),
            "candidate_source_counts": detect_obs.get("content", {}).get("candidate_sources"),
            "best_candidate_rank": best.get("rank") if best else None,
            "best_candidate_iou": best.get("iou_with_gt") if best else 0.0,
            "best_candidate_center_in_gt": best.get("center_in_gt") if best else False,
            "detect_candidate_hit": detect_candidate_hit,
            "candidate_recall_tier": recall_tier,
            "oracle_gt_injected": oracle_gt_injected,
            **recall,
        }
        records.append(make_record(row=row, image_rel=image_rel, trace=[
            {"action": detect_action, "observation": detect_obs},
            {"action": click_action, "observation": click_obs},
        ], final=final, quality=quality))
    return records


def build_docvqa(out_dir: Path, limit: int, max_scan: int, ocr_engine: str) -> list[dict[str, Any]]:
    ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation", streaming=True)
    records = []
    for out_idx, (src_idx, ex) in enumerate(take_stream(ds, limit, max_scan)):
        image_rel = f"images/docvqa/{out_idx:04d}.png"
        image_path = out_dir / image_rel
        save_image(ex["image"], image_path)
        answer = first_text(ex.get("answers", ""))
        row = {
            "id": f"docvqa_{out_idx:04d}",
            "source": "lmms-lab/DocVQA",
            "source_index": src_idx,
            "task_type": "doc_qa",
            "question": str(ex.get("question", "")),
            "answer": answer,
            "answers": ex.get("answers", [answer]),
            "meta": {
                "question_id": ex.get("questionId"),
                "question_types": ex.get("question_types"),
                "doc_id": ex.get("docId"),
            },
        }
        ocr_action = action_turn("Read document text before answering.", "ocr", {"engine": ocr_engine, "languages": ["en"], "max_regions": 120})
        ocr_obs = run_local_tool(image_path, image_rel, ocr_action["action"], "ev_001")
        add_ocr_span_fields(ocr_obs)
        spans = ocr_obs.get("content", {}).get("spans") or []
        answers = row["answers"] if isinstance(row["answers"], list) else [row["answers"]]
        ocr_text = " ".join(str(span.get("text", "")) for span in spans)
        answer_in_ocr_text = any(answer_in_text(ans, ocr_text) for ans in answers)
        answer_locations = find_answer_evidence_spans(answers, spans, "ev_001")
        answer_position_found = bool(answer_locations)
        support_phrase = "The answer is localized in ev_001 OCR spans." if answer_position_found else "ev_001 provides OCR evidence, but the exact answer string was not localized by the local OCR backend."
        final = final_turn(support_phrase, ["ev_001"], answer)
        quality_tags = [
            "ocr_available" if bool(ocr_obs.get("content", {}).get("available")) else "ocr_unavailable",
            "answer_in_ocr" if answer_in_ocr_text else "answer_missing_from_ocr",
            "answer_position_found" if answer_position_found else "answer_position_missing",
        ]
        quality = {
            "evidence_closed": True,
            "tool_sequence": ["ocr"],
            "quality_tier": "answer_text_localized" if answer_position_found else "ocr_without_localized_answer",
            "quality_tags": quality_tags,
            "ocr_available": bool(ocr_obs.get("content", {}).get("available")),
            "ocr_span_count": len(spans),
            "answer_in_ocr": answer_in_ocr_text,
            "answer_evidence_position_found": answer_position_found,
            "answer_evidence_span_count": sum(len(item.get("span_ids", [])) for item in answer_locations),
            "answer_evidence_locations": answer_locations,
            "strong_evidence": answer_position_found,
        }
        records.append(make_record(row=row, image_rel=image_rel, trace=[{"action": ocr_action, "observation": ocr_obs}], final=final, quality=quality))
    return records


def build_chartqa(out_dir: Path, limit: int, max_scan: int, ocr_engine: str) -> list[dict[str, Any]]:
    ds = load_dataset("HuggingFaceM4/ChartQA", split="test", streaming=True)
    records = []
    for out_idx, (src_idx, ex) in enumerate(take_stream(ds, limit, max_scan)):
        image_rel = f"images/chartqa/{out_idx:04d}.png"
        image_path = out_dir / image_rel
        save_image(ex["image"], image_path)
        answer = first_text(ex.get("label", ""))
        row = {
            "id": f"chartqa_{out_idx:04d}",
            "source": "HuggingFaceM4/ChartQA",
            "source_index": src_idx,
            "task_type": "chart_qa",
            "question": str(ex.get("query", "")),
            "answer": answer,
            "answers": ex.get("label", [answer]),
            "meta": {"human_or_machine": ex.get("human_or_machine")},
        }
        detect_action = action_turn("Detect chart bars or colored regions before answering.", "detect", {"mode": "bar", "max_results": 30, "min_area": 20})
        detect_obs = run_local_tool(image_path, image_rel, detect_action["action"], "ev_001")
        ocr_action = action_turn("Read chart labels and numbers with OCR.", "ocr", {"engine": ocr_engine, "languages": ["en"], "max_regions": 100})
        ocr_obs = run_local_tool(image_path, image_rel, ocr_action["action"], "ev_002")
        add_ocr_span_fields(ocr_obs)
        spans = ocr_obs.get("content", {}).get("spans") or []
        answers = row["answers"] if isinstance(row["answers"], list) else [row["answers"]]
        ocr_text = " ".join(str(span.get("text", "")) for span in spans)
        answer_in_ocr_text = any(answer_in_text(ans, ocr_text) for ans in answers)
        answer_locations = find_answer_evidence_spans(answers, spans, "ev_002")
        answer_position_found = bool(answer_locations)
        bar_count = int(detect_obs.get("content", {}).get("count") or 0)
        chart_structure_evidence = bar_count > 0
        step = "ev_001 provides chart structure candidates and ev_002 provides chart text/number OCR."
        if answer_position_found:
            step += " The final answer is localized in ev_002 OCR spans."
        elif answer_in_ocr_text:
            step += " The final answer appears in OCR text but was not localized to a compact span window."
        final = final_turn(step, ["ev_001", "ev_002"], answer)
        if answer_position_found and chart_structure_evidence:
            quality_tier = "answer_text_localized_with_chart_structure"
        elif answer_position_found:
            quality_tier = "answer_text_localized"
        elif chart_structure_evidence:
            quality_tier = "chart_structure_only"
        else:
            quality_tier = "weak_evidence"
        quality_tags = [
            "chart_structure_detected" if chart_structure_evidence else "chart_structure_missing",
            "ocr_available" if bool(ocr_obs.get("content", {}).get("available")) else "ocr_unavailable",
            "answer_in_ocr" if answer_in_ocr_text else "answer_missing_from_ocr",
            "answer_position_found" if answer_position_found else "answer_position_missing",
        ]
        quality = {
            "evidence_closed": True,
            "tool_sequence": ["detect", "ocr"],
            "quality_tier": quality_tier,
            "quality_tags": quality_tags,
            "bar_candidate_count": bar_count,
            "chart_structure_evidence": chart_structure_evidence,
            "ocr_available": bool(ocr_obs.get("content", {}).get("available")),
            "ocr_span_count": len(spans),
            "answer_in_ocr": answer_in_ocr_text,
            "answer_evidence_position_found": answer_position_found,
            "answer_evidence_span_count": sum(len(item.get("span_ids", [])) for item in answer_locations),
            "answer_evidence_locations": answer_locations,
            "strong_evidence": answer_position_found,
        }
        records.append(make_record(row=row, image_rel=image_rel, trace=[
            {"action": detect_action, "observation": detect_obs},
            {"action": ocr_action, "observation": ocr_obs},
        ], final=final, quality=quality))
    return records


def build_ai2d(out_dir: Path, limit: int, max_scan: int) -> list[dict[str, Any]]:
    ds = load_dataset("lmms-lab/ai2d", split="test", streaming=True)
    records = []
    for out_idx, (src_idx, ex) in enumerate(take_stream(ds, limit, max_scan)):
        image_rel = f"images/ai2d/{out_idx:04d}.png"
        image_path = out_dir / image_rel
        save_image(ex["image"], image_path)
        options = [str(option) for option in ex.get("options", [])]
        raw_answer = str(ex.get("answer", ""))
        answer = raw_answer
        answer_letter = None
        if raw_answer.isdigit():
            answer_idx = int(raw_answer)
            if 0 <= answer_idx < len(options):
                answer = options[answer_idx]
                answer_letter = chr(65 + answer_idx)
        option_text = "\n".join(f"{chr(65 + j)}. {option}" for j, option in enumerate(options))
        question = str(ex.get("question", ""))
        if option_text:
            question = f"{question}\n{option_text}"
        row = {
            "id": f"ai2d_{out_idx:04d}",
            "source": "lmms-lab/ai2d",
            "source_index": src_idx,
            "task_type": "science_diagram_qa",
            "question": question,
            "answer": answer,
            "choices": options,
            "meta": {"raw_answer": raw_answer, "answer_letter": answer_letter},
        }
        inspect_action = action_turn("Inspect the diagram metadata and visual density before choosing the option.", "inspect", {})
        inspect_obs = run_local_tool(image_path, image_rel, inspect_action["action"], "ev_001")
        final_answer = answer_letter or answer
        final = final_turn("ev_001 confirms the diagram was inspected; the answer follows the dataset option label.", ["ev_001"], final_answer)
        quality = {
            "evidence_closed": True,
            "tool_sequence": ["inspect"],
            "option_count": len(options),
            "answer_letter_available": answer_letter is not None,
            "strong_evidence": False,
        }
        records.append(make_record(row=row, image_rel=image_rel, trace=[{"action": inspect_action, "observation": inspect_obs}], final=final, quality=quality))
    return records


def summarize(records: list[dict[str, Any]], elapsed_sec: float) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_task[record["task_type"]].append(record)

    def mean(items: list[bool | int | float]) -> float | None:
        return sum(float(x) for x in items) / len(items) if items else None

    task_summary = {}
    for task, items in by_task.items():
        qualities = [item["quality"] for item in items]
        task_summary[task] = {
            "count": len(items),
            "evidence_closed_rate": mean([q.get("evidence_closed", False) for q in qualities]),
            "strong_evidence_rate": mean([q.get("strong_evidence", False) for q in qualities]),
            "avg_tool_calls": mean([len(q.get("tool_sequence", [])) for q in qualities]),
        }
        if task == "gui_grounding":
            task_summary[task].update({
                "candidate_recall_at_1": mean([q.get("candidate_recall_at_1", False) for q in qualities]),
                "candidate_recall_at_3": mean([q.get("candidate_recall_at_3", False) for q in qualities]),
                "candidate_recall_at_5": mean([q.get("candidate_recall_at_5", False) for q in qualities]),
                "candidate_recall_at_10": mean([q.get("candidate_recall_at_10", False) for q in qualities]),
                "candidate_recall_at_30": mean([q.get("candidate_recall_at_30", False) for q in qualities]),
                "detected_candidate_rate": mean([q.get("detect_candidate_hit", False) for q in qualities]),
                "oracle_gt_injected_rate": mean([q.get("selected_source") == "oracle_gt_bbox" for q in qualities]),
                "final_pointing_rate": mean([q.get("final_pointing", False) for q in qualities]),
                "selected_pointing_rate": mean([q.get("selected_pointing", False) for q in qualities]),
                "avg_selected_iou": mean([q.get("selected_iou_with_gt", 0.0) for q in qualities]),
                "avg_candidate_count": mean([q.get("candidate_count", 0) for q in qualities]),
                "avg_candidate_pool_size": mean([q.get("candidate_pool_size", 0) or 0 for q in qualities]),
            })
        if task in {"doc_qa", "chart_qa"}:
            task_summary[task].update({
                "ocr_available_rate": mean([q.get("ocr_available", False) for q in qualities]),
                "answer_in_ocr_rate": mean([q.get("answer_in_ocr", False) for q in qualities]),
                "answer_evidence_position_rate": mean([q.get("answer_evidence_position_found", False) for q in qualities]),
                "avg_answer_evidence_spans": mean([q.get("answer_evidence_span_count", 0) for q in qualities]),
                "avg_ocr_spans": mean([q.get("ocr_span_count", 0) for q in qualities]),
            })
        if task == "chart_qa":
            task_summary[task]["avg_bar_candidates"] = mean([q.get("bar_candidate_count", 0) for q in qualities])
            task_summary[task]["chart_structure_evidence_rate"] = mean([q.get("chart_structure_evidence", False) for q in qualities])
            task_summary[task]["chart_structure_only_rate"] = mean([q.get("quality_tier") == "chart_structure_only" for q in qualities])

    all_quality = [record["quality"] for record in records]
    return {
        "name": "evitool_traces_1k",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "count": len(records),
        "task_counts": dict(Counter(record["task_type"] for record in records)),
        "sources": sorted(set(record["source"] for record in records)),
        "elapsed_sec": round(elapsed_sec, 3),
        "evidence_closed_rate": mean([q.get("evidence_closed", False) for q in all_quality]),
        "strong_evidence_rate": mean([q.get("strong_evidence", False) for q in all_quality]),
        "avg_tool_calls": mean([len(q.get("tool_sequence", [])) for q in all_quality]),
        "by_task": task_summary,
    }


def pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{100 * value:.2f}%"


def write_report(path: Path, summary: dict[str, Any], records_path: str) -> None:
    by = summary["by_task"]
    lines = [
        "# EviTool-Trace-1K 数据集构建报告",
        "",
        f"生成时间：{summary['created_at']}",
        "",
        "## 1. 数据集概览",
        "",
        f"- 输出文件：`{records_path}`",
        f"- 样本总数：{summary['count']}",
        f"- 开源来源：{', '.join(summary['sources'])}",
        f"- 全局 evidence closed rate：{pct(summary['evidence_closed_rate'])}",
        f"- 全局 strong evidence rate：{pct(summary['strong_evidence_rate'])}",
        f"- 平均工具调用数：{summary['avg_tool_calls']:.3f}",
        "",
        "| Task | Count | Avg Tools | Evidence Closed | Strong Evidence |",
        "|---|---:|---:|---:|---:|",
    ]
    for task, item in by.items():
        lines.append(
            f"| {task} | {item['count']} | {item['avg_tool_calls']:.3f} | "
            f"{pct(item['evidence_closed_rate'])} | {pct(item['strong_evidence_rate'])} |"
        )
    lines.extend([
        "",
        "## 2. 构建方法",
        "",
        "本数据集没有使用远程 teacher 模型生成答案标签。答案、选项和 GUI bbox 来自公开数据集原始标注；视觉证据来自本地 EviTool 工具实际执行结果。",
        "",
        "- ScreenSpot-v2：使用融合版 `detect(mode=ui)` 生成 UI 边缘、矩形控件、文本视觉区、图标和布局候选；候选召回失败时注入 GT bbox 作为 bootstrap oracle trace，并在 `quality.selected_source` / `quality.oracle_gt_injected` 中标明。",
        f"- DocVQA：使用 `ocr(engine={summary.get('ocr_engine', 'easyocr' )})` 读取页面文本，最终答案来自 DocVQA 标注；`answer_in_ocr` 衡量本地 OCR 是否召回答案，`answer_evidence_locations` 保存支持答案的 OCR span 位置。",
        "- ChartQA：使用 `detect(mode=bar)` 获取图表结构候选，并用 OCR 读取文本/数字；`bar_candidate_count` 仅表示结构候选，强证据优先由答案 OCR span 位置决定。",
        "- AI2D：使用 `inspect` 建立最小工具协议 trace，答案来自多选标注。该子集主要用于训练协议格式，不作为强视觉证据子集。",
        "",
        "每条样本均保存 user、assistant action、tool observation、assistant final 的 action-observation-final 消息序列，并保存结构化 `trace/final/quality` 字段。",
        "",
        "## 3. 质量指标",
        "",
    ])
    if "gui_grounding" in by:
        gui = by["gui_grounding"]
        lines.extend([
            "### 3.1 GUI Grounding",
            "",
            f"- Candidate Recall@1：{pct(gui.get('candidate_recall_at_1'))}",
            f"- Candidate Recall@3：{pct(gui.get('candidate_recall_at_3'))}",
            f"- Candidate Recall@5：{pct(gui.get('candidate_recall_at_5'))}",
            f"- Candidate Recall@10：{pct(gui.get('candidate_recall_at_10'))}",
            f"- Candidate Recall@30：{pct(gui.get('candidate_recall_at_30'))}",
            f"- Detected Candidate Rate：{pct(gui.get('detected_candidate_rate'))}",
            f"- Oracle GT 注入比例：{pct(gui.get('oracle_gt_injected_rate'))}",
            f"- 平均候选数：{gui.get('avg_candidate_count', 0.0):.3f}",
            f"- 平均融合候选池：{gui.get('avg_candidate_pool_size', 0.0):.3f}",
            f"- Final Pointing Rate：{pct(gui.get('final_pointing_rate'))}",
            f"- Selected Pointing Rate：{pct(gui.get('selected_pointing_rate'))}",
            f"- 平均 selected IoU：{gui.get('avg_selected_iou', 0.0):.4f}",
            "",
            "解释：Candidate recall 反映当前本地 `detect(ui)` 是否能把 GT 目标放进候选集；Detected Candidate Rate 是无需注入 GT 的样本比例；Oracle GT 注入比例越高，说明 detect 工具仍需增强。Final pointing 使用 GT center，因此用于训练 click/point 协议，不能当作模型推理成绩。",
            "",
        ])
    for task in ("doc_qa", "chart_qa"):
        if task in by:
            item = by[task]
            lines.extend([
                f"### 3.2 {task}",
                "",
                f"- OCR 可用率：{pct(item.get('ocr_available_rate'))}",
                f"- Answer-in-OCR Rate：{pct(item.get('answer_in_ocr_rate'))}",
                f"- Answer Evidence Position Rate：{pct(item.get('answer_evidence_position_rate'))}",
                f"- 平均答案证据 spans：{item.get('avg_answer_evidence_spans', 0.0):.3f}",
                f"- 平均 OCR spans：{item.get('avg_ocr_spans', 0.0):.3f}",
            ])
            if task == "chart_qa":
                lines.append(f"- 平均 bar candidates：{item.get('avg_bar_candidates', 0.0):.3f}")
                lines.append(f"- Chart Structure Evidence Rate：{pct(item.get('chart_structure_evidence_rate'))}")
                lines.append(f"- Chart Structure Only Rate：{pct(item.get('chart_structure_only_rate'))}")
            lines.append("")
    lines.extend([
        "## 4. 使用建议",
        "",
        "第一轮 SFT 建议优先使用 GUI 中 `oracle_gt_injected=false` 的样本，以及 DocVQA/ChartQA 中 `answer_evidence_position_found=true` 的强证据子集。AI2D 与弱 OCR 样本可用于协议格式训练，但不应在论文中声称其每个视觉推理步骤都被强证据完全支持。",
        "",
        "下一步应重新构建数据集并观察 Candidate Recall@K、Oracle GT 注入比例、Answer Evidence Position Rate。之后再用允许的强模型做少量 hard case trace repair，而不是全量 teacher 生成。",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint
    random.seed(args.seed)

    out_dir = Path(args.output_dir)
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    builders = [
        ("screenspot", args.screenspot, lambda: build_screenspot(out_dir, args.screenspot, args.max_scan)),
        ("docvqa", args.docvqa, lambda: build_docvqa(out_dir, args.docvqa, args.max_scan, args.ocr_engine)),
        ("chartqa", args.chartqa, lambda: build_chartqa(out_dir, args.chartqa, args.max_scan, args.ocr_engine)),
        ("ai2d", args.ai2d, lambda: build_ai2d(out_dir, args.ai2d, args.max_scan)),
    ]

    records: list[dict[str, Any]] = []
    for name, limit, builder in builders:
        if limit <= 0:
            continue
        print(f"Building {name}: target={limit}", flush=True)
        built = builder()
        records.extend(built)
        print(f"Built {name}: {len(built)}", flush=True)

    records_path = out_dir / "train_traces_1000.jsonl"
    write_jsonl(records_path, records)
    summary = summarize(records, time.time() - started)
    summary["ocr_engine"] = args.ocr_engine
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(out_dir / "DATASET_REPORT.md", summary, str(records_path))

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
