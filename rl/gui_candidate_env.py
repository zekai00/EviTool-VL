"""GUI candidate-selection environment and reward helpers."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from tools.common import bbox_center, bbox_iou, point_in_bbox
from tools.external_detectors import run_external_detectors


def load_gui_rows(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("task_type") != "gui_grounding":
                continue
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def instruction_text(row: dict[str, Any]) -> str:
    return str((row.get("meta") or {}).get("instruction") or row.get("question") or "")


def normalize_candidates(items: list[dict[str, Any]], max_candidates: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for idx, item in enumerate(items[:max_candidates]):
        bbox = item.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        candidates.append(
            {
                "candidate_id": f"c{len(candidates):02d}",
                "rank": len(candidates) + 1,
                "bbox": [int(round(float(v))) for v in bbox],
                "score": round(float(item.get("score", 0.0) or 0.0), 4),
                "label": str(item.get("label") or item.get("source") or "candidate"),
                "source": str(item.get("source") or item.get("external_provider") or "unknown"),
                "text": str(item.get("text") or ""),
            }
        )
    return candidates


def generate_omniparser_candidates(
    image_path: str | Path,
    *,
    max_candidates: int = 30,
    omniparser_root: str | Path = "third_party/OmniParser",
    omniparser_weights_dir: str | Path = "third_party/OmniParser/weights",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    items, meta = run_external_detectors(
        image_path,
        providers=["omniparser"],
        max_results=max_candidates,
        omniparser_root=omniparser_root,
        omniparser_weights_dir=omniparser_weights_dir,
        omniparser_use_caption=False,
    )
    return normalize_candidates(items, max_candidates), meta


def build_candidate_prompt(row: dict[str, Any], candidates: list[dict[str, Any]]) -> str:
    lines = [
        "Select the UI target by candidate_id.",
        "Return only JSON in this exact schema: {\"candidate_id\": \"c00\"}",
        f"Instruction: {instruction_text(row)}",
        "Candidates:",
    ]
    for cand in candidates:
        text = f" text={cand['text']!r}" if cand.get("text") else ""
        lines.append(
            f"- {cand['candidate_id']}: bbox={cand['bbox']} score={cand['score']:.4f} "
            f"source={cand['source']} label={cand['label']}{text}"
        )
    return "\n".join(lines)


def draw_candidate_overlay(
    image_path: str | Path,
    candidates: list[dict[str, Any]],
    output_path: str | Path,
    *,
    max_candidates: int = 30,
) -> str:
    """Save an image annotated with candidate ids and boxes."""
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    palette = [
        (230, 57, 70),
        (29, 53, 87),
        (42, 157, 143),
        (244, 162, 97),
        (131, 56, 236),
        (255, 183, 3),
        (0, 119, 182),
        (106, 153, 78),
    ]
    for idx, cand in enumerate(candidates[:max_candidates]):
        bbox = cand.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        color = palette[idx % len(palette)]
        label = str(cand.get("candidate_id") or f"c{idx:02d}")
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        label_y = max(0, y1 - text_h - 4)
        draw.rectangle([x1, label_y, x1 + text_w + 8, label_y + text_h + 4], fill=color)
        draw.text((x1 + 4, label_y + 2), label, fill=(255, 255, 255), font=font)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return str(output)


def summarize_candidate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    oracle_metrics = [r.get("oracle_metrics") or {} for r in records]
    return {
        "count": len(records),
        "avg_candidates": mean([float(r.get("candidate_count") or 0) for r in records]),
        "oracle_hit_rate": mean([float(m.get("hit") or 0.0) for m in oracle_metrics]),
        "oracle_pointing_rate": mean([float(m.get("pointing") or 0.0) for m in oracle_metrics]),
        "oracle_iou50_rate": mean([float(m.get("iou_50") or 0.0) for m in oracle_metrics]),
        "avg_oracle_iou": mean([float(m.get("iou") or 0.0) for m in oracle_metrics]),
    }


def _normalize_candidate_id(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    match = re.fullmatch(r"c(\d{1,3})", text)
    if match:
        return f"c{int(match.group(1)):02d}"
    if re.fullmatch(r"\d{1,3}", text):
        return f"c{int(text):02d}"
    return text if text else None


def parse_candidate_id(text: Any) -> tuple[str | None, bool]:
    if isinstance(text, dict):
        value = text.get("candidate_id") or text.get("id") or text.get("candidate")
        return (_normalize_candidate_id(value), True) if value is not None else (None, False)
    raw = str(text or "").strip()
    if not raw:
        return None, False
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            value = obj.get("candidate_id") or obj.get("id") or obj.get("candidate")
            return (_normalize_candidate_id(value), True) if value is not None else (None, False)
    except Exception:
        pass
    match = re.search(r"\bc(\d{1,3})\b", raw.lower())
    if match:
        return f"c{int(match.group(1)):02d}", True
    number = re.search(r"\b\d{1,3}\b", raw)
    if number:
        return f"c{int(number.group(0)):02d}", True
    return None, False


def candidate_metrics(row: dict[str, Any], candidate: dict[str, Any] | None) -> dict[str, Any]:
    gt = row.get("answer_bbox")
    if candidate is None or not isinstance(gt, list) or len(gt) != 4:
        return {"iou": 0.0, "iou_50": False, "pointing": False, "hit": False}
    bbox = candidate["bbox"]
    iou = bbox_iou(bbox, gt)
    pointing = point_in_bbox(bbox_center(bbox), gt)
    return {
        "iou": iou,
        "iou_50": iou >= 0.5,
        "pointing": pointing,
        "hit": iou >= 0.3 or pointing,
    }


def oracle_candidate(row: dict[str, Any], candidates: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    best: dict[str, Any] | None = None
    best_metrics: dict[str, Any] = {"iou": 0.0, "iou_50": False, "pointing": False, "hit": False}
    best_key = (-1, -1, -1.0, 9999)
    for cand in candidates:
        metrics = candidate_metrics(row, cand)
        key = (int(metrics["hit"]), int(metrics["iou_50"]), float(metrics["iou"]), -int(cand["rank"]))
        if key > best_key:
            best = cand
            best_metrics = metrics
            best_key = key
    return best, best_metrics


def score_candidate_action(row: dict[str, Any], candidates: list[dict[str, Any]], action_text: Any) -> dict[str, Any]:
    candidate_id, parseable = parse_candidate_id(action_text)
    by_id = {cand["candidate_id"]: cand for cand in candidates}
    selected = by_id.get(candidate_id or "")
    valid = selected is not None
    metrics = candidate_metrics(row, selected)

    format_component = 1.0 if parseable else 0.0
    valid_component = 1.0 if valid else 0.0
    pointing_component = 1.0 if metrics["pointing"] else 0.0
    iou50_component = 1.0 if metrics["iou_50"] else 0.0
    iou_component = min(1.0, float(metrics["iou"]) / 0.5)
    format_penalty = 0.5 if not parseable else 0.0
    invalid_penalty = 0.4 if parseable and not valid else 0.0

    total = (
        0.15 * format_component
        + 0.15 * valid_component
        + 0.30 * pointing_component
        + 0.25 * iou50_component
        + 0.15 * iou_component
        - format_penalty
        - invalid_penalty
    )
    return {
        "total": round(float(total), 6),
        "format": round(format_component, 6),
        "valid": round(valid_component, 6),
        "pointing": round(pointing_component, 6),
        "iou_50": round(iou50_component, 6),
        "iou_shaped": round(iou_component, 6),
        "format_penalty": round(format_penalty, 6),
        "invalid_penalty": round(invalid_penalty, 6),
        "candidate_id": candidate_id,
        "selected_rank": selected.get("rank") if selected else None,
        "selected_bbox": selected.get("bbox") if selected else None,
        "metrics": metrics,
    }


def policy_action(
    policy: str,
    row: dict[str, Any],
    candidates: list[dict[str, Any]],
    rng: random.Random,
) -> dict[str, str]:
    if policy == "top1":
        candidate_id = candidates[0]["candidate_id"] if candidates else "c00"
    elif policy == "oracle":
        best, _ = oracle_candidate(row, candidates)
        candidate_id = best["candidate_id"] if best else "c00"
    elif policy == "random":
        candidate_id = rng.choice(candidates)["candidate_id"] if candidates else "c00"
    elif policy == "invalid":
        candidate_id = "c99"
    else:
        raise ValueError(f"unknown policy: {policy}")
    return {"candidate_id": candidate_id}
