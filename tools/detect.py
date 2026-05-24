"""Lightweight visual detection tools based on OpenCV."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .common import bbox_area, bbox_center, bbox_iou, crop_pil, image_size, load_image, parse_color, pil_to_cv, sort_boxes_reading_order


UI_MODE_ALIASES = {"ui", "button", "icon", "screen", "gui"}
TEXT_MODE_ALIASES = {"text", "text_region", "ocr"}
BAR_MODE_ALIASES = {"bar", "chart", "bars"}
DIAGRAM_MODE_ALIASES = {"diagram", "science_diagram", "ai2d"}
COLOR_MODE_ALIASES = {"color", "colour"}
KNOWN_MODE_ALIASES = UI_MODE_ALIASES | TEXT_MODE_ALIASES | BAR_MODE_ALIASES | DIAGRAM_MODE_ALIASES | COLOR_MODE_ALIASES | {"layout"}

SOURCE_PRIORITY = {
    "query_prior": 1.02,
    "ocr_text": 0.95,
    "text_expanded": 0.9,
    "text_visual": 0.82,
    "row_container": 0.8,
    "ui_rect": 0.76,
    "ui_edge": 0.72,
    "icon_visual": 0.68,
    "diagram_region": 0.6,
    "layout": 0.45,
}


def _clip_region(image, bbox):
    if bbox is None:
        return image, (0, 0), None
    crop, clipped = crop_pil(image, bbox)
    return crop, (clipped[0], clipped[1]), clipped


def _contour_boxes(mask: np.ndarray, offset: tuple[int, int], min_area: float, max_area_ratio: float, max_results: int) -> list[dict[str, Any]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape[:2]
    boxes = []
    ox, oy = offset
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        area = float(bw * bh)
        if area < min_area:
            continue
        if area > max_area_ratio * w * h:
            continue
        boxes.append({"bbox": [x + ox, y + oy, x + bw + ox, y + bh + oy], "score": min(1.0, area / max(1.0, w * h)), "area": area})
    boxes = sorted(boxes, key=lambda item: item["area"], reverse=True)
    return boxes[:max_results]


def _intersection_area(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def _tokenize(text: str | None) -> set[str]:
    if not text:
        return set()
    stopwords = {"the", "a", "an", "to", "for", "of", "on", "in", "at", "and", "or", "click", "tap", "select", "button", "icon"}
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) >= 2 and token not in stopwords}


def _query_text_bonus(candidate_text: str | None, query: str | None) -> float:
    query_tokens = _tokenize(query)
    text_tokens = _tokenize(candidate_text)
    if not query_tokens or not text_tokens:
        return 0.0
    overlap = len(query_tokens & text_tokens)
    return min(0.5, 0.18 * overlap)


def _tag_candidates(candidates: list[dict[str, Any]], source: str, label: str | None = None, score_boost: float = 0.0) -> list[dict[str, Any]]:
    tagged = []
    for item in candidates:
        bbox = item.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        candidate = {**item}
        candidate["bbox"] = [int(round(float(v))) for v in bbox]
        if label is not None:
            candidate["label"] = label
        sources = set(candidate.get("sources") or [])
        sources.add(source)
        candidate["source"] = source
        candidate["sources"] = sorted(sources)
        candidate["score"] = round(min(1.0, float(candidate.get("score", 0.0)) + score_boost), 4)
        tagged.append(candidate)
    return tagged


def _same_candidate(a: dict[str, Any], b: dict[str, Any]) -> bool:
    a_box = a.get("bbox")
    b_box = b.get("bbox")
    if not isinstance(a_box, list) or not isinstance(b_box, list):
        return False
    iou = bbox_iou(a_box, b_box)
    if iou >= 0.82:
        return True
    inter = _intersection_area(a_box, b_box)
    smaller = min(bbox_area(a_box), bbox_area(b_box))
    larger = max(bbox_area(a_box), bbox_area(b_box))
    if smaller <= 0:
        return False
    return inter / smaller >= 0.92 and smaller / max(larger, 1.0) >= 0.65


def _dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for item in sorted(candidates, key=lambda x: float(x.get("score", 0.0)), reverse=True):
        merged = False
        for existing in kept:
            if not _same_candidate(existing, item):
                continue
            sources = sorted(set(existing.get("sources") or []) | set(item.get("sources") or []))
            existing["sources"] = sources
            existing["source_count"] = len(sources)
            existing["score"] = round(max(float(existing.get("score", 0.0)), float(item.get("score", 0.0))), 4)
            if not existing.get("text") and item.get("text"):
                existing["text"] = item["text"]
            merged = True
            break
        if not merged:
            item["source_count"] = len(item.get("sources") or [])
            kept.append(item)
    return kept


def _candidate_rank_score(item: dict[str, Any], image_shape: tuple[int, int], query: str | None) -> float:
    bbox = item.get("bbox") or [0, 0, 0, 0]
    image_h, image_w = image_shape
    area_ratio = bbox_area(bbox) / max(1.0, float(image_h * image_w))
    x1, y1, x2, y2 = [float(v) for v in bbox]
    width, height = max(1.0, x2 - x1), max(1.0, y2 - y1)
    aspect = width / height
    source_priority = max(SOURCE_PRIORITY.get(source, 0.4) for source in item.get("sources") or [item.get("source")])
    size_bonus = min(0.22, np.sqrt(max(area_ratio, 0.0)) * 1.8)
    tiny_penalty = 0.18 if area_ratio < 0.00003 else 0.0
    huge_penalty = 0.28 if area_ratio > 0.35 else 0.0
    aspect_penalty = 0.12 if aspect < 0.08 or aspect > 16.0 else 0.0
    text_bonus = _query_text_bonus(item.get("text"), query)
    return source_priority + 0.35 * float(item.get("score", 0.0)) + size_bonus + text_bonus - tiny_penalty - huge_penalty - aspect_penalty


def _rank_candidates(candidates: list[dict[str, Any]], image_bgr: np.ndarray, query: str | None, max_results: int) -> list[dict[str, Any]]:
    h, w = image_bgr.shape[:2]
    for item in candidates:
        item["center"] = [round(float(v), 3) for v in bbox_center(item["bbox"])]
        item["rank_score"] = round(float(_candidate_rank_score(item, (h, w), query)), 4)
    ranked = sorted(candidates, key=lambda item: (-float(item.get("rank_score", 0.0)), item["bbox"][1], item["bbox"][0]))
    return ranked[:max_results]


def _layout(image_bgr: np.ndarray, offset: tuple[int, int], min_area: float, max_results: int) -> list[dict[str, Any]]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    boxes = _contour_boxes(closed, offset, min_area, 0.9, max_results * 2)
    for item in boxes:
        item["label"] = "layout_region"
    return sort_boxes_reading_order(boxes)[:max_results]


def _text_regions(image_bgr: np.ndarray, offset: tuple[int, int], min_area: float, max_results: int) -> list[dict[str, Any]]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (16, 4))
    mask = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)
    boxes = _contour_boxes(mask, offset, min_area, 0.5, max_results)
    for item in boxes:
        item["label"] = "text_region"
    return sort_boxes_reading_order(boxes)


def _ui_elements(image_bgr: np.ndarray, offset: tuple[int, int], min_area: float, max_results: int) -> list[dict[str, Any]]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.dilate(edges, kernel, iterations=1)
    boxes = _contour_boxes(mask, offset, min_area, 0.3, max_results * 3)
    filtered = []
    for item in boxes:
        x1, y1, x2, y2 = item["bbox"]
        w, h = x2 - x1, y2 - y1
        aspect = w / max(h, 1)
        if 0.15 <= aspect <= 12.0:
            item["label"] = "ui_element_candidate"
            filtered.append(item)
    return sort_boxes_reading_order(filtered)[:max_results]


def _clip_bbox_to_image(bbox: list[float], image_shape: tuple[int, int], offset: tuple[int, int]) -> list[int] | None:
    h, w = image_shape
    ox, oy = offset
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1 = max(float(ox), min(float(ox + w), x1))
    y1 = max(float(oy), min(float(oy + h), y1))
    x2 = max(float(ox), min(float(ox + w), x2))
    y2 = max(float(oy), min(float(oy + h), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]


def _expanded_text_candidates(image_bgr: np.ndarray, offset: tuple[int, int], min_area: float, max_results: int) -> list[dict[str, Any]]:
    text_boxes = _text_regions(image_bgr, offset, max(8.0, min_area * 0.4), max_results * 2)
    h, w = image_bgr.shape[:2]
    candidates = []
    for item in text_boxes:
        x1, y1, x2, y2 = [float(v) for v in item["bbox"]]
        bw, bh = x2 - x1, y2 - y1
        if bw < 3 or bh < 3:
            continue
        tight_pad_x = max(6.0, bh * 0.8)
        tight_pad_y = max(4.0, bh * 0.45)
        tight = _clip_bbox_to_image([x1 - tight_pad_x, y1 - tight_pad_y, x2 + tight_pad_x, y2 + tight_pad_y], (h, w), offset)
        if tight and bbox_area(tight) >= min_area:
            candidates.append({"bbox": tight, "score": min(1.0, 0.55 + min(0.35, bbox_area(tight) / max(1.0, h * w) * 12)), "label": "text_expanded_candidate"})

        row_pad_x_left = max(18.0, min(0.18 * w, bw * 1.5))
        row_pad_x_right = max(22.0, min(0.32 * w, bw * 3.0))
        row_pad_y = max(6.0, bh * 0.9)
        row = _clip_bbox_to_image([x1 - row_pad_x_left, y1 - row_pad_y, x2 + row_pad_x_right, y2 + row_pad_y], (h, w), offset)
        if row and bbox_area(row) >= min_area:
            rw, rh = row[2] - row[0], row[3] - row[1]
            if rh <= max(48, 0.18 * h) and rw <= 0.85 * w:
                candidates.append({"bbox": row, "score": min(1.0, 0.48 + min(0.28, bbox_area(row) / max(1.0, h * w) * 5)), "label": "row_container_candidate"})
    return candidates[:max_results]


def _query_prior_candidates(image_bgr: np.ndarray, offset: tuple[int, int], query: str | None, max_results: int) -> list[dict[str, Any]]:
    tokens = _tokenize(query)
    if not tokens:
        return []
    h, w = image_bgr.shape[:2]
    y_values = sorted({0, 8, 15, 30, 45, 60, 75, 90, 120, 150, 180, 210, 240, 255, 285, 315, 330, 345, 375, 405})
    y_values = [y for y in y_values if y < h - 8]
    right_slots = [(w - 50, w), (w - 96, w - 46), (w - 142, w - 92), (w - 188, w - 138)]
    left_slots = [(0, 52), (14, 66), (42, 94), (76, 128)]
    center_color_slots = [(int(0.62 * w), int(0.62 * w) + 34), (int(0.66 * w), int(0.66 * w) + 34)]

    slot_groups: list[tuple[str, list[tuple[int, int]], int]] = []
    if tokens & {"close", "exit", "dismiss"}:
        slot_groups.append(("window_close_prior", [right_slots[0]], 40))
    if tokens & {"minimize", "minimise"}:
        slot_groups.append(("window_minimize_prior", [right_slots[2], right_slots[1]], 38))
    if tokens & {"maximize", "maximise", "restore"}:
        slot_groups.append(("window_maximize_prior", [right_slots[1], right_slots[2]], 38))
    if tokens & {"settings", "more", "download", "downloads", "account", "profile", "shortcut", "toolbar", "pin", "tab"}:
        slot_groups.append(("right_toolbar_prior", [right_slots[0], right_slots[1]], 42))
    if tokens & {"refresh", "reload", "save", "undo", "redo", "back", "forward"}:
        slot_groups.append(("left_toolbar_prior", [left_slots[0], left_slots[1]], 40))
    if tokens & {"fill", "red", "blue", "green", "color", "colour"}:
        slot_groups.append(("color_toolbar_prior", center_color_slots, 32))

    candidates = []
    for label, slots, height in slot_groups:
        for y in y_values:
            for x1, x2 in slots:
                bbox = _clip_bbox_to_image([x1, y, x2, y + height], (h, w), offset)
                if bbox:
                    candidates.append({"bbox": bbox, "score": 0.94, "label": label})
                if len(candidates) >= max_results:
                    return candidates
    return candidates[:max_results]


def _rect_controls(image_bgr: np.ndarray, offset: tuple[int, int], min_area: float, max_results: int) -> list[dict[str, Any]]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 35, 130)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape[:2]
    ox, oy = offset
    boxes = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        area = float(bw * bh)
        if area < min_area or area > 0.45 * w * h or bw < 5 or bh < 5:
            continue
        aspect = bw / max(bh, 1)
        if not 0.12 <= aspect <= 14.0:
            continue
        contour_area = abs(float(cv2.contourArea(contour)))
        extent = contour_area / max(area, 1.0)
        score = min(1.0, 0.35 + extent + 0.5 * min(0.2, area / max(1.0, w * h)))
        boxes.append({"bbox": [x + ox, y + oy, x + bw + ox, y + bh + oy], "score": score, "area": area, "label": "ui_rect_candidate"})
    return sorted(boxes, key=lambda item: item["score"], reverse=True)[:max_results]


def _icon_regions(image_bgr: np.ndarray, offset: tuple[int, int], min_area: float, max_results: int) -> list[dict[str, Any]]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    mask = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 7)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
    boxes = _contour_boxes(mask, offset, max(8.0, min_area * 0.5), 0.08, max_results * 4)
    h, w = gray.shape[:2]
    filtered = []
    for item in boxes:
        x1, y1, x2, y2 = item["bbox"]
        bw, bh = x2 - x1, y2 - y1
        aspect = bw / max(bh, 1)
        if 0.2 <= aspect <= 5.0 and bw <= 0.35 * w and bh <= 0.35 * h:
            item["label"] = "icon_candidate"
            filtered.append(item)
    return sorted(filtered, key=lambda item: item["area"], reverse=True)[:max_results]


def _ocr_candidates(
    image_path: str | Path,
    bbox: list[int] | None,
    max_results: int,
    engine: str,
    languages: list[str] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from . import ocr as ocr_tool

    image = load_image(image_path)
    if bbox is None:
        offset = (0, 0)
        image_shape = (image.height, image.width)
    else:
        offset = (int(bbox[0]), int(bbox[1]))
        image_shape = (max(1, int(bbox[3] - bbox[1])), max(1, int(bbox[2] - bbox[0])))
    content, _, _ = ocr_tool.run(image_path, bbox=bbox, engine=engine, languages=languages or ["en"], max_regions=max_results)
    candidates = []
    for span in content.get("spans") or []:
        bbox_value = span.get("bbox")
        if not isinstance(bbox_value, list) or len(bbox_value) != 4:
            continue
        text = str(span.get("text", ""))
        score = float(span.get("score", 0.0))
        candidates.append({
            "bbox": bbox_value,
            "score": score,
            "label": "ocr_text",
            "text": text,
            "ocr_source": span.get("source"),
        })
        x1, y1, x2, y2 = [float(v) for v in bbox_value]
        height = max(1.0, y2 - y1)
        tight = _clip_bbox_to_image([x1 - height * 0.8, y1 - height * 0.55, x2 + height * 0.8, y2 + height * 0.55], image_shape, offset)
        if tight:
            candidates.append({"bbox": tight, "score": min(1.0, score + 0.08), "label": "ocr_text_expanded", "text": text, "ocr_source": span.get("source")})
        row = _clip_bbox_to_image([x1 - max(18.0, height * 2.0), y1 - height * 0.9, x2 + max(24.0, height * 4.0), y2 + height * 0.9], image_shape, offset)
        if row:
            row_w, row_h = row[2] - row[0], row[3] - row[1]
            if row_h <= max(56, 0.2 * image_shape[0]) and row_w <= 0.9 * image_shape[1]:
                candidates.append({"bbox": row, "score": min(1.0, score + 0.03), "label": "ocr_row_container", "text": text, "ocr_source": span.get("source")})
    for region in content.get("candidate_regions") or []:
        bbox_value = region.get("bbox")
        if not isinstance(bbox_value, list) or len(bbox_value) != 4:
            continue
        candidates.append({"bbox": bbox_value, "score": 0.1, "label": "ocr_candidate_region", "text": str(region.get("text", ""))})
    meta = {
        "ocr_engine": content.get("engine"),
        "ocr_available": bool(content.get("available")),
        "ocr_errors": content.get("errors") or [],
    }
    return candidates[:max_results], meta


def _source_counts(candidates: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in candidates:
        for source in item.get("sources") or [item.get("source")]:
            if source:
                counts[str(source)] = counts.get(str(source), 0) + 1
    return counts


def _ui_fused(
    image_bgr: np.ndarray,
    offset: tuple[int, int],
    min_area: float,
    max_results: int,
    query: str | None,
    *,
    image_path: str | Path,
    bbox: list[int] | None,
    include_ocr: bool,
    ocr_engine: str,
    ocr_languages: list[str] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_limit = max(max_results * 3, 60)
    candidates: list[dict[str, Any]] = []
    candidates.extend(_tag_candidates(_query_prior_candidates(image_bgr, offset, query, source_limit), "query_prior", score_boost=0.04))
    candidates.extend(_tag_candidates(_ui_elements(image_bgr, offset, min_area, source_limit), "ui_edge", score_boost=0.1))
    candidates.extend(_tag_candidates(_rect_controls(image_bgr, offset, min_area, source_limit), "ui_rect", score_boost=0.08))
    candidates.extend(_tag_candidates(_text_regions(image_bgr, offset, min_area, source_limit), "text_visual", score_boost=0.12))
    expanded_text = _expanded_text_candidates(image_bgr, offset, min_area, source_limit)
    candidates.extend(_tag_candidates([item for item in expanded_text if item.get("label") != "row_container_candidate"], "text_expanded", score_boost=0.1))
    candidates.extend(_tag_candidates([item for item in expanded_text if item.get("label") == "row_container_candidate"], "row_container", score_boost=0.05))
    candidates.extend(_tag_candidates(_icon_regions(image_bgr, offset, min_area, source_limit), "icon_visual", score_boost=0.05))
    candidates.extend(_tag_candidates(_layout(image_bgr, offset, min_area, max_results), "layout", score_boost=0.02))

    ocr_meta: dict[str, Any] = {"ocr_fused": False}
    if include_ocr:
        try:
            ocr_items, ocr_meta = _ocr_candidates(image_path, bbox, source_limit, ocr_engine, ocr_languages)
            ocr_meta["ocr_fused"] = True
            candidates.extend(_tag_candidates(ocr_items, "ocr_text", score_boost=0.18))
        except Exception as exc:
            ocr_meta = {"ocr_fused": False, "ocr_error": f"{type(exc).__name__}: {exc}"}

    deduped = _dedupe_candidates(candidates)
    ranked = _rank_candidates(deduped, image_bgr, query, max_results)
    return ranked, {
        "candidate_pool_size": len(deduped),
        "candidate_sources": _source_counts(ranked),
        **ocr_meta,
    }


def _select_mode(mode: str | None, query: str | None) -> str:
    selected = (mode or "layout").lower().strip()
    query_mode = (query or "").lower().strip()
    if selected in {"", "auto", "layout"} and query_mode in KNOWN_MODE_ALIASES:
        return query_mode
    return selected or "layout"


def _diagram_regions(image_bgr: np.ndarray, offset: tuple[int, int], min_area: float, max_results: int, query: str | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_limit = max(max_results * 3, 60)
    candidates: list[dict[str, Any]] = []
    candidates.extend(_tag_candidates(_layout(image_bgr, offset, max(20.0, min_area), source_limit), "diagram_region", score_boost=0.05))
    candidates.extend(_tag_candidates(_text_regions(image_bgr, offset, max(8.0, min_area * 0.4), source_limit), "text_visual", score_boost=0.1))
    candidates.extend(_tag_candidates(_expanded_text_candidates(image_bgr, offset, max(8.0, min_area * 0.4), source_limit), "text_expanded", score_boost=0.06))
    candidates.extend(_tag_candidates(_icon_regions(image_bgr, offset, max(8.0, min_area * 0.5), source_limit), "icon_visual", score_boost=0.03))
    deduped = _dedupe_candidates(candidates)
    ranked = _rank_candidates(deduped, image_bgr, query, max_results)
    return ranked, {
        "candidate_pool_size": len(deduped),
        "candidate_sources": _source_counts(ranked),
    }


def _bars(image_bgr: np.ndarray, offset: tuple[int, int], min_area: float, max_results: int) -> list[dict[str, Any]]:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    # Keep saturated colored regions, common for chart bars. This is heuristic.
    mask = cv2.inRange(hsv, np.array([0, 35, 35]), np.array([179, 255, 255]))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    boxes = _contour_boxes(mask, offset, min_area, 0.5, max_results * 2)
    candidates = []
    for item in boxes:
        x1, y1, x2, y2 = item["bbox"]
        w, h = x2 - x1, y2 - y1
        if h >= 8 and w >= 4:
            item["label"] = "bar_candidate"
            item["height"] = h
            candidates.append(item)
    return sorted(candidates, key=lambda item: (item["bbox"][0], item["bbox"][1]))[:max_results]


def _color(image_bgr: np.ndarray, offset: tuple[int, int], min_area: float, max_results: int, color: str | list[int], tolerance: int) -> list[dict[str, Any]]:
    rgb = np.array(parse_color(color), dtype=np.float32)
    rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    dist = np.sqrt(np.sum((rgb_image - rgb) ** 2, axis=2))
    mask = (dist <= tolerance).astype(np.uint8) * 255
    boxes = _contour_boxes(mask, offset, min_area, 0.9, max_results)
    for item in boxes:
        item["label"] = "color_region"
        item["target_color"] = list(map(int, rgb))
        item["tolerance"] = tolerance
    return boxes


def run(
    image_path: str | Path,
    mode: str = "layout",
    bbox: list[float] | None = None,
    query: str | None = None,
    max_results: int = 30,
    min_area: float = 80,
    color: str | list[int] | None = None,
    tolerance: int = 40,
    include_ocr: bool = False,
    ocr_engine: str = "auto",
    ocr_languages: list[str] | None = None,
    **_: Any,
) -> tuple[dict[str, Any], dict[str, Any], list[int] | None]:
    image = load_image(image_path)
    region, offset, clipped = _clip_region(image, bbox)
    image_bgr = pil_to_cv(region)
    selected = _select_mode(mode, query)
    mode_meta: dict[str, Any] = {}
    if selected in TEXT_MODE_ALIASES:
        detections = _text_regions(image_bgr, offset, min_area, max_results)
        used_mode = "text"
    elif selected in UI_MODE_ALIASES:
        detections, mode_meta = _ui_fused(
            image_bgr,
            offset,
            min_area,
            max_results,
            query,
            image_path=image_path,
            bbox=clipped,
            include_ocr=include_ocr,
            ocr_engine=ocr_engine,
            ocr_languages=ocr_languages,
        )
        used_mode = "ui"
    elif selected in BAR_MODE_ALIASES:
        detections = _bars(image_bgr, offset, min_area, max_results)
        used_mode = "bar"
    elif selected in DIAGRAM_MODE_ALIASES:
        detections, mode_meta = _diagram_regions(image_bgr, offset, min_area, max_results, query)
        used_mode = "diagram"
    elif selected in COLOR_MODE_ALIASES:
        if color is None:
            raise ValueError("color mode requires color")
        detections = _color(image_bgr, offset, min_area, max_results, color, tolerance)
        used_mode = "color"
    else:
        detections = _layout(image_bgr, offset, min_area, max_results)
        used_mode = "layout"
    content = {
        "mode": used_mode,
        "query": query,
        "image_size": list(image_size(image)),
        "region_size": [region.width, region.height],
        "detections": detections,
        "count": len(detections),
        **mode_meta,
    }
    return content, {}, clipped
