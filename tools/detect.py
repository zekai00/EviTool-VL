"""Lightweight visual detection tools based on OpenCV."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .common import bbox_area, crop_pil, image_size, load_image, parse_color, pil_to_cv, sort_boxes_reading_order


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
    **_: Any,
) -> tuple[dict[str, Any], dict[str, Any], list[int] | None]:
    image = load_image(image_path)
    region, offset, clipped = _clip_region(image, bbox)
    image_bgr = pil_to_cv(region)
    selected = (query or mode or "layout").lower().strip()
    if selected in {"text", "text_region", "ocr"}:
        detections = _text_regions(image_bgr, offset, min_area, max_results)
        used_mode = "text"
    elif selected in {"ui", "button", "icon", "screen", "gui"}:
        detections = _ui_elements(image_bgr, offset, min_area, max_results)
        used_mode = "ui"
    elif selected in {"bar", "chart", "bars"}:
        detections = _bars(image_bgr, offset, min_area, max_results)
        used_mode = "bar"
    elif selected in {"color", "colour"}:
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
    }
    return content, {}, clipped
