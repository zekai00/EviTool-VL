"""Geometry and chart-oriented measurement tool."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .common import bbox_area, bbox_center, bbox_iou, clip_bbox, image_size, load_image, point_in_bbox, xyxy_to_xywh


def _measure_bbox(bbox: list[float], size: tuple[int, int]) -> dict[str, Any]:
    clipped = clip_bbox(bbox, size)
    x1, y1, x2, y2 = clipped
    w, h = x2 - x1, y2 - y1
    return {
        "bbox": clipped,
        "xywh": xyxy_to_xywh(clipped),
        "width": w,
        "height": h,
        "area": w * h,
        "center": bbox_center(clipped),
        "area_ratio": (w * h) / max(1, size[0] * size[1]),
    }


def _point_distance(a: list[float], b: list[float]) -> float:
    return math.sqrt((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2)


def _angle(a: list[float], b: list[float]) -> float:
    return math.degrees(math.atan2(float(b[1]) - float(a[1]), float(b[0]) - float(a[0])))


def _relative(a: list[float], b: list[float]) -> dict[str, Any]:
    ca, cb = bbox_center(a), bbox_center(b)
    dx, dy = cb[0] - ca[0], cb[1] - ca[1]
    return {
        "center_a": ca,
        "center_b": cb,
        "dx_b_minus_a": dx,
        "dy_b_minus_a": dy,
        "b_is_left_of_a": cb[0] < ca[0],
        "b_is_right_of_a": cb[0] > ca[0],
        "b_is_above_a": cb[1] < ca[1],
        "b_is_below_a": cb[1] > ca[1],
        "center_distance": _point_distance(ca, cb),
        "iou": bbox_iou(a, b),
    }


def run(
    image_path: str | Path,
    bbox: list[float] | None = None,
    bboxes: list[list[float]] | None = None,
    points: list[list[float]] | None = None,
    mode: str = "auto",
    **_: Any,
) -> tuple[dict[str, Any], dict[str, Any], list[int] | None]:
    image = load_image(image_path)
    size = image_size(image)
    clipped = clip_bbox(bbox, size) if bbox is not None else None
    boxes = []
    if bbox is not None:
        boxes.append(clipped)
    if bboxes:
        boxes.extend(clip_bbox(b, size) for b in bboxes)

    content: dict[str, Any] = {"image_size": list(size), "mode": mode}
    if boxes:
        measures = [_measure_bbox(box, size) for box in boxes]
        content["bboxes"] = measures
        if len(boxes) >= 2:
            content["pairwise"] = [_relative(boxes[i], boxes[j]) for i in range(len(boxes)) for j in range(i + 1, len(boxes))]
        if mode in {"compare_height", "bars", "bar_height"}:
            ranked = sorted(measures, key=lambda item: item["height"], reverse=True)
            content["height_ranking"] = ranked
            content["tallest"] = ranked[0] if ranked else None
    if points:
        content["points"] = points
        if len(points) >= 2:
            content["point_distances"] = [
                {"from": i, "to": j, "distance": _point_distance(points[i], points[j]), "angle_degrees": _angle(points[i], points[j])}
                for i in range(len(points))
                for j in range(i + 1, len(points))
            ]
        if boxes:
            content["point_in_bbox"] = [
                {"point_index": i, "bbox_index": j, "inside": point_in_bbox(point, box)}
                for i, point in enumerate(points)
                for j, box in enumerate(boxes)
            ]
    return content, {}, clipped
