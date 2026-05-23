"""Visualization and marking tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import ImageDraw, ImageFont

from .common import bbox_center, clip_bbox, image_size, load_image, save_image, stable_hash


COLORS = ["red", "lime", "cyan", "yellow", "magenta", "orange", "white"]


def _draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], label: str, color: str) -> None:
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    x, y = xy
    draw.text((x + 2, y + 2), label, fill=color, font=font)


def run(
    image_path: str | Path,
    bboxes: list[list[float]] | None = None,
    bbox: list[float] | None = None,
    points: list[list[float]] | None = None,
    labels: list[str] | None = None,
    out_dir: str | Path | None = None,
    evidence_id: str | None = None,
    width: int = 3,
    **_: Any,
) -> tuple[dict[str, Any], dict[str, Any], list[int] | None]:
    image = load_image(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    size = image_size(image)
    boxes = []
    if bbox is not None:
        boxes.append(bbox)
    if bboxes:
        boxes.extend(bboxes)
    clipped_boxes = []
    for i, box in enumerate(boxes):
        clipped = clip_bbox(box, size)
        clipped_boxes.append(clipped)
        color = COLORS[i % len(COLORS)]
        draw.rectangle(tuple(clipped), outline=color, width=width)
        label = labels[i] if labels and i < len(labels) else f"box_{i}"
        _draw_label(draw, (clipped[0], clipped[1]), label, color)
    if points:
        for i, point in enumerate(points):
            x, y = int(point[0]), int(point[1])
            color = COLORS[(i + len(clipped_boxes)) % len(COLORS)]
            r = max(3, width * 2)
            draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=width)
            label = labels[len(clipped_boxes) + i] if labels and len(clipped_boxes) + i < len(labels) else f"pt_{i}"
            _draw_label(draw, (x + r, y + r), label, color)
    stem = evidence_id or stable_hash({"tool": "visualize", "image": str(image_path), "bboxes": clipped_boxes, "points": points})
    path = save_image(image, out_dir, stem=f"{stem}_mark")
    content = {"image_size": list(size), "bboxes": clipped_boxes, "points": points or []}
    artifacts = {"marked_path": path} if path else {}
    return content, artifacts, clipped_boxes[0] if clipped_boxes else None
