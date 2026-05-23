"""Virtual click/select tool for GUI grounding traces.

This tool does not interact with a real UI. It records the point or bbox that a
model wants to select, turning it into evidence that can be referenced in the
final answer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import bbox_center, clip_bbox, image_size, load_image, point_in_bbox


def run(
    image_path: str | Path,
    bbox: list[float] | None = None,
    point: list[float] | None = None,
    label: str | None = None,
    **_: Any,
) -> tuple[dict[str, Any], dict[str, Any], list[int] | None]:
    image = load_image(image_path)
    size = image_size(image)
    clipped = clip_bbox(bbox, size) if bbox is not None else None
    if point is None and clipped is not None:
        point = bbox_center(clipped)
    if point is None:
        raise ValueError("click requires either point or bbox")
    inside_image = point_in_bbox(point, [0, 0, size[0], size[1]])
    content = {
        "point": [float(point[0]), float(point[1])],
        "bbox": clipped,
        "label": label,
        "inside_image": inside_image,
        "message": "Virtual click/select target recorded. Use this evidence id in the final bbox answer if appropriate.",
    }
    return content, {}, clipped
