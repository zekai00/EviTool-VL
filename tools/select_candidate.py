"""Virtual candidate-selection tool for GUI grounding traces.

The tool records that the model selected a candidate id from a prior detect
observation. It is intentionally lightweight: real environments should resolve
candidate ids from state, while offline traces may provide the resolved bbox.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import bbox_center, clip_bbox, image_size, load_image, point_in_bbox


def run(
    image_path: str | Path,
    candidate_id: str,
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
    content: dict[str, Any] = {
        "candidate_id": str(candidate_id),
        "bbox": clipped,
        "label": label,
        "selected": True,
        "message": "Candidate selected from a prior detect observation. Cite this evidence id in the final answer.",
    }
    if point is not None:
        content["point"] = [float(point[0]), float(point[1])]
        content["inside_image"] = point_in_bbox(point, [0, 0, size[0], size[1]])
    return content, {}, clipped
