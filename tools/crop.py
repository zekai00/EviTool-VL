"""Crop tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import crop_pil, image_size, load_image, save_image, stable_hash


def run(image_path: str | Path, bbox: list[float], out_dir: str | Path | None = None, pad: int = 0, evidence_id: str | None = None) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    image = load_image(image_path)
    cropped, clipped = crop_pil(image, bbox, pad=pad)
    stem = evidence_id or stable_hash({"tool": "crop", "image": str(image_path), "bbox": clipped})
    crop_path = save_image(cropped, out_dir, stem=f"{stem}_crop")
    content = {
        "original_size": list(image_size(image)),
        "crop_size": [cropped.width, cropped.height],
        "pad": pad,
    }
    artifacts = {"crop_path": crop_path} if crop_path else {}
    return content, artifacts, clipped
