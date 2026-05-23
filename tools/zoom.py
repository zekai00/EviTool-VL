"""Zoom tool built on top of crop."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from .common import crop_pil, image_size, load_image, save_image, stable_hash


def run(
    image_path: str | Path,
    bbox: list[float],
    out_dir: str | Path | None = None,
    scale: float = 2.0,
    pad: int = 0,
    resample: str = "bicubic",
    evidence_id: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    if scale <= 0:
        raise ValueError("scale must be positive")
    image = load_image(image_path)
    cropped, clipped = crop_pil(image, bbox, pad=pad)
    method = Image.Resampling.BICUBIC if resample == "bicubic" else Image.Resampling.NEAREST
    zoomed = cropped.resize((max(1, int(cropped.width * scale)), max(1, int(cropped.height * scale))), method)
    stem = evidence_id or stable_hash({"tool": "zoom", "image": str(image_path), "bbox": clipped, "scale": scale})
    zoom_path = save_image(zoomed, out_dir, stem=f"{stem}_zoom")
    content = {
        "original_size": list(image_size(image)),
        "crop_size": [cropped.width, cropped.height],
        "zoom_size": [zoomed.width, zoomed.height],
        "scale": scale,
        "pad": pad,
    }
    artifacts = {"zoom_path": zoom_path} if zoom_path else {}
    return content, artifacts, clipped
