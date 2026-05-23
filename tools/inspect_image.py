"""Image inspection tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .common import load_image, pil_to_cv


def run(image_path: str | Path, **_: Any) -> tuple[dict[str, Any], dict[str, Any], None]:
    image = load_image(image_path)
    bgr = pil_to_cv(image)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    arr = np.array(image.convert("RGB"))
    channel_mean = arr.reshape(-1, 3).mean(axis=0).tolist()
    channel_std = arr.reshape(-1, 3).std(axis=0).tolist()
    content = {
        "path": str(image_path),
        "mode": image.mode,
        "size": [image.width, image.height],
        "aspect_ratio": image.width / max(1, image.height),
        "channel_mean_rgb": [round(float(x), 3) for x in channel_mean],
        "channel_std_rgb": [round(float(x), 3) for x in channel_std],
        "grayscale_mean": round(float(gray.mean()), 3),
        "grayscale_std": round(float(gray.std()), 3),
        "non_white_ratio": round(float((gray < 245).mean()), 6),
        "non_black_ratio": round(float((gray > 10).mean()), 6),
    }
    return content, {}, None
