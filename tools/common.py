"""Shared image and geometry helpers for local visual tools."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image


BBox = list[int]
Point = list[float]


def load_image(path: str | Path) -> Image.Image:
    image = Image.open(path)
    if image.mode not in {"RGB", "RGBA", "L"}:
        image = image.convert("RGB")
    return image


def to_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB") if image.mode != "RGB" else image


def pil_to_cv(image: Image.Image) -> np.ndarray:
    rgb = np.array(to_rgb(image))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def cv_to_pil(image: np.ndarray) -> Image.Image:
    if image.ndim == 2:
        return Image.fromarray(image)
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def image_size(image: Image.Image) -> tuple[int, int]:
    return int(image.width), int(image.height)


def maybe_denormalize_bbox(bbox: Iterable[float], size: tuple[int, int]) -> list[float]:
    values = [float(v) for v in bbox]
    if len(values) != 4:
        raise ValueError(f"bbox must contain 4 values, got {len(values)}")
    width, height = size
    if all(0.0 <= v <= 1.0 for v in values):
        x1, y1, x2, y2 = values
        return [x1 * width, y1 * height, x2 * width, y2 * height]
    return values


def xywh_to_xyxy(bbox: Iterable[float]) -> list[float]:
    x, y, w, h = [float(v) for v in bbox]
    return [x, y, x + w, y + h]


def xyxy_to_xywh(bbox: Iterable[float]) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return [x1, y1, x2 - x1, y2 - y1]


def clip_bbox(bbox: Iterable[float], size: tuple[int, int], pad: int = 0) -> BBox:
    width, height = size
    x1, y1, x2, y2 = maybe_denormalize_bbox(bbox, size)
    x1 -= pad
    y1 -= pad
    x2 += pad
    y2 += pad
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1 = max(0, min(width, int(math.floor(x1))))
    y1 = max(0, min(height, int(math.floor(y1))))
    x2 = max(0, min(width, int(math.ceil(x2))))
    y2 = max(0, min(height, int(math.ceil(y2))))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"empty bbox after clipping: {[x1, y1, x2, y2]}")
    return [x1, y1, x2, y2]


def crop_pil(image: Image.Image, bbox: Iterable[float], pad: int = 0) -> tuple[Image.Image, BBox]:
    clipped = clip_bbox(bbox, image_size(image), pad=pad)
    return image.crop(tuple(clipped)), clipped


def bbox_area(bbox: Iterable[float]) -> float:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_center(bbox: Iterable[float]) -> Point:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]


def bbox_iou(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    denom = bbox_area([ax1, ay1, ax2, ay2]) + bbox_area([bx1, by1, bx2, by2]) - inter
    return inter / denom if denom > 0 else 0.0


def point_in_bbox(point: Iterable[float], bbox: Iterable[float]) -> bool:
    x, y = [float(v) for v in point]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return x1 <= x <= x2 and y1 <= y <= y2


def parse_color(value: str | list[int] | tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return tuple(int(max(0, min(255, v))) for v in value)
    if not isinstance(value, str):
        raise ValueError("color must be '#rrggbb', 'r,g,b', a known color name, or [r,g,b]")
    text = value.strip().lower()
    named = {
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "red": (255, 0, 0),
        "green": (0, 128, 0),
        "blue": (0, 0, 255),
        "yellow": (255, 255, 0),
        "orange": (255, 165, 0),
        "purple": (128, 0, 128),
        "gray": (128, 128, 128),
        "grey": (128, 128, 128),
        "cyan": (0, 255, 255),
        "magenta": (255, 0, 255),
        "brown": (165, 42, 42),
    }
    if text in named:
        return named[text]
    if text.startswith("#") and len(text) == 7:
        return int(text[1:3], 16), int(text[3:5], 16), int(text[5:7], 16)
    nums = [int(x) for x in re.findall(r"\d+", text)]
    if len(nums) >= 3:
        return tuple(max(0, min(255, n)) for n in nums[:3])
    raise ValueError(f"cannot parse color: {value}")


def stable_hash(value: Any, length: int = 10) -> str:
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha1(data).hexdigest()[:length]


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_image(image: Image.Image, out_dir: str | Path | None, stem: str, suffix: str = ".png") -> str | None:
    if out_dir is None:
        return None
    directory = ensure_dir(out_dir)
    path = directory / f"{stem}{suffix}"
    image.save(path)
    return str(path)


def sort_boxes_reading_order(boxes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(boxes, key=lambda item: (item["bbox"][1], item["bbox"][0], -bbox_area(item["bbox"])))
